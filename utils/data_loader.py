# ============================================================
# HSR ENGINE v4 – Concavity-based Storage for SRTM 30 m
# - HSR_static: topographic storage volume (m³)
# - HSR_rain: rainfall-adjusted storage volume (m³)
# - Parcel-level zonal stats using Grid-demo.shp
# ============================================================

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pysheds.grid import Grid
from scipy import ndimage
from scipy.ndimage import grey_closing
import matplotlib.pyplot as plt
from rasterstats import zonal_stats

import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    "DEM_PATH": "/content/drive/MyDrive/Updated_LU/Dem-demo.tif",
    "CN_PATH": "/content/drive/MyDrive/Updated_LU/CN.tif",
    "PARCELS_PATH": "/content/drive/MyDrive/Updated_LU/Grid/Grid-demo.shp",
    "OUTPUT_DIR": "/content/drive/MyDrive/Updated_LU/HSR_v4",
    "RAINFALL_MM": 100.0,           # storm depth (mm)
    "TARGET_RESOLUTION": 30,        # m
    "CONCAVITY_WINDOW": 7,          # neighbourhood size (cells) for concavity (7x7 ~ 210 m)
    "DEBUG": True
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# -------------------------
# UTILS
# -------------------------
def debug_plot(arr, title):
    if not CONFIG["DEBUG"]:
        return
    plt.figure(figsize=(7, 6))
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmin, vmax = np.percentile(finite, [2, 98])
        plt.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar()
        print(f"{title}  ->  min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}, mean={np.nanmean(arr):.3f}")
    else:
        plt.imshow(arr, cmap="viridis")
        plt.colorbar()
        print(f"{title}: all values are NaN")
    plt.title(title)
    plt.axis("off")
    plt.show()

def reproject_raster(src_path, dst_path, target_crs, res):
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=res
        )
        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": w,
            "height": h,
            "compress": "lzw"
        })

        is_int = np.issubdtype(src.dtypes[0], np.integer)
        resampling = Resampling.nearest if is_int else Resampling.bilinear

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    rasterio.band(src, i),
                    rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling
                )

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        prof = src.profile
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof

def save_raster(arr, profile, out_path):
    prof = profile.copy()
    prof.update({
        "dtype": "float32",
        "nodata": np.nan,
        "count": 1
    })
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)

# ============================================================
# STEP 1 — REPROJECT DEM & CN
# ============================================================
print("\nSTEP 1 — Reprojecting DEM and CN...")

with rasterio.open(CONFIG["DEM_PATH"]) as dem_src:
    bounds = dem_src.bounds
    lon_center = (bounds.left + bounds.right) / 2

utm_zone = int((lon_center + 180) / 6) + 1
target_crs = f"EPSG:326{utm_zone}"

dem_utm_path = os.path.join(CONFIG["OUTPUT_DIR"], "dem_utm.tif")
cn_utm_path  = os.path.join(CONFIG["OUTPUT_DIR"], "cn_utm.tif")
res = CONFIG["TARGET_RESOLUTION"]

reproject_raster(CONFIG["DEM_PATH"], dem_utm_path, target_crs, res)
reproject_raster(CONFIG["CN_PATH"],  cn_utm_path,  target_crs, res)

dem, dem_profile = load_raster(dem_utm_path)
cn,  cn_profile  = load_raster(cn_utm_path)

cell_size = dem_profile["transform"][0]
cell_area = cell_size * cell_size

debug_plot(dem, "DEM (UTM)")
debug_plot(cn,  "Curve Number (UTM)")

# ============================================================
# STEP 2 — FLOW-ROUTING DEM (for accumulation)
# ============================================================
print("\nSTEP 2 — Preparing DEM for flow routing...")

grid = Grid.from_raster(dem_utm_path, data_name="dem")
dem_raw = grid.read_raster(dem_utm_path)

dem_filled = grid.fill_depressions(dem_raw)
dem_flat   = grid.resolve_flats(dem_filled)

# ============================================================
# STEP 3 — TOPOGRAPHIC CONCAVITY (STORAGE DEPTH)
# ============================================================
print("\nSTEP 3 — Detecting concavities (storage depth)...")

dem_array = np.array(dem_raw, dtype="float64")
mask_valid = np.isfinite(dem_array)
fill_value = np.nanmean(dem_array)

dem_for_morph = dem_array.copy()
dem_for_morph[~mask_valid] = fill_value

win = CONFIG["CONCAVITY_WINDOW"]
dem_closed = grey_closing(dem_for_morph, size=(win, win))
dem_closed[~mask_valid] = np.nan

storage_depth = dem_closed - dem_array
storage_depth[~mask_valid] = np.nan
storage_depth = np.where(storage_depth > 0, storage_depth, 0)

debug_plot(storage_depth, "Topographic concavity depth (m)")

positive = storage_depth[storage_depth > 0]
if positive.size == 0:
    print("  No concavities detected at this scale; HSR will be zero.")
    dep_labels = np.zeros_like(storage_depth, dtype="int32")
    n_deps = 0
else:
    # dynamic threshold: keep deeper half of concavities
    DEP_THRESHOLD = np.percentile(positive, 50)
    print(f"  Concavity depth stats: min={positive.min():.3f}, max={positive.max():.3f}, median={np.median(positive):.3f}")
    print(f"  Using DEP_THRESHOLD = {DEP_THRESHOLD:.3f} m")

    concavity_mask = storage_depth >= DEP_THRESHOLD
    dep_labels, n_deps = ndimage.label(concavity_mask, structure=np.ones((3,3)))
    print(f"  Number of concavity patches (storage units): {n_deps}")

debug_plot(dep_labels.astype(float), "Concavity patches (labels)")

# ============================================================
# STEP 4 — STATIC STORAGE VOLUME (HSR_static)
# ============================================================
print("\nSTEP 4 — Computing static storage volume (HSR_static)...")

if n_deps == 0:
    storage_volumes = np.array([])
    HSR_static = np.zeros_like(dem_array, dtype="float64")
else:
    storage_volumes = ndimage.sum(storage_depth * cell_area,
                                  dep_labels,
                                  index=np.arange(1, n_deps + 1))
    HSR_static = np.zeros_like(dem_array, dtype="float64")
    for dep_id, vol in enumerate(storage_volumes, start=1):
        HSR_static[dep_labels == dep_id] = vol

save_raster(HSR_static, dem_profile, os.path.join(CONFIG["OUTPUT_DIR"], "HSR_static.tif"))
debug_plot(HSR_static, "HSR_static (m³)")

# ============================================================
# STEP 5 — RUNOFF (SCS–CN)
# ============================================================
print("\nSTEP 5 — Computing runoff (SCS–CN)...")

rain = CONFIG["RAINFALL_MM"]
CN = np.where((cn > 0) & np.isfinite(cn), cn, np.nan)

S = (25400.0 / CN) - 254.0
Ia = 0.2 * S

runoff = np.where(
    np.isfinite(CN) & (rain > Ia),
    ((rain - Ia) ** 2) / ((rain - Ia) + S),
    0.0
)
runoff = np.where(np.isfinite(runoff), runoff, 0.0)

debug_plot(runoff, "Runoff depth (mm)")

runoff_path = os.path.join(CONFIG["OUTPUT_DIR"], "runoff_weights.tif")
save_raster(runoff, dem_profile, runoff_path)

# ============================================================
# STEP 6 — WEIGHTED FLOW ACCUMULATION
# ============================================================
print("\nSTEP 6 — Weighted flow accumulation...")

runoff_grid = grid.read_raster(runoff_path)
fdir = grid.flowdir(dem_flat)
wacc = grid.accumulation(fdir, weights=runoff_grid)
wacc_array = np.array(wacc, dtype="float64")

debug_plot(np.log1p(wacc_array), "Weighted accumulation (log1p)")

# ============================================================
# STEP 7 — INFLOW VOLUME & HSR_rain
# ============================================================
print("\nSTEP 7 — Computing inflow and HSR_rain...")

if n_deps == 0:
    inflow_mm = np.array([])
    inflow_volumes = np.array([])
    HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
else:
    # Approximate inflow as MAX accumulation within each concavity
    inflow_mm = ndimage.maximum(wacc_array,
                                dep_labels,
                                index=np.arange(1, n_deps + 1))
    inflow_volumes = inflow_mm * cell_area / 1000.0  # mm * m² → m³

    HSR_rain = np.minimum(storage_volumes, inflow_volumes)
    HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
    for dep_id, val in enumerate(HSR_rain, start=1):
        HSR_rain_map[dep_labels == dep_id] = val

save_raster(HSR_rain_map, dem_profile, os.path.join(CONFIG["OUTPUT_DIR"], "HSR_rain.tif"))
debug_plot(HSR_rain_map, "HSR_rain (m³)")

# ============================================================
# STEP 8 — PARCEL-LEVEL ZONAL STATISTICS
# ============================================================
print("\nSTEP 8 — Parcel-level HSR (zonal statistics)...")

if not os.path.exists(CONFIG["PARCELS_PATH"]):
    raise FileNotFoundError(f"Parcel file not found: {CONFIG['PARCELS_PATH']}")

parcels = gpd.read_file(CONFIG["PARCELS_PATH"])

static_stats = zonal_stats(
    parcels,
    os.path.join(CONFIG["OUTPUT_DIR"], "HSR_static.tif"),
    stats=["sum", "mean", "max"],
    nodata=np.nan
)
rain_stats = zonal_stats(
    parcels,
    os.path.join(CONFIG["OUTPUT_DIR"], "HSR_rain.tif"),
    stats=["sum", "mean", "max"],
    nodata=np.nan
)

parcels["HSR_static_sum"]  = [d["sum"]  for d in static_stats]
parcels["HSR_static_mean"] = [d["mean"] for d in static_stats]
parcels["HSR_static_max"]  = [d["max"]  for d in static_stats]

parcels["HSR_rain_sum"]  = [d["sum"]  for d in rain_stats]
parcels["HSR_rain_mean"] = [d["mean"] for d in rain_stats]
parcels["HSR_rain_max"]  = [d["max"]  for d in rain_stats]

out_parcels = os.path.join(CONFIG["OUTPUT_DIR"], "Grid_HSR_parcels.shp")
parcels.to_file(out_parcels)

print("\n✅ HSR ENGINE v4 COMPLETE")
print("   Outputs:")
print("   - HSR_static.tif      (concavity storage, m³)")
print("   - HSR_rain.tif        (rainfall-adjusted storage, m³)")
print("   - Grid_HSR_parcels.shp (parcel-level HSR stats)")
print(f"\nAll saved in: {CONFIG['OUTPUT_DIR']}")
