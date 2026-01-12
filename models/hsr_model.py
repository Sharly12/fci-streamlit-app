# models/hsr_model.py
import os
import tempfile
import numpy as np
import rasterio
import geopandas as gpd

from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.ndimage import grey_closing
from scipy import ndimage
from pysheds.grid import Grid
from rasterstats import zonal_stats


def _load_raster(path):
    """Load raster into numpy array + profile, and convert nodata to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr, profile


def _save_raster(array, profile, out_path):
    """Save float raster as GeoTIFF."""
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array.astype("float32"), 1)


def _reproject_raster(src_path, dst_path, target_crs, target_res, resampling=Resampling.bilinear):
    """Reproject a raster to target CRS + resolution."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res,
        )

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=np.nan,
            dtype="float32",
            compress="deflate",
        )

        src_data = src.read(1).astype("float64")
        if src.nodata is not None:
            src_data[src_data == src.nodata] = np.nan

        dst_data = np.full((height, width), np.nan, dtype="float64")

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dst_data.astype("float32"), 1)


def _pick_utm_crs_from_raster(profile):
    """
    Choose UTM CRS based on raster center if raster is geographic.
    If raster is already projected, return that CRS unchanged.
    """
    crs = profile.get("crs", None)
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]

    # center coords in raster CRS
    cx = transform.c + (width * transform.a) / 2.0
    cy = transform.f + (height * transform.e) / 2.0

    if crs is not None and crs.is_projected:
        return crs

    # assume geographic lon/lat
    lon = cx
    lat = cy
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return rasterio.crs.CRS.from_epsg(epsg)


def _norm(x):
    x = np.asarray(x, dtype="float64")
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    y[~finite] = 0
    return y


def run_hsr_analysis(
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    rainfall_mm: float = 100.0,
    concavity_window: int = 7,
):
    # ------------------------------------------------------------------
    # TEMP WORKSPACE
    # ------------------------------------------------------------------
    tmp_dir = os.path.join(tempfile.gettempdir(), "hsr_engine")
    os.makedirs(tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — REPROJECT DEM + CN to common UTM grid (30 m)
    # ------------------------------------------------------------------
    _, dem_prof_raw = _load_raster(dem_path)
    target_crs = _pick_utm_crs_from_raster(dem_prof_raw)
    target_res = 30  # m

    dem_utm_path = os.path.join(tmp_dir, "dem_utm.tif")
    cn_utm_path = os.path.join(tmp_dir, "cn_utm.tif")

    _reproject_raster(dem_path, dem_utm_path, target_crs, target_res)
    _reproject_raster(cn_path, cn_utm_path, target_crs, target_res)

    dem, dem_profile = _load_raster(dem_utm_path)
    cn, _ = _load_raster(cn_utm_path)

    cell_size = dem_profile["transform"][0]
    cell_area = cell_size * cell_size

    # ------------------------------------------------------------------
    # STEP 2 — PREP (same logic as your original)
    # ------------------------------------------------------------------
    dem_array = np.array(dem, dtype="float64")
    dem_nanmask = ~np.isfinite(dem_array)
    dem_filled = dem_array.copy()
    dem_filled[dem_nanmask] = np.nanmin(dem_array[np.isfinite(dem_array)])

    # ------------------------------------------------------------------
    # STEP 3 — CONCAVITY DETECTION (grey closing)
    # ------------------------------------------------------------------
    w = int(concavity_window)
    if w % 2 == 0:
        w += 1

    closed = grey_closing(dem_filled, size=(w, w))
    dep_depth = closed - dem_filled

    dep_threshold = 0.01  # same as original file
    depressions = dep_depth > dep_threshold
    dep_labels, n_deps = ndimage.label(depressions)

    # ------------------------------------------------------------------
    # STEP 4 — HSR_static
    # ------------------------------------------------------------------
    if n_deps == 0:
        storage_volumes = np.array([])
        HSR_static = np.zeros_like(dem_array, dtype="float64")
    else:
        mean_depth = ndimage.mean(dep_depth, dep_labels, index=np.arange(1, n_deps + 1))
        mean_depth = np.array(mean_depth, dtype="float64")

        pixels = ndimage.sum(np.ones_like(dep_depth), dep_labels, index=np.arange(1, n_deps + 1))
        pixels = np.array(pixels, dtype="float64")

        storage_volumes = mean_depth * pixels * cell_area  # m³

        HSR_static = np.zeros_like(dem_array, dtype="float64")
        for dep_id, vol in enumerate(storage_volumes, start=1):
            HSR_static[dep_labels == dep_id] = vol

    hsr_static_path = os.path.join(tmp_dir, "HSR_static.tif")
    _save_raster(HSR_static, dem_profile, hsr_static_path)

    # ------------------------------------------------------------------
    # STEP 5 — RUNOFF (SCS–CN)
    # ------------------------------------------------------------------
    rain = float(rainfall_mm)
    CN = np.where((cn > 0) & np.isfinite(cn), cn, np.nan)

    S = (25400.0 / CN) - 254.0
    Ia = 0.2 * S
    Q = np.where(rain > Ia, ((rain - Ia) ** 2) / (rain + 0.8 * S), 0.0)

    runoff = np.where(np.isfinite(Q), Q, 0.0)

    runoff_path = os.path.join(tmp_dir, "runoff_weights.tif")
    _save_raster(runoff, dem_profile, runoff_path)

    # ------------------------------------------------------------------
    # STEP 6 — WEIGHTED FLOW ACCUMULATION
    # ------------------------------------------------------------------
    grid = Grid.from_raster(dem_utm_path, data_name="dem")
    dem_flat = grid.read_raster(dem_utm_path)

    runoff_grid = grid.read_raster(runoff_path)
    fdir = grid.flowdir(dem_flat)
    wacc = grid.accumulation(fdir, weights=runoff_grid)
    wacc_array = np.array(wacc, dtype="float64")

    # ------------------------------------------------------------------
    # STEP 7 — HSR_rain
    # ------------------------------------------------------------------
    if n_deps == 0:
        HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
    else:
        inflow_mm = ndimage.maximum(wacc_array, dep_labels, index=np.arange(1, n_deps + 1))
        inflow_volumes = inflow_mm * cell_area / 1000.0  # m³
        HSR_rain = np.minimum(storage_volumes, inflow_volumes)

        HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
        for dep_id, val in enumerate(HSR_rain, start=1):
            HSR_rain_map[dep_labels == dep_id] = val

    hsr_rain_path = os.path.join(tmp_dir, "HSR_rain.tif")
    _save_raster(HSR_rain_map, dem_profile, hsr_rain_path)

    # ------------------------------------------------------------------
    # STEP 8 — PARCEL-LEVEL ZONAL STATISTICS
    # ------------------------------------------------------------------
    if not os.path.exists(parcels_path):
        raise FileNotFoundError(f"Parcel file not found: {parcels_path}")

    parcels = gpd.read_file(parcels_path)

    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define it and re-save.")

    # Reproject parcels to UTM CRS used by HSR rasters
    if str(parcels.crs) != str(dem_profile["crs"]):
        parcels = parcels.to_crs(dem_profile["crs"])

    static_stats = zonal_stats(parcels, hsr_static_path, stats=["sum", "mean", "max"], nodata=np.nan)
    rain_stats = zonal_stats(parcels, hsr_rain_path, stats=["sum", "mean", "max"], nodata=np.nan)

    parcels["HSR_static_sum"] = [d.get("sum", 0.0) for d in static_stats]
    parcels["HSR_static_mean"] = [d.get("mean", 0.0) for d in static_stats]
    parcels["HSR_static_max"] = [d.get("max", 0.0) for d in static_stats]

    parcels["HSR_rain_sum"] = [d.get("sum", 0.0) for d in rain_stats]
    parcels["HSR_rain_mean"] = [d.get("mean", 0.0) for d in rain_stats]
    parcels["HSR_rain_max"] = [d.get("max", 0.0) for d in rain_stats]

    parcels["Rainfall_mm"] = rain

    # HSR index (normalized)
    parcels["HSR_static_norm"] = _norm(parcels["HSR_static_sum"].values)
    parcels["HSR_rain_norm"] = _norm(parcels["HSR_rain_sum"].values)
    parcels["HSR_index"] = 0.5 * parcels["HSR_static_norm"] + 0.5 * parcels["HSR_rain_norm"]

    diagnostics = {
        "utm_crs": str(dem_profile["crs"]),
        "cell_size_m": float(cell_size),
        "cell_area_m2": float(cell_area),
        "n_concavities": int(n_deps),
        "concavity_window": int(concavity_window),
        "depth_threshold_m": float(dep_threshold) if dep_threshold is not None else None,
        "rainfall_mm": rain,
    }

    # ✅ NEW (only change): return the file paths of outputs your code already generated
    outputs = {
        "workspace": tmp_dir,
        "dem_utm_tif": dem_utm_path,
        "cn_utm_tif": cn_utm_path,
        "runoff_tif": runoff_path,
        "hsr_static_tif": hsr_static_path,
        "hsr_rain_tif": hsr_rain_path,
    }

    return parcels, diagnostics, outputs
