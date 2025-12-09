# models/hsr_model.py
"""
Hydrological Storage Role (HSR) model for the Streamlit app.

This is a direct adaptation of your HSR ENGINE v4:

- Reprojects DEM and CN to a common UTM grid (30 m)
- Detects concavities using grey closing (7x7 default)
- Computes:
    * HSR_static  = topographic storage volume (m³)
    * HSR_rain    = rainfall-adjusted storage volume (m³)
- Aggregates parcel-level zonal stats.

Inputs:
    dem_path      : path to DEM raster (any projection)
    cn_path       : path to Curve Number raster
    parcels_path  : path to parcel polygons (GeoJSON/Shapefile/etc.)
    rainfall_mm   : storm depth (mm)
    concavity_win : neighborhood size in cells (default 7)
"""

import os
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pysheds.grid import Grid
from scipy import ndimage
from scipy.ndimage import grey_closing
from rasterstats import zonal_stats


def _load_raster(path):
    """Read a single-band raster as float64 + profile, respecting nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        prof = src.profile
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof


def _save_raster(arr, profile, out_path):
    """Save float raster with NaN as nodata."""
    prof = profile.copy()
    prof.update(
        dtype="float32",
        nodata=np.nan,
        count=1,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def _reproject_raster(src_path, dst_path, target_crs, res):
    """Reproject any raster to target_crs and resolution."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=res,
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
        )

        is_int = np.issubdtype(src.dtypes[0], np.integer)
        resampling = Resampling.nearest if is_int else Resampling.bilinear

        data = src.read(1).astype("float64")
        dst_data = np.empty((height, width), dtype="float64")

        reproject(
            source=data,
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

    return dst_path


def run_hsr_analysis(
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    rainfall_mm: float = 100.0,
    concavity_window: int = 7,
):
    """
    Run the full HSR ENGINE v4 logic and return:

    - parcels_hsr : GeoDataFrame with HSR_* fields
    - diagnostics : dict with key diagnostics (CRS, concavities, etc.)
    """

    # ------------------------------------------------------------------
    # TEMP WORKING DIRECTORY
    # ------------------------------------------------------------------
    tmp_dir = os.path.join(tempfile.gettempdir(), "hsr_engine")
    os.makedirs(tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — REPROJECT DEM & CN TO UTM (30 m)
    # ------------------------------------------------------------------
    with rasterio.open(dem_path) as dem_src:
        bounds = dem_src.bounds
        lon_center = (bounds.left + bounds.right) / 2.0
        utm_zone = int((lon_center + 180) / 6) + 1
        target_crs = f"EPSG:326{utm_zone}"

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
    # STEP 2 — FLOW-ROUTING DEM (for accumulation)
    # ------------------------------------------------------------------
    grid = Grid.from_raster(dem_utm_path, data_name="dem")
    dem_raw = grid.read_raster(dem_utm_path)

    dem_filled = grid.fill_depressions(dem_raw)
    dem_flat = grid.resolve_flats(dem_filled)

    # ------------------------------------------------------------------
    # STEP 3 — TOPOGRAPHIC CONCAVITY (STORAGE DEPTH)
    # ------------------------------------------------------------------
    dem_array = np.array(dem_raw, dtype="float64")
    mask_valid = np.isfinite(dem_array)
    fill_value = np.nanmean(dem_array)

    dem_for_morph = dem_array.copy()
    dem_for_morph[~mask_valid] = fill_value

    win = concavity_window
    dem_closed = grey_closing(dem_for_morph, size=(win, win))
    dem_closed[~mask_valid] = np.nan

    storage_depth = dem_closed - dem_array
    storage_depth[~mask_valid] = np.nan
    storage_depth = np.where(storage_depth > 0, storage_depth, 0)

    positive = storage_depth[storage_depth > 0]
    if positive.size == 0:
        dep_labels = np.zeros_like(storage_depth, dtype="int32")
        n_deps = 0
        dep_threshold = None
    else:
        # dynamic threshold: keep deeper half of concavities
        dep_threshold = float(np.percentile(positive, 50))
        concavity_mask = storage_depth >= dep_threshold
        dep_labels, n_deps = ndimage.label(
            concavity_mask, structure=np.ones((3, 3), dtype=int)
        )

    # ------------------------------------------------------------------
    # STEP 4 — STATIC STORAGE VOLUME (HSR_static)
    # ------------------------------------------------------------------
    if n_deps == 0:
        storage_volumes = np.array([])
        HSR_static = np.zeros_like(dem_array, dtype="float64")
    else:
        storage_volumes = ndimage.sum(
            storage_depth * cell_area,
            dep_labels,
            index=np.arange(1, n_deps + 1),
        )
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

    runoff = np.where(
        np.isfinite(CN) & (rain > Ia),
        ((rain - Ia) ** 2) / ((rain - Ia) + S),
        0.0,
    )
    runoff = np.where(np.isfinite(runoff), runoff, 0.0)

    runoff_path = os.path.join(tmp_dir, "runoff_weights.tif")
    _save_raster(runoff, dem_profile, runoff_path)

    # ------------------------------------------------------------------
    # STEP 6 — WEIGHTED FLOW ACCUMULATION
    # ------------------------------------------------------------------
    runoff_grid = grid.read_raster(runoff_path)
    fdir = grid.flowdir(dem_flat)
    wacc = grid.accumulation(fdir, weights=runoff_grid)
    wacc_array = np.array(wacc, dtype="float64")

    # ------------------------------------------------------------------
    # STEP 7 — INFLOW VOLUME & HSR_rain
    # ------------------------------------------------------------------
    if n_deps == 0:
        inflow_mm = np.array([])
        inflow_volumes = np.array([])
        HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
    else:
        inflow_mm = ndimage.maximum(
            wacc_array,
            dep_labels,
            index=np.arange(1, n_deps + 1),
        )
        inflow_volumes = inflow_mm * cell_area / 1000.0  # mm * m² → m³

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

    static_stats = zonal_stats(
        parcels,
        hsr_static_path,
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )
    rain_stats = zonal_stats(
        parcels,
        hsr_rain_path,
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )

    parcels["HSR_static_sum"] = [d.get("sum", 0.0) for d in static_stats]
    parcels["HSR_static_mean"] = [d.get("mean", 0.0) for d in static_stats]
    parcels["HSR_static_max"] = [d.get("max", 0.0) for d in static_stats]

    parcels["HSR_rain_sum"] = [d.get("sum", 0.0) for d in rain_stats]
    parcels["HSR_rain_mean"] = [d.get("mean", 0.0) for d in rain_stats]
    parcels["HSR_rain_max"] = [d.get("max", 0.0) for d in rain_stats]

    parcels["Rainfall_mm"] = rain

    # Simple normalized index (optional)
    def _norm(x):
        x = np.asarray(x, dtype="float64")
        if x.size == 0:
            return x
        amin = np.nanmin(x)
        amax = np.nanmax(x)
        if not np.isfinite(amin) or not np.isfinite(amax) or np.isclose(amin, amax):
            return np.zeros_like(x)
        return (x - amin) / (amax - amin)

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

    return parcels, diagnostics
