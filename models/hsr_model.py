# models/hsr_model.py
# HSR ENGINE v4 – Concavity-based Storage for SRTM 30 m
# - HSR_static: topographic storage volume (m³)
# - HSR_rain: rainfall-adjusted storage volume (m³)
# - Parcel-level zonal stats using shared parcels (GeoJSON)

import numpy as np
import geopandas as gpd
from pathlib import Path
import tempfile

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pysheds.grid import Grid
from scipy import ndimage
from scipy.ndimage import grey_closing
from rasterstats import zonal_stats


def _reproject_raster(src_path, dst_path, target_crs, res):
    """Reproject a raster to target_crs at a given resolution."""
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=res,
        )
        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": w,
                "height": h,
                "compress": "lzw",
            }
        )

        is_int = np.issubdtype(src.dtypes[0], np.integer)
        resampling = Resampling.nearest if is_int else Resampling.bilinear

        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    rasterio.band(src, i),
                    rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )


def _load_raster(path):
    """Load a raster band as float64 array with NaNs for nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def run_hsr_analysis(
    rainfall_mm: float,
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    target_resolution: float = 30.0,
    concavity_window: int = 7,
):
    """
    Run the HSR engine and return:
      - parcels_gdf with HSR columns
      - diagnostics dict

    Parameters
    ----------
    rainfall_mm : float
        Storm depth in mm.
    dem_path : str
        Path to DEM raster (same as used by FCI).
    cn_path : str
        Path to Curve Number raster.
    parcels_path : str
        Path to parcels vector (GeoJSON, Shapefile, etc.).
    target_resolution : float
        Target DEM/CN resolution in meters after reprojection.
    concavity_window : int
        Window size (cells) for concavity detection (e.g., 7 for 7x7).
    """

    # ============================================================
    # STEP 1 — REPROJECT DEM & CN TO UTM
    # ============================================================
    with rasterio.open(dem_path) as dem_src:
        bounds = dem_src.bounds
        lon_center = (bounds.left + bounds.right) / 2.0
        utm_zone = int((lon_center + 180) / 6) + 1
        target_crs = f"EPSG:326{utm_zone}"

    tmp_dir = Path(tempfile.gettempdir()) / "hsr_engine"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dem_utm_path = tmp_dir / "dem_utm.tif"
    cn_utm_path = tmp_dir / "cn_utm.tif"

    _reproject_raster(dem_path, dem_utm_path, target_crs, target_resolution)
    _reproject_raster(cn_path, cn_utm_path, target_crs, target_resolution)

    dem, dem_profile = _load_raster(dem_utm_path)
    cn, _ = _load_raster(cn_utm_path)

    transform = dem_profile["transform"]
    cell_size = transform[0]
    cell_area = cell_size * cell_size

    # ============================================================
    # STEP 2 — FLOW-ROUTING DEM (for accumulation)
    # ============================================================
    grid = Grid.from_raster(str(dem_utm_path), data_name="dem")
    dem_raw = grid.read_raster(str(dem_utm_path))

    dem_filled = grid.fill_depressions(dem_raw)
    dem_flat = grid.resolve_flats(dem_filled)

    # ============================================================
    # STEP 3 — TOPOGRAPHIC CONCAVITY (STORAGE DEPTH)
    # ============================================================
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
    storage_depth = np.where(storage_depth > 0, storage_depth, 0.0)

    positive = storage_depth[storage_depth > 0]

    if positive.size == 0:
        # No concavities detected
        dep_labels = np.zeros_like(storage_depth, dtype="int32")
        n_deps = 0
        dep_threshold = None
        storage_volumes = np.array([])
        inflow_volumes = np.array([])
        HSR_static = np.zeros_like(dem_array, dtype="float64")
        HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
    else:
        # Dynamic threshold: keep deeper half of concavities
        dep_threshold = float(np.percentile(positive, 50))
        concavity_mask = storage_depth >= dep_threshold

        dep_labels, n_deps = ndimage.label(
            concavity_mask, structure=np.ones((3, 3), dtype="int32")
        )

        # ============================================================
        # STEP 4 — STATIC STORAGE VOLUME (HSR_static)
        # ============================================================
        storage_volumes = ndimage.sum(
            storage_depth * cell_area,
            dep_labels,
            index=np.arange(1, n_deps + 1),
        )

        HSR_static = np.zeros_like(dem_array, dtype="float64")
        for dep_id, vol in enumerate(storage_volumes, start=1):
            HSR_static[dep_labels == dep_id] = vol

        # ============================================================
        # STEP 5 — RUNOFF (SCS–CN)
        # ============================================================
        rain = rainfall_mm
        CN = np.where((cn > 0) & np.isfinite(cn), cn, np.nan)

        S = (25400.0 / CN) - 254.0
        Ia = 0.2 * S

        runoff = np.where(
            np.isfinite(CN) & (rain > Ia),
            ((rain - Ia) ** 2) / ((rain - Ia) + S),
            0.0,
        )
        runoff = np.where(np.isfinite(runoff), runoff, 0.0)

        # Save temporary runoff raster so we can reuse pysheds grid
        runoff_path = tmp_dir / "runoff_weights.tif"
        runoff_profile = dem_profile.copy()
        runoff_profile.update(dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(runoff_path, "w", **runoff_profile) as dst:
            dst.write(runoff.astype("float32"), 1)

        # ============================================================
        # STEP 6 — WEIGHTED FLOW ACCUMULATION
        # ============================================================
        runoff_grid = grid.read_raster(str(runoff_path))
        fdir = grid.flowdir(dem_flat)
        wacc = grid.accumulation(fdir, weights=runoff_grid)
        wacc_array = np.array(wacc, dtype="float64")

        # ============================================================
        # STEP 7 — INFLOW VOLUME & HSR_rain
        # ============================================================
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

    # ============================================================
    # STEP 8 — PARCEL-LEVEL ZONAL STATISTICS
    # ============================================================
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define a CRS.")

    # Reproject parcels to the UTM CRS of the DEM
    dem_crs = dem_profile["crs"]
    if parcels.crs.to_string() != dem_crs.to_string():
        parcels = parcels.to_crs(dem_crs)

    def _safe_get(d, key):
        return d.get(key, 0.0) if isinstance(d, dict) else 0.0

    # Static storage zonal stats
    static_stats = zonal_stats(
        vectors=parcels.geometry,
        raster=HSR_static,
        affine=dem_profile["transform"],
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )
    # Rainfall-adjusted storage zonal stats
    rain_stats = zonal_stats(
        vectors=parcels.geometry,
        raster=HSR_rain_map,
        affine=dem_profile["transform"],
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )

    parcels["HSR_static_sum"] = [_safe_get(d, "sum") for d in static_stats]
    parcels["HSR_static_mean"] = [_safe_get(d, "mean") for d in static_stats]
    parcels["HSR_static_max"] = [_safe_get(d, "max") for d in static_stats]

    parcels["HSR_rain_sum"] = [_safe_get(d, "sum") for d in rain_stats]
    parcels["HSR_rain_mean"] = [_safe_get(d, "mean") for d in rain_stats]
    parcels["HSR_rain_max"] = [_safe_get(d, "max") for d in rain_stats]

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    total_storage_m3 = float(np.sum(storage_volumes)) if storage_volumes.size else 0.0
    total_rain_storage_m3 = (
        float(np.sum(np.minimum(storage_volumes, inflow_volumes)))
        if (storage_volumes.size and inflow_volumes.size)
        else 0.0
    )

    diagnostics = {
        "rainfall_mm": float(rainfall_mm),
        "target_crs": str(dem_crs),
        "cell_size_m": float(cell_size),
        "cell_area_m2": float(cell_area),
        "concavity_window": int(concavity_window),
        "n_concavities": int(n_deps),
        "dep_threshold_m": float(dep_threshold) if dep_threshold is not None else None,
        "total_storage_m3": total_storage_m3,
        "total_rain_storage_m3": total_rain_storage_m3,
    }

    return parcels, diagnostics
