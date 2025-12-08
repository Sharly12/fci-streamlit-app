# models/hsr_model.py
"""
Hydrological Storage Role (HSR) ENGINE v4
----------------------------------------
- Reprojects DEM & CN to UTM
- Detects concavities (storage depth)
- Computes:
    HSR_static (m³)  – purely topographic storage
    HSR_rain   (m³)  – rainfall-filled storage using SCS–CN + weighted accumulation
- Aggregates parcel-level zonal statistics

This is a direct refactor of your Colab HSR ENGINE v4 for use in the web app.
"""

import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pysheds.grid import Grid
from scipy import ndimage
from scipy.ndimage import grey_closing
from rasterstats import zonal_stats


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    out_dir = _project_root() / "outputs" / "individual" / "hsr"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def _save_raster(arr: np.ndarray, profile: dict, out_path: str):
    prof = profile.copy()
    prof.update(
        {
            "dtype": "float32",
            "nodata": np.nan,
            "count": 1,
            "compress": "lzw",
        }
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def _reproject_raster(src_path: str, dst_path: str, target_crs: str, res: float):
    """Reproject raster to target_crs and resolution."""
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


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def run_hsr_analysis(
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    rainfall_mm: float = 100.0,
    concavity_window: int = 7,
    target_resolution: float = 30.0,
    output_dir: str | None = None,
):
    """
    Run full HSR ENGINE v4.

    Parameters
    ----------
    dem_path : str
        Path to DEM raster (e.g. Dem-demo.tif)
    cn_path : str
        Path to Curve Number raster (CN.tif)
    parcels_path : str
        Path to parcels vector (GeoJSON / Shapefile)
    rainfall_mm : float
        Design storm depth (mm)
    concavity_window : int
        Neighbourhood size in cells (e.g. 7 for 7x7 window)
    target_resolution : float
        Target DEM/CN resolution (m)
    output_dir : str or None
        If None, defaults to outputs/individual/hsr under the repo.

    Returns
    -------
    parcels_gdf : GeoDataFrame
        Parcels with HSR_* attributes attached.
    diagnostics : dict
        Key numeric diagnostics for UI display.
    """
    if output_dir is None:
        out_dir = _default_output_dir()
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # STEP 1 – Reproject DEM & CN to UTM
    # ----------------------------------------------------------
    with rasterio.open(dem_path) as dem_src:
        bounds = dem_src.bounds
        lon_center = (bounds.left + bounds.right) / 2.0

    utm_zone = int((lon_center + 180.0) / 6.0) + 1
    target_crs = f"EPSG:326{utm_zone}"

    dem_utm_path = out_dir / "dem_utm.tif"
    cn_utm_path = out_dir / "cn_utm.tif"

    _reproject_raster(dem_path, str(dem_utm_path), target_crs, target_resolution)
    _reproject_raster(cn_path, str(cn_utm_path), target_crs, target_resolution)

    dem, dem_profile = _load_raster(str(dem_utm_path))
    cn, _ = _load_raster(str(cn_utm_path))

    cell_size = dem_profile["transform"][0]
    cell_area = float(cell_size * cell_size)

    # ----------------------------------------------------------
    # STEP 2 – Flow-routing DEM (for accumulation)
    # ----------------------------------------------------------
    grid = Grid.from_raster(str(dem_utm_path), data_name="dem")
    dem_raw = grid.read_raster(str(dem_utm_path))

    dem_filled = grid.fill_depressions(dem_raw)
    dem_flat = grid.resolve_flats(dem_filled)

    # ----------------------------------------------------------
    # STEP 3 – Topographic concavity (storage depth)
    # ----------------------------------------------------------
    dem_array = np.asarray(dem_raw, dtype="float64")
    mask_valid = np.isfinite(dem_array)
    fill_value = np.nanmean(dem_array)

    dem_for_morph = dem_array.copy()
    dem_for_morph[~mask_valid] = fill_value

    win = int(concavity_window)
    dem_closed = grey_closing(dem_for_morph, size=(win, win))
    dem_closed[~mask_valid] = np.nan

    storage_depth = dem_closed - dem_array
    storage_depth[~mask_valid] = np.nan
    storage_depth = np.where(storage_depth > 0, storage_depth, 0.0)

    positive = storage_depth[storage_depth > 0]
    if positive.size == 0:
        depth_threshold = 0.0
        dep_labels = np.zeros_like(storage_depth, dtype="int32")
        n_deps = 0
    else:
        # dynamic threshold: keep deeper half of concavities
        depth_threshold = float(np.percentile(positive, 50.0))
        concavity_mask = storage_depth >= depth_threshold
        dep_labels, n_deps = ndimage.label(concavity_mask, structure=np.ones((3, 3)))

    # ----------------------------------------------------------
    # STEP 4 – Static storage volume (HSR_static)
    # ----------------------------------------------------------
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

    hsr_static_path = out_dir / "HSR_static.tif"
    _save_raster(HSR_static, dem_profile, str(hsr_static_path))

    # ----------------------------------------------------------
    # STEP 5 – Runoff (SCS–CN)
    # ----------------------------------------------------------
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

    runoff_path = out_dir / "runoff_weights.tif"
    _save_raster(runoff, dem_profile, str(runoff_path))

    # ----------------------------------------------------------
    # STEP 6 – Weighted flow accumulation
    # ----------------------------------------------------------
    runoff_grid = grid.read_raster(str(runoff_path))
    fdir = grid.flowdir(dem_flat)
    wacc = grid.accumulation(fdir, weights=runoff_grid)
    wacc_array = np.asarray(wacc, dtype="float64")

    # ----------------------------------------------------------
    # STEP 7 – Inflow volume & HSR_rain
    # ----------------------------------------------------------
    if n_deps == 0:
        inflow_mm = np.array([])
        inflow_volumes = np.array([])
        HSR_rain_map = np.zeros_like(dem_array, dtype="float64")
    else:
        # Approximate inflow as MAX accumulation within each concavity
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

    hsr_rain_path = out_dir / "HSR_rain.tif"
    _save_raster(HSR_rain_map, dem_profile, str(hsr_rain_path))

    # ----------------------------------------------------------
    # STEP 8 – Parcel-level zonal statistics
    # ----------------------------------------------------------
    if not os.path.exists(parcels_path):
        raise FileNotFoundError(f"Parcel file not found: {parcels_path}")

    parcels = gpd.read_file(parcels_path)

    # Make sure CRS matches the HSR rasters
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS defined.")
    if parcels.crs.to_string() != dem_profile["crs"].to_string():
        parcels = parcels.to_crs(dem_profile["crs"])

    static_stats = zonal_stats(
        parcels,
        str(hsr_static_path),
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )
    rain_stats = zonal_stats(
        parcels,
        str(hsr_rain_path),
        stats=["sum", "mean", "max"],
        nodata=np.nan,
    )

    parcels["HSR_static_sum"] = [d["sum"] for d in static_stats]
    parcels["HSR_static_mean"] = [d["mean"] for d in static_stats]
    parcels["HSR_static_max"] = [d["max"] for d in static_stats]

    parcels["HSR_rain_sum"] = [d["sum"] for d in rain_stats]
    parcels["HSR_rain_mean"] = [d["mean"] for d in rain_stats]
    parcels["HSR_rain_max"] = [d["max"] for d in rain_stats]

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    total_static = float(np.nansum(HSR_static))
    total_rain = float(np.nansum(HSR_rain_map))

    diagnostics = {
        "rainfall_mm": rain,
        "cell_size_m": float(cell_size),
        "concavity_window": int(concavity_window),
        "depth_threshold_m": float(depth_threshold),
        "num_patches": int(n_deps),
        "total_static_storage": total_static,
        "total_rain_filled": total_rain,
        "hsr_static_path": str(hsr_static_path),
        "hsr_rain_path": str(hsr_rain_path),
        "utm_crs": target_crs,
    }

    return parcels, diagnostics
