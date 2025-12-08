# models/hsr_model.py
# ============================================================
# HSR ENGINE v4 – Concavity-based Storage for SRTM 30 m
# Adapted for Streamlit app:
# - HSR_static: topographic storage volume (m³)
# - HSR_rain: rainfall-adjusted storage volume (m³)
# - Parcel-level zonal stats using grid-network.geojson
# ============================================================

import os
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from pysheds.grid import Grid
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from rasterstats import zonal_stats
from scipy import ndimage
from scipy.ndimage import grey_closing


# -------------------------
# Helpers
# -------------------------
def _reproject_raster(src_path, dst_path, target_crs, res):
    """Reproject raster to target CRS and resolution (m)."""
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


def _load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        prof = src.profile
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof


def _save_raster(arr, profile, out_path):
    prof = profile.copy()
    prof.update({"dtype": "float32", "nodata": np.nan, "count": 1})
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


# -------------------------
# Main HSR function
# -------------------------
def run_hsr_analysis(
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    rainfall_mm: float = 100.0,
    concavity_window: int = 7,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Run HSR ENGINE v4 and return:
      - parcels_hsr: GeoDataFrame with HSR_* fields
      - diagnostics: dict with key metrics for UI
    """

    # Use a temporary working folder (no permanent files)
    with tempfile.TemporaryDirectory() as tmpdir:
        # ============================================================
        # STEP 1 — REPROJECT DEM & CN TO UTM @ 30 m
        # ============================================================
        with rasterio.open(dem_path) as dem_src:
            bounds = dem_src.bounds
            lon_center = (bounds.left + bounds.right) / 2.0

        utm_zone = int((lon_center + 180) / 6) + 1
        target_crs = f"EPSG:326{utm_zone}"
        target_res = 30  # metres

        dem_utm_path = os.path.join(tmpdir, "dem_utm.tif")
        cn_utm_path = os.path.join(tmpdir, "cn_utm.tif")

        _reproject_raster(dem_path, dem_utm_path, target_crs, target_res)
        _reproject_raster(cn_path, cn_utm_path, target_crs, target_res)

        dem, dem_profile = _load_raster(dem_utm_path)
        cn, _ = _load_raster(cn_utm_path)

        cell_size = abs(dem_profile["transform"][0])
        cell_area = cell_size * cell_size

        # ============================================================
        # STEP 2 — FLOW-ROUTING DEM (for accumulation)
        # ============================================================
        grid = Grid.from_raster(dem_utm_path, data_name="dem")
        dem_raw = grid.read_raster(dem_utm_path)

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
        storage_depth = np.where(storage_depth > 0, storage_depth, 0)

        positive = storage_depth[storage_depth > 0]
        if positive.size == 0:
            depth_threshold = 0.0
            dep_labels = np.zeros_like(storage_depth, dtype="int32")
            n_deps = 0
        else:
            # Same as Colab: keep deeper half of concavities (50th percentile)
            depth_threshold = float(np.percentile(positive, 50.0))
            concavity_mask = storage_depth >= depth_threshold
            dep_labels, n_deps = ndimage.label(
                concavity_mask, structure=np.ones((3, 3))
            )

        # ============================================================
        # STEP 4 — STATIC STORAGE VOLUME (HSR_static)
        # ============================================================
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

        hsr_static_path = os.path.join(tmpdir, "HSR_static.tif")
        _save_raster(HSR_static, dem_profile, hsr_static_path)

        # ============================================================
        # STEP 5 — RUNOFF (SCS–CN)
        # ============================================================
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

        runoff_path = os.path.join(tmpdir, "runoff_weights.tif")
        _save_raster(runoff, dem_profile, runoff_path)

        # ============================================================
        # STEP 6 — WEIGHTED FLOW ACCUMULATION
        # ============================================================
        runoff_grid = grid.read_raster(runoff_path)
        fdir = grid.flowdir(dem_flat)
        wacc = grid.accumulation(fdir, weights=runoff_grid)
        wacc_array = np.array(wacc, dtype="float64")

        # ============================================================
        # STEP 7 — INFLOW VOLUME & HSR_rain
        # ============================================================
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

        hsr_rain_path = os.path.join(tmpdir, "HSR_rain.tif")
        _save_raster(HSR_rain_map, dem_profile, hsr_rain_path)

        # ============================================================
        # STEP 8 — PARCEL-LEVEL ZONAL STATISTICS
        # ============================================================
        if not os.path.exists(parcels_path):
            raise FileNotFoundError(f"Parcel file not found: {parcels_path}")

        parcels_orig = gpd.read_file(parcels_path)
        if parcels_orig.crs is None:
            raise ValueError("Parcels layer has no CRS defined.")

        # Work in same CRS as rasters (UTM)
        parcels = parcels_orig.to_crs(dem_profile["crs"])

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

        parcels["HSR_static_sum"] = [d["sum"] for d in static_stats]
        parcels["HSR_static_mean"] = [d["mean"] for d in static_stats]
        parcels["HSR_static_max"] = [d["max"] for d in static_stats]

        parcels["HSR_rain_sum"] = [d["sum"] for d in rain_stats]
        parcels["HSR_rain_mean"] = [d["mean"] for d in rain_stats]
        parcels["HSR_rain_max"] = [d["max"] for d in rain_stats]

        # Return parcels in original CRS (for mapping with other models)
        parcels_out = parcels.to_crs(parcels_orig.crs)

        # Diagnostics for UI
        diagnostics = {
            "rainfall_mm": rain,
            "concavity_window": int(concavity_window),
            "cell_size_m": float(cell_size),
            "depth_threshold_m": float(depth_threshold),
            "num_patches": int(n_deps),
            "total_static_storage": float(
                np.nansum(parcels_out["HSR_static_sum"])
            ),
            "total_rain_filled": float(
                np.nansum(parcels_out["HSR_rain_sum"])
            ),
        }

        return parcels_out, diagnostics
