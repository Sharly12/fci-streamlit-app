# models/pec_model.py
"""
Parcel Elevation Context (PEC) model for the Streamlit app.

This implementation avoids WhiteboxTools so it runs safely on
Streamlit Cloud (no external executables).

Core indicators (per parcel):
- dem_min, dem_max, dem_mean, relief
- slope_mean  (derived from DEM using numpy gradients)
- PREI (local relative elevation = DEM - neighbourhood mean)
- HAND_proxy (elevation above local minimum within a given radius)

These are combined into PEC classes:

1. Low-lying Depressed (Retention Priority)
2. Flat & Pressured (High Flood Exposure Risk)
3. Locally High & Disconnected
4. Moderate / Context-Dependent

A rainfall depth (mm) can be supplied to slightly shift PREI and HAND
thresholds, following your original Colab logic.
"""

from pathlib import Path
import os

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import mapping
from scipy.ndimage import uniform_filter, minimum_filter


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _minmax(arr):
    arr = np.asarray(arr, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _classify_pec_static(row):
    """Baseline (no rainfall adjustment) PEC classification."""
    prei = row["prei"]
    hand = row["hand_score"]
    relief = row["relief"]
    flat_flag = bool(row["slp_mean"] < 1.5)

    if prei <= -0.5 and hand <= 1.5 and relief <= 3:
        return "Low-lying Depressed (Retention Priority)"
    elif flat_flag and hand <= 3.0:
        return "Flat & Pressured (High Flood Exposure Risk)"
    elif prei > 0.5 and hand > 5.0:
        return "Locally High & Disconnected"
    else:
        return "Moderate / Context-Dependent"


def _classify_pec_rainfall(row, rainfall_mm: float):
    """
    Rainfall-aware PEC classification, following your previous rules:
      - PREI is reduced slightly with rainfall.
      - HAND thresholds increase slightly with rainfall.
    """
    prei_val = row["prei"] - 0.002 * rainfall_mm
    hand = row["hand_score"]
    relief = row["relief"]
    flat_flag = bool(row["slp_mean"] < 1.5)

    low_thresh = 1.5 + 0.01 * rainfall_mm
    med_thresh = 3.0 + 0.01 * rainfall_mm

    if prei_val <= -0.5 and hand <= low_thresh and relief <= 3:
        return "Low-lying Depressed (Retention Priority)"
    elif flat_flag and hand <= med_thresh:
        return "Flat & Pressured (High Flood Exposure Risk)"
    elif prei_val > 0.5 and hand > 5.0:
        return "Locally High & Disconnected"
    else:
        return "Moderate / Context-Dependent"


def _write_float_raster(arr, meta_template, transform, path):
    """Write a float32 GeoTIFF from a numpy array using a template profile."""
    nodata = -9999.0
    prof = meta_template.copy()
    prof.update(
        height=arr.shape[0],
        width=arr.shape[1],
        transform=transform,
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw",
    )
    data = arr.astype("float32").copy()
    data[~np.isfinite(data)] = nodata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data, 1)
    return str(path), nodata


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    rainfall_mm: float = 0.0,
    prei_radius_m: float = 250.0,
    hand_radius_m: float = 150.0,
):
    """
    Run full PEC workflow for the app (no Whitebox).

    Args
    ----
    dem_path        : path to DEM (GeoTIFF)
    parcels_path    : path to parcel grid (GeoJSON / Shapefile)
    rainfall_mm     : rainfall depth for rainfall-aware PEC.
                      Set 0 for static PEC.
    prei_radius_m   : neighbourhood radius for PREI (m)
    hand_radius_m   : neighbourhood radius for HAND proxy (m)

    Returns
    -------
    parcels_pec : GeoDataFrame with PEC indicators and class
    diagnostics : dict
    """

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "individual" / "pec"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load parcels & clip DEM to study extent
    # ------------------------------------------------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define it and re-save.")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_meta = src.meta.copy()
        # Reproject parcels to DEM CRS and clip extent
        parcels = parcels.to_crs(dem_crs)
        boundary_geom = [mapping(parcels.unary_union)]
        out_image, out_transform = mask(src, boundary_geom, crop=True)

    # DEM array (single band)
    dem = out_image[0].astype("float64")
    nodata_val = dem_meta.get("nodata", None)

    if nodata_val is not None:
        dem = np.where(dem == nodata_val, np.nan, dem)
    # Remove weird extreme negatives
    dem = np.where(dem < -1000, np.nan, dem)

    # If everything is NaN, bail out
    if not np.isfinite(dem).any():
        raise ValueError("DEM appears empty or all nodata within parcel extent.")

    cellsize = float(abs(out_transform.a))

    # Ensure grid_id
    parcels = parcels.reset_index(drop=True)
    if "grid_id" not in parcels.columns:
        parcels["grid_id"] = np.arange(1, len(parcels) + 1, dtype="int32")

    grid_gdf = parcels.copy()

    # ------------------------------------------------------------------
    # 2. Compute slope from DEM
    # ------------------------------------------------------------------
    dem_filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    # gradient expects rows=y, cols=x
    dy, dx = np.gradient(dem_filled, cellsize, cellsize)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~np.isfinite(dem)] = np.nan

    # ------------------------------------------------------------------
    # 3. PREI (DEM - neighbourhood mean)
    # ------------------------------------------------------------------
    prei_radius_pix = max(1, int(prei_radius_m / cellsize))
    prei_kernel = prei_radius_pix * 2 + 1

    local_mean = uniform_filter(dem_filled, size=prei_kernel, mode="nearest")
    prei_grid = dem_filled - local_mean
    prei_grid[~np.isfinite(dem)] = np.nan

    # ------------------------------------------------------------------
    # 4. HAND proxy (DEM - local minimum)
    # ------------------------------------------------------------------
    hand_radius_pix = max(1, int(hand_radius_m / cellsize))
    hand_kernel = hand_radius_pix * 2 + 1

    local_min = minimum_filter(dem_filled, size=hand_kernel, mode="nearest")
    hand_grid = dem_filled - local_min
    hand_grid[hand_grid < 0] = 0.0
    hand_grid[~np.isfinite(dem)] = np.nan

    # ------------------------------------------------------------------
    # 5. Write intermediate rasters & zonal statistics
    # ------------------------------------------------------------------
    dem_clip_path, dem_nodata = _write_float_raster(
        dem, dem_meta, out_transform, out_dir / "pec_dem_clip.tif"
    )
    slope_path, slope_nodata = _write_float_raster(
        slope_deg, dem_meta, out_transform, out_dir / "pec_slope_deg.tif"
    )
    prei_path, prei_nodata = _write_float_raster(
        prei_grid, dem_meta, out_transform, out_dir / "pec_prei.tif"
    )
    hand_path, hand_nodata = _write_float_raster(
        hand_grid, dem_meta, out_transform, out_dir / "pec_hand_proxy.tif"
    )

    # DEM stats
    dem_stats = zonal_stats(
        grid_gdf, dem_clip_path, stats=["min", "max", "mean"], nodata=dem_nodata
    )
    grid_gdf["dem_min"] = [d["min"] for d in dem_stats]
    grid_gdf["dem_max"] = [d["max"] for d in dem_stats]
    grid_gdf["dem_mean"] = [d["mean"] for d in dem_stats]

    # Relief
    grid_gdf["relief"] = grid_gdf["dem_max"] - grid_gdf["dem_min"]

    # Slope stats
    slp_stats = zonal_stats(
        grid_gdf, slope_path, stats=["mean"], nodata=slope_nodata
    )
    grid_gdf["slp_mean"] = [d["mean"] for d in slp_stats]
    grid_gdf["flat_flag"] = grid_gdf["slp_mean"] < 1.5

    # PREI stats (relative elevation)
    prei_stats = zonal_stats(
        grid_gdf, prei_path, stats=["mean", "min", "max"], nodata=prei_nodata
    )
    grid_gdf["rel_mean"] = [d["mean"] for d in prei_stats]
    grid_gdf["rel_min"] = [d["min"] for d in prei_stats]
    grid_gdf["rel_max"] = [d["max"] for d in prei_stats]

    # HAND proxy stats
    hand_stats = zonal_stats(
        grid_gdf, hand_path, stats=["min", "mean"], nodata=hand_nodata
    )
    grid_gdf["hand_min"] = [d["min"] for d in hand_stats]
    grid_gdf["hand_mean"] = [d["mean"] for d in hand_stats]

    # PREI index & hand_score (align with your earlier naming)
    grid_gdf["prei"] = grid_gdf["rel_mean"]
    grid_gdf["hand_score"] = grid_gdf["hand_min"]

    # Fill NaNs to avoid weird comparisons in classification
    cols_to_fill = ["prei", "hand_score", "relief", "slp_mean"]
    grid_gdf[cols_to_fill] = grid_gdf[cols_to_fill].fillna(0.0)

    # ------------------------------------------------------------------
    # 6. PEC classification (static or rainfall-aware)
    # ------------------------------------------------------------------
    if rainfall_mm is None or rainfall_mm <= 0:
        pec_classes = grid_gdf.apply(_classify_pec_static, axis=1)
    else:
        pec_classes = grid_gdf.apply(
            lambda r: _classify_pec_rainfall(r, rainfall_mm=rainfall_mm),
            axis=1,
        )

    grid_gdf["pec_class"] = pec_classes

    pec_mapping = {
        "Low-lying Depressed (Retention Priority)": 1,
        "Flat & Pressured (High Flood Exposure Risk)": 2,
        "Locally High & Disconnected": 3,
        "Moderate / Context-Dependent": 4,
    }
    grid_gdf["pec_code"] = grid_gdf["pec_class"].map(pec_mapping).astype("Int32")

    # Some normalized helper indices if you want later plots
    grid_gdf["prei_norm"] = _minmax(grid_gdf["prei"].values)
    grid_gdf["hand_norm"] = _minmax(grid_gdf["hand_score"].values)

    diagnostics = {
        "n_parcels": int(len(grid_gdf)),
        "dem_res_m": float(cellsize),
        "prei_radius_m": float(prei_radius_m),
        "hand_radius_m": float(hand_radius_m),
        "prei_radius_pixels": int(prei_radius_pix),
        "hand_radius_pixels": int(hand_radius_pix),
        "rainfall_mm": float(rainfall_mm),
        "pec_class_counts": grid_gdf["pec_class"].value_counts().to_dict(),
    }

    return grid_gdf, diagnostics
