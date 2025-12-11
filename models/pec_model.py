# models/pec_model.py
"""
Parcel Elevation Context (PEC) model for the Streamlit app.

Re-implements your original PEC workflow WITHOUT WhiteboxTools, using:
- rasterio + numpy + scipy + pysheds + rasterstats
- Same indicators: relief, PREI (relative elevation), HAND-like metric
- Same parcel-level classification rules and class labels.
"""

import os
import tempfile
from typing import Tuple, Dict, Any

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import mapping
from scipy.ndimage import uniform_filter, distance_transform_edt
from pysheds.grid import Grid


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _safe_stat_list(stats, key, fill=0.0):
    """Extract a list of 'key' values from zonal_stats output."""
    vals = []
    for d in stats:
        v = d.get(key)
        if v is None or np.isnan(v):
            vals.append(fill)
        else:
            vals.append(float(v))
    return vals


def _compute_slope_deg(dem: np.ndarray, cellsize: float) -> np.ndarray:
    """Simple Horn-style slope in degrees from DEM."""
    arr = dem.astype("float64")
    # Fill NaNs with global mean for stability
    mean_val = np.nanmean(arr)
    arr = np.where(np.isfinite(arr), arr, mean_val)

    dzdx = np.gradient(arr, axis=1) / cellsize
    dzdy = np.gradient(arr, axis=0) / cellsize
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~np.isfinite(slope_deg)] = 0.0
    return slope_deg.astype("float32")


def _compute_relative_elevation(
    dem: np.ndarray, cellsize: float, radius_m: float
) -> np.ndarray:
    """DEM – local mean within radius_m neighbourhood."""
    arr = dem.astype("float64")
    mean_val = np.nanmean(arr)
    arr_filled = np.where(np.isfinite(arr), arr, mean_val)

    radius_px = max(1, int(radius_m / cellsize))
    # kernel size similar to your 250 m mean filter
    size = 2 * radius_px + 1

    local_mean = uniform_filter(arr_filled, size=size, mode="nearest")
    rel = arr - local_mean
    rel[~np.isfinite(rel)] = np.nan
    return rel.astype("float32")


def _compute_hand_like(
    dem: np.ndarray, transform, stream_threshold: float
) -> np.ndarray:
    """
    HAND-like metric (height above nearest drainage) without Whitebox.

    Steps:
    1) Use pysheds to build D8 flow accumulation.
    2) Define streams as cells with accumulation >= stream_threshold.
    3) Use Euclidean distance transform to find nearest stream cell
       for every pixel and subtract its elevation.
    """
    # Write DEM temporarily for pysheds
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dem = os.path.join(tmpdir, "pec_dem.tif")
        # Reuse transform & metadata from array via rasterio in-memory write
        height, width = dem.shape
        with rasterio.open(
            tmp_dem,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(dem.astype("float32"), 1)

        grid = Grid.from_raster(tmp_dem, data_name="dem")
        dem_ps = grid.read_raster(tmp_dem, data_name="dem").astype("float64")

        dem_filled = grid.fill_depressions(dem_ps)
        dem_flat = grid.resolve_flats(dem_filled)
        fdir = grid.flowdir(dem_flat)
        fac = grid.accumulation(fdir)
        fac_arr = np.array(fac, dtype="float64")

    valid = np.isfinite(dem)
    streams = (fac_arr >= stream_threshold) & valid

    # If no streams at this threshold, relax it to top 1% accumulation
    if not streams.any():
        if np.isfinite(fac_arr).any():
            thr = np.nanpercentile(fac_arr, 99)
            streams = (fac_arr >= thr) & valid
        else:
            return np.zeros_like(dem, dtype="float32")

    # Distance transform to nearest stream cell
    dist, (ir, ic) = distance_transform_edt(~streams, return_indices=True)
    nearest_stream_elev = dem[ir, ic]

    hand = dem - nearest_stream_elev
    hand[~valid] = np.nan
    hand[streams] = 0.0
    return hand.astype("float32")


# --------------------------------------------------------------------
# Main PEC routine
# --------------------------------------------------------------------
def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    prei_radius_m: float = 250.0,
    stream_threshold: float = 400.0,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Compute Parcel Elevation Context (PEC) indicators & classes.

    Parameters
    ----------
    dem_path : str
        Path to DEM raster (GeoTIFF).
    parcels_path : str
        Path to parcel grid (GeoJSON or SHP).
    prei_radius_m : float, optional
        Radius for neighbourhood mean (PREI) in metres, default 250.
    stream_threshold : float, optional
        Flow accumulation threshold for stream initiation.

    Returns
    -------
    parcels_pec : GeoDataFrame
        Parcels with PEC indicators and 'pec_class'.
    diagnostics : dict
        Summary statistics and counts.
    """
    # -----------------------------
    # Load parcels & DEM (clipped)
    # -----------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.empty:
        raise ValueError(f"No features found in parcels file: {parcels_path}")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        full_transform = src.transform

    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS defined.")

    if dem_crs is not None and parcels.crs != dem_crs:
        parcels = parcels.to_crs(dem_crs)

    if "grid_id" not in parcels.columns:
        parcels["grid_id"] = np.arange(1, len(parcels) + 1)

    # Clip DEM to union of parcels (like your original script)
    boundary_geom = [mapping(parcels.unary_union)]
    with rasterio.open(dem_path) as src:
        dem_clip, dem_transform = mask(src, boundary_geom, crop=True)
        dem_meta = src.meta.copy()

    dem_arr = dem_clip[0].astype("float64")
    if dem_meta.get("nodata") is not None:
        dem_arr = np.where(dem_arr == dem_meta["nodata"], np.nan, dem_arr)

    cellsize = dem_transform.a  # pixel size in X

    # -----------------------------
    # Slope & relative elevation
    # -----------------------------
    slope_deg = _compute_slope_deg(dem_arr, cellsize)
    rel_elev = _compute_relative_elevation(dem_arr, cellsize, prei_radius_m)

    # -----------------------------
    # HAND-like metric
    # -----------------------------
    hand_arr = _compute_hand_like(dem_arr, dem_transform, stream_threshold)

    # -----------------------------
    # Zonal statistics (DEM, slope, rel, HAND)
    # -----------------------------
    grid_gdf = parcels.copy()

    dem_stats = zonal_stats(
        grid_gdf,
        dem_arr,
        affine=dem_transform,
        stats=["min", "max", "mean", "median", "std"],
        nodata=np.nan,
    )
    slope_stats = zonal_stats(
        grid_gdf,
        slope_deg,
        affine=dem_transform,
        stats=["mean", "median"],
        nodata=np.nan,
    )
    rel_stats = zonal_stats(
        grid_gdf,
        rel_elev,
        affine=dem_transform,
        stats=["mean", "min", "max"],
        nodata=np.nan,
    )
    hand_stats = zonal_stats(
        grid_gdf,
        hand_arr,
        affine=dem_transform,
        stats=["min", "mean"],
        nodata=np.nan,
    )

    grid_gdf["dem_min"] = _safe_stat_list(dem_stats, "min")
    grid_gdf["dem_max"] = _safe_stat_list(dem_stats, "max")
    grid_gdf["dem_mean"] = _safe_stat_list(dem_stats, "mean")
    grid_gdf["dem_median"] = _safe_stat_list(dem_stats, "median")
    grid_gdf["dem_std"] = _safe_stat_list(dem_stats, "std")

    grid_gdf["slp_mean"] = _safe_stat_list(slope_stats, "mean")
    grid_gdf["slp_median"] = _safe_stat_list(slope_stats, "median")

    grid_gdf["rel_mean"] = _safe_stat_list(rel_stats, "mean")
    grid_gdf["rel_min"] = _safe_stat_list(rel_stats, "min")
    grid_gdf["rel_max"] = _safe_stat_list(rel_stats, "max")

    grid_gdf["hand_min"] = _safe_stat_list(hand_stats, "min")
    grid_gdf["hand_mean"] = _safe_stat_list(hand_stats, "mean")

    # -----------------------------
    # PEC indicators & classes
    # (same rules as your script)
    # -----------------------------
    grid_gdf["relief"] = grid_gdf["dem_max"] - grid_gdf["dem_min"]
    grid_gdf["flat_flag"] = grid_gdf["slp_mean"] < 1.5
    grid_gdf["prei"] = grid_gdf["rel_mean"]
    grid_gdf["hand_score"] = grid_gdf["hand_min"]
    grid_gdf["retain_tag"] = np.where(
        (grid_gdf["prei"] <= -0.5) & (grid_gdf["hand_score"] <= 1.5),
        1,
        0,
    )

    cols_to_fill = ["prei", "hand_score", "relief", "slp_mean"]
    grid_gdf[cols_to_fill] = grid_gdf[cols_to_fill].fillna(0.0)

    def classify_pec(row):
        if (
            row["prei"] <= -0.5
            and row["hand_score"] <= 1.5
            and row["relief"] <= 3
        ):
            return "Low-lying Depressed (Retention Priority)"
        elif row["flat_flag"] and row["hand_score"] <= 3:
            return "Flat & Pressured (High Flood Exposure Risk)"
        elif row["prei"] > 0.5 and row["hand_score"] > 5:
            return "Locally High & Disconnected"
        else:
            return "Moderate / Context-Dependent"

    grid_gdf["pec_class"] = grid_gdf.apply(classify_pec, axis=1)

    # Diagnostics
    class_counts = grid_gdf["pec_class"].value_counts().to_dict()
    diagnostics = {
        "n_parcels": int(len(grid_gdf)),
        "prei_radius_m": float(prei_radius_m),
        "stream_threshold": float(stream_threshold),
        "dem_res_m": float(cellsize),
        "class_counts": class_counts,
        "dem_min_global": float(np.nanmin(dem_arr)),
        "dem_max_global": float(np.nanmax(dem_arr)),
        "hand_min_global": float(np.nanmin(hand_arr)),
        "hand_max_global": float(np.nanmax(hand_arr)),
    }

    return grid_gdf, diagnostics
