# models/pec_model.py
"""
Parcel Elevation Context (PEC) model for the Streamlit app.

Core ideas:
- Use DEM to derive elevation, slope, local relative elevation and HAND.
- Combine indicators into a parcel-level PEC class:
    1. Low-lying Depressed (Retention Priority)
    2. Flat & Pressured (High Flood Exposure Risk)
    3. Locally High & Disconnected
    4. Moderate / Context-Dependent

Rainfall-aware variant:
- Adjusts PREI (local relative elevation) and HAND thresholds for
  a chosen rainfall depth (mm).
"""

import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
from shapely.geometry import mapping
from whitebox import WhiteboxTools


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
    Rainfall-aware PEC classification (matches your Colab logic):
      - HAND thresholds increase slightly with rainfall.
      - PREI is shifted down slightly with rainfall.
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


def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    rainfall_mm: float = 0.0,
    neighbourhood_radius_m: float = 250.0,
    stream_threshold: int = 400,
):
    """
    Run full PEC workflow for the app.

    Args
    ----
    dem_path  : path to DEM (GeoTIFF)
    parcels_path : path to parcel grid (GeoJSON / Shapefile)
    rainfall_mm  : rainfall depth for rainfall-aware PEC.
                   Set 0 for purely static PEC.
    neighbourhood_radius_m : radius for neighbourhood mean elevation (PREI)
    stream_threshold : flow accumulation threshold for defining streams (cell count)

    Returns
    -------
    parcels_pec : GeoDataFrame with PEC indicators + class
    diagnostics : dict
    """

    # ------------------------------------------------------------------
    # 0. Setup working directory & Whitebox
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "individual" / "pec"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_clipped_path = out_dir / "pec_dem_clipped.tif"
    dem_filled_path = out_dir / "pec_dem_filled.tif"
    slope_path = out_dir / "pec_slope_deg.tif"
    dem_mean_path = out_dir / "pec_dem_mean.tif"
    dem_rel_path = out_dir / "pec_dem_relative.tif"
    fdir_path = out_dir / "pec_fdir.tif"
    facc_path = out_dir / "pec_facc.tif"
    streams_path = out_dir / "pec_streams.tif"
    hand_path = out_dir / "pec_hand.tif"

    wbt = WhiteboxTools()
    wbt.work_dir = str(out_dir)
    wbt.verbose = False

    # ------------------------------------------------------------------
    # 1. Load parcels & clip DEM to study extent
    # ------------------------------------------------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define it and re-save.")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_res = src.res[0]  # assume square pixels
        boundary_geom = [mapping(parcels.to_crs(dem_crs).unary_union)]

        # clip DEM
        out_image, out_transform = mask(src, boundary_geom, crop=True)
        dem_meta = src.meta.copy()

    # Replace extreme invalid DEM values (< -1000) with 0
    dem_clip_clean = np.where(out_image < -1000, 0, out_image)

    dem_meta.update(
        height=dem_clip_clean.shape[1],
        width=dem_clip_clean.shape[2],
        transform=out_transform,
        dtype=dem_clip_clean.dtype,
    )

    with rasterio.open(dem_clipped_path, "w", **dem_meta) as dst:
        dst.write(dem_clip_clean)

    # Align parcels to DEM CRS
    parcels = parcels.to_crs(dem_crs)
    parcels = parcels.reset_index(drop=True)
    if "grid_id" not in parcels.columns:
        parcels["grid_id"] = np.arange(1, len(parcels) + 1, dtype="int32")

    # Use parcels as the grid
    grid_gdf = parcels.copy()

    # ------------------------------------------------------------------
    # 2. Fill sinks & compute slope
    # ------------------------------------------------------------------
    wbt.fill_depressions(str(dem_clipped_path), str(dem_filled_path))

    wbt.slope(str(dem_filled_path), str(slope_path), zfactor=1.0)

    # ------------------------------------------------------------------
    # 3. Zonal statistics: DEM & slope
    # ------------------------------------------------------------------
    dem_stats = zonal_stats(
        grid_gdf,
        str(dem_filled_path),
        stats=["min", "max", "mean", "median", "std"],
        geojson_out=True,
    )
    slope_stats = zonal_stats(
        grid_gdf,
        str(slope_path),
        stats=["mean", "median"],
        geojson_out=True,
    )

    dem_df = gpd.GeoDataFrame.from_features(dem_stats)
    slope_df = gpd.GeoDataFrame.from_features(slope_stats)

    grid_gdf["dem_min"] = dem_df["min"]
    grid_gdf["dem_max"] = dem_df["max"]
    grid_gdf["dem_mean"] = dem_df["mean"]
    grid_gdf["dem_median"] = dem_df["median"]
    grid_gdf["dem_std"] = dem_df["std"]
    grid_gdf["slp_mean"] = slope_df["mean"]
    grid_gdf["slp_median"] = slope_df["median"]

    # ------------------------------------------------------------------
    # 4. Local relative elevation (DEM - neighbourhood mean)
    # ------------------------------------------------------------------
    radius_pixels = max(1, int(neighbourhood_radius_m / dem_res))

    wbt.mean_filter(
        str(dem_filled_path),
        str(dem_mean_path),
        filterx=radius_pixels,
        filtery=radius_pixels,
    )

    with rasterio.open(dem_filled_path) as src_a, rasterio.open(dem_mean_path) as src_b:
        a = src_a.read(1).astype("float64")
        b = src_b.read(1).astype("float64")

        nodata_a = src_a.nodata
        nodata_b = src_b.nodata

        if nodata_a is not None:
            a = np.where(a == nodata_a, np.nan, a)
        if nodata_b is not None:
            b = np.where(b == nodata_b, np.nan, b)

        if a.shape != b.shape:
            raise ValueError("DEM and DEM-mean rasters have different shapes.")

        dem_rel = a - b
        profile = src_a.profile.copy()
        profile.update(dtype=rasterio.float32, nodata=-9999.0, count=1, compress="lzw")
        out = np.where(np.isfinite(dem_rel), dem_rel.astype("float32"), profile["nodata"])

        with rasterio.open(dem_rel_path, "w", **profile) as dst:
            dst.write(out, 1)

    rel_stats = zonal_stats(
        grid_gdf,
        str(dem_rel_path),
        stats=["mean", "min", "max"],
        geojson_out=True,
    )
    rel_df = gpd.GeoDataFrame.from_features(rel_stats)

    grid_gdf["rel_mean"] = rel_df["mean"]
    grid_gdf["rel_min"] = rel_df["min"]
    grid_gdf["rel_max"] = rel_df["max"]

    # ------------------------------------------------------------------
    # 5. Flow direction, accumulation, HAND using Whitebox
    # ------------------------------------------------------------------
    wbt.d8_pointer(dem=str(dem_filled_path), output=str(fdir_path), esri_pntr=False)
    wbt.d8_flow_accumulation(
        i=str(dem_filled_path),
        output=str(facc_path),
        out_type="cells",
        pntr=False,
    )
    wbt.extract_streams(
        flow_accum=str(facc_path),
        output=str(streams_path),
        threshold=float(stream_threshold),
    )
    wbt.elevation_above_stream(
        dem=str(dem_filled_path),
        streams=str(streams_path),
        output=str(hand_path),
    )

    hand_stats = zonal_stats(
        grid_gdf,
        str(hand_path),
        stats=["min", "mean"],
        geojson_out=True,
    )
    hand_df = gpd.GeoDataFrame.from_features(hand_stats)

    grid_gdf["hand_min"] = hand_df["min"]
    grid_gdf["hand_mean"] = hand_df["mean"]

    # ------------------------------------------------------------------
    # 6. PEC indicators & classification
    # ------------------------------------------------------------------
    grid_gdf["relief"] = grid_gdf["dem_max"] - grid_gdf["dem_min"]
    grid_gdf["flat_flag"] = grid_gdf["slp_mean"] < 1.5

    # PREI (Parcel Relative Elevation Index)
    grid_gdf["prei"] = grid_gdf["rel_mean"]
    grid_gdf["hand_score"] = grid_gdf["hand_min"]

    # Replace NaNs with 0 for key numeric fields
    cols_to_fill = ["prei", "hand_score", "relief", "slp_mean"]
    grid_gdf[cols_to_fill] = grid_gdf[cols_to_fill].fillna(0.0)

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

    # Small normalized helper indices for optional plots
    grid_gdf["prei_norm"] = _minmax(grid_gdf["prei"].values)
    grid_gdf["hand_norm"] = _minmax(grid_gdf["hand_score"].values)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics = {
        "n_parcels": int(len(grid_gdf)),
        "dem_res_m": float(dem_res),
        "neighbourhood_radius_m": float(neighbourhood_radius_m),
        "radius_pixels": int(radius_pixels),
        "stream_threshold_cells": int(stream_threshold),
        "rainfall_mm": float(rainfall_mm),
        "pec_class_counts": grid_gdf["pec_class"].value_counts().to_dict(),
    }

    return grid_gdf, diagnostics
