# models/pec_model.py
"""
Parcel Elevation Context (PEC) model – pysheds-based (no WhiteboxTools).

This is a function-style version of your PEC workflow, designed for use
inside the Streamlit app.

It reproduces your logic:

1. Clip DEM to parcel extent
2. DEM conditioning (fill pits/depressions, resolve flats) with pysheds
3. Slope (degrees) from gradients
4. Local relative elevation (PREI = DEM − neighbourhood mean, ~250 m window)
5. HAND-like score via flow direction, accumulation & stream mask
6. Zonal stats per parcel
7. PEC indicators + classification into 4 classes
8. Optional rainfall-adjusted PEC (thresholds modified by rainfall_mm)
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from shapely.geometry import mapping
from pysheds.grid import Grid
from scipy.ndimage import uniform_filter
import pandas as pd


# Default processing CRS (metric). Matches your original PEC script.
DEFAULT_METRIC_EPSG = 5234


def _pick_metric_crs(parcels_crs, dem_crs):
    """Choose a metric (projected, metres) CRS for slope/PREI/HAND calculations."""
    if parcels_crs is not None and parcels_crs.is_projected:
        return parcels_crs
    if dem_crs is not None and dem_crs.is_projected:
        return dem_crs
    return rasterio.crs.CRS.from_epsg(DEFAULT_METRIC_EPSG)


def _reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs: rasterio.crs.CRS,
    resampling: Resampling = Resampling.bilinear,
):
    """Reproject a single-band GeoTIFF to dst_crs and write to dst_path."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("DEM has no CRS; cannot reproject.")

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=-9999.0,
            count=1,
            compress="lzw",
        )

        dest = np.full((height, width), profile["nodata"], dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=profile["nodata"],
        )

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dest, 1)

    return dst_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ensure_grid_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure a 'grid_id' column exists."""
    if "grid_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["grid_id"] = np.arange(1, len(gdf) + 1, dtype="int32")
    return gdf


def _classify_pec_static(row):
    """Static PEC rules (no rainfall adjustment)."""
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
    Rainfall-adjusted PEC rules, matching your script:

    - PREI lowered slightly with rainfall.
    - HAND thresholds increase with rainfall.
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


# ------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------
def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    rainfall_mm: float = 0.0,
    neighbourhood_radius_m: float = 250.0,
    stream_threshold: int = 400,
):
    """
    Run the full PEC workflow.

    Returns
    -------
    parcels_pec : GeoDataFrame
    diagnostics : dict
    """

    # Where to drop intermediate rasters (optional, mainly for debugging)
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "individual" / "pec"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 0. Load parcels & clip DEM to their extent
    # --------------------------------------------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS; please define CRS and re-save.")

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM has no CRS; please define CRS before running PEC.")

        dem_crs_in = src.crs
        dem_meta_in = src.meta.copy()

        # Choose a metric CRS so DEM resolution + slope are in metres (critical).
        metric_crs = _pick_metric_crs(parcels.crs, dem_crs_in)

        # Mask DEM to parcels in DEM CRS first (cheap & avoids reprojection of whole DEM).
        parcels_in_dem = parcels.to_crs(dem_crs_in)
        boundary_geom = [mapping(parcels_in_dem.unary_union)]
        out_image, out_transform = mask(src, boundary_geom, crop=True)

    # DEM as float, clean nodata & extreme negatives
    dem = out_image[0].astype("float64")
    nodata_in = dem_meta_in.get("nodata", None)
    if nodata_in is not None:
        dem = np.where(dem == nodata_in, np.nan, dem)
    dem = np.where(dem < -1000, np.nan, dem)

    if not np.isfinite(dem).any():
        raise ValueError("DEM is empty or all nodata within the parcel extent.")

    # Save clipped DEM (in original DEM CRS)
    dem_meta_clip = dem_meta_in.copy()
    dem_meta_clip.update(
        height=dem.shape[0],
        width=dem.shape[1],
        transform=out_transform,
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="lzw",
    )
    dem_clipped = dem.astype("float32").copy()
    dem_clipped[~np.isfinite(dem_clipped)] = dem_meta_clip["nodata"]

    dem_clipped_src_path = out_dir / "dem_clipped_src_crs.tif"
    with rasterio.open(dem_clipped_src_path, "w", **dem_meta_clip) as dst:
        dst.write(dem_clipped, 1)

    # If DEM CRS is geographic (degrees) or differs from our chosen metric CRS,
    # reproject the *clipped* DEM to metric CRS for all subsequent calculations.
    if dem_crs_in.is_geographic or dem_crs_in != metric_crs:
        dem_clipped_path = out_dir / "dem_clipped.tif"
        _reproject_raster(
            src_path=dem_clipped_src_path,
            dst_path=dem_clipped_path,
            dst_crs=metric_crs,
            resampling=Resampling.bilinear,
        )
        dem_crs = metric_crs
    else:
        dem_clipped_path = dem_clipped_src_path
        dem_crs = dem_crs_in

    # --------------------------------------------------------------
    # 1. DEM conditioning with pysheds (fill pits / flats)
    # --------------------------------------------------------------
    grid_cond = Grid.from_raster(str(dem_clipped_path))
    dem_raw = grid_cond.read_raster(str(dem_clipped_path))

    pit_filled = grid_cond.fill_pits(dem_raw)
    flooded = grid_cond.fill_depressions(pit_filled)
    inflated = grid_cond.resolve_flats(flooded)

    with rasterio.open(dem_clipped_path) as src:
        filled_profile = src.profile.copy()

    filled_profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    dem_filled_arr = np.where(
        np.isfinite(inflated), inflated.astype("float32"), filled_profile["nodata"]
    )

    dem_filled_path = out_dir / "dem_filled.tif"
    with rasterio.open(dem_filled_path, "w", **filled_profile) as dst:
        dst.write(dem_filled_arr, 1)

    with rasterio.open(dem_filled_path) as src:
        dem_res = float(abs(src.res[0]))

    # --------------------------------------------------------------
    # 2. Use parcels as grid; ensure grid_id
    # --------------------------------------------------------------
    grid_gdf = _ensure_grid_id(parcels.to_crs(dem_crs)).copy()

    # --------------------------------------------------------------
    # 3. Slope (degrees) from DEM gradients
    # --------------------------------------------------------------
    slope_path = out_dir / "slope_deg.tif"
    with rasterio.open(dem_filled_path) as src:
        dem_data = src.read(1, masked=True).astype("float64")
        dem_profile = src.profile.copy()
        res_x, res_y = src.res

    gy, gx = np.gradient(dem_data.filled(np.nan), res_y, res_x)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    slope_arr = np.where(np.isfinite(dem_data), slope_deg, np.nan)

    slope_profile = dem_profile.copy()
    slope_profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    slope_out = np.where(
        np.isfinite(slope_arr), slope_arr.astype("float32"), slope_profile["nodata"]
    )
    with rasterio.open(slope_path, "w", **slope_profile) as dst:
        dst.write(slope_out, 1)

    # --------------------------------------------------------------
    # 4. Zonal stats: DEM + slope
    # --------------------------------------------------------------
    dem_stats = zonal_stats(
        grid_gdf,
        str(dem_filled_path),
        stats=["min", "max", "mean", "median", "std"],
        nodata=filled_profile["nodata"],
    )
    slope_stats = zonal_stats(
        grid_gdf,
        str(slope_path),
        stats=["mean", "median"],
        nodata=slope_profile["nodata"],
    )

    grid_gdf["dem_min"] = [d["min"] for d in dem_stats]
    grid_gdf["dem_max"] = [d["max"] for d in dem_stats]
    grid_gdf["dem_mean"] = [d["mean"] for d in dem_stats]
    grid_gdf["dem_median"] = [d["median"] for d in dem_stats]
    grid_gdf["dem_std"] = [d["std"] for d in dem_stats]
    grid_gdf["slp_mean"] = [d["mean"] for d in slope_stats]
    grid_gdf["slp_median"] = [d["median"] for d in slope_stats]

    # --------------------------------------------------------------
    # 5. Local relative elevation (DEM – neighbourhood mean)
    # --------------------------------------------------------------
    dem_mean_path = out_dir / "dem_mean_250m.tif"

    radius_pixels = max(1, int(neighbourhood_radius_m / dem_res))
    kernel_size = radius_pixels * 2 + 1

    with rasterio.open(dem_filled_path) as src:
        dem_arr = src.read(1, masked=True).astype("float64")
        mean_profile = src.profile.copy()

    data = dem_arr.filled(np.nan)
    valid_mask = np.isfinite(data)

    data_filled = np.where(valid_mask, data, 0.0)
    sum_filtered = uniform_filter(
        data_filled, size=kernel_size, mode="constant", cval=0.0
    )
    count_filtered = uniform_filter(
        valid_mask.astype("float64"), size=kernel_size, mode="constant", cval=0.0
    )

    mean_arr = np.where(count_filtered > 0, sum_filtered / count_filtered, np.nan)

    mean_profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    mean_out = np.where(
        np.isfinite(mean_arr), mean_arr.astype("float32"), mean_profile["nodata"]
    )
    with rasterio.open(dem_mean_path, "w", **mean_profile) as dst:
        dst.write(mean_out, 1)

    # Relative elevation
    dem_rel_path = out_dir / "dem_relative.tif"
    with rasterio.open(dem_filled_path) as src_a, rasterio.open(dem_mean_path) as src_b:
        a = src_a.read(1).astype("float64")
        b = src_b.read(1).astype("float64")
        prof = src_a.profile.copy()
        nodata_a = src_a.nodata
        nodata_b = src_b.nodata

        if nodata_a is not None:
            a = np.where(a == nodata_a, np.nan, a)
        if nodata_b is not None:
            b = np.where(b == nodata_b, np.nan, b)

        dem_rel = a - b
        prof.update(dtype=rasterio.float32, nodata=-9999.0, count=1, compress="lzw")
        out_rel = np.where(
            np.isfinite(dem_rel), dem_rel.astype("float32"), prof["nodata"]
        )
        with rasterio.open(dem_rel_path, "w", **prof) as dst:
            dst.write(out_rel, 1)

    rel_stats = zonal_stats(
        grid_gdf,
        str(dem_rel_path),
        stats=["mean", "min", "max"],
        nodata=prof["nodata"],
    )
    grid_gdf["rel_mean"] = [d["mean"] for d in rel_stats]
    grid_gdf["rel_min"] = [d["min"] for d in rel_stats]
    grid_gdf["rel_max"] = [d["max"] for d in rel_stats]

    # --------------------------------------------------------------
    # 6. Flow direction / accumulation / streams / HAND with pysheds
    # --------------------------------------------------------------
    hand_path = out_dir / "hand.tif"

    grid = Grid.from_raster(str(dem_filled_path))
    dem_for_flow = grid.read_raster(str(dem_filled_path))
    dem_for_flow = grid.resolve_flats(dem_for_flow)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(dem_for_flow, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    stream_mask = acc >= stream_threshold
    hand = grid.compute_hand(fdir, dem_for_flow, stream_mask)

    with rasterio.open(dem_filled_path) as src:
        base_profile = src.profile.copy()

    hand_profile = base_profile.copy()
    hand_profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    hand_out = hand.astype("float32")
    hand_out[~np.isfinite(hand_out)] = hand_profile["nodata"]
    with rasterio.open(hand_path, "w", **hand_profile) as dst:
        dst.write(hand_out, 1)

    hand_stats = zonal_stats(
        grid_gdf, str(hand_path), stats=["min", "mean"], nodata=hand_profile["nodata"]
    )
    grid_gdf["hand_min"] = [d["min"] for d in hand_stats]
    grid_gdf["hand_mean"] = [d["mean"] for d in hand_stats]

    # --------------------------------------------------------------
    # 7. PEC indicators & classification
    # --------------------------------------------------------------
    grid_gdf["relief"] = grid_gdf["dem_max"] - grid_gdf["dem_min"]
    grid_gdf["flat_flag"] = grid_gdf["slp_mean"] < 1.5
    grid_gdf["prei"] = grid_gdf["rel_mean"]
    grid_gdf["hand_score"] = grid_gdf["hand_min"]
    grid_gdf["retain_tag"] = np.where(
        (grid_gdf["prei"] <= -0.5) & (grid_gdf["hand_score"] <= 1.5), 1, 0
    )

    cols_to_fill = ["prei", "hand_score", "relief", "slp_mean"]
    grid_gdf[cols_to_fill] = grid_gdf[cols_to_fill].fillna(0.0)

    grid_gdf["pec_class_static"] = grid_gdf.apply(_classify_pec_static, axis=1)

    if rainfall_mm and rainfall_mm > 0:
        grid_gdf["pec_class_rainfall"] = grid_gdf.apply(
            lambda r: _classify_pec_rainfall(r, rainfall_mm), axis=1
        )
        grid_gdf["pec_class"] = grid_gdf["pec_class_rainfall"]
    else:
        grid_gdf["pec_class_rainfall"] = grid_gdf["pec_class_static"]
        grid_gdf["pec_class"] = grid_gdf["pec_class_static"]

    pec_mapping = {
        "Low-lying Depressed (Retention Priority)": 1,
        "Flat & Pressured (High Flood Exposure Risk)": 2,
        "Locally High & Disconnected": 3,
        "Moderate / Context-Dependent": 4,
    }
    grid_gdf["pec_code"] = grid_gdf["pec_class"].map(pec_mapping).astype("Int32")

    # --------------------------------------------------------------
    # 8. Diagnostics (helps explain missing classes in the app UI)
    # --------------------------------------------------------------
    _counts = grid_gdf["pec_class"].value_counts().to_dict()
    class_counts_full = {k: int(_counts.get(k, 0)) for k in pec_mapping.keys()}

    if rainfall_mm and rainfall_mm > 0:
        low_thresh_used = 1.5 + 0.01 * rainfall_mm
        med_thresh_used = 3.0 + 0.01 * rainfall_mm
        prei_used = grid_gdf["prei"] - 0.002 * rainfall_mm
    else:
        low_thresh_used = 1.5
        med_thresh_used = 3.0
        prei_used = grid_gdf["prei"]

    flat_count = int((grid_gdf["slp_mean"] < 1.5).sum())
    hand_le_med = int((grid_gdf["hand_score"] <= med_thresh_used).sum())
    flat_and_hand = int(
        ((grid_gdf["slp_mean"] < 1.5) & (grid_gdf["hand_score"] <= med_thresh_used)).sum()
    )
    low_lying_rule = (
        (prei_used <= -0.5)
        & (grid_gdf["hand_score"] <= low_thresh_used)
        & (grid_gdf["relief"] <= 3)
    )
    low_lying_rule_count = int(low_lying_rule.sum())

    diagnostics = {
        "n_parcels": int(len(grid_gdf)),
        "dem_res_m": float(dem_res),
        "neighbourhood_radius_m": float(neighbourhood_radius_m),
        "neighbourhood_radius_pixels": int(radius_pixels),
        "stream_threshold": int(stream_threshold),
        "rainfall_mm": float(rainfall_mm),

        "dem_crs_input": str(dem_crs_in),
        "dem_crs_processing": str(dem_crs),
        "dem_reprojected": bool(dem_crs_in != dem_crs),
        "dem_input_is_geographic": bool(getattr(dem_crs_in, "is_geographic", False)),

        "pec_class_counts": class_counts_full,

        "flat_count_slope_lt_1p5": flat_count,
        "hand_le_med_thresh": hand_le_med,
        "flat_and_hand": flat_and_hand,
        "low_lying_rule_count": low_lying_rule_count,
        "flat_pressured_count": class_counts_full.get(
            "Flat & Pressured (High Flood Exposure Risk)", 0
        ),
        "hand_low_thresh_used": float(low_thresh_used),
        "hand_med_thresh_used": float(med_thresh_used),
    }

    return grid_gdf, diagnostics
