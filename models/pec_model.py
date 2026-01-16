# models/pec_model.py
"""
PEC model (Parcel Elevation Context) for Streamlit app.

Key fixes/features:
- Automatically ensures DEM processing happens in a METRIC CRS (meters).
  This fixes slope/PREI/HAND calculations when DEM is in degrees (EPSG:4326).
- Writes ALL intermediate/output GeoTIFFs to outputs/individual/pec
- Exports parcel-level KML (fallback GeoJSON if KML driver missing)
- Returns diagnostics including file paths for easy ZIP download in the UI
"""

from pathlib import Path
import os
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


DEFAULT_METRIC_EPSG = 5234


# ---------------------------------------------------------------------
# CRS / raster helpers
# ---------------------------------------------------------------------
def _pick_metric_crs(parcels_crs, dem_crs):
    """Pick a projected CRS (meters). Prefer parcels CRS if projected; else DEM CRS if projected; else EPSG:5234."""
    if parcels_crs is not None and getattr(parcels_crs, "is_projected", False):
        return parcels_crs
    if dem_crs is not None and getattr(dem_crs, "is_projected", False):
        return dem_crs
    return rasterio.crs.CRS.from_epsg(DEFAULT_METRIC_EPSG)


def _reproject_raster_singleband(src_path: Path, dst_path: Path, dst_crs, resampling=Resampling.bilinear):
    """Reproject a single-band GeoTIFF to dst_crs and write."""
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


def _ensure_grid_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure 'grid_id' exists for parcels."""
    if "grid_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["grid_id"] = np.arange(1, len(gdf) + 1, dtype="int32")
    return gdf


# ---------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------
def _classify_pec_static(row):
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


# ---------------------------------------------------------------------
# Raster writers
# ---------------------------------------------------------------------
def _write_float_tif(path: Path, arr: np.ndarray, profile: dict, nodata=-9999.0):
    prof = profile.copy()
    prof.update(dtype="float32", nodata=float(nodata), count=1, compress="lzw")
    out = arr.astype("float32")
    out[~np.isfinite(out)] = prof["nodata"]
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def _write_int_tif(path: Path, arr: np.ndarray, profile: dict, dtype="int32", nodata_val=-9999):
    prof = profile.copy()
    prof.update(dtype=dtype, nodata=nodata_val, count=1, compress="lzw")
    out = arr.astype(dtype)
    out[~np.isfinite(out)] = nodata_val
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def _write_uint8_tif(path: Path, arr: np.ndarray, profile: dict, nodata_val=0):
    prof = profile.copy()
    prof.update(dtype="uint8", nodata=nodata_val, count=1, compress="lzw")
    out = arr.astype("uint8")
    out[~np.isfinite(out)] = nodata_val
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


# ---------------------------------------------------------------------
# Main model entry
# ---------------------------------------------------------------------
def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    rainfall_mm: float = 0.0,
    neighbourhood_radius_m: float = 250.0,
    stream_threshold: int = 400,
):
    """
    Run PEC and return (GeoDataFrame, diagnostics).
    """
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "outputs" / "individual" / "pec"
    out_dir.mkdir(parents=True, exist_ok=True)

    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define CRS and re-save.")

    # --------------------------------------------------------------
    # 1) Clip DEM to parcels in DEM CRS first
    # --------------------------------------------------------------
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM has no CRS. Please define CRS before running PEC.")

        dem_crs_in = src.crs
        dem_meta_in = src.meta.copy()

        # Select metric CRS for processing
        metric_crs = _pick_metric_crs(parcels.crs, dem_crs_in)

        parcels_in_dem = parcels.to_crs(dem_crs_in)
        boundary_geom = [mapping(parcels_in_dem.unary_union)]
        out_image, out_transform = mask(src, boundary_geom, crop=True)

    dem_clip = out_image[0].astype("float64")
    nodata_in = dem_meta_in.get("nodata", None)
    if nodata_in is not None:
        dem_clip = np.where(dem_clip == nodata_in, np.nan, dem_clip)
    dem_clip = np.where(dem_clip < -1000, np.nan, dem_clip)

    if not np.isfinite(dem_clip).any():
        raise ValueError("DEM is empty or all nodata inside parcel extent.")

    # Save clipped DEM in source CRS
    dem_meta_clip = dem_meta_in.copy()
    dem_meta_clip.update(
        height=dem_clip.shape[0],
        width=dem_clip.shape[1],
        transform=out_transform,
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="lzw",
    )
    dem_clip_out = dem_clip.astype("float32")
    dem_clip_out[~np.isfinite(dem_clip_out)] = dem_meta_clip["nodata"]

    dem_clipped_src_path = out_dir / "dem_clipped_src_crs.tif"
    with rasterio.open(dem_clipped_src_path, "w", **dem_meta_clip) as dst:
        dst.write(dem_clip_out, 1)

    # --------------------------------------------------------------
    # 2) Reproject clipped DEM to metric CRS if needed
    # --------------------------------------------------------------
    dem_reprojected = False
    if getattr(dem_crs_in, "is_geographic", False) or (dem_crs_in != metric_crs):
        dem_clipped_path = out_dir / "dem_clipped.tif"
        _reproject_raster_singleband(
            dem_clipped_src_path, dem_clipped_path, metric_crs, resampling=Resampling.bilinear
        )
        dem_processing_path = dem_clipped_path
        dem_crs_processing = metric_crs
        dem_reprojected = True
    else:
        dem_processing_path = dem_clipped_src_path
        dem_crs_processing = dem_crs_in

    # Ensure parcels in processing CRS
    grid_gdf = parcels.to_crs(dem_crs_processing)
    grid_gdf = _ensure_grid_id(grid_gdf).copy()

    # --------------------------------------------------------------
    # 3) DEM conditioning (pysheds)
    # --------------------------------------------------------------
    grid_cond = Grid.from_raster(str(dem_processing_path))
    dem_raw = grid_cond.read_raster(str(dem_processing_path))

    pit_filled = grid_cond.fill_pits(dem_raw)
    flooded = grid_cond.fill_depressions(pit_filled)
    inflated = grid_cond.resolve_flats(flooded)

    with rasterio.open(dem_processing_path) as src:
        base_profile = src.profile.copy()
        dem_res_m = float(abs(src.res[0]))

    dem_filled_path = out_dir / "dem_filled.tif"
    _write_float_tif(dem_filled_path, inflated, base_profile, nodata=-9999.0)

    # --------------------------------------------------------------
    # 4) Slope (degrees)
    # --------------------------------------------------------------
    slope_path = out_dir / "slope_deg.tif"
    with rasterio.open(dem_filled_path) as src:
        dem_data = src.read(1, masked=True).astype("float64")
        dem_profile = src.profile.copy()
        res_x, res_y = src.res

    arr = dem_data.filled(np.nan)
    gy, gx = np.gradient(arr, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg = np.where(np.isfinite(arr), slope_deg, np.nan)

    _write_float_tif(slope_path, slope_deg, dem_profile, nodata=-9999.0)

    # --------------------------------------------------------------
    # 5) Zonal stats: DEM + slope
    # --------------------------------------------------------------
    dem_stats = zonal_stats(
        grid_gdf, str(dem_filled_path),
        stats=["min", "max", "mean", "median", "std"],
        nodata=-9999.0
    )
    slope_stats = zonal_stats(
        grid_gdf, str(slope_path),
        stats=["mean", "median"],
        nodata=-9999.0
    )

    grid_gdf["dem_min"] = [d["min"] for d in dem_stats]
    grid_gdf["dem_max"] = [d["max"] for d in dem_stats]
    grid_gdf["dem_mean"] = [d["mean"] for d in dem_stats]
    grid_gdf["dem_median"] = [d["median"] for d in dem_stats]
    grid_gdf["dem_std"] = [d["std"] for d in dem_stats]
    grid_gdf["slp_mean"] = [d["mean"] for d in slope_stats]
    grid_gdf["slp_median"] = [d["median"] for d in slope_stats]

    # --------------------------------------------------------------
    # 6) PREI: neighbourhood mean (250 m default) + relative elevation
    # --------------------------------------------------------------
    radius_pixels = max(1, int(neighbourhood_radius_m / dem_res_m))
    kernel_size = radius_pixels * 2 + 1

    dem_mean_path = out_dir / "dem_mean_250m.tif"
    with rasterio.open(dem_filled_path) as src:
        dem_arr = src.read(1, masked=True).astype("float64")
        mean_profile = src.profile.copy()

    data = dem_arr.filled(np.nan)
    valid_mask = np.isfinite(data)

    data_filled = np.where(valid_mask, data, 0.0)
    sum_filtered = uniform_filter(data_filled, size=kernel_size, mode="constant", cval=0.0)
    count_filtered = uniform_filter(valid_mask.astype("float64"), size=kernel_size, mode="constant", cval=0.0)

    mean_arr = np.where(count_filtered > 0, sum_filtered / count_filtered, np.nan)
    _write_float_tif(dem_mean_path, mean_arr, mean_profile, nodata=-9999.0)

    dem_rel_path = out_dir / "dem_relative.tif"
    with rasterio.open(dem_filled_path) as src_a, rasterio.open(dem_mean_path) as src_b:
        a = src_a.read(1).astype("float64")
        b = src_b.read(1).astype("float64")
        prof = src_a.profile.copy()
        na = src_a.nodata
        nb = src_b.nodata

        if na is not None:
            a = np.where(a == na, np.nan, a)
        if nb is not None:
            b = np.where(b == nb, np.nan, b)

        dem_rel = a - b

    _write_float_tif(dem_rel_path, dem_rel, prof, nodata=-9999.0)

    rel_stats = zonal_stats(
        grid_gdf, str(dem_rel_path),
        stats=["mean", "min", "max"],
        nodata=-9999.0
    )
    grid_gdf["rel_mean"] = [d["mean"] for d in rel_stats]
    grid_gdf["rel_min"] = [d["min"] for d in rel_stats]
    grid_gdf["rel_max"] = [d["max"] for d in rel_stats]

    # --------------------------------------------------------------
    # 7) Flow direction / accumulation / streams / HAND (pysheds)
    # --------------------------------------------------------------
    grid = Grid.from_raster(str(dem_filled_path))
    dem_flow = grid.read_raster(str(dem_filled_path))
    dem_flow = grid.resolve_flats(dem_flow)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(dem_flow, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    stream_mask = acc >= stream_threshold
    hand = grid.compute_hand(fdir, dem_flow, stream_mask)

    # Save rasters requested
    fdir_path = out_dir / "flowdir.tif"
    facc_path = out_dir / "flowacc.tif"
    streams_path = out_dir / "streams.tif"
    hand_path = out_dir / "hand.tif"

    # Use base profile from dem_filled
    with rasterio.open(dem_filled_path) as src:
        flow_profile = src.profile.copy()

    # flowdir is integer
    _write_int_tif(fdir_path, fdir, flow_profile, dtype="int32", nodata_val=-9999)
    # flowacc is float
    _write_float_tif(facc_path, acc, flow_profile, nodata=-9999.0)
    # streams mask 0/1
    _write_uint8_tif(streams_path, stream_mask.astype("uint8"), flow_profile, nodata_val=0)
    # hand
    _write_float_tif(hand_path, hand, flow_profile, nodata=-9999.0)

    hand_stats = zonal_stats(
        grid_gdf, str(hand_path),
        stats=["min", "mean"],
        nodata=-9999.0
    )
    grid_gdf["hand_min"] = [d["min"] for d in hand_stats]
    grid_gdf["hand_mean"] = [d["mean"] for d in hand_stats]

    # --------------------------------------------------------------
    # 8) PEC indicators + classification
    # --------------------------------------------------------------
    grid_gdf["relief"] = grid_gdf["dem_max"] - grid_gdf["dem_min"]
    grid_gdf["flat_flag"] = grid_gdf["slp_mean"] < 1.5
    grid_gdf["prei"] = grid_gdf["rel_mean"]
    grid_gdf["hand_score"] = grid_gdf["hand_min"]
    grid_gdf["retain_tag"] = np.where(
        (grid_gdf["prei"] <= -0.5) & (grid_gdf["hand_score"] <= 1.5),
        1, 0
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
    # 9) Export parcel KML (fallback GeoJSON)
    # --------------------------------------------------------------
    kml_path = out_dir / "parcel_level_PEC.kml"
    geojson_path = out_dir / "parcel_level_PEC.geojson"
    kml_written = False

    kml_fields = [c for c in ["grid_id", "pec_class", "pec_code", "prei", "hand_score", "relief", "slp_mean"] if c in grid_gdf.columns]
    gdf_kml = grid_gdf[kml_fields + ["geometry"]].to_crs(epsg=4326).copy()

    # Make sure KML-friendly field types
    for c in kml_fields:
        if c in ["pec_class"]:
            gdf_kml[c] = gdf_kml[c].astype(str)
        elif c in ["pec_code"]:
            gdf_kml[c] = gdf_kml[c].astype("Int64").astype(str)
        else:
            # keep numeric as is; KML driver can be picky, but usually OK
            pass

    try:
        gdf_kml.to_file(kml_path, driver="KML")
        kml_written = True
    except Exception:
        # fallback
        gdf_kml.to_file(geojson_path, driver="GeoJSON")
        kml_written = False

    # --------------------------------------------------------------
    # 10) Diagnostics + output file list (for ZIP download)
    # --------------------------------------------------------------
    counts = grid_gdf["pec_class"].value_counts().to_dict()
    class_counts_full = {k: int(counts.get(k, 0)) for k in pec_mapping.keys()}

    # debug counts: why Flat&Pressured could be zero
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
    flat_and_hand = int(((grid_gdf["slp_mean"] < 1.5) & (grid_gdf["hand_score"] <= med_thresh_used)).sum())
    low_lying_rule = (
        (prei_used <= -0.5)
        & (grid_gdf["hand_score"] <= low_thresh_used)
        & (grid_gdf["relief"] <= 3)
    )
    low_lying_rule_count = int(low_lying_rule.sum())

    output_files = [
        str(dem_clipped_src_path),
        str(out_dir / "dem_clipped.tif"),     # exists only if reprojected
        str(dem_filled_path),
        str(slope_path),
        str(dem_mean_path),
        str(dem_rel_path),
        str(fdir_path),
        str(facc_path),
        str(streams_path),
        str(hand_path),
    ]
    output_files = [f for f in output_files if os.path.exists(f)]

    diagnostics = {
        "n_parcels": int(len(grid_gdf)),
        "dem_res_m": float(dem_res_m),
        "neighbourhood_radius_m": float(neighbourhood_radius_m),
        "neighbourhood_radius_pixels": int(radius_pixels),
        "stream_threshold": int(stream_threshold),
        "rainfall_mm": float(rainfall_mm),

        "dem_crs_input": str(dem_crs_in),
        "dem_crs_processing": str(dem_crs_processing),
        "dem_reprojected": bool(dem_reprojected),
        "dem_input_is_geographic": bool(getattr(dem_crs_in, "is_geographic", False)),

        "pec_class_counts": class_counts_full,

        "flat_count_slope_lt_1p5": flat_count,
        "hand_le_med_thresh": hand_le_med,
        "flat_and_hand": flat_and_hand,
        "low_lying_rule_count": low_lying_rule_count,
        "flat_pressured_count": class_counts_full.get("Flat & Pressured (High Flood Exposure Risk)", 0),
        "hand_low_thresh_used": float(low_thresh_used),
        "hand_med_thresh_used": float(med_thresh_used),

        "pec_out_dir": str(out_dir),
        "pec_output_files": output_files,
        "pec_kml_path": str(kml_path) if (kml_written and os.path.exists(kml_path)) else None,
        "pec_geojson_path": str(geojson_path) if ((not kml_written) and os.path.exists(geojson_path)) else None,
    }

    return grid_gdf, diagnostics
