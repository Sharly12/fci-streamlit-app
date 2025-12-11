# models/pec_model.py
"""
Parcel Elevation Context (PEC) model – Streamlit version (no Whitebox / no pysheds)

Inputs
------
- dem_path: path to DEM raster (GeoTIFF)
- parcels_path: path to parcel grid (GeoJSON / Shapefile)
- prei_radius_m: radius [m] for neighbourhood mean (PREI)

Outputs
-------
- parcels: GeoDataFrame with PEC indicators + "pec_class" + "pec_code"
- diagnostics: dict with summary information
"""

import os
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from scipy.ndimage import uniform_filter


# ------------------------------------------------------------------
# Helper: mean filter that respects NaNs
# ------------------------------------------------------------------
def _mean_filter_nan(arr: np.ndarray, radius_pixels: int) -> np.ndarray:
    """
    Approximate neighbourhood mean with a box filter while ignoring NaNs.
    """
    if radius_pixels < 1:
        return arr.astype("float32")

    size = int(radius_pixels) * 2 + 1

    valid = np.isfinite(arr).astype("float32")
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)

    sum_ = uniform_filter(arr_filled, size=size, mode="nearest")
    count = uniform_filter(valid, size=size, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_ / count
    mean[count == 0] = np.nan
    return mean.astype("float32")


# ------------------------------------------------------------------
# Main PEC function
# ------------------------------------------------------------------
def run_pec_analysis(
    dem_path: str,
    parcels_path: str,
    prei_radius_m: float = 250.0,
):
    # -----------------------------
    # 1) Load DEM + parcels
    # -----------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS defined.")

    with rasterio.open(dem_path) as src:
        dem_full = src.read(1).astype("float32")
        dem_crs = src.crs
        dem_transform = src.transform
        dem_res = src.res[0]
        dem_nodata = src.nodata

    # Align CRS
    if parcels.crs != dem_crs:
        parcels = parcels.to_crs(dem_crs)

    # Ensure grid_id
    if "grid_id" not in parcels.columns:
        parcels["grid_id"] = np.arange(1, len(parcels) + 1, dtype="int32")

    # -----------------------------
    # 2) Clip DEM to parcel extent
    # -----------------------------
    boundary_geom = [parcels.unary_union.__geo_interface__]

    with rasterio.open(dem_path) as src:
        dem_clip, clip_transform = mask(src, boundary_geom, crop=True)
        profile_clip = src.profile.copy()

    dem_clip = dem_clip[0].astype("float32")

    # Replace extreme invalid values (like -9999) if present
    if dem_nodata is not None:
        dem_clip = np.where(dem_clip == dem_nodata, np.nan, dem_clip)
    dem_clip = np.where(dem_clip < -1000, np.nan, dem_clip)

    profile_clip.update(
        height=dem_clip.shape[0],
        width=dem_clip.shape[1],
        transform=clip_transform,
        dtype="float32",
        nodata=-9999.0,
    )

    # Write clipped DEM to temp file for zonal_stats
    tmp_dem = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_dem.close()
    with rasterio.open(tmp_dem.name, "w", **profile_clip) as dst:
        out = np.where(np.isfinite(dem_clip), dem_clip, profile_clip["nodata"])
        dst.write(out.astype("float32"), 1)

    # -----------------------------
    # 3) DEM stats per parcel
    # -----------------------------
    dem_stats = zonal_stats(
        parcels,
        tmp_dem.name,
        stats=["min", "max", "mean"],
        geojson_out=False,
        nodata=profile_clip["nodata"],
    )

    parcels["dem_min"] = [d["min"] for d in dem_stats]
    parcels["dem_max"] = [d["max"] for d in dem_stats]
    parcels["dem_mean"] = [d["mean"] for d in dem_stats]

    # -----------------------------
    # 4) Slope (simple gradient)
    # -----------------------------
    dy, dx = np.gradient(dem_clip, dem_res, dem_res)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_deg = np.degrees(slope_rad).astype("float32")

    tmp_slope = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_slope.close()
    with rasterio.open(tmp_slope.name, "w", **profile_clip) as dst:
        out = np.where(np.isfinite(slope_deg), slope_deg, profile_clip["nodata"])
        dst.write(out.astype("float32"), 1)

    slope_stats = zonal_stats(
        parcels,
        tmp_slope.name,
        stats=["mean"],
        geojson_out=False,
        nodata=profile_clip["nodata"],
    )
    parcels["slp_mean"] = [d["mean"] for d in slope_stats]

    # -----------------------------
    # 5) PREI – local relative elevation
    # -----------------------------
    radius_pix = max(1, int(prei_radius_m / dem_res))
    local_mean = _mean_filter_nan(dem_clip, radius_pix)
    dem_rel = dem_clip - local_mean

    tmp_rel = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_rel.close()
    with rasterio.open(tmp_rel.name, "w", **profile_clip) as dst:
        out = np.where(np.isfinite(dem_rel), dem_rel, profile_clip["nodata"])
        dst.write(out.astype("float32"), 1)

    rel_stats = zonal_stats(
        parcels,
        tmp_rel.name,
        stats=["mean", "min", "max"],
        geojson_out=False,
        nodata=profile_clip["nodata"],
    )

    parcels["rel_mean"] = [d["mean"] for d in rel_stats]
    parcels["rel_min"] = [d["min"] for d in rel_stats]
    parcels["rel_max"] = [d["max"] for d in rel_stats]

    # -----------------------------
    # 6) Indicators & HAND proxy
    # -----------------------------
    parcels["relief"] = parcels["dem_max"] - parcels["dem_min"]
    parcels["flat_flag"] = parcels["slp_mean"] < 1.5
    parcels["prei"] = parcels["rel_mean"]

    # HAND-like proxy: normalised PREI (0–10 “m”)
    prei_vals = parcels["prei"].values.astype("float32")
    valid = np.isfinite(prei_vals)
    if valid.any():
        pmin = float(prei_vals[valid].min())
        pmax = float(prei_vals[valid].max())
        rng = max(pmax - pmin, 1e-6)
        hand_proxy = (prei_vals - pmin) / rng * 10.0
    else:
        hand_proxy = np.zeros_like(prei_vals)

    parcels["hand_score"] = hand_proxy
    parcels["retain_tag"] = np.where(
        (parcels["prei"] <= -0.5) & (parcels["hand_score"] <= 1.5), 1, 0
    )

    # -----------------------------
    # 7) PEC class rules (same as original)
    # -----------------------------
    def classify_pec(row):
        if row["prei"] <= -0.5 and row["hand_score"] <= 1.5 and row["relief"] <= 3:
            return "Low-lying Depressed (Retention Priority)"
        elif row["flat_flag"] and row["hand_score"] <= 3:
            return "Flat & Pressured (High Flood Exposure Risk)"
        elif row["prei"] > 0.5 and row["hand_score"] > 5:
            return "Locally High & Disconnected"
        else:
            return "Moderate / Context-Dependent"

    parcels["pec_class"] = parcels.apply(classify_pec, axis=1)

    pec_mapping = {
        "Low-lying Depressed (Retention Priority)": 1,
        "Flat & Pressured (High Flood Exposure Risk)": 2,
        "Locally High & Disconnected": 3,
        "Moderate / Context-Dependent": 4,
    }
    parcels["pec_code"] = parcels["pec_class"].map(pec_mapping)

    diagnostics = {
        "n_parcels": int(len(parcels)),
        "prei_radius_m": float(prei_radius_m),
        "class_counts": parcels["pec_class"].value_counts().to_dict(),
    }

    # Clean temp files
    for p in (tmp_dem.name, tmp_slope.name, tmp_rel.name):
        try:
            os.remove(p)
        except OSError:
            pass

    return parcels, diagnostics
