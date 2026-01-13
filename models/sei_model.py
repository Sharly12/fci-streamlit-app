# models/sei_model.py
# Surrounding Exposure Index (SEI) – parcel-level implementation
# Area-weighted land-use criticality around each parcel (+ optional hazard)

import os
import re
import math
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

TIERS = ["Critical", "High", "Medium", "Low", "Very-Low"]


def normalize_minmax(arr, eps: float = 1e-9):
    """Safe min–max normalization to [0, 1]. Returns zeros if array is constant or empty."""
    arr = np.asarray(arr, dtype="float64")
    if arr.size == 0:
        return arr
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    rng = max(amax - amin, eps)
    return (arr - amin) / rng


def slug(s: object) -> str:
    """Normalize label for fuzzy prefix/contains matching."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[/\-\s]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# Land-use reclassification: raw → (canonical label, tier, weight)
RECLASS = {
    # Very-Low / Low impact natural/unused
    "abandoned":             ("Vacant/Abandoned",         "Very-Low", 0.15),
    "vacant buil":           ("Vacant Building",          "Very-Low", 0.25),
    "vacant lan":            ("Vacant Land",              "Very-Low", 0.20),
    "scrub":                 ("Scrub/Bare",               "Very-Low", 0.15),
    "vegetation":            ("Vegetation",               "Very-Low", 0.20),
    "water bodi":            ("Water Body",               "Very-Low", 0.20),
    "water body":            ("Water Body",               "Very-Low", 0.20),
    "marshy":                ("Wetland/Marsh",            "Very-Low", 0.20),
    "marshy p":              ("Wetland/Marsh",            "Very-Low", 0.20),

    # Agriculture / plantations
    "agricultur":            ("Agriculture (general)",    "Low",      0.35),
    "paddy":                 ("Agriculture - Paddy",      "Low",      0.35),
    "coconut":               ("Plantation - Coconut",     "Low",      0.30),
    "other plan":            ("Plantation - Other",       "Low",      0.30),
    "rubber":                ("Plantation - Rubber",      "Low",      0.30),

    # Parks & recreation
    "park and p":            ("Parks & Open Space",       "Low",      0.50),
    "recreation":            ("Recreation Facility",      "Medium",   0.60),
    "sports a":              ("Sports & Assembly",        "Medium",   0.60),

    # Residential
    "home garde":            ("Residential - Home Garden","Medium",   0.60),
    "homesteads":            ("Residential - Homesteads", "Medium",   0.60),
    "residentia":            ("Residential - General",    "High",     0.70),

    # Social/community/religious
    "religious":             ("Religious Facility",       "Medium",   0.70),
    "socio cult":            ("Socio-cultural Facility",  "High",     0.75),
    "sociocultu":            ("Socio-cultural Facility",  "High",     0.75),
    "community":             ("Community Facility",       "High",     0.80),

    # Institutions/admin/education/health
    "administra":            ("Administration/Institution","High",    0.80),
    "institute":             ("Administration/Institution","High",    0.80),
    "institutio":            ("Administration/Institution","High",    0.80),
    "educationa":            ("Education",                "Critical", 0.95),
    "health":                ("Health Facility",          "Critical", 1.00),
    "local auth":            ("Local Authority",          "High",     0.85),

    # Commercial/finance/tourism/industrial
    "commercial":            ("Commercial",               "Critical", 0.95),
    "bank all":              ("Financial/Banking",        "Critical", 0.90),
    "hotels":                ("Tourism - Hotels",         "High",     0.85),
    "tourism":               ("Tourism Facility",         "High",     0.80),
    "industrial":            ("Industrial",               "High",     0.80),

    # Roads/rail/transport/utilities
    "rda road":              ("Road - Arterial (RDA)",    "Critical", 0.90),
    "rda roads":             ("Road - Arterial (RDA)",    "Critical", 0.90),
    "uc road":               ("Road - UC (Collector)",    "High",     0.85),
    "other road":            ("Road - Local/Other",       "High",     0.80),
    "prda road":             ("Road - Regional/PRDA",     "High",     0.85),
    "road activ":            ("Road - Ops Yard",          "High",     0.80),
    "railway ac":            ("Railway - Facility",       "High",     0.85),
    "railway li":            ("Railway - Line",           "Critical", 0.90),
    "transporta":            ("Transport Facility",       "High",     0.90),
    "utility":               ("Utilities",                "Critical", 1.00),

    # Streams
    "streams":               ("Stream/Canal",             "Very-Low", 0.20),

    # Unclassified fallback
    "unclassifi":            ("Unclassified",             "Low",      0.30),
}


def _classify_lu_value(val_raw: object):
    """Map raw LU value → (canonical label, tier, weight)."""
    s = slug(val_raw)
    for key in sorted(RECLASS.keys(), key=len, reverse=True):
        if key in s:
            return RECLASS[key]
    return ("Unclassified", "Low", 0.30)


def run_sei_analysis(
    parcels_path: str,
    landuse_path: str,
    lu_field: str = "LU_All",
    buffer_m: float = 500.0,
    hazard_raster: str | None = None,
):
    """
    Compute Surrounding Exposure Index (SEI) for each parcel.

    Returns
    -------
    parcels_out : GeoDataFrame
        Parcels with SEI components & SEI score.
    diagnostics : dict
        Summary diagnostics.
    outputs : dict
        GeoTIFF export paths (ADDED; no analysis changes).
    """
    # --- Load data ---
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS defined.")

    lu = gpd.read_file(landuse_path)
    if lu.crs is None:
        raise ValueError("Land-use layer has no CRS defined.")

    if lu.crs != parcels.crs:
        lu = lu.to_crs(parcels.crs)

    if lu_field not in lu.columns:
        raise KeyError(f"'{lu_field}' not found in land-use attributes: {list(lu.columns)}")

    # --- Reclassify land-use ---
    canon, tier, weight, unmatched = [], [], [], []
    for v in lu[lu_field].values:
        c, t, w = _classify_lu_value(v)
        canon.append(c)
        tier.append(t)
        weight.append(w)
        if c == "Unclassified" and slug(v) not in ("unclassifi",):
            unmatched.append(v)

    lu = lu.copy()
    lu["SEI_canonical"] = canon
    lu["SEI_tier"]      = tier
    lu["SEI_weight"]    = weight

    # --- Build buffers ---
    parcels = parcels.copy()
    parcels["_buf_geom"] = parcels.geometry.buffer(buffer_m)

    # --- Area-weighted exposure within buffers ---
    lu_sindex = lu.sindex
    rows = []

    for idx, geom in parcels["_buf_geom"].items():
        if geom is None or geom.is_empty:
            rows.append((idx, 0.0, 0.0, {t: 0.0 for t in TIERS}))
            continue

        cand_idx = list(lu_sindex.intersection(geom.bounds))
        if not cand_idx:
            rows.append((idx, 0.0, 0.0, {t: 0.0 for t in TIERS}))
            continue

        sub = lu.iloc[cand_idx][["SEI_weight", "SEI_tier", "geometry"]]

        buf_gdf = gpd.GeoDataFrame({"_pid": [idx]}, geometry=[geom], crs=parcels.crs)
        inter = gpd.overlay(buf_gdf, sub, how="intersection")

        if inter.empty:
            rows.append((idx, 0.0, 0.0, {t: 0.0 for t in TIERS}))
            continue

        inter["area_m2"] = inter.geometry.area
        total_area = float(inter["area_m2"].sum())
        aw_sum     = float((inter["area_m2"] * inter["SEI_weight"]).sum())

        tier_areas = {t: 0.0 for t in TIERS}
        for t in TIERS:
            tier_areas[t] = float(inter.loc[inter["SEI_tier"] == t, "area_m2"].sum())

        rows.append((idx, total_area, aw_sum, tier_areas))

    # Attach stats
    nei_area_m2 = []
    nei_aw_sum = []
    tier_area_cols = {t: [] for t in TIERS}

    for idx, total_area, aw_sum, tier_areas in rows:
        nei_area_m2.append(total_area)
        nei_aw_sum.append(aw_sum)
        for t in TIERS:
            tier_area_cols[t].append(tier_areas[t])

    parcels["nei_area_m2"] = nei_area_m2
    parcels["nei_aw_sum"]  = nei_aw_sum
    for t in TIERS:
        col_name = f"nei_area_{t.replace('-', '_').lower()}"
        parcels[col_name] = tier_area_cols[t]

    # --- Hazard sampling (optional) ---
    if hazard_raster:
        zs = zonal_stats(
            vectors=list(parcels["_buf_geom"].values),
            raster=hazard_raster,
            stats=["mean", "max"],
            all_touched=True,
            nodata=None,
        )
        haz_mean, haz_max = [], []
        for d in zs:
            if isinstance(d, dict):
                haz_mean.append(d.get("mean", 0.0))
                haz_max.append(d.get("max", 0.0))
            else:
                haz_mean.append(0.0)
                haz_max.append(0.0)
        parcels["haz_mean"] = haz_mean
        parcels["haz_max"]  = haz_max
    else:
        parcels["haz_mean"] = 1.0
        parcels["haz_max"]  = 0.0

    # --- SEI components ---
    parcels["nei_aw_norm"] = normalize_minmax(parcels["nei_aw_sum"].values)

    haz_mean_vals = parcels["haz_mean"].values.astype("float64")
    haz_max_vals  = parcels["haz_max"].values.astype("float64")

    if hazard_raster and np.nanstd(haz_mean_vals) >= 1e-12:
        parcels["haz_norm"] = normalize_minmax(haz_mean_vals)
        parcels["haz_max_norm"] = normalize_minmax(haz_max_vals)
    else:
        parcels["haz_norm"] = 1.0
        parcels["haz_max_norm"] = 0.0

    ALPHA = 0.85
    BETA  = 0.15

    parcels["SEI_raw"] = parcels["nei_aw_norm"] * parcels["haz_norm"]

    if not hazard_raster:
        parcels["SEI"] = parcels["nei_aw_norm"]
    else:
        parcels["SEI"] = normalize_minmax(
            ALPHA * parcels["SEI_raw"].values + BETA * parcels["haz_max_norm"].values
        )

    # Diagnostics
    tier_cols = [c for c in parcels.columns if c.startswith("nei_area_")]
    tier_totals = parcels[tier_cols].sum().to_dict()

    diagnostics = {
        "n_parcels": int(len(parcels)),
        "buffer_m": float(buffer_m),
        "hazard_used": bool(hazard_raster),
        "tier_area_totals_m2": tier_totals,
        "unmatched_examples": list(pd.Series(unmatched).value_counts().head(10).index)
            if unmatched else [],
    }

    # Drop internal buffer geom
    parcels = parcels.drop(columns=["_buf_geom"])

    # ------------------------------------------------------------
    # ✅ EXPORT (ADDED ONLY; DOES NOT CHANGE ANALYSIS)
    # Rasterize SEI values to a GeoTIFF for visualization/download.
    # If hazard raster is provided, use it as the reference grid.
    # Otherwise create a simple grid over parcel bounds.
    # ------------------------------------------------------------
    tmp_dir = os.path.join(tempfile.gettempdir(), "sei_engine")
    os.makedirs(tmp_dir, exist_ok=True)

    sei_tif_path = os.path.join(tmp_dir, "SEI_index.tif")

    nodata = -9999.0

    if hazard_raster:
        with rasterio.open(hazard_raster) as src:
            ref_profile = src.profile.copy()
            ref_transform = src.transform
            ref_crs = src.crs
            height = src.height
            width = src.width

        gdf_ref = parcels.to_crs(ref_crs) if parcels.crs != ref_crs else parcels

        shapes = ((geom, float(val)) for geom, val in zip(gdf_ref.geometry, gdf_ref["SEI"].values))
        sei_grid = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=ref_transform,
            fill=nodata,
            dtype="float32",
            all_touched=True,
        )

        out_profile = ref_profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=nodata,
            compress="lzw",
        )

        with rasterio.open(sei_tif_path, "w", **out_profile) as dst:
            dst.write(sei_grid.astype("float32"), 1)

    else:
        # Create a reference grid over parcels extent
        bounds = parcels.total_bounds  # minx, miny, maxx, maxy
        minx, miny, maxx, maxy = map(float, bounds)

        # Choose resolution based on CRS type (export-only choice)
        try:
            is_geo = bool(parcels.crs and parcels.crs.is_geographic)
        except Exception:
            is_geo = False

        res = 0.00025 if is_geo else 30.0  # ~30m in degrees, or 30m in projected CRS
        width = max(1, int(math.ceil((maxx - minx) / res)))
        height = max(1, int(math.ceil((maxy - miny) / res)))

        transform = from_origin(minx, maxy, res, res)

        shapes = ((geom, float(val)) for geom, val in zip(parcels.geometry, parcels["SEI"].values))
        sei_grid = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype="float32",
            all_touched=True,
        )

        out_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": parcels.crs,
            "transform": transform,
            "nodata": nodata,
            "compress": "lzw",
        }

        with rasterio.open(sei_tif_path, "w", **out_profile) as dst:
            dst.write(sei_grid.astype("float32"), 1)

    outputs = {
        "workspace": tmp_dir,
        "sei_index_tif": sei_tif_path,
        "hazard_used": bool(hazard_raster),
    }

    return parcels, diagnostics, outputs
