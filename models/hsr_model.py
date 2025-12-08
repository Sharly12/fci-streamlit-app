# models/hsr_model.py
# ============================================================
# HSR ENGINE – Web-app version
# - Uses preloaded DEM/CN/parcels from utils.data_loader.load_base_data
# - Computes:
#     * HSR_static (m³) – pure concavity storage volume
#     * HSR_rain   (m³) – rainfall-limited storage (SCS–CN runoff)
# - Returns parcel-level stats + raster arrays for visualization
# ============================================================

import numpy as np
import pandas as pd
from pysheds.grid import Grid
from scipy import ndimage
from scipy.ndimage import grey_closing
from rasterstats import zonal_stats


# ---------- small helper ----------

def _normalize_minmax(arr, eps: float = 1e-9):
    arr = np.asarray(arr, dtype="float64")
    if arr.size == 0:
        return arr
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    rng = max(amax - amin, eps)
    return (arr - amin) / rng


# ---------- main engine ----------

def run_hsr_analysis(
    rainfall_mm: float,
    base_data: dict,
    concavity_window: int = 7,
):
    """
    Run HSR analysis using arrays from load_base_data.

    Parameters
    ----------
    rainfall_mm : float
        Design storm depth (mm).
    base_data : dict
        Output from utils.data_loader.load_base_data(...).
    concavity_window : int
        Size of morphological window (in cells) for concavity (e.g. 7 → 7x7).

    Returns
    -------
    parcels : GeoDataFrame
        Parcels with HSR attributes and 10-class HSR_rain classification.
    diagnostics : dict
        Summary stats and processing diagnostics.
    HSR_static : 2D np.ndarray
        Static storage volume per cell (m³).
    HSR_rain_map : 2D np.ndarray
        Rainfall-limited storage volume per cell (m³).
    """
    # --------------------------------------------------------
    # Unpack shared data
    # --------------------------------------------------------
    dem = base_data["dem"].astype("float64")
    cn = base_data["cn_aligned"].astype("float64")
    dem_profile = base_data["dem_profile"]
    dem_transform = base_data["dem_transform"]
    dem_crs = base_data["dem_crs"]
    parcels = base_data["parcels"].copy()
    dem_path = base_data["dem_path"]

    cell_size = float(abs(dem_transform[0]))
    cell_area = cell_size * cell_size

    diagnostics = {
        "rainfall_mm": float(rainfall_mm),
        "concavity_window": int(concavity_window),
        "cell_size_m": cell_size,
    }

    # --------------------------------------------------------
    # STEP 1 – Prepare grid for flow routing
    # --------------------------------------------------------
    grid = Grid.from_raster(dem_path, data_name="dem")
    dem_raw = grid.read_raster(dem_path).astype("float64")

    dem_filled = grid.fill_depressions(dem_raw)
    dem_flat = grid.resolve_flats(dem_filled)

    diagnostics["dem_min"] = float(np.nanmin(dem_raw))
    diagnostics["dem_max"] = float(np.nanmax(dem_raw))

    # --------------------------------------------------------
    # STEP 2 – Concavity / storage depth (m)
    # --------------------------------------------------------
    dem_array = dem_raw.copy()
    mask_valid = np.isfinite(dem_array)
    if not np.any(mask_valid):
        # Degenerate case
        HSR_static = np.zeros_like(dem_array)
        HSR_rain_map = np.zeros_like(dem_array)
        diagnostics["n_concavities"] = 0
        return parcels, diagnostics, HSR_static, HSR_rain_map

    fill_value = float(np.nanmean(dem_array[mask_valid]))
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
        dep_labels = np.zeros_like(storage_depth, dtype="int32")
        n_deps = 0
        DEP_THRESHOLD = float("nan")
    else:
        DEP_THRESHOLD = float(np.percentile(positive, 50.0))
        concavity_mask = storage_depth >= DEP_THRESHOLD
        dep_labels, n_deps = ndimage.label(concavity_mask, structure=np.ones((3, 3)))

    diagnostics["concavity_min"] = float(positive.min()) if positive.size > 0 else 0.0
    diagnostics["concavity_max"] = float(positive.max()) if positive.size > 0 else 0.0
    diagnostics["concavity_median"] = float(np.median(positive)) if positive.size > 0 else 0.0
    diagnostics["concavity_threshold_m"] = float(DEP_THRESHOLD)
    diagnostics["n_concavities"] = int(n_deps)

    # --------------------------------------------------------
    # STEP 3 – Static storage volume (m³) per cell
    # --------------------------------------------------------
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

    diagnostics["HSR_static_total_m3"] = float(np.nansum(HSR_static))

    # --------------------------------------------------------
    # STEP 4 – Runoff (SCS–CN, mm)
    # --------------------------------------------------------
    CN = np.where((cn > 0) & np.isfinite(cn), cn, np.nan)
    S = (25400.0 / CN) - 254.0
    Ia = 0.2 * S

    rain = float(rainfall_mm)
    runoff = np.where(
        np.isfinite(CN) & (rain > Ia),
        ((rain - Ia) ** 2) / ((rain - Ia) + S),
        0.0,
    )
    runoff = np.where(np.isfinite(runoff), runoff, 0.0)

    diagnostics["runoff_mean_mm"] = float(np.nanmean(runoff))
    diagnostics["runoff_max_mm"] = float(np.nanmax(runoff))

    # --------------------------------------------------------
    # STEP 5 – Weighted flow accumulation (mm)
    # --------------------------------------------------------
    runoff_grid = runoff.astype("float64")
    fdir = grid.flowdir(dem_flat)
    wacc = grid.accumulation(fdir, weights=runoff_grid)
    wacc_array = np.array(wacc, dtype="float64")

    diagnostics["wacc_max_mm"] = float(np.nanmax(wacc_array))

    # --------------------------------------------------------
    # STEP 6 – Inflow & HSR_rain (m³)
    # --------------------------------------------------------
    if n_deps == 0:
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

    diagnostics["HSR_rain_total_m3"] = float(np.nansum(HSR_rain_map))

    # --------------------------------------------------------
    # STEP 7 – Parcel-level zonal statistics (using in-memory arrays)
    # --------------------------------------------------------
    static_stats = zonal_stats(
        parcels.geometry,
        HSR_static,
        affine=dem_transform,
        stats=["sum", "mean", "max"],
        nodata=np.nan,
        all_touched=False,
    )
    rain_stats = zonal_stats(
        parcels.geometry,
        HSR_rain_map,
        affine=dem_transform,
        stats=["sum", "mean", "max"],
        nodata=np.nan,
        all_touched=False,
    )

    parcels["HSR_static_sum"] = [d.get("sum", 0.0) for d in static_stats]
    parcels["HSR_static_mean"] = [d.get("mean", 0.0) for d in static_stats]
    parcels["HSR_static_max"] = [d.get("max", 0.0) for d in static_stats]

    parcels["HSR_rain_sum"] = [d.get("sum", 0.0) for d in rain_stats]
    parcels["HSR_rain_mean"] = [d.get("mean", 0.0) for d in rain_stats]
    parcels["HSR_rain_max"] = [d.get("max", 0.0) for d in rain_stats]

    # --------------------------------------------------------
    # STEP 8 – Normalisation & 10-class classification (parcel level)
    # Higher HSR_rain → more storage → more protection value
    # --------------------------------------------------------
    parcels["HSR_static_norm"] = _normalize_minmax(parcels["HSR_static_sum"].values)
    parcels["HSR_rain_norm"] = _normalize_minmax(parcels["HSR_rain_sum"].values)

    # 10-class deciles on rainfall-adjusted storage
    parcels["HSR_rain_class_10"] = pd.qcut(
        parcels["HSR_rain_norm"].rank(method="first"),
        q=10,
        labels=[str(i) for i in range(1, 11)],  # "1" (lowest) … "10" (highest)
    )

    diagnostics["HSR_rain_min_norm"] = float(parcels["HSR_rain_norm"].min())
    diagnostics["HSR_rain_max_norm"] = float(parcels["HSR_rain_norm"].max())

    return parcels, diagnostics, HSR_static, HSR_rain_map
