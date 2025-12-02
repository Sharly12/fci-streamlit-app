# models/fci_model.py

import numpy as np
import pandas as pd
from collections import deque
from rasterstats import zonal_stats

# ------------------------------
# Configuration
# ------------------------------
ACCUM_Q = 0.90        # quantile for identifying flow corridors (top 10%)
EPS = 1e-9

# Component weights for structural FCI
WEIGHT_SUM = 0.4        # contribution from total accumulation
WEIGHT_CORRIDOR = 0.4   # contribution from corridor accumulation
WEIGHT_P90 = 0.2        # contribution from peak flow (P90)

# Maximum rainfall used for scaling (should match your slider max)
RAIN_MAX_FOR_SCALING = 250.0  # mm


# ------------------------------
# Helper functions
# ------------------------------
def nrcs_runoff_depth(P_mm, CN):
    """
    NRCS/SCS Curve Number runoff depth (mm).
    P_mm : scalar rainfall depth in mm
    CN   : 2D array of curve numbers
    """
    CN = np.clip(CN, 1.0, 100.0).astype("float32")
    S = (25400.0 / CN) - 254.0      # mm
    Ia = 0.2 * S                    # initial abstraction

    # Broadcast P_mm over the grid
    P = np.asarray(P_mm, dtype="float32")
    Pe = np.where(
        P > Ia,
        ((P - Ia) ** 2) / (P - Ia + S + EPS),
        0.0,
    )
    return Pe.astype("float32")


def normalize_minmax(x, eps=EPS):
    """
    Min-max normalization with safety:
    returns 0 if array is constant or empty.
    """
    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        return x
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    rng = max(x_max - x_min, eps)
    return (x - x_min) / rng


def safe_get(stats_dict, key, default=np.nan):
    """Safely extract value from a zonal_stats result (dict)."""
    if isinstance(stats_dict, dict):
        return stats_dict.get(key, default)
    return default


def detect_pysheds_flow_direction_scheme(fdir_array):
    """
    Detect which flow direction encoding scheme pysheds is using.
    Returns: (scheme_name, direction_map)
    """
    unique_vals = np.unique(fdir_array[fdir_array > 0])
    unique_vals = unique_vals[~np.isnan(unique_vals)]

    # Power-of-2 (classic D8)
    power_of_2_values = {1, 2, 4, 8, 16, 32, 64, 128}
    if set(unique_vals).issubset(power_of_2_values):
        direction_map = {
            1:   (0, 1),    # E
            2:   (1, 1),    # SE
            4:   (1, 0),    # S
            8:   (1, -1),   # SW
            16:  (0, -1),   # W
            32:  (-1, -1),  # NW
            64:  (-1, 0),   # N
            128: (-1, 1),   # NE
        }
        return "power_of_2", direction_map

    # Sequential 0–7 encoding
    if set(unique_vals).issubset(set(range(8))):
        direction_map = {
            0: (0, 1),     # E
            1: (1, 1),     # SE
            2: (1, 0),     # S
            3: (1, -1),    # SW
            4: (0, -1),    # W
            5: (-1, -1),   # NW
            6: (-1, 0),    # N
            7: (-1, 1),    # NE
        }
        return "sequential_0_7", direction_map

    # Fallback
    direction_map = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }
    return "unknown", direction_map


def accumulate_d8_validated(fdir, weights, valid_mask):
    """
    Enhanced D8 flow accumulation with validation and diagnostics.
    fdir        : 2D array of flow directions (pysheds)
    weights     : 2D array of runoff weights (mm or unit flow)
    valid_mask  : 2D boolean array of valid cells

    Returns:
      accumulation : 2D array
      diagnostics  : dict with mass-balance checks, etc.
    """
    H, W = fdir.shape
    N = H * W

    # Detect encoding
    scheme, direction_map = detect_pysheds_flow_direction_scheme(fdir)

    fdir = fdir.astype(np.int32, copy=False)
    weights = np.where(valid_mask, weights, 0.0).astype(np.float64, copy=False)

    def coords_to_index(r, c):
        return r * W + c

    # Build downstream connectivity
    downstream = np.full(N, -1, dtype=np.int64)
    valid_flat = valid_mask.ravel()

    for code, (dr, dc) in direction_map.items():
        flow_mask = (fdir == code) & valid_mask
        if not flow_mask.any():
            continue

        src_r, src_c = np.nonzero(flow_mask)
        dst_r = src_r + dr
        dst_c = src_c + dc

        in_bounds = (
            (dst_r >= 0) & (dst_r < H) &
            (dst_c >= 0) & (dst_c < W) &
            valid_mask[dst_r, dst_c]
        )
        if not in_bounds.any():
            continue

        src_idx = coords_to_index(src_r[in_bounds], src_c[in_bounds])
        dst_idx = coords_to_index(dst_r[in_bounds], dst_c[in_bounds])
        downstream[src_idx] = dst_idx

    # Incoming degree
    incoming_degree = np.zeros(N, dtype=np.int32)
    has_downstream = downstream >= 0
    np.add.at(incoming_degree, downstream[has_downstream], 1)

    accumulation = weights.ravel().copy()

    # Outlets: valid cells with no downstream
    outlets = valid_flat & (downstream == -1)
    num_outlets = int(np.sum(outlets))

    # Topological traversal
    queue = deque(list(np.nonzero((incoming_degree == 0) & valid_flat)[0]))
    processed = np.zeros(N, dtype=bool)
    processed[list(queue)] = True

    iterations = 0
    max_iterations = N * 2

    while queue and iterations < max_iterations:
        current = queue.popleft()
        dst = downstream[current]

        if dst >= 0:
            accumulation[dst] += accumulation[current]
            incoming_degree[dst] -= 1

            if (incoming_degree[dst] == 0 and
                valid_flat[dst] and
                not processed[dst]):
                processed[dst] = True
                queue.append(dst)

        iterations += 1

    # Diagnostics
    unprocessed = valid_flat & ~processed
    num_unprocessed = int(np.sum(unprocessed))

    total_input = float(np.sum(weights[valid_mask]))
    total_accumulated = float(np.sum(accumulation.reshape(H, W)[valid_mask]))
    outlet_accumulation = float(np.sum(accumulation[outlets]))

    mass_balance_error = (
        abs(total_accumulated - total_input) / (total_input + EPS) * 100.0
        if total_input > 0 else 0.0
    )

    diagnostics = {
        "scheme": scheme,
        "total_input": total_input,
        "total_accumulated": total_accumulated,
        "outlet_accumulation": outlet_accumulation,
        "num_outlets": num_outlets,
        "num_unprocessed": num_unprocessed,
        "mass_balance_error_pct": mass_balance_error,
        "iterations": iterations,
    }

    return accumulation.reshape(H, W), diagnostics


# ------------------------------
# Main FCI analysis
# ------------------------------
def run_fci_analysis(rainfall_mm, use_nrcs_runoff, base_data):
    """
    Run Flow Corridor Importance (FCI) analysis for a given rainfall.

    Parameters
    ----------
    rainfall_mm : float
        Design rainfall (mm).
    use_nrcs_runoff : bool
        If True, use NRCS SCS CN runoff; else simple CN scaling.
    base_data : dict
        Output of utils.data_loader.load_base_data, with keys:
        - dem, valid_mask, flow_directions, cn_aligned,
          dem_profile, dem_transform, dem_crs,
          height, width, parcels (GeoDataFrame)

    Returns
    -------
    parcels : GeoDataFrame
        Parcels with FCI, FCI_struct, risk classes, zonal stats.
    diagnostics : dict
        Flow accumulation diagnostics.
    corridor_threshold : float
        Accumulation threshold used to define corridors.
    corridor_cells : int
        Number of corridor cells.
    risk_counts : dict
        Counts of parcels in each risk class.
    flow_accumulation : 2D np.ndarray
        Flow accumulation grid.
    corridor_mask : 2D np.ndarray (uint8)
        1 where corridors, 0 otherwise.
    """
    dem = base_data["dem"]
    valid_mask = base_data["valid_mask"]
    flow_directions = base_data["flow_directions"]
    cn_aligned = base_data["cn_aligned"]
    dem_profile = base_data["dem_profile"]
    dem_transform = base_data["dem_transform"]
    dem_crs = base_data["dem_crs"]
    height = base_data["height"]
    width = base_data["width"]
    parcels = base_data["parcels"].copy()

    # --------------------------
    # 1) Runoff weights
    # --------------------------
    if use_nrcs_runoff:
        runoff_weights = nrcs_runoff_depth(rainfall_mm, cn_aligned)
    else:
        runoff_weights = (rainfall_mm * (cn_aligned / 100.0)).astype("float32")

    runoff_weights = np.where(valid_mask, np.nan_to_num(runoff_weights, nan=0.0), 0.0)
    runoff_weights = np.maximum(runoff_weights, 0.0).astype("float32")

    # --------------------------
    # 2) Flow accumulation
    # --------------------------
    flow_accumulation, diagnostics = accumulate_d8_validated(
        flow_directions, runoff_weights, valid_mask
    )

    # --------------------------
    # 3) Flow corridors (still relative, top 10%)
    # --------------------------
    positive = flow_accumulation[flow_accumulation > 0]
    if positive.size > 0:
        corridor_threshold = float(np.quantile(positive, ACCUM_Q))
    else:
        corridor_threshold = float("inf")

    corridor_mask = (flow_accumulation >= corridor_threshold).astype("uint8")
    corridor_cells = int(np.sum(corridor_mask))

    # --------------------------
    # 4) Zonal statistics per parcel
    # --------------------------
    # All accumulation
    zonal_all = zonal_stats(
        vectors=parcels.geometry,
        raster=flow_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["count", "sum", "mean", "max"],
        all_touched=False,
    )

    # Corridor-only accumulation
    corridor_accumulation = flow_accumulation * corridor_mask
    zonal_corridor = zonal_stats(
        vectors=parcels.geometry,
        raster=corridor_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["sum", "mean", "max"],
        all_touched=False,
    )

    # 90th percentile of accumulation
    zonal_p90_raw = zonal_stats(
        vectors=parcels.geometry,
        raster=flow_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["percentile_90"],
        all_touched=False,
    )

    # Attach zonal stats
    parcels["fci_count_cells"] = [safe_get(z, "count", 0) for z in zonal_all]
    parcels["fci_sum"] = [safe_get(z, "sum", 0.0) for z in zonal_all]
    parcels["fci_mean"] = [safe_get(z, "mean", 0.0) for z in zonal_all]
    parcels["fci_max"] = [safe_get(z, "max", 0.0) for z in zonal_all]
    parcels["fci_p90"] = [safe_get(z, "percentile_90", 0.0) for z in zonal_p90_raw]

    parcels["fci_corr_sum"] = [safe_get(z, "sum", 0.0) for z in zonal_corridor]
    parcels["fci_corr_mean"] = [safe_get(z, "mean", 0.0) for z in zonal_corridor]
    parcels["fci_corr_max"] = [safe_get(z, "max", 0.0) for z in zonal_corridor]

    # --------------------------
    # 5) Structural FCI (0–1, rainfall-independent)
    # --------------------------
    parcels["fci_sum_norm"] = normalize_minmax(parcels["fci_sum"].values)
    parcels["fci_corr_sum_norm"] = normalize_minmax(parcels["fci_corr_sum"].values)
    parcels["fci_p90_norm"] = normalize_minmax(parcels["fci_p90"].values)

    parcels["FCI_struct"] = (
        WEIGHT_SUM * parcels["fci_sum_norm"]
        + WEIGHT_CORRIDOR * parcels["fci_corr_sum_norm"]
        + WEIGHT_P90 * parcels["fci_p90_norm"]
    )

    # --------------------------
    # 6) Rainfall scaling – new FCI
    # --------------------------
    if RAIN_MAX_FOR_SCALING > 0:
        P_norm = float(
            np.clip(rainfall_mm / float(RAIN_MAX_FOR_SCALING), 0.0, 1.0)
        )
    else:
        P_norm = 0.0

    parcels["Rainfall_mm"] = float(rainfall_mm)
    parcels["P_norm"] = P_norm  # useful for debugging / plotting

    # NEW rainfall-scaled FCI used for risk & maps
    parcels["FCI"] = parcels["FCI_struct"] * P_norm

    # --------------------------
    # 7) Risk classification on scaled FCI
    # --------------------------
    # 10-class FCI for detailed mapping
    if parcels["FCI"].max() > 0:
        bins_10 = np.linspace(0.0, 1.0, 11)
        labels_10 = [str(i) for i in range(1, 11)]
        parcels["FCI_class_10"] = pd.cut(
            parcels["FCI"],
            bins=bins_10,
            labels=labels_10,
            include_lowest=True,
        )
    else:
        # All effectively zero: treat as lowest class
        parcels["FCI_class_10"] = "1"

    # 3-class risk (you can tune these thresholds)
    risk_bins = [0.0, 0.2, 0.5, 1.0]
    risk_labels = ["Low", "Medium", "High"]
    parcels["Risk"] = pd.cut(
        parcels["FCI"],
        bins=risk_bins,
        labels=risk_labels,
        include_lowest=True,
    )

    risk_counts = parcels["Risk"].value_counts().to_dict()

    return (
        parcels,
        diagnostics,
        corridor_threshold,
        corridor_cells,
        risk_counts,
        flow_accumulation,
        corridor_mask,
    )
