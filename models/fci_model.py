# models/fci_model.py
import numpy as np
from collections import deque
from rasterstats import zonal_stats
import streamlit as st

ACCUM_Q = 0.90
EPS = 1e-9

WEIGHT_SUM      = 0.4
WEIGHT_CORRIDOR = 0.4
WEIGHT_P90      = 0.2

FCI_LOW_THRESHOLD  = 0.33
FCI_HIGH_THRESHOLD = 0.66


def nrcs_runoff_depth(P_mm, CN):
    CN = np.clip(CN, 1.0, 100.0).astype("float32")
    S = (25400.0 / CN) - 254.0
    Ia = 0.2 * S
    Pe = np.where(P_mm > Ia, ((P_mm - Ia) ** 2) / (P_mm - Ia + S), 0.0)
    return Pe.astype("float32")


def normalize_minmax(x):
    x = np.asarray(x, dtype="float64")
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if x_max - x_min < EPS:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def safe_get(stats_dict, key, default=np.nan):
    if isinstance(stats_dict, dict):
        return stats_dict.get(key, default)
    return default


def detect_pysheds_flow_direction_scheme(fdir_array):
    unique_vals = np.unique(fdir_array[fdir_array > 0])
    unique_vals = unique_vals[~np.isnan(unique_vals)]

    st.write("🔍 Flow Direction Analysis:")
    st.write(f"&nbsp;&nbsp;Unique flow direction values: {unique_vals}")

    power_of_2_values = {1, 2, 4, 8, 16, 32, 64, 128}
    if set(unique_vals).issubset(power_of_2_values):
        st.write("&nbsp;&nbsp;✓ Detected: Power-of-2 encoding (D8 standard)")
        direction_map = {
            1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
            16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
        }
        return "power_of_2", direction_map

    if set(unique_vals).issubset(set(range(8))):
        st.write("&nbsp;&nbsp;✓ Detected: Sequential 0–7 encoding")
        direction_map = {
            0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1),
            4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1),
        }
        return "sequential_0_7", direction_map

    st.write("⚠️ WARNING: Unknown flow direction encoding – assuming power-of-2.")
    direction_map = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }
    return "unknown", direction_map


def accumulate_d8_validated(fdir, weights, valid_mask):
    H, W = fdir.shape
    N = H * W

    scheme, direction_map = detect_pysheds_flow_direction_scheme(fdir)

    fdir = fdir.astype(np.int32, copy=False)
    weights = np.where(valid_mask, weights, 0.0).astype(np.float64, copy=False)

    def coords_to_index(r, c):
        return r * W + c

    downstream = np.full(N, -1, dtype=np.int64)
    valid_flat = valid_mask.ravel()

    for direction_code, (dr, dc) in direction_map.items():
        flow_mask = (fdir == direction_code) & valid_mask
        if not flow_mask.any():
            continue

        source_rows, source_cols = np.nonzero(flow_mask)
        dest_rows = source_rows + dr
        dest_cols = source_cols + dc

        valid_destinations = (
            (dest_rows >= 0) & (dest_rows < H) &
            (dest_cols >= 0) & (dest_cols < W) &
            valid_mask[dest_rows, dest_cols]
        )
        if not valid_destinations.any():
            continue

        source_indices = coords_to_index(
            source_rows[valid_destinations],
            source_cols[valid_destinations],
        )
        dest_indices = coords_to_index(
            dest_rows[valid_destinations],
            dest_cols[valid_destinations],
        )
        downstream[source_indices] = dest_indices

    incoming_degree = np.zeros(N, dtype=np.int32)
    has_downstream = downstream >= 0
    np.add.at(incoming_degree, downstream[has_downstream], 1)

    accumulation = weights.ravel().copy()
    outlets = valid_flat & (downstream == -1)
    num_outlets = int(np.sum(outlets))

    queue = deque(list(np.nonzero((incoming_degree == 0) & valid_flat)[0]))
    processed = np.zeros(N, dtype=bool)
    processed[list(queue)] = True

    iterations = 0
    max_iterations = N * 2

    while queue and iterations < max_iterations:
        current_cell = queue.popleft()
        downstream_cell = downstream[current_cell]

        if downstream_cell >= 0:
            accumulation[downstream_cell] += accumulation[current_cell]
            incoming_degree[downstream_cell] -= 1

            if (incoming_degree[downstream_cell] == 0 and
                valid_flat[downstream_cell] and
                not processed[downstream_cell]):
                processed[downstream_cell] = True
                queue.append(downstream_cell)

        iterations += 1

    unprocessed = valid_flat & ~processed
    num_unprocessed = int(np.sum(unprocessed))

    total_input = float(np.sum(weights[valid_mask]))
    total_accumulated = float(np.sum(accumulation.reshape(H, W)[valid_mask]))
    outlet_accumulation = float(np.sum(accumulation[outlets]))
    mass_balance_error = (
        abs(total_accumulated - total_input) /
        (total_input + EPS) * 100.0
    )

    diagnostics = {
        "scheme": scheme,
        "total_input": total_input,
        "total_accumulated": total_accumulated,
        "outlet_accumulation": outlet_accumulation,
        "num_outlets": num_outlets,
        "num_unprocessed": num_unprocessed,
        "mass_balance_error_pct": float(mass_balance_error),
        "iterations": int(iterations),
    }

    return accumulation.reshape(H, W), diagnostics


def run_fci_analysis(rainfall_mm, use_nrcs_runoff, base_data):
    """
    Full hydrologic + FCI analysis for one rainfall scenario.

    Returns:
      parcels_result, diagnostics, corridor_threshold, corridor_cells,
      risk_counts, flow_accumulation, corridor_mask
    """
    dem = base_data["dem"]
    valid_mask = base_data["valid_mask"]
    flow_directions = base_data["flow_directions"]
    cn_aligned = base_data["cn_aligned"]
    dem_transform = base_data["dem_transform"]
    parcels = base_data["parcels"].copy()

    # --- Runoff weights ---
    if rainfall_mm <= 0:
        runoff_weights = np.zeros_like(cn_aligned, dtype="float32")
        st.warning("Rainfall is 0 mm – all runoff and FCI scores will be 0.")
    else:
        if use_nrcs_runoff:
            runoff_weights = nrcs_runoff_depth(rainfall_mm, cn_aligned)
        else:
            runoff_weights = (rainfall_mm * (cn_aligned / 100.0)).astype("float32")

    runoff_weights = np.where(valid_mask, np.nan_to_num(runoff_weights, nan=0.0), 0.0)
    runoff_weights = np.maximum(runoff_weights, 0.0).astype("float32")

    positive_runoff = runoff_weights[runoff_weights > 0]
    if positive_runoff.size > 0:
        st.write(
            f"💧 Runoff range (valid cells): {np.min(positive_runoff):.2f} – "
            f"{np.max(runoff_weights):.2f} mm"
        )
    else:
        st.write("💧 No positive runoff values (all zeros).")

    # --- Flow accumulation ---
    st.write("🌊 Computing D8 flow accumulation ...")
    flow_accumulation, diagnostics = accumulate_d8_validated(
        flow_directions, runoff_weights, valid_mask
    )

    # --- Corridors ---
    positive_accumulation = flow_accumulation[flow_accumulation > 0]
    if positive_accumulation.size > 0:
        corridor_threshold = float(np.quantile(positive_accumulation, ACCUM_Q))
    else:
        corridor_threshold = float("inf")

    corridor_mask = (flow_accumulation >= corridor_threshold).astype("uint8")
    corridor_cells = int(np.sum(corridor_mask))
    corridor_accumulation = flow_accumulation * corridor_mask

    # --- Zonal statistics ---
    st.write("📈 Calculating parcel-level zonal statistics ...")

    zonal_all = zonal_stats(
        vectors=parcels.geometry,
        raster=flow_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["count", "sum", "mean", "max"],
        all_touched=False,
    )

    zonal_corridor = zonal_stats(
        vectors=parcels.geometry,
        raster=corridor_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["sum", "mean", "max"],
        all_touched=False,
    )

    zonal_p90_raw = zonal_stats(
        vectors=parcels.geometry,
        raster=flow_accumulation,
        affine=dem_transform,
        nodata=0.0,
        stats=["percentile_90"],
        all_touched=False,
    )

    fci_count_cells = np.array([safe_get(z, "count", 0) for z in zonal_all], dtype="float64")
    fci_sum         = np.array([safe_get(z, "sum", 0.0)   for z in zonal_all], dtype="float64")
    fci_mean        = np.array([safe_get(z, "mean", 0.0)  for z in zonal_all], dtype="float64")
    fci_max         = np.array([safe_get(z, "max", 0.0)   for z in zonal_all], dtype="float64")
    fci_p90         = np.array([safe_get(z, "percentile_90", 0.0) for z in zonal_p90_raw],
                               dtype="float64")
    fci_corr_sum    = np.array([safe_get(z, "sum", 0.0)   for z in zonal_corridor],
                               dtype="float64")
    fci_corr_mean   = np.array([safe_get(z, "mean", 0.0)  for z in zonal_corridor],
                               dtype="float64")
    fci_corr_max    = np.array([safe_get(z, "max", 0.0)   for z in zonal_corridor],
                               dtype="float64")

    parcels["fci_count"]     = fci_count_cells
    parcels["fci_sum"]       = fci_sum
    parcels["fci_mean"]      = fci_mean
    parcels["fci_max"]       = fci_max
    parcels["fci_p90"]       = fci_p90
    parcels["fci_corr_sum"]  = fci_corr_sum
    parcels["fci_corr_mean"] = fci_corr_mean
    parcels["fci_corr_max"]  = fci_corr_max

    parcels["fci_sum_norm"]      = normalize_minmax(parcels["fci_sum"].values)
    parcels["fci_corr_sum_norm"] = normalize_minmax(parcels["fci_corr_sum"].values)
    parcels["fci_p90_norm"]      = normalize_minmax(parcels["fci_p90"].values)

    parcels["FCI"] = (
        WEIGHT_SUM      * parcels["fci_sum_norm"] +
        WEIGHT_CORRIDOR * parcels["fci_corr_sum_norm"] +
        WEIGHT_P90      * parcels["fci_p90_norm"]
    )
    parcels["FCI"] = parcels["FCI"].clip(0.0, 1.0)
    parcels["Rainfall_mm"] = rainfall_mm

    fci_class = np.floor(parcels["FCI"].values * 10).astype(int) + 1
    fci_class = np.clip(fci_class, 1, 10)
    parcels["FCI_class_10"] = fci_class

    risk = np.full(len(parcels), "Low", dtype=object)
    risk[parcels["FCI"] >= FCI_HIGH_THRESHOLD] = "High"
    medium_mask = ((parcels["FCI"] >= FCI_LOW_THRESHOLD) &
                   (parcels["FCI"] < FCI_HIGH_THRESHOLD))
    risk[medium_mask] = "Medium"
    parcels["Risk"] = risk

    labels, counts = np.unique(risk, return_counts=True)
    risk_counts = dict(zip(labels, counts))

    return (
        parcels,
        diagnostics,
        corridor_threshold,
        corridor_cells,
        risk_counts,
        flow_accumulation,
        corridor_mask,
    )
