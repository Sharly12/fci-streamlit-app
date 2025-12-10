# models/uds_model.py
"""
Upstream–Downstream Sensitivity (UDS) model for the Streamlit app.

Pipeline:
1. Build parcel-level hydrologic graph from DEM:
   - For each parcel, find the pour point (max flow accumulation cell).
   - Follow D8 flow directions to find downstream parcel → directed edge.
2. Compute structural UDS metrics:
   - uds_up   : number of upstream parcels (ancestors)
   - uds_down : number of downstream parcels (descendants)
   - uds_score      = 0.5 * (upstream area fraction) + 0.5 * (downstream parcel fraction)
   - uds_score_norm : min–max normalized uds_score
3. Extract Curve Number (CN) per parcel from CN raster.
4. For a chosen rainfall depth (mm), compute SCS–CN runoff depth.
5. Combine UDS + runoff:
   - runoff_norm       : min–max normalized runoff_mm
   - UDS_runoff_index  = 0.5 * uds_down + 0.5 * runoff_norm
   - UDS_runoff_norm   : min–max normalized UDS_runoff_index

Returns:
    parcels_uds : GeoDataFrame with all metrics
    diagnostics : dict
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pysheds.grid import Grid
from rasterstats import zonal_stats
import networkx as nx


def _minmax_norm(arr):
    """Safe min–max normalization for 1D numpy array."""
    arr = np.asarray(arr, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    amin = float(np.nanmin(finite))
    amax = float(np.nanmax(finite))
    if not np.isfinite(amin) or not np.isfinite(amax) or np.isclose(amin, amax):
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def run_uds_analysis(
    dem_path: str,
    cn_path: str,
    parcels_path: str,
    rainfall_mm: float = 100.0,
    max_steps: int = 5000,
):
    """
    Run full UDS + CN–runoff analysis.

    Args
    ----
    dem_path      : path to DEM raster
    cn_path       : path to Curve Number raster
    parcels_path  : path to parcel polygons (GeoJSON / Shapefile / etc.)
    rainfall_mm   : rainfall depth (mm) used for SCS–CN runoff
    max_steps     : max D8 steps when tracing flow from pour point

    Returns
    -------
    parcels : GeoDataFrame with UDS and runoff metrics
    diagnostics : dict of summary info
    """

    # --------------------------------------------------------------
    # LOAD PARCELS + ALIGN TO DEM CRS
    # --------------------------------------------------------------
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS. Please define it and re-save.")

    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_height, dem_width = dem_src.height, dem_src.width

    if str(parcels.crs) != str(dem_crs):
        parcels = parcels.to_crs(dem_crs)

    # Ensure a clean integer parcel ID field
    parcels = parcels.reset_index(drop=True)
    parcels["grid_id"] = np.arange(1, len(parcels) + 1, dtype="int32")

    n_parcels = len(parcels)

    # --------------------------------------------------------------
    # FLOW DIRECTION + ACCUMULATION FROM DEM
    # --------------------------------------------------------------
    gridDEM = Grid.from_raster(dem_path)
    dem = gridDEM.read_raster(dem_path)

    dem_resolved = gridDEM.resolve_flats(dem)
    fdir = gridDEM.flowdir(dem_resolved)
    acc = gridDEM.accumulation(fdir)

    # --------------------------------------------------------------
    # RASTERIZE PARCELS (grid_id per cell)
    # --------------------------------------------------------------
    shapes = (
        (geom, int(pid))
        for geom, pid in zip(parcels.geometry, parcels["grid_id"])
    )

    parcel_raster = rasterize(
        shapes,
        out_shape=(dem_height, dem_width),
        transform=dem_transform,
        fill=0,
        dtype="int32",
    )

    # --------------------------------------------------------------
    # FIND POUR POINTS (max accumulation cell in each parcel)
    # --------------------------------------------------------------
    parcel_pour = {}
    for pid in parcels["grid_id"]:
        mask_cell = parcel_raster == int(pid)
        if not mask_cell.any():
            continue
        vals = acc[mask_cell]
        rows, cols = np.where(mask_cell)
        idx = int(np.argmax(vals))
        parcel_pour[int(pid)] = (int(rows[idx]), int(cols[idx]))

    # --------------------------------------------------------------
    # BUILD FLOW GRAPH BETWEEN PARCELS
    # --------------------------------------------------------------
    d8_to_offset = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    }

    nrows, ncols = parcel_raster.shape
    G = nx.DiGraph()

    for pid in parcels["grid_id"]:
        G.add_node(int(pid))

    for pid, (r, c) in parcel_pour.items():
        cur_r, cur_c = int(r), int(c)

        for _ in range(max_steps):
            if cur_r < 0 or cur_r >= nrows or cur_c < 0 or cur_c >= ncols:
                break

            code = int(fdir[cur_r, cur_c])
            if code not in d8_to_offset:
                break

            dr, dc = d8_to_offset[code]
            nr, nc = cur_r + dr, cur_c + dc

            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                break

            downstream_pid = int(parcel_raster[nr, nc])

            # Found a downstream parcel different from source
            if downstream_pid != 0 and downstream_pid != pid:
                G.add_edge(int(pid), int(downstream_pid))
                break

            cur_r, cur_c = nr, nc

    # --------------------------------------------------------------
    # UDS METRICS
    # --------------------------------------------------------------
    parcels["area_m2"] = parcels.geometry.area
    total_area = float(parcels["area_m2"].sum())
    total_cells = float(len(parcels))

    uds_scores = []
    ups = []
    downs = []

    for pid in parcels["grid_id"]:
        up = nx.ancestors(G, int(pid))
        down = nx.descendants(G, int(pid))

        up_area = float(
            parcels.loc[parcels["grid_id"].isin(up), "area_m2"].sum()
        )
        up_norm = up_area / total_area if total_area > 0 else 0.0
        down_norm = len(down) / total_cells if total_cells > 0 else 0.0

        uds_val = 0.5 * up_norm + 0.5 * down_norm

        uds_scores.append(uds_val)
        ups.append(len(up))
        downs.append(len(down))

    parcels["uds_score"] = np.array(uds_scores, dtype="float64")
    parcels["uds_up"] = np.array(ups, dtype="int32")
    parcels["uds_down"] = np.array(downs, dtype="int32")
    parcels["uds_score_norm"] = _minmax_norm(parcels["uds_score"].values)

    # --------------------------------------------------------------
    # CN EXTRACTION VIA ZONAL STATS
    # --------------------------------------------------------------
    with rasterio.open(cn_path) as cn_src:
        cn_crs = cn_src.crs

    if str(parcels.crs) != str(cn_crs):
        parcels_cn = parcels.to_crs(cn_crs)
    else:
        parcels_cn = parcels.copy()

    zs = zonal_stats(
        parcels_cn,
        cn_path,
        stats=["mean"],
        nodata=None,
    )

    cn_vals = np.array(
        [
            d.get("mean", np.nan) if isinstance(d, dict) else np.nan
            for d in zs
        ],
        dtype="float64",
    )
    parcels["CN"] = cn_vals

    # --------------------------------------------------------------
    # SCS–CN RUNOFF FOR SELECTED RAINFALL
    # --------------------------------------------------------------
    P = float(rainfall_mm)
    CN = parcels["CN"].values.astype("float64")

    runoff_mm = np.zeros_like(CN, dtype="float64")

    valid = np.isfinite(CN) & (CN > 0)
    S = np.zeros_like(CN, dtype="float64")
    S[valid] = (25400.0 / CN[valid]) - 254.0
    Ia = 0.2 * S

    cond = valid & (P > Ia)
    runoff_mm[cond] = ((P - Ia[cond]) ** 2) / (P - Ia[cond] + S[cond])

    parcels["runoff_mm"] = runoff_mm
    parcels["runoff_norm"] = _minmax_norm(runoff_mm)

    # --------------------------------------------------------------
    # COMBINED UDS × RUNOFF HAZARD INDEX
    # --------------------------------------------------------------
    uds_down_arr = parcels["uds_down"].values.astype("float64")
    uds_runoff_index = 0.5 * uds_down_arr + 0.5 * parcels["runoff_norm"].values

    parcels["UDS_runoff_index"] = uds_runoff_index
    parcels["UDS_runoff_norm"] = _minmax_norm(uds_runoff_index)
    parcels["Rainfall_mm"] = P

    # --------------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------------
    diagnostics = {
        "n_parcels": int(n_parcels),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "parcels_with_pour": int(len(parcel_pour)),
        "total_area_m2": float(total_area),
        "rainfall_mm": P,
        "cn_valid_parcels": int(np.isfinite(CN[valid]).sum()),
    }

    return parcels, diagnostics
