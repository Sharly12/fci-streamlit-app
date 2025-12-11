# models/pec_model.py
import numpy as np
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from pysheds.grid import Grid
from scipy.ndimage import uniform_filter
from typing import Tuple, Dict

# Discrete PEC classes and colours (match original notebook)
PEC_COLORS = {
    "Low-lying Depressed (Retention Priority)": "blue",
    "Flat & Pressured (High Flood Exposure Risk)": "red",
    "Locally High & Disconnected": "green",
    "Moderate / Context-Dependent": "yellow",
}

PEC_CLASS_TO_CODE = {
    name: code for code, name in enumerate(PEC_COLORS.keys(), start=1)
}

# D8 encoding used by pysheds
_D8_OFFSETS = {
    1:   (0, 1),   # E
    2:   (1, 1),   # SE
    4:   (1, 0),   # S
    8:   (1, -1),  # SW
    16:  (0, -1),  # W
    32:  (-1, -1), # NW
    64:  (-1, 0),  # N
    128: (-1, 1),  # NE
}


def _compute_hand_from_streams(
    dem: np.ndarray,
    fdir: np.ndarray,
    streams: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Approximate HAND (height above nearest drainage) by traversing
    upstream from stream cells along the D8 flow network.
    """
    nrows, ncols = dem.shape
    N = nrows * ncols

    dem_flat = dem.reshape(N)
    valid_flat = valid_mask.reshape(N)
    streams_flat = streams.reshape(N)

    # Index of downstream cell for each cell (-1 if none)
    down_index = np.full(N, -1, dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            if not valid_mask[r, c]:
                continue
            code = int(fdir[r, c])
            if code not in _D8_OFFSETS:
                continue
            dr, dc = _D8_OFFSETS[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols and valid_mask[nr, nc]:
                down_index[idx] = nr * ncols + nc

    # Build upstream adjacency list
    upstream = [[] for _ in range(N)]
    for i in range(N):
        j = down_index[i]
        if j >= 0:
            upstream[j].append(i)

    # BFS from streams upstream
    from collections import deque
    hand_flat = np.full(N, np.nan, dtype="float32")
    q = deque()

    for i in range(N):
        if valid_flat[i] and streams_flat[i]:
            hand_flat[i] = 0.0
            q.append(i)

    # propagate upstream
    while q:
        j = q.popleft()
        for i in upstream[j]:
            if not valid_flat[i]:
                continue
            if not np.isnan(hand_flat[i]):
                continue
            drop = dem_flat[i] - dem_flat[j]
            if drop < 0:
                drop = 0.0
            hand_flat[i] = hand_flat[j] + drop
            q.append(i)

    # cells not connected to a stream: treat as 0 HAND
    mask_unset = np.isnan(hand_flat) & valid_flat
    hand_flat[mask_unset] = 0.0
    return hand_flat.reshape(nrows, ncols)


def run_pec_base(
    dem_path: str,
    parcels_path: str,
    prei_radius_m: float = 250.0,
    stream_threshold: float = 400.0,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Run the base PEC workflow (no rainfall adjustment yet).

    Returns (parcels_gdf, diagnostics).
    """
    # --- DEM ---
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        res_x = src.res[0]
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    valid_mask = np.isfinite(dem)

    # --- Parcels ---
    parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        raise ValueError("Parcels layer has no CRS.")
    if parcels.crs != crs:
        parcels = parcels.to_crs(crs)
    if "grid_id" not in parcels.columns:
        parcels["grid_id"] = np.arange(1, len(parcels) + 1)

    grid_gdf = parcels.copy()

    # --- Flow routing with pysheds ---
    grid = Grid.from_raster(dem_path)
    dem_flow = grid.read_raster(dem_path).astype("float64")
    dem_flow = np.where(~valid_mask, np.nan, dem_flow)

    dem_filled = grid.fill_depressions(dem_flow)
    dem_flat = grid.resolve_flats(dem_filled)
    fdir = grid.flowdir(dem_flat)
    acc = grid.accumulation(fdir)
    acc = np.where(~valid_mask, 0.0, acc)

    # --- Slope (degrees) ---
    cellsize_x = profile["transform"][0]
    cellsize_y = -profile["transform"][4]
    gy, gx = np.gradient(dem_filled, cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~valid_mask] = np.nan

    # --- PREI: DEM - neighbourhood mean ---
    radius_cells = max(1, int(round(prei_radius_m / res_x)))
    size = radius_cells * 2 + 1

    valid_float = np.isfinite(dem_filled).astype("float32")
    dem_zero = np.where(valid_float, dem_filled, 0.0)

    mean_dem = uniform_filter(dem_zero, size=size, mode="nearest")
    mean_mask = uniform_filter(valid_float, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        neigh_mean = np.where(
            mean_mask > 0,
            mean_dem / np.maximum(mean_mask, 1e-6),
            np.nan,
        )
    dem_rel = dem_filled - neigh_mean
    dem_rel[~valid_mask] = np.nan

    # --- Streams and HAND ---
    streams = acc >= float(stream_threshold)
    hand = _compute_hand_from_streams(dem_filled, fdir, streams, valid_mask)

    # --- Parcel zonal stats ---
    dem_stats = zonal_stats(
        grid_gdf,
        dem_filled,
        affine=transform,
        stats=["min", "max", "mean", "median", "std"],
        nodata=np.nan,
    )
    slope_stats = zonal_stats(
        grid_gdf,
        slope_deg,
        affine=transform,
        stats=["mean", "median"],
        nodata=np.nan,
    )
    rel_stats = zonal_stats(
        grid_gdf,
        dem_rel,
        affine=transform,
        stats=["mean", "min", "max"],
        nodata=np.nan,
    )
    hand_stats = zonal_stats(
        grid_gdf,
        hand,
        affine=transform,
        stats=["min", "mean"],
        nodata=np.nan,
    )

    grid_gdf["dem_min"] = [d["min"] for d in dem_stats]
    grid_gdf["dem_max"] = [d["max"] for d in dem_stats]
    grid_gdf["dem_mean"] = [d["mean"] for d in dem_stats]
    grid_gdf["dem_median"] = [d["median"] for d in dem_stats]
    grid_gdf["dem_std"] = [d["std"] for d in dem_stats]

    grid_gdf["slp_mean"] = [d["mean"] for d in slope_stats]
    grid_gdf["slp_median"] = [d["median"] for d in slope_stats]

    grid_gdf["rel_mean"] = [d["mean"] for d in rel_stats]
    grid_gdf["rel_min"] = [d["min"] for d in rel_stats]
    grid_gdf["rel_max"] = [d["max"] for d in rel_stats]

    grid_gdf["hand_min"] = [d["min"] for d in hand_stats]
    grid_gdf["hand_mean"] = [d["mean"] for d in hand_stats]

    # --- Derived indicators ("static" PEC) ---
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
    grid_gdf[cols_to_fill] = grid_gdf[cols_to_fill].fillna(0)

    diagnostics = {
        "n_parc
