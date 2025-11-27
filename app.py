# ============================================
# Flow Corridor Importance (FCI) – Streamlit Web App
# Full hydrologic analysis + rainfall slider (0–250 mm)
# Now with parcel FCI + flow accumulation + corridor layers
# ============================================

# requirements.txt (minimum):
# geopandas, rasterio, shapely, fiona, pyproj, rasterstats, pysheds,
# numpy, requests, streamlit, folium, streamlit-folium, tqdm, matplotlib

# 1) Imports
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling as RioResampling
from rasterio.transform import array_bounds
from rasterstats import zonal_stats
from pysheds.grid import Grid
from collections import deque
import streamlit as st
import requests
import tempfile
import folium
from streamlit_folium import folium_static
import pandas as pd
from rasterio.features import rasterize
from pyproj import Transformer
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

# ============================================
# 2) Configuration - Dropbox Direct Download Links
# ============================================

DEM_URL = (
    "https://www.dropbox.com/scl/fi/lrzt81x0d501w948j6etu/"
    "Dem-demo.tif?rlkey=vyzxwmgo55pqvmem7xyn9emp1&st=j790cjgz&dl=1"
)

PARCELS_GEOJSON_URL = (
    "https://www.dropbox.com/scl/fi/dv1br78ds9mz7zdtwc1sr/"
    "grid-network.geojson?rlkey=rbq4kyi8u9nl4byzz7wu6rkq4&st=qiyefkny&dl=1"
)

CN_URL = (
    "https://www.dropbox.com/scl/fi/xfseghib9vg31loxan294/"
    "CN.tif?rlkey=6o75z9l36l8viuiivxmgiame7&st=e7jfq1vi&dl=1"
)

# Analysis parameters
ACCUM_Q = 0.90          # Corridor threshold quantile (top 10%)
EPS = 1e-9

# FCI weighting
WEIGHT_SUM      = 0.4
WEIGHT_CORRIDOR = 0.4
WEIGHT_P90      = 0.2

# Risk thresholds on FCI (0–1)
FCI_LOW_THRESHOLD  = 0.33
FCI_HIGH_THRESHOLD = 0.66

# ============================================
# 3) Helper functions
# ============================================

def nrcs_runoff_depth(P_mm, CN):
    """
    NRCS / SCS Curve Number runoff.
    Q = (P - Ia)^2 / (P - Ia + S), Ia = 0.2*S, S = (25400/CN) - 254
    """
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
    """Detect flow direction encoding scheme used by pysheds."""
    unique_vals = np.unique(fdir_array[fdir_array > 0])
    unique_vals = unique_vals[~np.isnan(unique_vals)]

    st.write("🔍 Flow Direction Analysis:")
    st.write(f"&nbsp;&nbsp;Unique flow direction values: {unique_vals}")

    power_of_2_values = {1, 2, 4, 8, 16, 32, 64, 128}
    if set(unique_vals).issubset(power_of_2_values):
        st.write("&nbsp;&nbsp;✓ Detected: Power-of-2 encoding (D8 standard)")
        direction_map = {
            1: (0, 1),     2: (1, 1),   4: (1, 0),   8: (1, -1),
            16: (0, -1),   32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
        }
        return "power_of_2", direction_map

    if set(unique_vals).issubset(set(range(8))):
        st.write("&nbsp;&nbsp;✓ Detected: Sequential 0–7 encoding")
        direction_map = {
            0: (0, 1),     1: (1, 1),   2: (1, 0),   3: (1, -1),
            4: (0, -1),    5: (-1, -1), 6: (-1, 0),  7: (-1, 1),
        }
        return "sequential_0_7", direction_map

    st.write("⚠️ WARNING: Unknown flow direction encoding – assuming power-of-2.")
    direction_map = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
    }
    return "unknown", direction_map


def accumulate_d8_validated(fdir, weights, valid_mask):
    """
    Full D8 flow accumulation with validation diagnostics.
    Returns (accumulation_array, diagnostics_dict)
    """
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
            source_cols[valid_destinations]
        )
        dest_indices = coords_to_index(
            dest_rows[valid_destinations],
            dest_cols[valid_destinations]
        )
        downstream[source_indices] = dest_indices

    incoming_degree = np.zeros(N, dtype=np.int32)
    has_downstream = downstream >= 0
    np.add.at(incoming_degree, downstream[has_downstream], 1)

    accumulation = weights.ravel().copy()
    outlets = valid_flat & (downstream == -1)
    num_outlets = np.sum(outlets)

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
    num_unprocessed = np.sum(unprocessed)

    total_input = np.sum(weights[valid_mask])
    total_accumulated = np.sum(accumulation.reshape(H, W)[valid_mask])
    outlet_accumulation = np.sum(accumulation[outlets])

    mass_balance_error = (
        abs(total_accumulated - total_input) /
        (total_input + EPS) * 100.0
    )

    diagnostics = {
        "scheme": scheme,
        "total_input": float(total_input),
        "total_accumulated": float(total_accumulated),
        "outlet_accumulation": float(outlet_accumulation),
        "num_outlets": int(num_outlets),
        "num_unprocessed": int(num_unprocessed),
        "mass_balance_error_pct": float(mass_balance_error),
        "iterations": int(iterations),
    }

    return accumulation.reshape(H, W), diagnostics


def make_raster_overlays(flow_acc, corridor_mask, valid_mask,
                         dem_transform, dem_crs, height, width):
    """
    Prepare RGBA numpy arrays + WGS84 bounds for:
      - log10(flow_acc) overlay
      - corridor mask overlay
    """
    # --- Bounds in DEM CRS ---
    left, bottom, right, top = array_bounds(height, width, dem_transform)
    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(left, bottom)
    east, north = transformer.transform(right, top)
    bounds_wgs84 = [[south, west], [north, east]]

    # --- Flow accumulation (log10) ---
    flow_display = np.where(valid_mask, flow_acc, np.nan)
    flow_log = np.log10(flow_display + 1.0)
    flow_log = np.nan_to_num(flow_log, nan=0.0)
    if np.nanmax(flow_log) > 0:
        flow_norm = flow_log / np.nanmax(flow_log)
    else:
        flow_norm = flow_log

    flow_rgba = cm.Blues(flow_norm)  # (H, W, 4) floats 0–1

    # --- Corridor mask (red overlay) ---
    corr = np.where(corridor_mask == 1, 1.0, 0.0)
    corr_rgba = np.zeros((height, width, 4), dtype=float)
    corr_rgba[..., 0] = 1.0            # red channel
    corr_rgba[..., 3] = corr * 0.8     # alpha

    return flow_rgba, corr_rgba, bounds_wgs84

# ============================================
# 4) Data download & base prep
# ============================================

@st.cache_resource(show_spinner="📥 Downloading input data from Dropbox...")
def setup_data_environment():
    data_dir = os.path.join(tempfile.gettempdir(), "fci_data")
    os.makedirs(data_dir, exist_ok=True)

    dem_path = os.path.join(data_dir, "Dem-demo.tif")
    cn_path = os.path.join(data_dir, "CN.tif")
    parcels_path = os.path.join(data_dir, "parcels.geojson")

    def download_file(url, local_path, label):
        if os.path.exists(local_path):
            return
        st.info(f"Downloading {label} ...")
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    download_file(DEM_URL, dem_path, "DEM")
    download_file(CN_URL, cn_path, "Curve Number raster")
    download_file(PARCELS_GEOJSON_URL, parcels_path, "Parcels GeoJSON")

    return dem_path, parcels_path, cn_path


@st.cache_resource(show_spinner="🔧 Preparing DEM, CN, flow directions & parcels...")
def load_base_data(DEM_PATH_IN, PARCELS_PATH, CN_RASTER_PATH):
    with rasterio.open(DEM_PATH_IN) as dem_src:
        dem_profile = dem_src.profile.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_nodata = dem_src.nodata
        height, width = dem_src.height, dem_src.width

    grid = Grid.from_raster(DEM_PATH_IN)
    dem = grid.read_raster(DEM_PATH_IN).astype("float32")

    if dem_nodata is not None:
        valid_mask = (dem != dem_nodata) & np.isfinite(dem)
    else:
        valid_mask = np.isfinite(dem)

    dem_filled = grid.fill_depressions(dem)
    dem_conditioned = grid.resolve_flats(dem_filled)
    flow_directions = grid.flowdir(dem_conditioned)

    with rasterio.open(CN_RASTER_PATH) as cn_src:
        cn_data = cn_src.read(1).astype("float32")
        cn_nodata = cn_src.nodata
        cn_transform = cn_src.transform
        cn_crs = cn_src.crs

    cn_aligned = np.empty((height, width), dtype="float32")
    cn_aligned.fill(np.nan)

    reproject(
        source=cn_data,
        destination=cn_aligned,
        src_transform=cn_transform,
        src_crs=cn_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=RioResampling.nearest,  # CN is categorical
        src_nodata=cn_nodata,
        dst_nodata=np.nan,
    )

    parcels = gpd.read_file(PARCELS_PATH)
    if parcels.crs is None:
        raise ValueError("Parcels file has no CRS defined.")
    if parcels.crs.to_string() != dem_crs.to_string():
        parcels = parcels.to_crs(dem_crs)
    if "area_m2" not in parcels.columns:
        parcels["area_m2"] = parcels.geometry.area

    base_data = {
        "dem": dem,
        "valid_mask": valid_mask,
        "flow_directions": flow_directions,
        "cn_aligned": cn_aligned,
        "dem_profile": dem_profile,
        "dem_transform": dem_transform,
        "dem_crs": dem_crs,
        "height": height,
        "width": width,
        "parcels": parcels,
    }
    return base_data

# ============================================
# 5) Full FCI analysis for a single rainfall
# ============================================

def run_fci_analysis(rainfall_mm, use_nrcs_runoff, base_data):
    """
    Full hydrologic + FCI analysis for one rainfall scenario.
    Returns:
      parcels_result (GeoDataFrame),
      diagnostics dict,
      corridor_threshold,
      corridor_cells,
      risk_counts dict,
      flow_accumulation array,
      corridor_mask array
    """
    dem = base_data["dem"]
    valid_mask = base_data["valid_mask"]
    flow_directions = base_data["flow_directions"]
    cn_aligned = base_data["cn_aligned"]
    dem_transform = base_data["dem_transform"]
    dem_profile = base_data["dem_profile"]
    height = base_data["height"]
    width = base_data["width"]
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
        corridor_threshold = np.quantile(positive_accumulation, ACCUM_Q)
    else:
        corridor_threshold = np.inf

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

    # --- FCI composite ---
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

    # --- 10-class FCI ---
    fci_class = np.floor(parcels["FCI"].values * 10).astype(int) + 1
    fci_class = np.clip(fci_class, 1, 10)
    parcels["FCI_class_10"] = fci_class

    # --- Risk ---
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

# ============================================
# 6) Streamlit App
# ============================================

def main():
    st.set_page_config(layout="wide")
    st.title("🌊 Flow Corridor Importance (FCI) – Web Tool")

    st.sidebar.header("Analysis Parameters")

    rainfall_mm = st.sidebar.slider(
        "Design Rainfall (mm)", min_value=0.0, max_value=250.0,
        value=100.0, step=5.0
    )
    use_nrcs = st.sidebar.checkbox(
        "Use NRCS Runoff Method (recommended)", value=True
    )

    st.markdown(f"### 📊 Results for **{rainfall_mm:.1f} mm** rainfall")

    try:
        DEM_PATH_IN, PARCELS_PATH, CN_RASTER_PATH = setup_data_environment()
        base_data = load_base_data(DEM_PATH_IN, PARCELS_PATH, CN_RASTER_PATH)
    except Exception as e:
        st.error("Failed to set up geospatial data environment.")
        st.exception(e)
        return

    if st.button("Run FCI Analysis"):
        with st.spinner("Running full FCI analysis ..."):
            (
                parcels_result,
                diagnostics,
                corridor_threshold,
                corridor_cells,
                risk_counts,
                flow_accumulation,
                corridor_mask,
            ) = run_fci_analysis(rainfall_mm, use_nrcs, base_data)

        st.success("✅ Analysis completed successfully!")

        # --- Diagnostics summary ---
        st.subheader("Hydrologic Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Flow direction scheme: **{diagnostics['scheme']}**")
            st.write(f"- Total input runoff: **{diagnostics['total_input']:.2f} mm**")
            st.write(f"- Total accumulated: **{diagnostics['total_accumulated']:.2f} mm**")
            st.write(f"- Outlet accumulation: **{diagnostics['outlet_accumulation']:.2f} mm**")
            st.write(f"- Mass balance error: **{diagnostics['mass_balance_error_pct']:.3f}%**")
        with col2:
            st.write(f"- Number of outlets: **{diagnostics['num_outlets']}**")
            st.write(f"- Unprocessed cells: **{diagnostics['num_unprocessed']}**")
            st.write(f"- Processing iterations: **{diagnostics['iterations']}**")
            st.write(f"- Corridor threshold (top {int((1-ACCUM_Q)*100)}%): "
                     f"**{corridor_threshold:.2f}**")
            st.write(f"- Corridor cells: **{corridor_cells:,}**")

        # --- Risk distribution ---
        st.subheader("Parcel Risk Distribution")
        risk_df = pd.DataFrame(
            [
                {"Risk": lbl,
                 "Parcels": cnt,
                 "Percent": cnt / len(parcels_result) * 100.0}
                for lbl, cnt in risk_counts.items()
        ]).set_index("Risk")
        st.table(risk_df.style.format({"Percent": "{:.1f}"}))

        # --- Map with multiple layers ---
        st.subheader("🌐 Interactive FCI Map with Additional Layers")

        parcels_wgs84 = parcels_result.to_crs(epsg=4326).copy()
        parcels_wgs84["index"] = parcels_wgs84.index.astype(str)

        centers = parcels_wgs84.geometry.centroid
        center_x = centers.x.mean()
        center_y = centers.y.mean()

        m = folium.Map(
            location=[center_y, center_x],
            zoom_start=12,
            tiles="CartoDB positron",
        )

        # 1) Parcel-level FCI choropleth
        folium.Choropleth(
            geo_data=parcels_wgs84.to_json(),
            name="Parcels – FCI (0–1)",
            data=parcels_wgs84,
            columns=["index", "FCI"],
            key_on="feature.properties.index",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"FCI (Rainfall {rainfall_mm:.1f} mm)",
        ).add_to(m)

        # 2) Parcel outlines + tooltip
        style_function = lambda x: {
            "fillColor": "#00000000",
            "color": "black",
            "weight": 0.3,
        }
        highlight_function = lambda x: {
            "fillColor": "#00000000",
            "color": "yellow",
            "weight": 1.0,
        }

        tooltip = folium.features.GeoJsonTooltip(
            fields=[
                "FCI",
                "FCI_class_10",
                "Risk",
                "fci_sum",
                "fci_corr_sum",
                "fci_p90",
            ],
            aliases=[
                "FCI:",
                "FCI class (1–10):",
                "Risk:",
                "Total Accum.:",
                "Corridor Accum.:",
                "P90 Flow:",
            ],
            localize=True,
        )

        folium.GeoJson(
            parcels_wgs84.to_json(),
            name="Parcels – outlines & tooltip",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
        ).add_to(m)

        # 3) Flow accumulation & corridor overlays as raster layers
        flow_img, corr_img, bounds_wgs84 = make_raster_overlays(
            flow_accumulation,
            corridor_mask,
            base_data["valid_mask"],
            base_data["dem_transform"],
            base_data["dem_crs"],
            base_data["height"],
            base_data["width"],
        )

        folium.raster_layers.ImageOverlay(
            image=flow_img,
            bounds=bounds_wgs84,
            name="Flow accumulation (log10)",
            opacity=0.7,
            interactive=False,
        ).add_to(m)

        folium.raster_layers.ImageOverlay(
            image=corr_img,
            bounds=bounds_wgs84,
            name="Flow corridors (top 10%)",
            opacity=0.9,
            interactive=False,
        ).add_to(m)

        folium.LayerControl().add_to(m)
        folium_static(m, width=1000, height=600)

        # --- Top table + download ---
        st.subheader("Top 10 High-FCI Parcels")
        table_cols = [
            "FCI",
            "FCI_class_10",
            "Risk",
            "fci_sum",
            "fci_corr_sum",
            "fci_p90",
            "Rainfall_mm",
        ]
        table_df = parcels_result.sort_values("FCI", ascending=False)[table_cols]
        st.dataframe(
            table_df.head(10).style.format(
                {"FCI": "{:.3f}", "fci_sum": "{:.0f}",
                 "fci_corr_sum": "{:.0f}", "fci_p90": "{:.0f}"}
            )
        )

        csv_data = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full parcel results (CSV)",
            data=csv_data,
            file_name=f"FCI_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
