# pages/1_HSR_Analysis.py

import streamlit as st
from streamlit_folium import folium_static

from models.hsr_model import run_hsr_analysis
from utils.data_loader import get_data_paths
from utils.hsr_visualization import build_hsr_map

st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    "This model detects concave storage zones in the terrain and estimates "
    "their static capacity and rainfall-filled storage at parcel (grid) level."
)

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("HSR Parameters")
    rainfall_mm = st.slider(
        "Design rainfall (mm)",
        min_value=0.0,
        max_value=250.0,
        value=100.0,
        step=5.0,
    )
    concavity_window = st.slider(
        "Concavity window (cells)",
        min_value=3,
        max_value=15,
        value=7,
        step=2,
        help="Size of the moving window (e.g. 7 → 7×7 cells) used to detect concavities.",
    )

run_btn = st.button("Run HSR Analysis")

# ------------------------------------------------------------------
# Run model
# ------------------------------------------------------------------
if run_btn:
    dem_path, parcels_path, cn_path = get_data_paths()

    with st.spinner("Running HSR ENGINE v4 …"):
        parcels_hsr, diagnostics = run_hsr_analysis(
            dem_path=dem_path,
            cn_path=cn_path,
            parcels_path=parcels_path,
            rainfall_mm=rainfall_mm,
            concavity_window=concavity_window,
        )

    st.success("✅ HSR analysis complete")

    # ---- Safely extract diagnostic values (with fallbacks) ----
    rain_val = diagnostics.get("rainfall_mm", rainfall_mm)
    cell_size = diagnostics.get("cell_size_m", 30.0)
    conc_win = diagnostics.get("concavity_window", concavity_window)

    depth_thr = diagnostics.get(
        "depth_threshold_m",
        diagnostics.get("depth_threshold", 0.0),
    )

    num_patches = diagnostics.get("num_patches", len(parcels_hsr))

    total_static = diagnostics.get(
        "total_static_storage",
        float(parcels_hsr.get("HSR_static_sum", 0).sum()),
    )
    total_rain = diagnostics.get(
        "total_rain_filled",
        float(parcels_hsr.get("HSR_rain_sum", 0).sum()),
    )

    # ------------------------------------------------------------------
    # Diagnostics panel
    # ------------------------------------------------------------------
    st.subheader("Hydrological Storage Diagnostics")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rainfall (mm)", f"{rain_val:.1f}")
        st.metric("Cell size (m)", f"{cell_size:.1f}")

    with c2:
        st.metric("Concavity window (cells)", conc_win)
        st.metric("Depth threshold (m)", f"{depth_thr:.3f}")

    with c3:
        st.metric("Storage patches", num_patches)
        st.metric("Total storage (m³)", f"{total_static:.0f}")

    st.metric(
        "Total rainfall-filled storage (m³)",
        f"{total_rain:.0f}",
    )

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    st.subheader("Interactive HSR Map")
    hsr_map = build_hsr_map(parcels_hsr, rain_val)
    folium_static(hsr_map, width=1000, height=600)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    st.subheader("Top 10 parcels by rainfall-filled storage")
    cols = [
        "HSR_rain_sum",
        "HSR_static_sum",
        "HSR_rain_mean",
        "HSR_static_mean",
        "HSR_rain_max",
        "HSR_static_max",
    ]
    existing = [c for c in cols if c in parcels_hsr.columns]

    st.dataframe(
        parcels_hsr.sort_values("HSR_rain_sum", ascending=False)[existing]
        .head(10)
        .round(1)
    )

else:
    st.info(
        "Choose rainfall and concavity window on the left, then click "
        "**Run HSR Analysis**."
    )
