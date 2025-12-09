# pages/1_HSR_Analysis.py
import streamlit as st
import pandas as pd
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.hsr_model import run_hsr_analysis
from utils.hsr_visualization import build_hsr_map

st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    "This module estimates how much water can be temporarily stored in "
    "topographic concavities (HSR_static) and how much of that storage is "
    "actually filled under a given storm (HSR_rain), and aggregates the "
    "results at parcel level."
)

# ------------------------------------------------------------------
# Get shared DEM / Parcels / CN paths (from local or Dropbox)
# ------------------------------------------------------------------
DEM_PATH_IN, PARCELS_PATH, CN_RASTER_PATH = get_data_paths()

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("HSR Parameters")

    rainfall_mm = st.slider(
        "Design Rainfall (mm)",
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
        help="Size of the neighbourhood (e.g. 7 → 7×7 cells) used to detect concavities.",
    )

    target_res = st.number_input(
        "Target DEM resolution (m)",
        min_value=10.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
        help="DEM/CN will be reprojected to this resolution in UTM for HSR.",
    )

run_btn = st.button("Run HSR Analysis")

# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------
if run_btn:
    try:
        with st.spinner("Running HSR engine… this may take some time for large areas."):
            parcels_hsr, diagnostics = run_hsr_analysis(
                rainfall_mm=rainfall_mm,
                dem_path=DEM_PATH_IN,
                cn_path=CN_RASTER_PATH,
                parcels_path=PARCELS_PATH,
                target_resolution=target_res,
                concavity_window=concavity_window,
            )

        st.success("✅ HSR analysis complete")

        # Diagnostics
        st.subheader("Hydrological Storage Diagnostics")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rainfall (mm)", f"{diagnostics['rainfall_mm']:.1f}")
            st.metric("Cell size (m)", f"{diagnostics['cell_size_m']:.1f}")

        with c2:
            st.metric("Concavity window (cells)", diagnostics["concavity_window"])
            st.metric("Storage patches", diagnostics["n_concavities"])

        with c3:
            dep_th = diagnostics.get("dep_threshold_m", None)
            st.metric(
                "Depth threshold (m)",
                f"{dep_th:.3f}" if dep_th is not None else "N/A",
            )
            st.metric(
                "Total storage (m³)",
                f"{diagnostics['total_storage_m3']:.0f}",
            )

        st.write(
            f"Total rainfall-filled storage (m³): **{diagnostics['total_rain_storage_m3']:.0f}**"
        )

        # Map
        st.subheader("Interactive HSR Map")
        hsr_map = build_hsr_map(parcels_hsr, rainfall_mm=rainfall_mm)
        folium_static(hsr_map, width=1000, height=600)

        # Top parcels table
        st.subheader("Top 10 parcels by rainfall-adjusted storage (HSR_rain_sum)")
        cols = [
            "HSR_rain_sum",
            "HSR_rain_mean",
            "HSR_rain_max",
            "HSR_static_sum",
            "HSR_static_mean",
            "HSR_static_max",
        ]
        existing_cols = [c for c in cols if c in parcels_hsr.columns]

        table_df = (
            parcels_hsr.sort_values("HSR_rain_sum", ascending=False)[existing_cols]
            if "HSR_rain_sum" in parcels_hsr.columns
            else parcels_hsr[existing_cols]
        )

        st.dataframe(table_df.head(10).round(3))

        # CSV download
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download HSR parcel metrics (CSV)",
            data=csv_bytes,
            file_name=f"HSR_parcels_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("HSR analysis failed. Please check data and parameters.")
        st.exception(e)
else:
    st.info("Set rainfall and click **Run HSR Analysis** to start.")
