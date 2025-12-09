# pages/1_HSR_Analysis.py
import streamlit as st
import pandas as pd
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.hsr_model import run_hsr_analysis
from utils.hsr_visualization import build_hsr_map

st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    """
The Hydrological Storage Role (HSR) model estimates how much water can be
stored in local topographic concavities (HSR_static) and how much of that
storage is effectively used under a given storm depth (HSR_rain).
Results are reported at parcel level.
"""
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
        help="Neighbourhood size (e.g. 7×7 cells ≈ 210 m at 30 m resolution).",
    )

run_btn = st.button("Run HSR Analysis")

# ------------------------------------------------------------------
# Run the model
# ------------------------------------------------------------------
if run_btn:
    try:
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

        # Diagnostics
        st.subheader("Model diagnostics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rainfall (mm)", f"{diagnostics['rainfall_mm']:.0f}")
            st.metric("Concavity window (cells)", diagnostics["concavity_window"])
        with c2:
            st.metric("Cell size (m)", f"{diagnostics['cell_size_m']:.1f}")
            st.metric("Cell area (m²)", f"{diagnostics['cell_area_m2']:.1f}")
        with c3:
            st.metric("Concavity patches", diagnostics["n_concavities"])
            thresh = diagnostics.get("depth_threshold_m")
            st.metric(
                "Depth threshold (m)",
                f"{thresh:.3f}" if thresh is not None else "N/A",
            )

        # ------------------------------------------------------------------
        # Interactive map
        # ------------------------------------------------------------------
        st.subheader("Interactive HSR Map")
        hsr_map = build_hsr_map(parcels_hsr, rainfall_mm=rainfall_mm)
        folium_static(hsr_map, width=1000, height=600)

        # ------------------------------------------------------------------
        # Top parcels table
        # ------------------------------------------------------------------
        st.subheader("Top 10 parcels by HSR index")
        cols_show = [
            "HSR_index",
            "HSR_static_sum",
            "HSR_rain_sum",
            "HSR_static_mean",
            "HSR_rain_mean",
            "Rainfall_mm",
        ]
        existing_cols = [c for c in cols_show if c in parcels_hsr.columns]

        table_df = (
            parcels_hsr[existing_cols]
            .sort_values("HSR_index", ascending=False)
            .head(10)
        )

        st.dataframe(
            table_df.style.format(
                {
                    "HSR_index": "{:.3f}",
                    "HSR_static_sum": "{:.0f}",
                    "HSR_rain_sum": "{:.0f}",
                    "HSR_static_mean": "{:.1f}",
                    "HSR_rain_mean": "{:.1f}",
                }
            )
        )

        # CSV export (all parcels)
        export_cols = existing_cols
        export_df = parcels_hsr[export_cols].copy()

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download parcel HSR results (CSV)",
            data=csv_bytes,
            file_name=f"HSR_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("HSR analysis failed. Please check the logs.")
        st.exception(e)
else:
    st.info("Set rainfall and concavity window, then click **Run HSR Analysis**.")
