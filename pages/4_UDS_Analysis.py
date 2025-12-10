# pages/4_UDS_Analysis.py
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd

from utils.data_loader import get_data_paths
from models.uds_model import run_uds_analysis
from utils.uds_visualization import build_uds_map

st.title("🔀 Upstream–Downstream Sensitivity (UDS) Analysis")

st.write(
    """
This module quantifies how each parcel is positioned in the hydrological
network (upstream / downstream dependence) and how that structural role
interacts with runoff generated from Curve Number (CN) under a chosen
rainfall depth.

**Outputs include:**
- Structural UDS sensitivity (normalized `uds_score_norm`)
- Combined UDS × SCS–CN runoff hazard (`UDS_runoff_norm`)
"""
)

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("UDS Parameters")

    rainfall_mm = st.slider(
        "Rainfall depth for SCS–CN runoff (mm)",
        min_value=0.0,
        max_value=250.0,
        value=100.0,
        step=10.0,
    )

    max_steps = st.slider(
        "Max flow-trace steps from each parcel pour point",
        min_value=500,
        max_value=5000,
        value=3000,
        step=500,
        help="Upper limit on how far we follow D8 flow directions when "
             "searching for a downstream parcel.",
    )

run_btn = st.button("Run UDS Analysis")

# ------------------------------------------------------------------
# Run the model
# ------------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        with st.spinner("Running UDS + CN–runoff engine …"):
            parcels_uds, diagnostics = run_uds_analysis(
                dem_path=dem_path,
                cn_path=cn_path,
                parcels_path=parcels_path,
                rainfall_mm=rainfall_mm,
                max_steps=max_steps,
            )

        st.success("✅ UDS analysis complete")

        # ---------------- Diagnostics ----------------
        st.subheader("Model diagnostics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_uds)))
            st.metric("Parcels with pour points", diagnostics.get("parcels_with_pour", 0))
        with c2:
            st.metric("Graph nodes", diagnostics.get("n_nodes", 0))
            st.metric("Graph edges", diagnostics.get("n_edges", 0))
        with c3:
            st.metric("Rainfall (mm)", f"{diagnostics.get('rainfall_mm', rainfall_mm):.0f}")
            st.metric(
                "Parcels with valid CN",
                diagnostics.get("cn_valid_parcels", 0),
            )

        # ---------------- Map ----------------
        st.subheader("Interactive UDS Map")
        uds_map = build_uds_map(parcels_uds, rainfall_mm=rainfall_mm)
        folium_static(uds_map, width=1000, height=600)

        # ---------------- Tables & Download ----------------
        st.subheader("Top 10 parcels by UDS × CN runoff hazard")

        cols_show = [
            "grid_id",
            "uds_score",
            "uds_score_norm",
            "uds_up",
            "uds_down",
            "CN",
            "runoff_mm",
            "runoff_norm",
            "UDS_runoff_index",
            "UDS_runoff_norm",
            "Rainfall_mm",
        ]
        existing_cols = [c for c in cols_show if c in parcels_uds.columns]

        table_df = (
            parcels_uds[existing_cols]
            .sort_values("UDS_runoff_norm", ascending=False)
            .head(10)
        )

        st.dataframe(
            table_df.style.format(
                {
                    "uds_score": "{:.4f}",
                    "uds_score_norm": "{:.3f}",
                    "CN": "{:.1f}",
                    "runoff_mm": "{:.2f}",
                    "runoff_norm": "{:.3f}",
                    "UDS_runoff_index": "{:.3f}",
                    "UDS_runoff_norm": "{:.3f}",
                }
            )
        )

        # Full CSV download
        export_df = parcels_uds[existing_cols].copy()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full UDS results (CSV)",
            data=csv_bytes,
            file_name=f"UDS_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("UDS analysis failed. Please check the error below.")
        st.exception(e)
else:
    st.info(
        "Set rainfall and (optionally) max flow-trace steps, "
        "then click **Run UDS Analysis**."
    )
