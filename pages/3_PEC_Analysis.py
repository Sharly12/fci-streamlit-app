# pages/3_PEC_Analysis.py

import streamlit as st
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.pec_model import run_pec_analysis
from utils.pec_visualization import build_pec_map
import pandas as pd

st.title("🏔 Parcel Elevation Context (PEC) Analysis")

st.write(
    "The Parcel Elevation Context (PEC) index combines relative elevation, "
    "local slope and height above drainage to classify each parcel into "
    "retention priority, high exposure, locally high, or moderate categories."
)

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("PEC Parameters")

    buffer_radius_m = st.slider(
        "Neighbourhood mean radius (m)",
        min_value=100,
        max_value=600,
        value=250,
        step=50,
        help="Radius used for neighbourhood mean elevation (DEM − mean within radius).",
    )

    stream_threshold = st.slider(
        "Stream extraction threshold (flow-accumulation cells)",
        min_value=100,
        max_value=2000,
        value=400,
        step=100,
        help="Higher values = fewer, larger streams; lower values = denser stream network.",
    )

    rainfall_mm = st.slider(
        "Rainfall label for map (mm/h)",
        min_value=80,
        max_value=190,
        value=100,
        step=10,
        help="Used only as a label on the PEC map (classification is currently base PEC).",
    )

run_btn = st.button("Run PEC Analysis")

# ------------------------------------------------------------------
# Data paths (DEM & parcels come from shared loader)
# ------------------------------------------------------------------
dem_path, parcels_path, _ = get_data_paths()


# Cached wrapper so PEC is not recomputed every interaction
@st.cache_data(show_spinner="Running PEC model …")
def _run_pec_cached(dem_path, parcels_path, buffer_radius_m, stream_threshold):
    return run_pec_analysis(
        dem_path=dem_path,
        parcels_path=parcels_path,
        buffer_radius_m=buffer_radius_m,
        stream_threshold=stream_threshold,
    )


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
if run_btn:
    try:
        parcels_pec, diagnostics = _run_pec_cached(
            dem_path,
            parcels_path,
            buffer_radius_m,
            stream_threshold,
        )

        st.success("✅ PEC analysis complete")

        # ---------------- Diagnostics ----------------
        st.subheader("Diagnostics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Parcels analysed",
                diagnostics.get("n_parcels", len(parcels_pec)),
            )
        with c2:
            st.metric(
                "Neighbourhood radius (m)",
                f"{diagnostics.get('radius_m', buffer_radius_m):.0f}",
            )
        with c3:
            st.metric(
                "Stream threshold (cells)",
                diagnostics.get("stream_threshold", stream_threshold),
            )

        # ---------------- Map ----------------
        st.subheader("Interactive PEC Map")

        # 🔴 FIX: pass rainfall as 'rainfall_label' (not rainfall_mm)
        pec_map = build_pec_map(parcels_pec, rainfall_label=rainfall_mm)
        folium_static(pec_map, width=1000, height=600)

        # ---------------- Summary table ----------------
        st.subheader("PEC Class Summary")

        if "pec_class" in parcels_pec.columns:
            counts = (
                parcels_pec["pec_class"]
                .value_counts()
                .rename_axis("PEC class")
                .reset_index(name="Parcels")
            )
            st.dataframe(counts)
        else:
            st.info("No 'pec_class' column found in results.")

    except Exception as e:
        st.error("PEC analysis failed. Please see the error below.")
        st.exception(e)

else:
    st.info("Choose parameters and click **Run PEC Analysis** to start.")
