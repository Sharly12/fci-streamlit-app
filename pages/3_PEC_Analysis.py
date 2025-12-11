# pages/3_PEC_Analysis.py
import streamlit as st
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.pec_model import run_pec_analysis
from utils.pec_visualization import build_pec_map

st.title("🏔️ Parcel Elevation Context (PEC) Analysis")

st.write(
    """
The Parcel Elevation Context (PEC) model evaluates how each parcel sits in the
local terrain:

- **PREI** – local relative elevation (parcel vs. neighbourhood mean)
- **HAND-like score** – height above nearest drainage / valley bottom
- **Relief** – min–max elevation range within the parcel

These indicators are combined to classify parcels into four PEC categories.
"""
)

# --------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------
with st.sidebar:
    st.header("PEC Parameters")

    prei_radius_m = st.slider(
        "Neighbourhood radius for PREI (m)",
        min_value=100.0,
        max_value=500.0,
        value=250.0,
        step=25.0,
    )

    stream_threshold = st.slider(
        "Flow accumulation threshold for streams",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
    )

    rainfall_mm = st.slider(
        "Reference rainfall for legend (mm/h)",
        min_value=50.0,
        max_value=250.0,
        value=100.0,
        step=10.0,
    )

run_btn = st.button("Run PEC Analysis")

# --------------------------------------------------------------------
# Cached wrapper
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="🧮 Running PEC model …")
def _run_pec_cached(
    dem_path: str,
    parcels_path: str,
    prei_radius_m: float,
    stream_threshold: float,
):
    return run_pec_analysis(
        dem_path=dem_path,
        parcels_path=parcels_path,
        prei_radius_m=prei_radius_m,
        stream_threshold=stream_threshold,
    )


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        parcels_pec, diagnostics = _run_pec_cached(
            dem_path, parcels_path, prei_radius_m, stream_threshold
        )

        st.success("✅ PEC analysis complete.")

        # Diagnostics summary
        st.subheader("Diagnostics & summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_pec)))
        with c2:
            st.metric(
                "PREI radius (m)",
                f"{diagnostics.get('prei_radius_m', prei_radius_m):.0f}",
            )
        with c3:
            st.metric(
                "Stream threshold (cells)",
                f"{diagnostics.get('stream_threshold', stream_threshold):.0f}",
            )

        # Class counts
        class_counts = diagnostics.get("class_counts", {})
        if class_counts:
            st.write("**PEC class distribution (parcels)**")
            st.table(
                {
                    "Class": list(class_counts.keys()),
                    "Parcels": list(class_counts.values()),
                }
            )

        # Map
        st.subheader("Interactive PEC Map")
        pec_map = build_pec_map(parcels_pec, rainfall_label=rainfall_mm)
        folium_static(pec_map, width=1000, height=600)

        # Optional: show top rows for QA
        with st.expander("Show sample attribute table"):
            cols_show = [
                "grid_id",
                "pec_class",
                "prei",
                "hand_score",
                "relief",
                "dem_min",
                "dem_max",
                "slp_mean",
            ]
            existing = [c for c in cols_show if c in parcels_pec.columns]
            st.dataframe(parcels_pec[existing].head(20).round(3))

    except Exception as e:
        st.error("PEC analysis failed. Please see the error below.")
        st.exception(e)
else:
    st.info("Adjust parameters on the left and click **Run PEC Analysis**.")
