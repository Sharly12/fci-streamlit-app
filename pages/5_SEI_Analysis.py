# pages/5_SEI_Analysis.py
import streamlit as st
from streamlit_folium import folium_static

from models.sei_model import run_sei_analysis
from utils.sei_visualization import build_sei_map

st.title("📍 Surrounding Exposure Index (SEI) Analysis")

st.write(
    "The Surrounding Exposure Index (SEI) measures how critical the land-use "
    "context is around each parcel within a specified buffer distance. "
    "Optionally, a hazard raster (e.g. flood depth) can be included."
)

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.header("SEI Parameters")

    # ⚠️ Set these to where your data is in the repo or on the server
    default_parcels = "data/raw/Grid/Grid-demo.shp"
    default_landuse = "data/raw/LU_Demo.shp"

    parcels_path = st.text_input("Parcels layer path", default_parcels)
    landuse_path = st.text_input("Land-use layer path", default_landuse)
    lu_field = st.text_input("Land-use attribute field", "LU_All")

    buffer_m = st.slider("Neighbourhood radius (m)", 100.0, 1000.0, 500.0, 50.0)

    hazard_path = st.text_input(
        "Optional hazard raster (e.g., flood depth TIFF). Leave blank for LU-only SEI.",
        "",
    )
    use_hazard = bool(hazard_path.strip())

run_btn = st.button("Run SEI Analysis")

# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------
if run_btn:
    try:
        with st.spinner("Computing SEI …"):
            parcels_sei, diagnostics = run_sei_analysis(
                parcels_path=parcels_path,
                landuse_path=landuse_path,
                lu_field=lu_field,
                buffer_m=buffer_m,
                hazard_raster=hazard_path if use_hazard else None,
            )

        st.success("✅ SEI analysis complete")

        # Diagnostics
        st.subheader("Diagnostics & Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_sei)))
        with c2:
            st.metric("Buffer radius (m)", f"{diagnostics.get('buffer_m', buffer_m):.0f}")
        with c3:
            st.metric(
                "Hazard raster used",
                "Yes" if diagnostics.get("hazard_used", False) else "No",
            )

        unmatched = diagnostics.get("unmatched_examples", [])
        if unmatched:
            with st.expander("Unmatched land-use labels"):
                st.write(unmatched)

        # Map
        st.subheader("Interactive SEI Map")
        sei_map = build_sei_map(parcels_sei, buffer_m=buffer_m)
        folium_static(sei_map, width=1000, height=600)

        # Top parcels table
        st.subheader("Top 10 parcels by SEI")
        cols_show = [
            "SEI",
            "SEI_raw",
            "nei_aw_sum",
            "nei_aw_norm",
            "haz_mean",
            "haz_norm",
            "haz_max",
            "haz_max_norm",
        ]
        existing_cols = [c for c in cols_show if c in parcels_sei.columns]

        st.dataframe(
            parcels_sei[existing_cols]
            .sort_values("SEI", ascending=False)
            .head(10)
            .round(3)
        )

    except Exception as e:
        st.error("SEI analysis failed. Please check paths and parameters.")
        st.exception(e)
else:
    st.info("Configure inputs and click **Run SEI Analysis** to start.")
