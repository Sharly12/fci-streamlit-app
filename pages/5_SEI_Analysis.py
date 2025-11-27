import streamlit as st
from utils.data_loader import get_data_paths, load_base_data

st.title("📍 Surrounding Exposure Index (SEI)")

st.write(
    """
    The **Surrounding Exposure Index (SEI)** will capture how exposed a parcel is
    based on its neighbourhood: nearby flow corridors, low-lying areas, and other
    risk factors.

    This page is currently a **placeholder** for the SEI model.
    """
)

if st.button("Run SEI Analysis (placeholder)"):
    dem_path, parcels_path, cn_path = get_data_paths()
    base_data = load_base_data(dem_path, parcels_path, cn_path)

    st.success("Base DEM / CN / parcel data loaded successfully ✅")
    st.info(
        "Later, implement `run_sei_analysis(...)` in `models/sei_model.py` and "
        "call it here with `base_data`."
    )
else:
    st.info(
        "SEI model is under development. Full multi-model integration will come after "
        "HSR, PEC, and UDS are implemented."
    )
