import streamlit as st
from utils.data_loader import get_data_paths, load_base_data

st.title("⛰️ Parcel Elevation Context (PEC)")

st.write(
    """
    The **Parcel Elevation Context (PEC)** model will describe the topographic setting
    of each parcel (relative elevation, slope position, local relief), and how that
    relates to water accumulation and flood exposure.

    This page is currently a **placeholder** for the future PEC implementation.
    """
)

if st.button("Run PEC Analysis (placeholder)"):
    dem_path, parcels_path, cn_path = get_data_paths()
    base_data = load_base_data(dem_path, parcels_path, cn_path)

    st.success("Base DEM / CN / parcel data loaded successfully ✅")
    st.info(
        "Later, implement `run_pec_analysis(...)` in `models/pec_model.py` and "
        "call it here with `base_data`."
    )
else:
    st.info(
        "PEC model is under development. For now, you can explore the FCI model on its page."
    )
