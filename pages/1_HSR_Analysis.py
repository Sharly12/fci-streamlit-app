import streamlit as st
from utils.data_loader import get_data_paths, load_base_data

st.title("💧 Hydrological Storage Role (HSR)")

st.write(
    """
    The **Hydrological Storage Role (HSR)** model will analyse how much each parcel
    contributes to water storage and retention (depressions, wetlands, reservoirs, etc.).

    This page is currently a **placeholder**. You can already test that the data loading
    works and later plug in the real HSR model logic.
    """
)

if st.button("Run HSR Analysis (placeholder)"):
    # This just tests that data loading works and gives you a hook for later.
    dem_path, parcels_path, cn_path = get_data_paths()
    base_data = load_base_data(dem_path, parcels_path, cn_path)

    st.success("Base DEM / CN / parcel data loaded successfully ✅")
    st.info(
        "When the HSR model is implemented in `models/hsr_model.py`, "
        "you can call it here using `base_data`."
    )
else:
    st.info(
        "HSR model is under development. Use the **FCI Analysis** page for full functionality."
    )
