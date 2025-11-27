import streamlit as st
from utils.data_loader import get_data_paths, load_base_data

st.title("🔁 Upstream–Downstream Sensitivity (UDS)")

st.write(
    """
    The **Upstream–Downstream Sensitivity (UDS)** model will quantify how changes in
    upstream parcels (land use, storage, flow corridors) affect downstream parcels
    and infrastructure.

    This page is a **placeholder** until the UDS model is implemented.
    """
)

if st.button("Run UDS Analysis (placeholder)"):
    dem_path, parcels_path, cn_path = get_data_paths()
    base_data = load_base_data(dem_path, parcels_path, cn_path)

    st.success("Base DEM / CN / parcel data loaded successfully ✅")
    st.info(
        "Once `models/uds_model.py` is ready, you can plug the real UDS computation "
        "here using `base_data`."
    )
else:
    st.info(
        "UDS model is under development. Use the FCI page to explore flow corridors meanwhile."
    )
