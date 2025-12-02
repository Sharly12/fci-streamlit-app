# pages/5_SEI_Analysis.py
import os
import streamlit as st
from streamlit_folium import folium_static

from models.sei_model import run_sei_analysis
from utils.sei_visualization import build_sei_map
from utils.data_loader import get_data_paths, get_lu_path

st.title("📍 Surrounding Exposure Index (SEI) Analysis")

st.write(
    "The Surrounding Exposure Index (SEI) measures how critical the land-use "
    "context is around each parcel within a specified buffer distance. "
    "Optionally, a hazard raster (e.g. flood depth) can be included."
)

# ------------------------------------------------------------------
# Resolve shared paths from data_loader
#   - DEM_PATH_IN, DEFAULT_PARCELS_PATH, CN_RASTER_PATH come from Dropbox/local
#   - LU path comes from LU_Demo.geojson Dropbox/local
# ------------------------------------------------------------------
DEM_PATH_IN, DEFAULT_PARCELS_PATH, CN_RASTER_PATH = get_data_paths()
DEFAULT_LU_PATH = get_lu_path()

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.header("SEI Parameters")

    # Parcels: default to same file used by FCI (grid-network.geojson from Dropbox)
    parcels_path = st.text_input(
        "Parcels layer path",
        DEFAULT_PARCELS_PATH,
        help="Defaults to the parcels file downloaded via Dropbox (same as FCI).",
    )

    # Land-use: default to LU_Demo.geojson from Dropbox
    landuse_path = st.text_input(
        "Land-use layer path",
        DEFAULT_LU_PATH,
        help="Defaults to LU_Demo.geojson downloaded via Dropbox.",
    )

    lu_field = st.text_input(
        "Land-use attribute field",
        "LU_All",
        help="Name of the land-use column to reclassify (e.g. LU_All).",
    )

    buffer_m = st.slider(
        "Neighbourhood radius (m)",
        min_value=100.0,
        max_value=1000.0,
        value=500.0,
        step=50.0,
    )

    hazard_path = st.text_input(
        "Optional hazard raster (e.g., flood depth TIFF). "
        "Leave blank for LU-only SEI.",
        "",
    )
    use_hazard = bool(hazard_path.strip())

run_btn = st.button("Run SEI Analysis")

# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------
if run_btn:
    missing_messages = []

    # Check parcels file existence
    if not os.path.exists(parcels_path):
        missing_messages.append(f"❌ Parcels file not found: `{parcels_path}`")

    # Check land-use file existence
    if not os.path.exists(landuse_path):
        missing_messages.append(f"❌ Land-use file not found: `{landuse_path}`")

    # Check optional hazard raster
    haz_path_clean = hazard_path.strip()
    if use_hazard and haz_path_clean and not os.path.exists(haz_path_clean):
        missing_messages.append(f"❌ Hazard raster not found: `{haz_path_clean}`")

    if missing_messages:
        for msg in missing_messages:
            st.error(msg)
        st.info(
            "Please check that the file paths are correct. "
            "Parcels and land-use defaults are downloaded via Dropbox; "
            "hazard raster must be accessible in the app environment."
        )
    else:
        try:
            with st.spinner(
                "Computing SEI … this may take a moment for larger datasets."
            ):
                parcels_sei, diagnostics = run_sei_analysis(
                    parcels_path=parcels_path,
                    landuse_path=landuse_path,
                    lu_field=lu_field,
                    buffer_m=buffer_m,
                    hazard_raster=haz_path_clean if use_hazard else None,
                )

            st.success("✅ SEI analysis complete")

            # Diagnostics
            st.subheader("Diagnostics & Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Parcels analysed",
                    diagnostics.get("n_parcels", len(parcels_sei)),
                )
            with c2:
                st.metric(
                    "Buffer radius (m)",
                    f"{diagnostics.get('buffer_m', buffer_m):.0f}",
                )
            with c3:
                st.metric(
                    "Hazard raster used",
                    "Yes" if diagnostics.get("hazard_used", False) else "No",
                )

            unmatched = diagnostics.get("unmatched_examples", [])
            if unmatched:
                with st.expander(
                    "Unmatched land-use labels (fell back to 'Unclassified')"
                ):
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
            st.error("SEI analysis failed. Please check paths, CRS, and parameters.")
            st.exception(e)
else:
    st.info("Configure inputs and click **Run SEI Analysis** to start.")
