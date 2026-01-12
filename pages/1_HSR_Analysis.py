# pages/1_HSR_Analysis.py
import io
import zipfile

import streamlit as st
import pandas as pd
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.hsr_model import run_hsr_analysis
from utils.hsr_visualization import build_hsr_map
from utils.kml_export import gdf_to_kml_bytes


st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    """
The Hydrological Storage Role (HSR) model estimates how much water can be
stored in local topographic concavities (HSR_static) and how much of that
storage is effectively used under a given storm depth (HSR_rain).
"""
)

with st.expander("Model inputs", expanded=True):
    rainfall_mm = st.number_input(
        "Storm rainfall depth (mm)",
        min_value=0.0,
        max_value=1000.0,
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

# ------------------------------------------------------------
# RUN MODEL (store everything in session_state for downloading)
# ------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        with st.spinner("Running HSR ENGINE v4 …"):
            parcels_hsr, diagnostics, outputs = run_hsr_analysis(
                dem_path=dem_path,
                cn_path=cn_path,
                parcels_path=parcels_path,
                rainfall_mm=rainfall_mm,
                concavity_window=concavity_window,
            )

        # Save results (so buttons remain visible after Streamlit reruns)
        st.session_state["hsr_parcels"] = parcels_hsr
        st.session_state["hsr_diag"] = diagnostics
        st.session_state["hsr_outputs"] = outputs
        st.session_state["hsr_rainfall_mm"] = float(rainfall_mm)

        # Read GeoTIFFs into memory for download buttons
        with open(outputs["hsr_static_tif"], "rb") as f:
            st.session_state["hsr_static_tif_bytes"] = f.read()
        with open(outputs["hsr_rain_tif"], "rb") as f:
            st.session_state["hsr_rain_tif_bytes"] = f.read()

        # ZIP both rasters (optional convenience)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("HSR_static.tif", st.session_state["hsr_static_tif_bytes"])
            zf.writestr("HSR_rain.tif", st.session_state["hsr_rain_tif_bytes"])
        st.session_state["hsr_tifs_zip_bytes"] = zip_buf.getvalue()

        # Build parcel KML (KML must be EPSG:4326)
        parcels_wgs84 = parcels_hsr.to_crs(epsg=4326)
        kml_fields = [
            "HSR_index",
            "HSR_static_sum",
            "HSR_static_mean",
            "HSR_static_max",
            "HSR_rain_sum",
            "HSR_rain_mean",
            "HSR_rain_max",
            "Rainfall_mm",
        ]
        existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]

        st.session_state["hsr_kml_bytes"] = gdf_to_kml_bytes(
            parcels_wgs84,
            doc_name="HSR Parcels",
            fields=existing_fields,
        )

        # CSV export bytes (all parcels)
        cols_show = [
            "HSR_index",
            "HSR_static_sum",
            "HSR_rain_sum",
            "HSR_static_mean",
            "HSR_rain_mean",
            "Rainfall_mm",
        ]
        existing_cols = [c for c in cols_show if c in parcels_hsr.columns]
        export_df = parcels_hsr[existing_cols].copy()
        st.session_state["hsr_csv_bytes"] = export_df.to_csv(index=False).encode("utf-8")

        st.success("✅ HSR analysis complete")

    except Exception as e:
        st.error("HSR analysis failed. Please check the logs.")
        st.exception(e)

# ------------------------------------------------------------
# SHOW OUTPUTS + DOWNLOAD BUTTONS (if results exist)
# ------------------------------------------------------------
if "hsr_parcels" in st.session_state:
    parcels_hsr = st.session_state["hsr_parcels"]
    diagnostics = st.session_state["hsr_diag"]
    rr = int(st.session_state.get("hsr_rainfall_mm", 0))

    st.subheader("Downloads (right after running)")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download HSR_static (GeoTIFF)",
            data=st.session_state["hsr_static_tif_bytes"],
            file_name=f"HSR_static_{rr}mm.tif",
            mime="image/tiff",
        )
        st.download_button(
            "Download HSR_rain (GeoTIFF)",
            data=st.session_state["hsr_rain_tif_bytes"],
            file_name=f"HSR_rain_{rr}mm.tif",
            mime="image/tiff",
        )

    with c2:
        st.download_button(
            "Download both GeoTIFFs (ZIP)",
            data=st.session_state["hsr_tifs_zip_bytes"],
            file_name=f"HSR_rasters_{rr}mm.zip",
            mime="application/zip",
        )

    with c3:
        st.download_button(
            "Download parcel layer (KML)",
            data=st.session_state["hsr_kml_bytes"],
            file_name=f"HSR_parcels_{rr}mm.kml",
            mime="application/vnd.google-earth.kml+xml",
        )
        st.download_button(
            "Download parcel results (CSV)",
            data=st.session_state["hsr_csv_bytes"],
            file_name=f"HSR_results_{rr}mm.csv",
            mime="text/csv",
        )

    # Diagnostics
    st.subheader("Model diagnostics")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Rainfall (mm)", f"{diagnostics['rainfall_mm']:.0f}")
        st.metric("Concavity window (cells)", diagnostics["concavity_window"])
    with d2:
        st.metric("Cell size (m)", f"{diagnostics['cell_size_m']:.1f}")
        st.metric("Cell area (m²)", f"{diagnostics['cell_area_m2']:.1f}")
    with d3:
        st.metric("Concavity patches", diagnostics["n_concavities"])
        thresh = diagnostics.get("depth_threshold_m")
        st.metric("Depth threshold (m)", f"{thresh:.3f}" if thresh is not None else "N/A")

    # Interactive map
    st.subheader("Interactive HSR Map")
    hsr_map = build_hsr_map(parcels_hsr, rainfall_mm=st.session_state.get("hsr_rainfall_mm", rr))
    folium_static(hsr_map, width=1000, height=600)

    # Top parcels table
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
else:
    st.info("Set rainfall and concavity window, then click **Run HSR Analysis**.")
