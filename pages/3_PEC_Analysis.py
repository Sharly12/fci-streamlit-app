# pages/3_PEC_Analysis.py
import io
import os
import zipfile
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd

from utils.data_loader import get_data_paths
from models.pec_model import run_pec_analysis
from utils.pec_visualization import build_pec_map


st.title("🌄 PEC Analysis")

st.write(
    """
PEC (Parcel Elevation Context) classifies parcels based on local terrain position:

- **Slope** (degrees)
- **PREI**: Local relative elevation (DEM − neighbourhood mean)
- **HAND-like score**: elevation above drainage
- **Relief** within parcel

Classes:
1. Low-lying Depressed (Retention Priority)  
2. Flat & Pressured (High Flood Exposure Risk)  
3. Locally High & Disconnected  
4. Moderate / Context-Dependent  
"""
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("PEC Parameters")

    if st.button("Clear PEC cache"):
        st.cache_data.clear()
        st.success("PEC cache cleared")

    rainfall_mm = st.slider(
        "Rainfall depth for PEC adjustment (mm)",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=10.0,
    )

    neighbourhood_radius_m = st.slider(
        "Neighbourhood radius for PREI (m)",
        min_value=100.0,
        max_value=500.0,
        value=250.0,
        step=50.0,
    )

    stream_threshold = st.slider(
        "Stream extraction threshold (cell count)",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
    )

run_btn = st.button("Run PEC Analysis")


# ---------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running PEC model …")
def _run_cached(dem_path, parcels_path, rainfall_mm, neighbourhood_radius_m, stream_threshold):
    return run_pec_analysis(
        dem_path=dem_path,
        parcels_path=parcels_path,
        rainfall_mm=rainfall_mm,
        neighbourhood_radius_m=neighbourhood_radius_m,
        stream_threshold=stream_threshold,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if not run_btn:
    st.info("Set parameters in the sidebar and click **Run PEC Analysis**.")
else:
    try:
        dem_path, parcels_path, _ = get_data_paths()

        parcels_pec, diagnostics = _run_cached(
            dem_path,
            parcels_path,
            rainfall_mm,
            neighbourhood_radius_m,
            stream_threshold,
        )

        st.success("✅ PEC analysis complete")

        # ---------------- Diagnostics ----------------
        st.subheader("Diagnostics & Summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_pec)))
            st.metric("DEM resolution (m)", f"{diagnostics.get('dem_res_m', 0.0):.2f}")
        with c2:
            st.metric("Neighbourhood radius (m)", f"{diagnostics.get('neighbourhood_radius_m', 0.0):.0f}")
            st.metric("Radius (pixels)", diagnostics.get("neighbourhood_radius_pixels", 0))
        with c3:
            st.metric("Stream threshold (cells)", diagnostics.get("stream_threshold", 0))
            st.metric("Rainfall (mm)", f"{diagnostics.get('rainfall_mm', 0.0):.0f}")

        # Always show 4 classes (even if one has 0)
        risk_order = [
            "Low-lying Depressed (Retention Priority)",
            "Flat & Pressured (High Flood Exposure Risk)",
            "Locally High & Disconnected",
            "Moderate / Context-Dependent",
        ]
        class_counts = diagnostics.get("pec_class_counts", {}) or {}
        cc_df = pd.DataFrame({
            "PEC class": risk_order,
            "Parcels": [int(class_counts.get(k, 0)) for k in risk_order],
        })
        st.write("**Parcel counts by PEC class**")
        st.table(cc_df)

        # Debug expander
        with st.expander("Debug (why a class might be missing)"):
            st.write({
                "DEM CRS (input)": diagnostics.get("dem_crs_input"),
                "DEM CRS (processing)": diagnostics.get("dem_crs_processing"),
                "DEM reprojected": diagnostics.get("dem_reprojected"),
                "DEM input is geographic": diagnostics.get("dem_input_is_geographic"),
                "flat parcels (slope<1.5)": diagnostics.get("flat_count_slope_lt_1p5"),
                "hand<=med_thresh": diagnostics.get("hand_le_med_thresh"),
                "flat & hand": diagnostics.get("flat_and_hand"),
                "low-lying rule hits": diagnostics.get("low_lying_rule_count"),
                "Flat & Pressured final": diagnostics.get("flat_pressured_count"),
                "HAND low thresh used": diagnostics.get("hand_low_thresh_used"),
                "HAND med thresh used": diagnostics.get("hand_med_thresh_used"),
                "Outputs directory": diagnostics.get("pec_out_dir"),
            })

        # ---------------- Map ----------------
        st.subheader("Interactive PEC Map")
        pec_map = build_pec_map(parcels_pec, rainfall_mm=rainfall_mm)
        folium_static(pec_map, width=1000, height=600)

        # ---------------- Downloads (ZIP) ----------------
        st.subheader("Download outputs")

        export_cols = [c for c in ["grid_id", "pec_class", "pec_code", "prei", "hand_score", "relief", "slp_mean"] if c in parcels_pec.columns]
        csv_bytes = parcels_pec[export_cols].to_csv(index=False).encode("utf-8")

        files = list(diagnostics.get("pec_output_files", []) or [])

        kml_path = diagnostics.get("pec_kml_path")
        geojson_path = diagnostics.get("pec_geojson_path")
        if kml_path and os.path.exists(kml_path):
            files.append(kml_path)
        elif geojson_path and os.path.exists(geojson_path):
            files.append(geojson_path)

        def _build_zip(file_paths, csv_data: bytes):
            buff = io.BytesIO()
            with zipfile.ZipFile(buff, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for fp in file_paths:
                    if fp and os.path.exists(fp):
                        z.write(fp, arcname=os.path.basename(fp))
                z.writestr("PEC_results.csv", csv_data)
            buff.seek(0)
            return buff.getvalue()

        zip_bytes = _build_zip(files, csv_bytes)

        st.download_button(
            label="Download PEC package (ZIP: all TIFFs + parcel KML/GeoJSON + CSV)",
            data=zip_bytes,
            file_name=f"PEC_outputs_{int(rainfall_mm)}mm.zip",
            mime="application/zip",
        )

        # Optional: separate vector download buttons
        if kml_path and os.path.exists(kml_path):
            with open(kml_path, "rb") as f:
                st.download_button(
                    "Download parcel KML only",
                    data=f.read(),
                    file_name=os.path.basename(kml_path),
                    mime="application/vnd.google-earth.kml+xml",
                )
        elif geojson_path and os.path.exists(geojson_path):
            with open(geojson_path, "rb") as f:
                st.download_button(
                    "Download parcel GeoJSON (KML not supported in this environment)",
                    data=f.read(),
                    file_name=os.path.basename(geojson_path),
                    mime="application/geo+json",
                )

        # ---------------- Quick table preview ----------------
        st.subheader("Preview (first 20 parcels)")
        show_cols = [c for c in [
            "grid_id", "pec_class", "pec_code", "prei", "hand_score", "relief", "slp_mean",
            "dem_min", "dem_max", "hand_min", "hand_mean"
        ] if c in parcels_pec.columns]
        st.dataframe(parcels_pec[show_cols].head(20))

    except Exception as e:
        st.error("PEC analysis failed. See error below.")
        st.exception(e)
