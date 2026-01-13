# pages/1_HSR_Analysis.py
import os
import io
import zipfile
import tempfile
from xml.etree.ElementTree import Element, SubElement, tostring

import streamlit as st
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths
from models.hsr_model import run_hsr_analysis
from utils.hsr_visualization import build_hsr_map


def gdf_to_kml_bytes(gdf_wgs84, doc_name="HSR Parcels", fields=None) -> bytes:
    """
    Convert a parcels GeoDataFrame (must be EPSG:4326) to KML bytes.
    Writes selected fields into ExtendedData.
    """
    if fields is None:
        fields = []

    def _coords_to_kml(coords):
        return " ".join([f"{x},{y},0" for (x, y) in coords])

    def _polygon_to_kml(parent, poly: Polygon):
        poly_el = SubElement(parent, "Polygon")

        outer = SubElement(SubElement(poly_el, "outerBoundaryIs"), "LinearRing")
        SubElement(outer, "coordinates").text = _coords_to_kml(list(poly.exterior.coords))

        for ring in poly.interiors:
            inner = SubElement(SubElement(poly_el, "innerBoundaryIs"), "LinearRing")
            SubElement(inner, "coordinates").text = _coords_to_kml(list(ring.coords))

    kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = SubElement(kml, "Document")
    SubElement(doc, "name").text = doc_name

    gdf = gdf_wgs84[gdf_wgs84.geometry.notnull()].copy()

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        pm = SubElement(doc, "Placemark")
        SubElement(pm, "name").text = f"Parcel {idx}"

        ext = SubElement(pm, "ExtendedData")
        for f in fields:
            if f in row:
                data_el = SubElement(ext, "Data", name=str(f))
                SubElement(data_el, "value").text = "" if row[f] is None else str(row[f])

        if isinstance(geom, Polygon):
            _polygon_to_kml(pm, geom)
        elif isinstance(geom, MultiPolygon):
            mg = SubElement(pm, "MultiGeometry")
            for poly in geom.geoms:
                _polygon_to_kml(mg, poly)

    return tostring(kml, encoding="utf-8", xml_declaration=True)


st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    """
The Hydrological Storage Role (HSR) model estimates how much water can be
stored in local topographic concavities (HSR_static) and how much of that
storage is effectively used under a given storm depth (HSR_rain).
Results are reported at parcel level.
"""
)

# ------------------------------------------------------------------
# Sidebar controls (same as your original)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("HSR Parameters")

    rainfall_mm = st.slider(
        "Design rainfall (mm)",
        min_value=0.0,
        max_value=250.0,
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

    if st.button("Clear results"):
        for k in ["hsr_parcels", "hsr_diag", "hsr_outputs", "hsr_rainfall_mm"]:
            st.session_state.pop(k, None)
        st.rerun()

run_btn = st.button("Run HSR Analysis")

# ------------------------------------------------------------------
# Run the model (NO change to analysis; only store outputs for download)
# ------------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        with st.spinner("Running HSR ENGINE v4 …"):
            # ✅ Backward/forward compatible:
            # - if your model returns 2 values (old): (parcels, diagnostics)
            # - if it returns 3 values (new): (parcels, diagnostics, outputs)
            try:
                parcels_hsr, diagnostics, outputs = run_hsr_analysis(
                    dem_path=dem_path,
                    cn_path=cn_path,
                    parcels_path=parcels_path,
                    rainfall_mm=rainfall_mm,
                    concavity_window=concavity_window,
                )
            except ValueError:
                parcels_hsr, diagnostics = run_hsr_analysis(
                    dem_path=dem_path,
                    cn_path=cn_path,
                    parcels_path=parcels_path,
                    rainfall_mm=rainfall_mm,
                    concavity_window=concavity_window,
                )
                # fallback: your original code writes to this workspace with fixed names
                tmp_dir = os.path.join(tempfile.gettempdir(), "hsr_engine")
                outputs = {
                    "workspace": tmp_dir,
                    "hsr_static_tif": os.path.join(tmp_dir, "HSR_static.tif"),
                    "hsr_rain_tif": os.path.join(tmp_dir, "HSR_rain.tif"),
                }

        # Store so downloads stay visible after reruns
        st.session_state["hsr_parcels"] = parcels_hsr
        st.session_state["hsr_diag"] = diagnostics
        st.session_state["hsr_outputs"] = outputs
        st.session_state["hsr_rainfall_mm"] = float(rainfall_mm)

        st.success("✅ HSR analysis complete")

    except Exception as e:
        st.error("HSR analysis failed. Please check the logs.")
        st.exception(e)

# ------------------------------------------------------------------
# Show results + downloads (works even after reruns)
# ------------------------------------------------------------------
if "hsr_parcels" in st.session_state:
    parcels_hsr = st.session_state["hsr_parcels"]
    diagnostics = st.session_state["hsr_diag"]
    outputs = st.session_state["hsr_outputs"]
    rr = int(st.session_state.get("hsr_rainfall_mm", 0))

    # -----------------------------
    # Downloads
    # -----------------------------
    st.subheader("Downloads")

    static_path = outputs.get("hsr_static_tif")
    rain_path = outputs.get("hsr_rain_tif")

    static_bytes = None
    rain_bytes = None

    if static_path and os.path.exists(static_path):
        with open(static_path, "rb") as f:
            static_bytes = f.read()
    else:
        st.warning(f"HSR_static GeoTIFF not found at: {static_path}")

    if rain_path and os.path.exists(rain_path):
        with open(rain_path, "rb") as f:
            rain_bytes = f.read()
    else:
        st.warning(f"HSR_rain GeoTIFF not found at: {rain_path}")

    c1, c2, c3 = st.columns(3)

    with c1:
        if static_bytes:
            st.download_button(
                "Download HSR_static (GeoTIFF)",
                data=static_bytes,
                file_name=f"HSR_static_{rr}mm.tif",
                mime="image/tiff",
            )

    with c2:
        if rain_bytes:
            st.download_button(
                "Download HSR_rain (GeoTIFF)",
                data=rain_bytes,
                file_name=f"HSR_rain_{rr}mm.tif",
                mime="image/tiff",
            )

    with c3:
        if static_bytes and rain_bytes:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("HSR_static.tif", static_bytes)
                zf.writestr("HSR_rain.tif", rain_bytes)
            st.download_button(
                "Download both rasters (ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"HSR_rasters_{rr}mm.zip",
                mime="application/zip",
            )

    # Parcel KML (always generated from parcel results)
    parcels_wgs84 = parcels_hsr.to_crs(epsg=4326)
    kml_fields = [
        "HSR_index",
        "HSR_static_sum", "HSR_static_mean", "HSR_static_max",
        "HSR_rain_sum", "HSR_rain_mean", "HSR_rain_max",
        "Rainfall_mm",
    ]
    existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]
    kml_bytes = gdf_to_kml_bytes(parcels_wgs84, doc_name="HSR Parcels", fields=existing_fields)

    st.download_button(
        "Download parcel HSR layer (KML)",
        data=kml_bytes,
        file_name=f"HSR_parcels_{rr}mm.kml",
        mime="application/vnd.google-earth.kml+xml",
    )

    # -----------------------------
    # Diagnostics (unchanged)
    # -----------------------------
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

    # -----------------------------
    # Interactive map (unchanged)
    # -----------------------------
    st.subheader("Interactive HSR Map")
    hsr_map = build_hsr_map(parcels_hsr, rainfall_mm=rr)
    folium_static(hsr_map, width=1000, height=600)

    # -----------------------------
    # Table + CSV (unchanged)
    # -----------------------------
    st.subheader("Top 10 parcels by HSR index")

    export_cols = [
        "HSR_index",
        "HSR_static_sum",
        "HSR_rain_sum",
        "HSR_static_mean",
        "HSR_rain_mean",
        "Rainfall_mm",
    ]
    export_cols = [c for c in export_cols if c in parcels_hsr.columns]

    table_df = (
        parcels_hsr[export_cols]
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

    export_df = parcels_hsr[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download parcel HSR results (CSV)",
        data=csv_bytes,
        file_name=f"HSR_results_{rr}mm.csv",
        mime="text/csv",
    )

else:
    st.info("Set rainfall and concavity window, then click **Run HSR Analysis**.")
