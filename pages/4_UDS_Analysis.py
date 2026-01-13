# pages/4_UDS_Analysis.py
import io
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring

import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from utils.data_loader import get_data_paths
from models.uds_model import run_uds_analysis
from utils.uds_visualization import build_uds_map


def gdf_to_kml_bytes(gdf_wgs84, doc_name="UDS Parcels", fields=None) -> bytes:
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


st.title("🔀 Upstream–Downstream Sensitivity (UDS) Analysis")

st.write(
    """
This module quantifies how each parcel is positioned in the hydrological
network (upstream / downstream dependence) and how that structural role
interacts with runoff generated from Curve Number (CN) under a chosen
rainfall depth.

**Outputs include:**
- Structural UDS sensitivity (normalized `uds_score_norm`)
- Combined UDS × SCS–CN runoff hazard (`UDS_runoff_norm`)
"""
)

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("UDS Parameters")

    rainfall_mm = st.slider(
        "Rainfall depth for SCS–CN runoff (mm)",
        min_value=0.0,
        max_value=250.0,
        value=100.0,
        step=10.0,
    )

    max_steps = st.slider(
        "Max flow-trace steps from each parcel pour point",
        min_value=500,
        max_value=5000,
        value=3000,
        step=500,
        help="Upper limit on how far we follow D8 flow directions when "
             "searching for a downstream parcel.",
    )

run_btn = st.button("Run UDS Analysis")

# ------------------------------------------------------------------
# Run the model
# ------------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        with st.spinner("Running UDS + CN–runoff engine …"):
            parcels_uds, diagnostics, outputs = run_uds_analysis(
                dem_path=dem_path,
                cn_path=cn_path,
                parcels_path=parcels_path,
                rainfall_mm=rainfall_mm,
                max_steps=max_steps,
            )

        st.success("✅ UDS analysis complete")

        # ------------------------------------------------------------
        # ✅ Downloads (GeoTIFF + ZIP + KML) — export only (no analysis change)
        # ------------------------------------------------------------
        st.subheader("Downloads")

        with open(outputs["flow_direction_tif"], "rb") as f:
            fdir_bytes = f.read()
        with open(outputs["flow_accumulation_tif"], "rb") as f:
            acc_bytes = f.read()
        with open(outputs["parcel_id_grid_tif"], "rb") as f:
            pid_bytes = f.read()
        with open(outputs["uds_structural_norm_tif"], "rb") as f:
            uds_struct_bytes = f.read()
        with open(outputs["uds_hazard_norm_tif"], "rb") as f:
            uds_hazard_bytes = f.read()

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"UDS_flow_direction.tif", fdir_bytes)
            zf.writestr(f"UDS_flow_accumulation.tif", acc_bytes)
            zf.writestr(f"UDS_parcel_id_grid.tif", pid_bytes)
            zf.writestr(f"UDS_structural_norm_{int(rainfall_mm)}mm.tif", uds_struct_bytes)
            zf.writestr(f"UDS_hazard_norm_{int(rainfall_mm)}mm.tif", uds_hazard_bytes)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button(
                "Flow Dir (GeoTIFF)",
                data=fdir_bytes,
                file_name="UDS_flow_direction.tif",
                mime="image/tiff",
            )
        with c2:
            st.download_button(
                "Flow Accum (GeoTIFF)",
                data=acc_bytes,
                file_name="UDS_flow_accumulation.tif",
                mime="image/tiff",
            )
        with c3:
            st.download_button(
                "UDS Hazard Raster (GeoTIFF)",
                data=uds_hazard_bytes,
                file_name=f"UDS_hazard_norm_{int(rainfall_mm)}mm.tif",
                mime="image/tiff",
            )
        with c4:
            st.download_button(
                "All UDS rasters (ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"UDS_rasters_{int(rainfall_mm)}mm.zip",
                mime="application/zip",
            )

        parcels_wgs84 = parcels_uds.to_crs(epsg=4326)
        kml_fields = [
            "grid_id",
            "uds_score",
            "uds_score_norm",
            "uds_up",
            "uds_down",
            "CN",
            "runoff_mm",
            "runoff_norm",
            "UDS_runoff_index",
            "UDS_runoff_norm",
            "Rainfall_mm",
        ]
        existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]
        kml_bytes = gdf_to_kml_bytes(parcels_wgs84, doc_name="UDS Parcels", fields=existing_fields)

        st.download_button(
            "Download parcel UDS layer (KML)",
            data=kml_bytes,
            file_name=f"UDS_parcels_{int(rainfall_mm)}mm.kml",
            mime="application/vnd.google-earth.kml+xml",
        )

        # ---------------- Diagnostics ----------------
        st.subheader("Model diagnostics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_uds)))
            st.metric("Parcels with pour points", diagnostics.get("parcels_with_pour", 0))
        with c2:
            st.metric("Graph nodes", diagnostics.get("n_nodes", 0))
            st.metric("Graph edges", diagnostics.get("n_edges", 0))
        with c3:
            st.metric("Rainfall (mm)", f"{diagnostics.get('rainfall_mm', rainfall_mm):.0f}")
            st.metric("Parcels with valid CN", diagnostics.get("cn_valid_parcels", 0))

        # ---------------- Map ----------------
        st.subheader("Interactive UDS Map")
        uds_map = build_uds_map(parcels_uds, rainfall_mm=rainfall_mm)
        folium_static(uds_map, width=1000, height=600)

        # ---------------- Tables & Download ----------------
        st.subheader("Top 10 parcels by UDS × CN runoff hazard")

        cols_show = [
            "grid_id",
            "uds_score",
            "uds_score_norm",
            "uds_up",
            "uds_down",
            "CN",
            "runoff_mm",
            "runoff_norm",
            "UDS_runoff_index",
            "UDS_runoff_norm",
            "Rainfall_mm",
        ]
        existing_cols = [c for c in cols_show if c in parcels_uds.columns]

        table_df = (
            parcels_uds[existing_cols]
            .sort_values("UDS_runoff_norm", ascending=False)
            .head(10)
        )

        st.dataframe(
            table_df.style.format(
                {
                    "uds_score": "{:.4f}",
                    "uds_score_norm": "{:.3f}",
                    "CN": "{:.1f}",
                    "runoff_mm": "{:.2f}",
                    "runoff_norm": "{:.3f}",
                    "UDS_runoff_index": "{:.3f}",
                    "UDS_runoff_norm": "{:.3f}",
                }
            )
        )

        export_df = parcels_uds[existing_cols].copy()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full UDS results (CSV)",
            data=csv_bytes,
            file_name=f"UDS_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("UDS analysis failed. Please check the error below.")
        st.exception(e)
else:
    st.info(
        "Set rainfall and (optionally) max flow-trace steps, "
        "then click **Run UDS Analysis**."
    )
