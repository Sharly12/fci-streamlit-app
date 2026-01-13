# pages/3_PEC_Analysis.py
import io
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring

import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from utils.data_loader import get_data_paths
from models.pec_model import run_pec_analysis
from utils.pec_visualization import build_pec_map


def gdf_to_kml_bytes(gdf_wgs84, doc_name="PEC Parcels", fields=None) -> bytes:
    """
    Convert GeoDataFrame (must be EPSG:4326) into KML bytes.
    Stores selected fields into ExtendedData.
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


st.title("🌄 Parcel Elevation Context (PEC) Analysis")

st.write(
    """
The Parcel Elevation Context (PEC) model evaluates how each parcel sits
within its local terrain:

- **Elevation & relief** from a conditioned DEM  
- **Slope**, derived from DEM gradients  
- **Local relative elevation** (PREI = DEM − neighbourhood mean)  
- **HAND-like score**: elevation above nearby drainage / local minimum  

These are combined to classify each parcel into four PEC classes:

1. Low-lying Depressed (Retention Priority)  
2. Flat & Pressured (High Flood Exposure Risk)  
3. Locally High & Disconnected  
4. Moderate / Context-Dependent
"""
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("PEC Parameters")

    rainfall_mm = st.slider(
        "Rainfall depth for PEC adjustment (mm)",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=10.0,
        help="Set to 0 for static PEC (no rainfall adjustment).",
    )

    neighbourhood_radius_m = st.slider(
        "Neighbourhood radius for PREI (m)",
        min_value=100.0,
        max_value=500.0,
        value=250.0,
        step=50.0,
        help="Radius for computing neighbourhood mean elevation.",
    )

    stream_threshold = st.slider(
        "Stream extraction threshold (cell count)",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Higher = fewer streams; lower = more streams.",
    )

run_btn = st.button("Run PEC Analysis")


# --------------- Cached wrapper -------------------
@st.cache_data(show_spinner="Running PEC engine …")
def _run_pec_cached(
    dem_path, parcels_path, rainfall_mm, neighbourhood_radius_m, stream_threshold
):
    return run_pec_analysis(
        dem_path=dem_path,
        parcels_path=parcels_path,
        rainfall_mm=rainfall_mm,
        neighbourhood_radius_m=neighbourhood_radius_m,
        stream_threshold=stream_threshold,
    )


# --------------- Main execution -------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        # ✅ now returns 3 outputs
        parcels_pec, diagnostics, outputs = _run_pec_cached(
            dem_path,
            parcels_path,
            rainfall_mm,
            neighbourhood_radius_m,
            stream_threshold,
        )

        st.success("✅ PEC analysis complete")

        # ------------------------------------------------------------
        # ✅ Downloads (GeoTIFF + ZIP + parcel KML) — export only
        # ------------------------------------------------------------
        st.subheader("Downloads")

        # Read GeoTIFFs produced by the model
        def _read_bytes(path):
            with open(path, "rb") as f:
                return f.read()

        dem_filled_bytes = _read_bytes(outputs["dem_filled_tif"])
        slope_bytes = _read_bytes(outputs["slope_deg_tif"])
        prei_bytes = _read_bytes(outputs["prei_tif"])
        hand_bytes = _read_bytes(outputs["hand_tif"])
        acc_bytes = _read_bytes(outputs["flow_accumulation_tif"])
        streams_bytes = _read_bytes(outputs["streams_mask_tif"])

        # ZIP bundle
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("PEC_dem_filled.tif", dem_filled_bytes)
            zf.writestr("PEC_slope_deg.tif", slope_bytes)
            zf.writestr("PEC_PREI_relative_elevation.tif", prei_bytes)
            zf.writestr("PEC_HAND.tif", hand_bytes)
            zf.writestr(f"PEC_flow_accumulation_stream{int(stream_threshold)}.tif", acc_bytes)
            zf.writestr(f"PEC_streams_mask_stream{int(stream_threshold)}.tif", streams_bytes)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "DEM filled (GeoTIFF)",
                data=dem_filled_bytes,
                file_name="PEC_dem_filled.tif",
                mime="image/tiff",
            )
            st.download_button(
                "Slope (GeoTIFF)",
                data=slope_bytes,
                file_name="PEC_slope_deg.tif",
                mime="image/tiff",
            )
        with c2:
            st.download_button(
                "PREI / Relative Elevation (GeoTIFF)",
                data=prei_bytes,
                file_name=f"PEC_PREI_{int(neighbourhood_radius_m)}m.tif",
                mime="image/tiff",
            )
            st.download_button(
                "HAND (GeoTIFF)",
                data=hand_bytes,
                file_name=f"PEC_HAND_stream{int(stream_threshold)}.tif",
                mime="image/tiff",
            )
        with c3:
            st.download_button(
                "Flow Accumulation (GeoTIFF)",
                data=acc_bytes,
                file_name=f"PEC_flow_accumulation_stream{int(stream_threshold)}.tif",
                mime="image/tiff",
            )
            st.download_button(
                "All PEC rasters (ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"PEC_rasters_{int(rainfall_mm)}mm.zip",
                mime="application/zip",
            )

        # Parcel layer as KML (KML requires WGS84)
        parcels_wgs84 = parcels_pec.to_crs(epsg=4326)
        kml_fields = [
            "grid_id",
            "pec_class",
            "pec_code",
            "prei",
            "hand_score",
            "relief",
            "slp_mean",
            "dem_min",
            "dem_max",
            "hand_min",
            "hand_mean",
        ]
        existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]
        kml_bytes = gdf_to_kml_bytes(parcels_wgs84, doc_name="PEC Parcels", fields=existing_fields)

        st.download_button(
            "Download parcel PEC layer (KML)",
            data=kml_bytes,
            file_name=f"PEC_parcels_{int(rainfall_mm)}mm.kml",
            mime="application/vnd.google-earth.kml+xml",
        )

        # Diagnostics
        st.subheader("Diagnostics & Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_pec)))
            st.metric("DEM resolution (m)", f"{diagnostics.get('dem_res_m', float('nan')):.2f}")
        with c2:
            st.metric(
                "Neighbourhood radius (m)",
                f"{diagnostics.get('neighbourhood_radius_m', neighbourhood_radius_m):.0f}",
            )
            st.metric("Radius (pixels)", diagnostics.get("neighbourhood_radius_pixels", 0))
        with c3:
            st.metric("Stream threshold (cells)", diagnostics.get("stream_threshold", stream_threshold))
            st.metric("Rainfall (mm)", f"{diagnostics.get('rainfall_mm', 0):.0f}")

        class_counts = diagnostics.get("pec_class_counts", {})
        if class_counts:
            st.write("**Parcel counts by PEC class**")
            cc_df = (
                pd.DataFrame([{"PEC class": k, "Parcels": v} for k, v in class_counts.items()])
                .sort_values("Parcels", ascending=False)
                .reset_index(drop=True)
            )
            st.table(cc_df)

        # Map
        st.subheader("Interactive PEC Map")
        pec_map = build_pec_map(parcels_pec, rainfall_mm=rainfall_mm)
        folium_static(pec_map, width=1000, height=600)

        # Table + CSV download
        st.subheader("Top parcels by PEC (higher risk first)")
        risk_order = [
            "Low-lying Depressed (Retention Priority)",
            "Flat & Pressured (High Flood Exposure Risk)",
            "Locally High & Disconnected",
            "Moderate / Context-Dependent",
        ]
        parcels_pec["pec_rank"] = parcels_pec["pec_class"].apply(
            lambda c: risk_order.index(c) if c in risk_order else len(risk_order)
        )

        cols = [
            "grid_id",
            "pec_class",
            "prei",
            "hand_score",
            "relief",
            "slp_mean",
            "dem_min",
            "dem_max",
            "hand_min",
            "hand_mean",
        ]
        existing_cols = [c for c in cols if c in parcels_pec.columns]

        table_df = parcels_pec.sort_values(["pec_rank", "hand_score"])[existing_cols].head(10).copy()

        st.dataframe(
            table_df.style.format(
                {
                    "prei": "{:.2f}",
                    "hand_score": "{:.2f}",
                    "relief": "{:.1f}",
                    "slp_mean": "{:.2f}",
                    "dem_min": "{:.1f}",
                    "dem_max": "{:.1f}",
                    "hand_min": "{:.2f}",
                    "hand_mean": "{:.2f}",
                }
            )
        )

        export_cols = [
            "grid_id",
            "pec_class",
            "pec_code",
            "prei",
            "hand_score",
            "relief",
            "slp_mean",
            "dem_min",
            "dem_max",
            "hand_min",
            "hand_mean",
        ]
        export_cols = [c for c in export_cols if c in parcels_pec.columns]
        csv_bytes = parcels_pec[export_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download full PEC results (CSV)",
            data=csv_bytes,
            file_name=f"PEC_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("PEC analysis failed. Please see the error below.")
        st.exception(e)
else:
    st.info(
        "Set parameters in the sidebar and click **Run PEC Analysis** "
        "to compute parcel elevation context."
    )
