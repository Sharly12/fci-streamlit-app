# pages/5_SEI_Analysis.py
import os
import io
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring

import streamlit as st
from streamlit_folium import folium_static
from shapely.geometry import Polygon, MultiPolygon

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
# ------------------------------------------------------------------
DEM_PATH_IN, DEFAULT_PARCELS_PATH, CN_RASTER_PATH = get_data_paths()
DEFAULT_LU_PATH = get_lu_path()


def gdf_to_kml_bytes(gdf_wgs84, doc_name="SEI Parcels", fields=None) -> bytes:
    """Convert GeoDataFrame (must be EPSG:4326) to KML bytes with selected fields."""
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


# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.header("SEI Parameters")

    parcels_path = st.text_input(
        "Parcels layer path",
        DEFAULT_PARCELS_PATH,
        help="Defaults to the parcels file downloaded via Dropbox (same as FCI).",
    )

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
        "Optional hazard raster (e.g., flood depth TIFF). Leave blank for LU-only SEI.",
        "",
    )
    use_hazard = bool(hazard_path.strip())

run_btn = st.button("Run SEI Analysis")

# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------
if run_btn:
    missing_messages = []

    if not os.path.exists(parcels_path):
        missing_messages.append(f"❌ Parcels file not found: `{parcels_path}`")

    if not os.path.exists(landuse_path):
        missing_messages.append(f"❌ Land-use file not found: `{landuse_path}`")

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
            with st.spinner("Computing SEI … this may take a moment for larger datasets."):
                # ✅ now returns outputs too (GeoTIFF path)
                parcels_sei, diagnostics, outputs = run_sei_analysis(
                    parcels_path=parcels_path,
                    landuse_path=landuse_path,
                    lu_field=lu_field,
                    buffer_m=buffer_m,
                    hazard_raster=haz_path_clean if use_hazard else None,
                )

            st.success("✅ SEI analysis complete")

            # ------------------------------------------------------------
            # ✅ Downloads (GeoTIFF + ZIP + parcel KML + CSV)
            # ------------------------------------------------------------
            st.subheader("Downloads")

            # SEI raster GeoTIFF
            with open(outputs["sei_index_tif"], "rb") as f:
                sei_tif_bytes = f.read()

            # ZIP (just the SEI raster for now)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"SEI_index_buffer{int(buffer_m)}m.tif", sei_tif_bytes)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download SEI raster (GeoTIFF)",
                    data=sei_tif_bytes,
                    file_name=f"SEI_index_buffer{int(buffer_m)}m.tif",
                    mime="image/tiff",
                )
            with c2:
                st.download_button(
                    "Download SEI rasters (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name=f"SEI_rasters_buffer{int(buffer_m)}m.zip",
                    mime="application/zip",
                )

            # Parcel KML (WGS84)
            parcels_wgs84 = parcels_sei.to_crs(epsg=4326)
            kml_fields = [
                "SEI",
                "SEI_raw",
                "nei_aw_sum",
                "nei_aw_norm",
                "haz_mean",
                "haz_norm",
                "haz_max",
                "haz_max_norm",
            ]
            existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]
            kml_bytes = gdf_to_kml_bytes(parcels_wgs84, doc_name="SEI Parcels", fields=existing_fields)

            with c3:
                st.download_button(
                    "Download parcel SEI layer (KML)",
                    data=kml_bytes,
                    file_name=f"SEI_parcels_buffer{int(buffer_m)}m.kml",
                    mime="application/vnd.google-earth.kml+xml",
                )

            # CSV export
            export_cols = [c for c in kml_fields if c in parcels_sei.columns]
            csv_bytes = parcels_sei[export_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full SEI results (CSV)",
                data=csv_bytes,
                file_name=f"SEI_results_buffer{int(buffer_m)}m.csv",
                mime="text/csv",
            )

            # Diagnostics
            st.subheader("Diagnostics & Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_sei)))
            with c2:
                st.metric("Buffer radius (m)", f"{diagnostics.get('buffer_m', buffer_m):.0f}")
            with c3:
                st.metric("Hazard raster used", "Yes" if diagnostics.get("hazard_used", False) else "No")

            unmatched = diagnostics.get("unmatched_examples", [])
            if unmatched:
                with st.expander("Unmatched land-use labels (fell back to 'Unclassified')"):
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
