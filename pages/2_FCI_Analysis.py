# pages/2_FCI_Analysis.py
import io
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from shapely.geometry import Polygon, MultiPolygon
from streamlit_folium import folium_static

from utils.data_loader import get_data_paths, load_base_data
from models.fci_model import run_fci_analysis
from utils.visualization import build_fci_map


def gdf_to_kml_bytes(gdf_wgs84, doc_name="FCI Parcels", fields=None) -> bytes:
    """
    Convert GeoDataFrame (must be EPSG:4326) to KML bytes with selected fields in ExtendedData.
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


def array_to_geotiff_bytes(arr, base_data, *, dtype, nodata, set_invalid=None) -> bytes:
    """
    Export a 2D numpy array to GeoTIFF bytes using base_data's transform & CRS.
    - set_invalid: if provided, invalid cells (where valid_mask=False) are set to this value
    """
    valid_mask = base_data["valid_mask"]
    dem_profile = base_data["dem_profile"].copy()

    out = arr
    if set_invalid is not None:
        out = np.where(valid_mask, arr, set_invalid)

    meta = dem_profile.copy()
    meta.update(
        driver="GTiff",
        height=int(base_data["height"]),
        width=int(base_data["width"]),
        count=1,
        dtype=dtype,
        crs=base_data["dem_crs"],
        transform=base_data["dem_transform"],
        nodata=nodata,
        compress="lzw",
    )

    with MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            dst.write(out.astype(dtype), 1)
        return memfile.read()


st.title("📈 Flow Corridor Importance (FCI) Analysis")

st.write(
    "This page runs the full Flow Corridor Importance model: "
    "NRCS runoff → D8 flow accumulation → flow corridors → parcel-level indices.\n\n"
    "- **FCI_struct** = structural importance (0–1) based on flow network and corridors.\n"
    "- **FCI** = rainfall-scaled FCI = FCI_struct × (Rainfall / 250 mm), used for risk."
)

# --- User controls ---
st.sidebar.header("FCI Parameters")
rainfall_mm = st.sidebar.slider(
    "Design Rainfall (mm)", min_value=0.0, max_value=250.0, value=100.0, step=5.0
)
use_nrcs = st.sidebar.checkbox(
    "Use NRCS Runoff Method (recommended)", value=True
)

# --- Load shared DEM/CN/parcels ---
dem_path, parcels_path, cn_path = get_data_paths()
base_data = load_base_data(dem_path, parcels_path, cn_path)

if st.button("Run FCI Analysis"):
    with st.spinner("Running FCI model ..."):
        (
            parcels_result,
            diagnostics,
            corridor_threshold,
            corridor_cells,
            risk_counts,
            flow_accumulation,
            corridor_mask,
        ) = run_fci_analysis(rainfall_mm, use_nrcs, base_data)

    st.success("✅ FCI analysis complete")

    # ------------------------------------------------------------------
    # ✅ Downloads (GeoTIFF + KML) — EXPORT ONLY, NO ANALYSIS CHANGES
    # ------------------------------------------------------------------
    st.subheader("Downloads")

    # GeoTIFF exports (model visualization layers)
    # 1) Flow accumulation (float) — set invalid to -9999 to avoid mixing with real zeros
    flow_tif = array_to_geotiff_bytes(
        flow_accumulation,
        base_data,
        dtype="float32",
        nodata=-9999.0,
        set_invalid=-9999.0,
    )

    # 2) Corridor mask (0/1)
    corr_tif = array_to_geotiff_bytes(
        corridor_mask.astype("uint8"),
        base_data,
        dtype="uint8",
        nodata=0,
        set_invalid=0,
    )

    # 3) Optional corridor-only accumulation (nice for GIS)
    corridor_acc = (flow_accumulation * corridor_mask).astype("float32")
    corr_acc_tif = array_to_geotiff_bytes(
        corridor_acc,
        base_data,
        dtype="float32",
        nodata=-9999.0,
        set_invalid=-9999.0,
    )

    # ZIP all rasters
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"FCI_flow_accumulation_{int(rainfall_mm)}mm.tif", flow_tif)
        zf.writestr(f"FCI_corridor_mask_{int(rainfall_mm)}mm.tif", corr_tif)
        zf.writestr(f"FCI_corridor_accumulation_{int(rainfall_mm)}mm.tif", corr_acc_tif)
    zip_bytes = zip_buf.getvalue()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download Flow Accum (GeoTIFF)",
            data=flow_tif,
            file_name=f"FCI_flow_accumulation_{int(rainfall_mm)}mm.tif",
            mime="image/tiff",
        )
    with c2:
        st.download_button(
            "Download Corridor Mask (GeoTIFF)",
            data=corr_tif,
            file_name=f"FCI_corridor_mask_{int(rainfall_mm)}mm.tif",
            mime="image/tiff",
        )
    with c3:
        st.download_button(
            "Download Corridor Accum (GeoTIFF)",
            data=corr_acc_tif,
            file_name=f"FCI_corridor_accumulation_{int(rainfall_mm)}mm.tif",
            mime="image/tiff",
        )
    with c4:
        st.download_button(
            "Download All FCI Rasters (ZIP)",
            data=zip_bytes,
            file_name=f"FCI_rasters_{int(rainfall_mm)}mm.zip",
            mime="application/zip",
        )

    # Parcel layer as KML (FCI results)
    parcels_wgs84 = parcels_result.to_crs(epsg=4326)
    kml_fields = [
        "FCI",
        "FCI_struct",
        "FCI_class_10",
        "Risk",
        "fci_sum",
        "fci_corr_sum",
        "fci_p90",
        "Rainfall_mm",
    ]
    existing_fields = [c for c in kml_fields if c in parcels_wgs84.columns]
    kml_bytes = gdf_to_kml_bytes(parcels_wgs84, doc_name="FCI Parcels", fields=existing_fields)

    st.download_button(
        "Download parcel FCI layer (KML)",
        data=kml_bytes,
        file_name=f"FCI_parcels_{int(rainfall_mm)}mm.kml",
        mime="application/vnd.google-earth.kml+xml",
    )

    # Diagnostics
    st.subheader("Hydrologic Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Flow direction scheme: **{diagnostics['scheme']}**")
        st.write(f"- Total input runoff: **{diagnostics['total_input']:.2f} mm**")
        st.write(f"- Total accumulated: **{diagnostics['total_accumulated']:.2f} mm**")
        st.write(f"- Outlet accumulation: **{diagnostics['outlet_accumulation']:.2f} mm**")
        st.write(f"- Mass balance error: **{diagnostics['mass_balance_error_pct']:.3f}%**")
    with c2:
        st.write(f"- Number of outlets: **{diagnostics['num_outlets']}**")
        st.write(f"- Unprocessed cells: **{diagnostics['num_unprocessed']}**")
        st.write(f"- Processing iterations: **{diagnostics['iterations']}**")
        st.write(f"- Corridor threshold (top 10%): **{corridor_threshold:.2f}**")
        st.write(f"- Corridor cells: **{corridor_cells:,}**")

    # Risk distribution
    st.subheader("Parcel Risk Distribution (based on rainfall-scaled FCI)")
    risk_df = pd.DataFrame(
        [
            {"Risk": lbl, "Parcels": cnt, "Percent": cnt / len(parcels_result) * 100.0}
            for lbl, cnt in risk_counts.items()
        ]
    ).set_index("Risk")
    st.table(risk_df.style.format({"Percent": "{:.1f}"}))

    # Map
    st.subheader("Interactive Map – Parcels + Flow Accumulation + Corridors")
    fci_map = build_fci_map(
        parcels_result,
        flow_accumulation,
        corridor_mask,
        base_data,
        rainfall_mm,
    )
    folium_static(fci_map, width=1000, height=600)

    # Top parcels + + CSV
    st.subheader("Top 10 High-Risk Parcels")
    cols = [
        "FCI",          # rainfall-scaled
        "FCI_struct",   # structural component
        "FCI_class_10",
        "Risk",
        "fci_sum",
        "fci_corr_sum",
        "fci_p90",
        "Rainfall_mm",
    ]
    existing_cols = [c for c in cols if c in parcels_result.columns]
    table_df = parcels_result.sort_values("FCI", ascending=False)[existing_cols]
    st.dataframe(
        table_df.head(10).style.format(
            {
                "FCI": "{:.3f}",
                "FCI_struct": "{:.3f}",
                "fci_sum": "{:.0f}",
                "fci_corr_sum": "{:.0f}",
                "fci_p90": "{:.0f}",
            }
        )
    )

    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full parcel results (CSV)",
        data=csv_bytes,
        file_name=f"FCI_results_{int(rainfall_mm)}mm.csv",
        mime="text/csv",
    )
else:
    st.info("Set rainfall and click **Run FCI Analysis** to start.")
