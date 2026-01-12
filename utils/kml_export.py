# utils/kml_export.py
from xml.etree.ElementTree import Element, SubElement, tostring
from shapely.geometry import Polygon, MultiPolygon

_KML_NS = "http://www.opengis.net/kml/2.2"


def _coords_to_kml(coords):
    # coords = [(lon,lat), ...]
    return " ".join([f"{x},{y},0" for (x, y) in coords])


def _polygon_to_kml(parent, poly: Polygon):
    poly_el = SubElement(parent, "Polygon")

    outer = SubElement(SubElement(poly_el, "outerBoundaryIs"), "LinearRing")
    SubElement(outer, "coordinates").text = _coords_to_kml(list(poly.exterior.coords))

    # holes (optional)
    for ring in poly.interiors:
        inner = SubElement(SubElement(poly_el, "innerBoundaryIs"), "LinearRing")
        SubElement(inner, "coordinates").text = _coords_to_kml(list(ring.coords))


def gdf_to_kml_bytes(gdf_wgs84, *, doc_name="HSR Parcels", name_field=None, fields=None) -> bytes:
    """
    Convert a GeoDataFrame to KML bytes.
    IMPORTANT: gdf_wgs84 must be EPSG:4326 (lon/lat). KML requires WGS84.
    """
    gdf = gdf_wgs84[gdf_wgs84.geometry.notnull()].copy()

    kml = Element("kml", xmlns=_KML_NS)
    doc = SubElement(kml, "Document")
    SubElement(doc, "name").text = doc_name

    if fields is None:
        fields = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        pm = SubElement(doc, "Placemark")

        # name
        if name_field and name_field in row and row[name_field] is not None:
            SubElement(pm, "name").text = str(row[name_field])
        else:
            SubElement(pm, "name").text = f"Parcel {idx}"

        # attributes
        ext = SubElement(pm, "ExtendedData")
        for f in fields:
            if f not in row:
                continue
            val = row[f]
            data_el = SubElement(ext, "Data", name=str(f))
            SubElement(data_el, "value").text = "" if val is None else str(val)

        # geometry
        if isinstance(geom, Polygon):
            _polygon_to_kml(pm, geom)
        elif isinstance(geom, MultiPolygon):
            mg = SubElement(pm, "MultiGeometry")
            for poly in geom.geoms:
                _polygon_to_kml(mg, poly)

    return tostring(kml, encoding="utf-8", xml_declaration=True)
