# utils/pec_visualization.py

import folium
import geopandas as gpd

# Fixed colour scheme – matches your original Matplotlib map
PEC_COLORS = {
    "Low-lying Depressed (Retention Priority)": "#2166ac",   # blue
    "Flat & Pressured (High Flood Exposure Risk)": "#b2182b", # red
    "Locally High & Disconnected": "#1a9850",                # green
    "Moderate / Context-Dependent": "#ffffbf",               # yellow
}


def build_pec_map(parcels_gdf: gpd.GeoDataFrame, rainfall_label=None):
    """
    Build an interactive PEC map (Folium) with the same colours as the
    original PEC plots.
    """
    if parcels_gdf.crs is None:
        raise ValueError("PEC parcels must have a CRS.")

    parcels_wgs = parcels_gdf.to_crs(epsg=4326)
    centroid = parcels_wgs.geometry.unary_union.centroid

    if rainfall_label is None:
        title = "Parcel-level PEC classification"
    else:
        title = f"Parcel-level PEC classification (reference rainfall {rainfall_label:.0f} mm/h)"

    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    def style_function(feature):
        cls = feature["properties"].get("pec_class", "Moderate / Context-Dependent")
        color = PEC_COLORS.get(cls, "#ffffbf")
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.8,
        }

    folium.GeoJson(
        parcels_wgs,
        name="PEC parcels",
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["grid_id", "pec_class"],
            aliases=["Parcel ID", "PEC class"],
            sticky=False,
        ),
    ).add_to(m)

    # HTML legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index:9999;
        background-color: rgba(255,255,255,0.9);
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 12px;
        box-shadow: 0 0 5px rgba(0,0,0,0.3);
    ">
      <b>{title}</b><br>
    """
    for label, color in PEC_COLORS.items():
        legend_html += (
            f'<div><span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border:1px solid #555;margin-right:4px;"></span>'
            f'{label}</div>'
        )
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m
