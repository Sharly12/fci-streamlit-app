# utils/pec_visualization.py
import folium
import geopandas as gpd

PEC_COLORS = {
    "Low-lying Depressed (Retention Priority)": "#1f78b4",
    "Flat & Pressured (High Flood Exposure Risk)": "#e31a1c",
    "Locally High & Disconnected": "#33a02c",
    "Moderate / Context-Dependent": "#ffd92f",
}


def build_pec_map(parcels_pec: gpd.GeoDataFrame, rainfall_mm: float = 0.0):
    if parcels_pec.crs is None:
        raise ValueError("PEC GeoDataFrame has no CRS.")

    gdf = parcels_pec.to_crs(epsg=4326).copy()
    centroids = gdf.geometry.centroid
    center = [centroids.y.mean(), centroids.x.mean()]

    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    def style_fn(feature):
        cls = feature["properties"].get("pec_class")
        color = PEC_COLORS.get(cls, "#999999")
        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    tooltip_fields = []
    aliases = []
    for col, alias in [
        ("grid_id", "Grid ID"),
        ("pec_class", "PEC class"),
        ("prei", "PREI"),
        ("hand_score", "HAND score"),
        ("relief", "Relief (m)"),
        ("slp_mean", "Mean slope (deg)"),
    ]:
        if col in gdf.columns:
            tooltip_fields.append(col)
            aliases.append(alias)

    folium.GeoJson(
        gdf,
        name="PEC parcels",
        style_function=style_fn,
        highlight_function=lambda f: {"weight": 1.0, "color": "black"},
        tooltip=folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=aliases,
            localize=True,
        ),
    ).add_to(m)

    title_suffix = f" — {int(rainfall_mm)} mm" if rainfall_mm and rainfall_mm > 0 else ""
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 10px 12px;
        border-radius: 6px;
        box-shadow: 0 0 4px rgba(0,0,0,0.3);
        font-size: 13px;
    ">
      <b>PEC classes{title_suffix}</b><br>
    """

    for label, color in PEC_COLORS.items():
        legend_html += f"""
        <div style="margin-top: 4px;">
          <span style="display:inline-block;width:14px;height:14px;
                       background:{color};border:1px solid #555;
                       margin-right:6px;"></span>{label}
        </div>
        """

    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m
