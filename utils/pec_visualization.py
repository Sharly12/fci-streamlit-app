# utils/pec_visualization.py
import folium
import numpy as np
from branca.colormap import LinearColormap


PEC_COLORS = {
    "Low-lying Depressed (Retention Priority)": "#3182bd",  # blue
    "Flat & Pressured (High Flood Exposure Risk)": "#de2d26",  # red
    "Locally High & Disconnected": "#31a354",  # green
    "Moderate / Context-Dependent": "#fed976",  # yellow/orange
}


def build_pec_map(parcels_pec, rainfall_mm: float = 0.0):
    """
    Build interactive PEC map with categorical styling and tooltips.
    """

    gdf = parcels_pec.to_crs(epsg=4326)
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0

    m = folium.Map(
        location=[cy, cx],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB positron", control=False).add_to(m)

    def style_fn(feat):
        cls = feat["properties"].get("pec_class", "Moderate / Context-Dependent")
        color = PEC_COLORS.get(cls, "#999999")
        return {
            "fillColor": color,
            "color": "#444444",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=[
            "grid_id",
            "pec_class",
            "prei",
            "hand_score",
            "relief",
        ],
        aliases=[
            "Parcel ID:",
            "PEC class:",
            "PREI (rel. elev.):",
            "HAND score:",
            "Relief (max-min, m):",
        ],
        localize=True,
        sticky=False,
    )

    folium.GeoJson(
        gdf,
        name=f"PEC classification (rainfall {int(rainfall_mm)} mm)",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Discrete legend (simple HTML)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 10px;
        z-index: 9999;
        background-color: white;
        padding: 8px 10px;
        border: 1px solid #ccc;
        border-radius: 6px;
        box-shadow: 0 0 3px rgba(0,0,0,0.3);
        font-size: 12px;
    ">
      <b>PEC Classes</b><br>
      <span style="background:{c1};width:12px;height:12px;display:inline-block;margin-right:4px;"></span>
      Low-lying Depressed (Retention Priority)<br>
      <span style="background:{c2};width:12px;height:12px;display:inline-block;margin-right:4px;"></span>
      Flat &amp; Pressured (High Flood Exposure Risk)<br>
      <span style="background:{c3};width:12px;height:12px;display:inline-block;margin-right:4px;"></span>
      Locally High &amp; Disconnected<br>
      <span style="background:{c4};width:12px;height:12px;display:inline-block;margin-right:4px;"></span>
      Moderate / Context-Dependent
    </div>
    """.format(
        c1=PEC_COLORS["Low-lying Depressed (Retention Priority)"],
        c2=PEC_COLORS["Flat & Pressured (High Flood Exposure Risk)"],
        c3=PEC_COLORS["Locally High & Disconnected"],
        c4=PEC_COLORS["Moderate / Context-Dependent"],
    )

    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m
