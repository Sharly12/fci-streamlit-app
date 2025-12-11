# utils/pec_visualization.py

import geopandas as gpd
import folium


def build_pec_map(parcels_pec, rainfall_label=None):
    """
    Build an interactive PEC map with the SAME colours as the
    original matplotlib PEC script.

    Parameters
    ----------
    parcels_pec : GeoDataFrame
        Output of the PEC model. Must contain a 'pec_class' column.
    rainfall_label : optional
        If not None, a small note is added to the layer name
        (e.g. 'Rainfall 100 mm/h'). This keeps compatibility with
        the current page code, which may pass a label.
    """
    if parcels_pec is None or len(parcels_pec) == 0:
        raise ValueError("parcels_pec is empty – nothing to map.")

    if "pec_class" not in parcels_pec.columns:
        raise ValueError("GeoDataFrame must contain a 'pec_class' column.")

    if parcels_pec.crs is None:
        raise ValueError("PEC GeoDataFrame has no CRS defined.")

    # --- Colour scheme: EXACT match to your original script ---
    pec_colors = {
        "Low-lying Depressed (Retention Priority)": "#0000FF",  # blue
        "Flat & Pressured (High Flood Exposure Risk)": "#FF0000",  # red
        "Locally High & Disconnected": "#008000",  # green
        "Moderate / Context-Dependent": "#FFFF00",  # yellow
    }

    # Work in WGS84 for Folium
    gdf = parcels_pec.copy().to_crs(epsg=4326)

    # Add a colour column used only for styling
    gdf["__color__"] = gdf["pec_class"].map(pec_colors).fillna("#999999")

    # Map centre
    center_y = gdf.geometry.centroid.y.mean()
    center_x = gdf.geometry.centroid.x.mean()

    layer_name = "Parcel Elevation Context (PEC)"
    if rainfall_label is not None:
        layer_name += f" – Rainfall {rainfall_label} mm/h"

    m = folium.Map(
        location=[center_y, center_x],
        zoom_start=13,
        tiles="cartodbpositron",
        control_scale=True,
    )

    # Style function uses the pre-computed colour
    def style_fn(feature):
        return {
            "fillColor": feature["properties"].get("__color__", "#999999"),
            "color": "#000000",
            "weight": 0.3,
            "fillOpacity": 0.8,
        }

    tooltip = folium.features.GeoJsonTooltip(
        fields=["pec_class"],
        aliases=["PEC class:"],
        sticky=True,
    )

    folium.GeoJson(
        gdf.to_json(),
        name=layer_name,
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    # --- Legend with the same four colours & labels ---
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #444444;
        padding: 10px;
        border-radius: 4px;
        font-size: 13px;
    ">
      <b>PEC class</b><br>
      <div style="margin-top:4px">
        <div>
          <span style="background:#0000FF; width:12px; height:12px;
                       display:inline-block; margin-right:4px;"></span>
          Low-lying Depressed (Retention Priority)
        </div>
        <div>
          <span style="background:#FF0000; width:12px; height:12px;
                       display:inline-block; margin-right:4px;"></span>
          Flat &amp; Pressured (High Flood Exposure Risk)
        </div>
        <div>
          <span style="background:#008000; width:12px; height:12px;
                       display:inline-block; margin-right:4px;"></span>
          Locally High &amp; Disconnected
        </div>
        <div>
          <span style="background:#FFFF00; width:12px; height:12px;
                       display:inline-block; margin-right:4px;"></span>
          Moderate / Context-Dependent
        </div>
      </div>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    return m
