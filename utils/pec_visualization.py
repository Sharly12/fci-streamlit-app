# utils/pec_visualization.py
import folium
import geopandas as gpd

# Exact colour scheme from your original PEC plots
PEC_COLORS = {
    "Low-lying Depressed (Retention Priority)": "#0000ff",  # blue
    "Flat & Pressured (High Flood Exposure Risk)": "#ff0000",  # red
    "Locally High & Disconnected": "#008000",  # green
    "Moderate / Context-Dependent": "#ffff00",  # yellow
}


def build_pec_map(parcels_gdf: gpd.GeoDataFrame, rainfall_label: float | None = None):
    """
    Build a Folium map for PEC classes.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Output of run_pec_analysis (must contain 'pec_class').
    rainfall_label : float, optional
        Only used for legend text (no re-computation of classes).
    """
    if parcels_gdf.crs is None:
        raise ValueError("Parcels GeoDataFrame must have a CRS.")

    parcels_wgs = parcels_gdf.to_crs(epsg=4326)

    center = [
        parcels_wgs.geometry.centroid.y.mean(),
        parcels_wgs.geometry.centroid.x.mean(),
    ]
    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    def style_fn(feature):
        cls = feature["properties"].get(
            "pec_class", "Moderate / Context-Dependent"
        )
        return {
            "fillColor": PEC_COLORS.get(cls, "#ffffcc"),
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.8,
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
            "Parcel ID",
            "PEC class",
            "PREI (rel. elev.)",
            "HAND-like score",
            "Relief (m)",
        ],
        localize=True,
    )

    folium.GeoJson(
        parcels_wgs,
        name="PEC parcels",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Legend HTML
    title = "Parcel-level PEC classification"
    if rainfall_label is not None:
        title += f" (reference rainfall {rainfall_label:.0f} mm/h)"

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px 14px;
        border-radius: 6px;
        box-shadow: 0 0 8px rgba(0,0,0,0.3);
        font-size: 12px;
        ">
        <b>{title}</b><br>
        <table style="border-collapse: collapse; margin-top: 4px;">
            <tr><td><span style="display:inline-block;width:14px;height:14px;background:{PEC_COLORS['Low-lying Depressed (Retention Priority)']};border:1px solid #000;"></span></td><td style="padding-left:4px;">Low-lying Depressed (Retention Priority)</td></tr>
            <tr><td><span style="display:inline-block;width:14px;height:14px;background:{PEC_COLORS['Flat & Pressured (High Flood Exposure Risk)']};border:1px solid #000;"></span></td><td style="padding-left:4px;">Flat & Pressured (High Flood Exposure Risk)</td></tr>
            <tr><td><span style="display:inline-block;width:14px;height:14px;background:{PEC_COLORS['Locally High & Disconnected']};border:1px solid #000;"></span></td><td style="padding-left:4px;">Locally High & Disconnected</td></tr>
            <tr><td><span style="display:inline-block;width:14px;height:14px;background:{PEC_COLORS['Moderate / Context-Dependent']};border:1px solid #000;"></span></td><td style="padding-left:4px;">Moderate / Context-Dependent</td></tr>
        </table>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    return m
