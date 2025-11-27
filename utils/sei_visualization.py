# utils/sei_visualization.py
# SEI-specific map builder (kept separate so FCI visualisation stays untouched)

import geopandas as gpd
import pandas as pd
import folium
from branca.colormap import linear
from folium.features import GeoJson, GeoJsonTooltip


def build_sei_map(parcels_sei: gpd.GeoDataFrame, buffer_m: float):
    """
    Build a Folium map for SEI results.

    Parameters
    ----------
    parcels_sei : GeoDataFrame
        Parcels with a 'SEI' column.
    buffer_m : float
        Buffer radius used in SEI calculation (for legend text only).

    Returns
    -------
    folium.Map
    """
    if parcels_sei.empty:
        raise ValueError("No parcels to map.")
    if "SEI" not in parcels_sei.columns:
        raise KeyError("GeoDataFrame must contain an 'SEI' column.")

    gdf = parcels_sei.to_crs(epsg=4326).copy()
    gdf["index"] = gdf.index.astype(str)

    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # 5-class SEI classification
    try:
        gdf["SEI_class"] = pd.qcut(
            gdf["SEI"].rank(method="first"),
            q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )
    except ValueError:
        gdf["SEI_class"] = "All parcels"

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="cartodbpositron",
        control_scale=True,
    )

    vmin, vmax = float(gdf["SEI"].min()), float(gdf["SEI"].max())
    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = f"Surrounding Exposure Index (SEI) – buffer {buffer_m:.0f} m"

    # Choropleth by SEI
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["index", "SEI"],
        key_on="feature.properties.index",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.3,
        highlight=True,
        legend_name="SEI",
    ).add_to(m)

    tooltip = GeoJsonTooltip(
        fields=["SEI", "SEI_class", "nei_aw_sum", "haz_mean", "haz_max"],
        aliases=[
            "SEI score:",
            "Class:",
            "Weighted LU exposure:",
            "Mean hazard:",
            "Max hazard:",
        ],
        localize=True,
    )

    gj = GeoJson(
        gdf,
        name="Parcels (SEI)",
        tooltip=tooltip,
        style_function=lambda x: {
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.0,
        },
        highlight_function=lambda x: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0.1,
        },
    )
    gj.add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    return m
