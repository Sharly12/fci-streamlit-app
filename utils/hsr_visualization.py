# utils/hsr_visualization.py

import folium
import geopandas as gpd


def build_hsr_map(parcels_gdf: gpd.GeoDataFrame, rainfall_mm: float) -> folium.Map:
    """
    Build an interactive Folium map with:
      - HSR_static_sum (Blues)
      - HSR_rain_sum (YlOrRd)
    """

    # Reproject to WGS84 for web mapping
    parcels_wgs84 = parcels_gdf.to_crs(epsg=4326).reset_index(drop=True)
    parcels_wgs84["index"] = parcels_wgs84.index.astype(str)

    # Map centre
    centroids = parcels_wgs84.geometry.centroid
    center_y = centroids.y.mean()
    center_x = centroids.x.mean()

    m = folium.Map(
        location=[center_y, center_x],
        zoom_start=13,
        tiles="cartodbpositron",
    )

    geojson = parcels_wgs84.to_json()

    # Static storage layer
    folium.Choropleth(
        geo_data=geojson,
        name="HSR – Static storage (sum)",
        data=parcels_wgs84,
        columns=["index", "HSR_static_sum"],
        key_on="feature.properties.index",
        fill_color="Blues",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="HSR_static_sum (m³)",
    ).add_to(m)

    # Rainfall-adjusted storage layer
    folium.Choropleth(
        geo_data=geojson,
        name=f"HSR – Rainfall-adjusted storage (sum, {rainfall_mm:.0f} mm)",
        data=parcels_wgs84,
        columns=["index", "HSR_rain_sum"],
        key_on="feature.properties.index",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=f"HSR_rain_sum (m³) – Rainfall {rainfall_mm:.0f} mm",
    ).add_to(m)

    # Thin grid outlines + tooltip
    style_fn = lambda x: {
        "fillColor": "transparent",
        "color": "#555555",
        "weight": 0.4,
        "fillOpacity": 0.0,
    }
    tooltip = folium.GeoJsonTooltip(
        fields=["HSR_static_sum", "HSR_rain_sum"],
        aliases=["Static storage (m³):", "Rain-filled storage (m³):"],
        localize=True,
    )

    folium.GeoJson(
        geojson,
        name="Grid",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
