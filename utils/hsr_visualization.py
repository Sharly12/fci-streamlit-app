# utils/hsr_visualization.py

import numpy as np
import folium
import geopandas as gpd


def build_hsr_map(parcels_gdf: gpd.GeoDataFrame, rainfall_mm: float):
    """
    Build an interactive Folium map with:
    - HSR_rain_sum (rainfall-adjusted storage, m³)
    - HSR_static_sum (static storage, m³)
    """

    # Reproject to WGS84 for web mapping
    parcels_wgs = parcels_gdf.to_crs(epsg=4326).copy()
    parcels_wgs["idx"] = np.arange(len(parcels_wgs)).astype(str)

    # Map centre
    centroids = parcels_wgs.geometry.centroid
    center_y = float(centroids.y.mean())
    center_x = float(centroids.x.mean())

    m = folium.Map(location=[center_y, center_x], zoom_start=13, tiles="cartodbpositron")

    # ------------------------------------------------------------------
    # Rainfall-adjusted storage layer
    # ------------------------------------------------------------------
    if "HSR_rain_sum" in parcels_wgs.columns:
        rain_fg = folium.FeatureGroup(
            name=f"HSR – Rainfall-adjusted storage (sum, m³)", show=False
        )
        folium.Choropleth(
            geo_data=parcels_wgs.to_json(),
            data=parcels_wgs,
            columns=["idx", "HSR_rain_sum"],
            key_on="feature.properties.idx",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            nan_fill_opacity=0,
            legend_name=f"HSR_rain_sum (m³) – Rainfall {rainfall_mm:.0f} mm",
        ).add_to(rain_fg)
        rain_fg.add_to(m)

    # ------------------------------------------------------------------
    # Static storage layer
    # ------------------------------------------------------------------
    if "HSR_static_sum" in parcels_wgs.columns:
        static_fg = folium.FeatureGroup(
            name="HSR – Static storage (sum, m³)", show=True
        )
        folium.Choropleth(
            geo_data=parcels_wgs.to_json(),
            data=parcels_wgs,
            columns=["idx", "HSR_static_sum"],
            key_on="feature.properties.idx",
            fill_color="Blues",
            fill_opacity=0.7,
            line_opacity=0.2,
            nan_fill_opacity=0,
            legend_name="HSR_static_sum (m³)",
        ).add_to(static_fg)
        static_fg.add_to(m)

    # Tooltip layer (same geometry, just outlines)
    tooltip_fields = []
    if "HSR_rain_sum" in parcels_wgs.columns:
        tooltip_fields.append("HSR_rain_sum")
    if "HSR_static_sum" in parcels_wgs.columns:
        tooltip_fields.append("HSR_static_sum")

    if tooltip_fields:
        folium.GeoJson(
            parcels_wgs.to_json(),
            name="Parcel info",
            style_function=lambda x: {
                "fillOpacity": 0.0,
                "color": "black",
                "weight": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[f"{f} (m³)" for f in tooltip_fields],
                localize=True,
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
