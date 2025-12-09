# utils/hsr_visualization.py
import numpy as np
import geopandas as gpd
import folium
import branca.colormap as cm


def build_hsr_map(parcels_hsr: gpd.GeoDataFrame, rainfall_mm: float) -> folium.Map:
    """
    Build an interactive Folium map for HSR results.

    - Colors parcels by HSR_rain_sum (m³)
    - Adds optional overlay for HSR_static_sum
    """
    if parcels_hsr.empty:
        # Fallback empty map (center of world)
        return folium.Map(location=[0, 0], zoom_start=2)

    # Reproject to WGS84 for web mapping
    parcels_wgs84 = parcels_hsr.to_crs(epsg=4326)

    # Map center
    bounds = parcels_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2.0
    center_lon = (bounds[0] + bounds[2]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Determine range for HSR_rain_sum
    if "HSR_rain_sum" in parcels_wgs84.columns:
        vals = parcels_wgs84["HSR_rain_sum"].replace(
            [np.inf, -np.inf], np.nan
        )
        if vals.notna().any():
            vmin = float(vals.min())
            vmax = float(vals.max())
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Color scale
    colormap = cm.LinearColormap(
        colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=vmin,
        vmax=vmax,
        caption=f"HSR_rain_sum (m³) – Rainfall {rainfall_mm:.0f} mm",
    )

    def style_rain(feature):
        val = feature["properties"].get("HSR_rain_sum", None)
        try:
            val_f = float(val) if val is not None else None
        except (TypeError, ValueError):
            val_f = None

        if val_f is None or np.isnan(val_f):
            color = "#cccccc"
        else:
            color = colormap(val_f)

        return {
            "fillColor": color,
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.7,
        }

    rain_layer = folium.GeoJson(
        parcels_wgs84,
        name="HSR – Rainfall-adjusted storage (sum)",
        style_function=style_rain,
        highlight_function=lambda x: {"weight": 2, "color": "black"},
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "HSR_rain_sum",
                "HSR_rain_mean",
                "HSR_rain_max",
                "HSR_static_sum",
                "HSR_static_mean",
                "HSR_static_max",
            ],
            aliases=[
                "HSR_rain_sum (m³):",
                "HSR_rain_mean (m³):",
                "HSR_rain_max (m³):",
                "HSR_static_sum (m³):",
                "HSR_static_mean (m³):",
                "HSR_static_max (m³):",
            ],
            localize=True,
        ),
    )
    rain_layer.add_to(m)

    # Optional static overlay
    if "HSR_static_sum" in parcels_wgs84.columns:
        static_layer = folium.GeoJson(
            parcels_wgs84,
            name="HSR – Static storage (sum)",
            style_function=lambda feature: {
                "fillColor": "#3182bd",
                "color": "#08519c",
                "weight": 0.5,
                "fillOpacity": 0.3,
            },
            highlight_function=lambda x: {"weight": 2, "color": "black"},
        )
        static_layer.add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m
