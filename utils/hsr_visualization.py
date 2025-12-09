# utils/hsr_visualization.py
import numpy as np
import folium
from branca.colormap import LinearColormap


def _add_choropleth_layer(gdf_wgs84, column, name, colors, m):
    """Internal helper: add a single choropleth layer for a parcel column."""
    if column not in gdf_wgs84.columns:
        return

    values = gdf_wgs84[column].values.astype("float64")
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    # Avoid zero range
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    cmap.caption = name

    def style_fn(feat):
        val = feat["properties"].get(column)
        if val is None or not np.isfinite(val):
            return {
                "fillColor": "#00000000",
                "color": "#777777",
                "weight": 0.2,
                "fillOpacity": 0.0,
            }
        return {
            "fillColor": cmap(val),
            "color": "#555555",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=[column],
        aliases=[name + ":"],
        localize=True,
        sticky=False,
    )

    gj = folium.GeoJson(
        gdf_wgs84,
        name=name,
        style_function=style_fn,
        tooltip=tooltip,
    )
    gj.add_to(m)
    cmap.add_to(m)


def build_hsr_map(parcels_hsr, rainfall_mm: float):
    """
    Build an interactive Folium map for HSR results.

    Layers:
      - HSR_static_sum  (m³)  – Static storage
      - HSR_rain_sum    (m³)  – Rainfall-adjusted storage
    """
    # Reproject to WGS84 for web mapping
    parcels_wgs84 = parcels_hsr.to_crs(epsg=4326)

    bounds = parcels_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Base layers
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB positron", control=False).add_to(m)

    # Static storage
    _add_choropleth_layer(
        parcels_wgs84,
        column="HSR_static_sum",
        name="HSR – Static storage (sum, m³)",
        colors=["#f7fbff", "#08306b"],
        m=m,
    )

    # Rainfall-adjusted storage
    _add_choropleth_layer(
        parcels_wgs84,
        column="HSR_rain_sum",
        name=f"HSR_rain_sum (m³) – Rainfall {int(rainfall_mm)} mm",
        colors=["#fff5eb", "#7f0000"],
        m=m,
    )

    folium.LayerControl(collapsed=False).add_to(m)
    return m
