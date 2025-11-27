# utils/visualization.py
import folium
from streamlit_folium import folium_static
from rasterio.transform import array_bounds
from pyproj import Transformer
import matplotlib.cm as cm
import numpy as np


def make_raster_overlays(flow_acc, corridor_mask, valid_mask,
                         dem_transform, dem_crs, height, width):
    """
    Return:
      flow_rgba  – RGBA array for log10(flow_acc) in Blues,
      corr_rgba  – RGBA array for corridor mask in red,
      bounds_wgs – [[south, west], [north, east]] in WGS84.
    """
    left, bottom, right, top = array_bounds(height, width, dem_transform)
    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(left, bottom)
    east, north = transformer.transform(right, top)
    bounds_wgs = [[south, west], [north, east]]

    flow_display = np.where(valid_mask, flow_acc, np.nan)
    flow_log = np.log10(flow_display + 1.0)
    flow_log = np.nan_to_num(flow_log, nan=0.0)
    if np.nanmax(flow_log) > 0:
        flow_norm = flow_log / np.nanmax(flow_log)
    else:
        flow_norm = flow_log
    flow_rgba = cm.Blues(flow_norm)

    corr = np.where(corridor_mask == 1, 1.0, 0.0)
    corr_rgba = np.zeros((height, width, 4), dtype=float)
    corr_rgba[..., 0] = 1.0
    corr_rgba[..., 3] = corr * 0.8

    return flow_rgba, corr_rgba, bounds_wgs


def build_fci_map(parcels_result, flow_accumulation, corridor_mask,
                  base_data, rainfall_mm):
    """Create Folium map with parcels + flow accumulation + corridor overlays."""
    parcels_wgs84 = parcels_result.to_crs(epsg=4326).copy()
    parcels_wgs84["index"] = parcels_wgs84.index.astype(str)

    centers = parcels_wgs84.geometry.centroid
    center_x = centers.x.mean()
    center_y = centers.y.mean()

    m = folium.Map(
        location=[center_y, center_x],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # Choropleth by FCI
    folium.Choropleth(
        geo_data=parcels_wgs84.to_json(),
        name="Parcels – FCI (0–1)",
        data=parcels_wgs84,
        columns=["index", "FCI"],
        key_on="feature.properties.index",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"FCI (Rainfall {rainfall_mm:.1f} mm)",
    ).add_to(m)

    # Parcel outlines + tooltip
    style_function = lambda x: {
        "fillColor": "#00000000",
        "color": "black",
        "weight": 0.3,
    }
    highlight_function = lambda x: {
        "fillColor": "#00000000",
        "color": "yellow",
        "weight": 1.0,
    }

    tooltip = folium.features.GeoJsonTooltip(
        fields=[
            "FCI",
            "FCI_class_10",
            "Risk",
            "fci_sum",
            "fci_corr_sum",
            "fci_p90",
        ],
        aliases=[
            "FCI:",
            "FCI class (1–10):",
            "Risk:",
            "Total Accum.:",
            "Corridor Accum.:",
            "P90 Flow:",
        ],
        localize=True,
    )

    folium.GeoJson(
        parcels_wgs84.to_json(),
        name="Parcels – outlines & tooltip",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
    ).add_to(m)

    # Flow accumulation & corridors as raster overlays
    flow_img, corr_img, bounds_wgs84 = make_raster_overlays(
        flow_accumulation,
        corridor_mask,
        base_data["valid_mask"],
        base_data["dem_transform"],
        base_data["dem_crs"],
        base_data["height"],
        base_data["width"],
    )

    folium.raster_layers.ImageOverlay(
        image=flow_img,
        bounds=bounds_wgs84,
        name="Flow accumulation (log10)",
        opacity=0.7,
        interactive=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=corr_img,
        bounds=bounds_wgs84,
        name="Flow corridors (top 10%)",
        opacity=0.9,
        interactive=False,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
