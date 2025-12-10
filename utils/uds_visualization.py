# utils/uds_visualization.py
import numpy as np
import folium
from branca.colormap import LinearColormap


def build_uds_map(parcels_uds, rainfall_mm: float):
    """
    Build an interactive Folium map for UDS results.

    Layers:
      - UDS structural sensitivity (uds_score_norm)
      - Combined UDS × CN runoff (UDS_runoff_norm) for the chosen rainfall
    """
    # Reproject to WGS84 for web
    gdf = parcels_uds.to_crs(epsg=4326)

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Base tiles
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB positron", control=False).add_to(m)

    # ------------------------
    # Layer 1: Structural UDS
    # ------------------------
    if "uds_score_norm" in gdf.columns:
        vals = gdf["uds_score_norm"].values.astype("float64")
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                vmax = vmin + 1.0

            cmap_struct = LinearColormap(
                colors=["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
                vmin=vmin,
                vmax=vmax,
            )
            cmap_struct.caption = "UDS structural sensitivity (normalized)"

            def style_struct(feat):
                v = feat["properties"].get("uds_score_norm")
                if v is None or not np.isfinite(v):
                    return {
                        "fillColor": "#00000000",
                        "color": "#777777",
                        "weight": 0.2,
                        "fillOpacity": 0.0,
                    }
                return {
                    "fillColor": cmap_struct(v),
                    "color": "#555555",
                    "weight": 0.3,
                    "fillOpacity": 0.7,
                }

            tooltip_struct = folium.GeoJsonTooltip(
                fields=["grid_id", "uds_score", "uds_up", "uds_down"],
                aliases=[
                    "Parcel ID:",
                    "UDS score:",
                    "Upstream parcels:",
                    "Downstream parcels:",
                ],
                localize=True,
                sticky=False,
            )

            gj_struct = folium.GeoJson(
                gdf,
                name="UDS structural sensitivity",
                style_function=style_struct,
                tooltip=tooltip_struct,
            )
            gj_struct.add_to(m)
            cmap_struct.add_to(m)

    # ------------------------
    # Layer 2: UDS × CN runoff
    # ------------------------
    if "UDS_runoff_norm" in gdf.columns:
        vals = gdf["UDS_runoff_norm"].values.astype("float64")
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                vmax = vmin + 1.0

            cmap_hazard = LinearColormap(
                colors=["#ffffe5", "#fee391", "#fe9929", "#cc4c02", "#4a1486"],
                vmin=vmin,
                vmax=vmax,
            )
            cmap_hazard.caption = f"UDS × CN runoff (normalized, {int(rainfall_mm)} mm)"

            def style_hazard(feat):
                v = feat["properties"].get("UDS_runoff_norm")
                if v is None or not np.isfinite(v):
                    return {
                        "fillColor": "#00000000",
                        "color": "#777777",
                        "weight": 0.2,
                        "fillOpacity": 0.0,
                    }
                return {
                    "fillColor": cmap_hazard(v),
                    "color": "#555555",
                    "weight": 0.3,
                    "fillOpacity": 0.7,
                }

            tooltip_hazard = folium.GeoJsonTooltip(
                fields=["grid_id", "uds_down", "runoff_mm", "UDS_runoff_norm"],
                aliases=[
                    "Parcel ID:",
                    "Downstream parcels:",
                    "Runoff depth (mm):",
                    "UDS×runoff (norm):",
                ],
                localize=True,
                sticky=False,
            )

            gj_hazard = folium.GeoJson(
                gdf,
                name=f"UDS × CN runoff ({int(rainfall_mm)} mm)",
                style_function=style_hazard,
                tooltip=tooltip_hazard,
            )
            gj_hazard.add_to(m)
            cmap_hazard.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
