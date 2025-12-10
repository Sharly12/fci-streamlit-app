# pages/3_PEC_Analysis.py
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd

from utils.data_loader import get_data_paths
from models.pec_model import run_pec_analysis
from utils.pec_visualization import build_pec_map

st.title("🌄 Parcel Elevation Context (PEC) Analysis")

st.write(
    """
The Parcel Elevation Context (PEC) model characterises each parcel's
topographic setting using:

- **Elevation & relief** (DEM min / max / mean)
- **Slope** (mean parcel slope)
- **Local relative elevation** (PREI = DEM - neighbourhood mean, ~250 m)
- **Hydrologic position** via HAND (elevation above streams)

These indicators are combined into a **PEC class**:

1. Low-lying Depressed (Retention Priority)  
2. Flat & Pressured (High Flood Exposure Risk)  
3. Locally High & Disconnected  
4. Moderate / Context-Dependent  

You can optionally apply a rainfall depth so that thresholds and PREI
are adjusted to reflect more intense events.
"""
)

# --------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------
with st.sidebar:
    st.header("PEC Parameters")

    rainfall_mm = st.slider(
        "Rainfall depth for PEC adjustment (mm)",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=10.0,
        help="Set to 0 for purely static PEC (no rainfall adjustment).",
    )

    neighbourhood_radius_m = st.slider(
        "Neighbourhood radius for PREI (m)",
        min_value=100.0,
        max_value=500.0,
        value=250.0,
        step=50.0,
        help="Radius for computing neighbourhood mean elevation.",
    )

    stream_threshold = st.slider(
        "Flow accumulation threshold for streams (cells)",
        min_value=100,
        max_value=2000,
        value=400,
        step=100,
        help="Higher values = only major streams; lower = more streams.",
    )

run_btn = st.button("Run PEC Analysis")

# --------------------------------------------------------------
# Cached wrapper
# --------------------------------------------------------------
@st.cache_data(show_spinner="Running PEC engine …")
def _run_pec_cached(
    dem_path, parcels_path, rainfall_mm, neighbourhood_radius_m, stream_threshold
):
    return run_pec_analysis(
        dem_path=dem_path,
        parcels_path=parcels_path,
        rainfall_mm=rainfall_mm,
        neighbourhood_radius_m=neighbourhood_radius_m,
        stream_threshold=stream_threshold,
    )


# --------------------------------------------------------------
# Execute
# --------------------------------------------------------------
if run_btn:
    try:
        dem_path, parcels_path, cn_path = get_data_paths()

        parcels_pec, diagnostics = _run_pec_cached(
            dem_path,
            parcels_path,
            rainfall_mm,
            neighbourhood_radius_m,
            stream_threshold,
        )

        st.success("✅ PEC analysis complete")

        # ---------- Diagnostics ----------
        st.subheader("Diagnostics & Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parcels analysed", diagnostics.get("n_parcels", len(parcels_pec)))
            st.metric(
                "DEM resolution (m)",
                f"{diagnostics.get('dem_res_m', float('nan')):.2f}",
            )
        with c2:
            st.metric(
                "Neighbourhood radius (m)",
                f"{diagnostics.get('neighbourhood_radius_m', neighbourhood_radius_m):.0f}",
            )
            st.metric(
                "Radius (pixels)",
                diagnostics.get("radius_pixels", 0),
            )
        with c3:
            st.metric(
                "Stream threshold (cells)",
                diagnostics.get("stream_threshold_cells", stream_threshold),
            )
            st.metric("Rainfall (mm)", f"{diagnostics.get('rainfall_mm', rainfall_mm):.0f}")

        # Class counts
        class_counts = diagnostics.get("pec_class_counts", {})
        if class_counts:
            st.write("**Parcel counts by PEC class**")
            cc_df = (
                pd.DataFrame(
                    [
                        {"PEC class": k, "Parcels": v}
                        for k, v in class_counts.items()
                    ]
                )
                .sort_values("Parcels", ascending=False)
                .reset_index(drop=True)
            )
            st.table(cc_df)

        # ---------- Map ----------
        st.subheader("Interactive PEC Map")
        pec_map = build_pec_map(parcels_pec, rainfall_mm=rainfall_mm)
        folium_static(pec_map, width=1000, height=600)

        # ---------- Tables & download ----------
        st.subheader("Parcel-level PEC indicators (top 10 by risk)")

        # Order: Retention + High risk first
        risk_order = [
            "Low-lying Depressed (Retention Priority)",
            "Flat & Pressured (High Flood Exposure Risk)",
            "Locally High & Disconnected",
            "Moderate / Context-Dependent",
        ]
        parcels_pec["pec_rank"] = parcels_pec["pec_class"].apply(
            lambda c: risk_order.index(c) if c in risk_order else len(risk_order)
        )

        cols = [
            "grid_id",
            "pec_class",
            "prei",
            "hand_score",
            "relief",
            "slp_mean",
            "dem_min",
            "dem_max",
            "hand_min",
            "hand_mean",
        ]
        existing_cols = [c for c in cols if c in parcels_pec.columns]

        table_df = (
            parcels_pec.sort_values(["pec_rank", "hand_score"])
            [existing_cols]
            .head(10)
        )

        st.dataframe(
            table_df.style.format(
                {
                    "prei": "{:.2f}",
                    "hand_score": "{:.2f}",
                    "relief": "{:.1f}",
                    "slp_mean": "{:.2f}",
                    "dem_min": "{:.1f}",
                    "dem_max": "{:.1f}",
                    "hand_min": "{:.2f}",
                    "hand_mean": "{:.2f}",
                }
            )
        )

        # CSV download of all parcels
        export_cols = ["grid_id", "pec_class", "pec_code", "prei", "hand_score", "relief"]
        export_cols = [c for c in export_cols if c in parcels_pec.columns]
        export_df = parcels_pec[export_cols].copy()

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full PEC results (CSV)",
            data=csv_bytes,
            file_name=f"PEC_results_{int(rainfall_mm)}mm.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("PEC analysis failed. Please see the error below.")
        st.exception(e)
else:
    st.info(
        "Adjust parameters in the sidebar and click **Run PEC Analysis** "
        "to compute parcel elevation context."
    )
