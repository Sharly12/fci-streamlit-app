# pages/2_FCI_Analysis.py
import streamlit as st
import pandas as pd
from utils.data_loader import get_data_paths, load_base_data
from models.fci_model import run_fci_analysis
from utils.visualization import build_fci_map
from streamlit_folium import folium_static

st.title("📈 Flow Corridor Importance (FCI) Analysis")

st.write(
    "This page runs the full Flow Corridor Importance model: "
    "NRCS runoff → D8 flow accumulation → flow corridors → parcel-level FCI."
)

# --- User controls ---
st.sidebar.header("FCI Parameters")
rainfall_mm = st.sidebar.slider(
    "Design Rainfall (mm)", min_value=0.0, max_value=250.0, value=100.0, step=5.0
)
use_nrcs = st.sidebar.checkbox(
    "Use NRCS Runoff Method (recommended)", value=True
)

# --- Load shared DEM/CN/parcels ---
dem_path, parcels_path, cn_path = get_data_paths()
base_data = load_base_data(dem_path, parcels_path, cn_path)

if st.button("Run FCI Analysis"):
    with st.spinner("Running FCI model ..."):
        (
            parcels_result,
            diagnostics,
            corridor_threshold,
            corridor_cells,
            risk_counts,
            flow_accumulation,
            corridor_mask,
        ) = run_fci_analysis(rainfall_mm, use_nrcs, base_data)

    st.success("✅ FCI analysis complete")

    # Diagnostics
    st.subheader("Hydrologic Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Flow direction scheme: **{diagnostics['scheme']}**")
        st.write(f"- Total input runoff: **{diagnostics['total_input']:.2f} mm**")
        st.write(f"- Total accumulated: **{diagnostics['total_accumulated']:.2f} mm**")
        st.write(f"- Outlet accumulation: **{diagnostics['outlet_accumulation']:.2f} mm**")
        st.write(f"- Mass balance error: **{diagnostics['mass_balance_error_pct']:.3f}%**")
    with c2:
        st.write(f"- Number of outlets: **{diagnostics['num_outlets']}**")
        st.write(f"- Unprocessed cells: **{diagnostics['num_unprocessed']}**")
        st.write(f"- Processing iterations: **{diagnostics['iterations']}**")
        st.write(f"- Corridor threshold (top 10%): **{corridor_threshold:.2f}**")
        st.write(f"- Corridor cells: **{corridor_cells:,}**")

    # Risk distribution
    st.subheader("Parcel Risk Distribution")
    risk_df = pd.DataFrame(
        [
            {"Risk": lbl, "Parcels": cnt, "Percent": cnt / len(parcels_result) * 100.0}
            for lbl, cnt in risk_counts.items()
        ]
    ).set_index("Risk")
    st.table(risk_df.style.format({"Percent": "{:.1f}"}))

    # Map
    st.subheader("Interactive Map – Parcels + Flow Accumulation + Corridors")
    fci_map = build_fci_map(
        parcels_result,
        flow_accumulation,
        corridor_mask,
        base_data,
        rainfall_mm,
    )
    folium_static(fci_map, width=1000, height=600)

    # Top parcels + CSV
    st.subheader("Top 10 High-FCI Parcels")
    cols = [
        "FCI",
        "FCI_class_10",
        "Risk",
        "fci_sum",
        "fci_corr_sum",
        "fci_p90",
        "Rainfall_mm",
    ]
    table_df = parcels_result.sort_values("FCI", ascending=False)[cols]
    st.dataframe(
        table_df.head(10).style.format(
            {"FCI": "{:.3f}", "fci_sum": "{:.0f}",
             "fci_corr_sum": "{:.0f}", "fci_p90": "{:.0f}"}
        )
    )

    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full parcel results (CSV)",
        data=csv_bytes,
        file_name=f"FCI_results_{int(rainfall_mm)}mm.csv",
        mime="text/csv",
    )
else:
    st.info("Set rainfall and click **Run FCI Analysis** to start.")

