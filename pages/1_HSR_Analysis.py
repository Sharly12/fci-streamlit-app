# pages/1_HSR_Analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_loader import get_data_paths, load_base_data
from models.hsr_model import run_hsr_analysis


st.title("💧 Hydrological Storage Role (HSR) Analysis")

st.write(
    "This module estimates **topographic storage capacity** using concavities in the DEM.\n\n"
    "- **HSR_static**: purely topographic storage volume (m³)\n"
    "- **HSR_rain**: rainfall-limited storage (m³) based on SCS–CN runoff\n\n"
    "Higher HSR values indicate areas that **store more water** and therefore "
    "are more valuable to protect."
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("HSR Parameters")

    rainfall_mm = st.slider(
        "Design Rainfall (mm)",
        min_value=0.0,
        max_value=250.0,
        value=100.0,
        step=5.0,
    )

    concavity_window = st.slider(
        "Concavity window (cells)",
        min_value=3,
        max_value=15,
        value=7,
        step=2,
        help="Morphological window size (NxN) for detecting concavities.",
    )

run_btn = st.button("Run HSR Analysis")

# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------
if run_btn:
    # Shared data (DEM, CN, parcels)
    DEM_PATH, PARCELS_PATH, CN_PATH = get_data_paths()
    base_data = load_base_data(DEM_PATH, PARCELS_PATH, CN_PATH)

    with st.spinner("Running HSR engine ..."):
        (
            parcels_hsr,
            diagnostics,
            HSR_static,
            HSR_rain_map,
        ) = run_hsr_analysis(
            rainfall_mm=rainfall_mm,
            base_data=base_data,
            concavity_window=concavity_window,
        )

    st.success("✅ HSR analysis complete")

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------
    st.subheader("Diagnostics & Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rainfall (mm)", f"{diagnostics['rainfall_mm']:.1f}")
        st.metric("Cell size (m)", f"{diagnostics['cell_size_m']:.1f}")
    with c2:
        st.metric("Concavities detected", diagnostics["n_concavities"])
        st.metric("Concavity threshold (m)", f"{diagnostics['concavity_threshold_m']:.3f}")
    with c3:
        st.metric("Total static storage (m³)", f"{diagnostics['HSR_static_total_m3']:.0f}")
        st.metric("Total rain-limited storage (m³)", f"{diagnostics['HSR_rain_total_m3']:.0f}")

    # -----------------------------------------------------
    # Static visualisation (HSR_static & HSR_rain)
    # -----------------------------------------------------
    st.subheader("HSR Maps")

    def _plot_hsr_maps(hsr_static, hsr_rain):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1 – HSR_static (log scale for visibility)
        ax = axes[0]
        data = np.where(np.isfinite(hsr_static), hsr_static, np.nan)
        v = data[data > 0]
        if v.size > 0:
            img = ax.imshow(
                np.log10(v.min() + (data - v.min()) + 1e-6),
                cmap="Blues",
            )
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="log10(HSR_static + c)")
        else:
            img = ax.imshow(data, cmap="Blues")
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("HSR_static (m³ – concavity storage)")
        ax.axis("off")

        # Panel 2 – HSR_rain (log scale)
        ax = axes[1]
        data_r = np.where(np.isfinite(hsr_rain), hsr_rain, np.nan)
        vr = data_r[data_r > 0]
        if vr.size > 0:
            img2 = ax.imshow(
                np.log10(vr.min() + (data_r - vr.min()) + 1e-6),
                cmap="Greens",
            )
            fig.colorbar(img2, ax=ax, fraction=0.046, pad=0.04, label="log10(HSR_rain + c)")
        else:
            img2 = ax.imshow(data_r, cmap="Greens")
            fig.colorbar(img2, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"HSR_rain (m³ – {rainfall_mm:.0f} mm event)")
        ax.axis("off")

        # Panel 3 – 10-class HSR_rain (pixel level)
        ax = axes[2]
        flat = data_r.flatten()
        valid = flat[np.isfinite(flat) & (flat > 0)]
        if valid.size > 0:
            # Deciles on pixel values
            qs = np.quantile(valid, np.linspace(0, 1, 11))
            class_raster = np.zeros_like(data_r, dtype=int)
            for i in range(10):
                mask = (data_r >= qs[i]) & (data_r <= qs[i + 1])
                class_raster[mask] = i + 1
            img3 = ax.imshow(class_raster, cmap="viridis")
            fig.colorbar(img3, ax=ax, fraction=0.046, pad=0.04, label="HSR_rain decile (1–10)")
        else:
            img3 = ax.imshow(np.zeros_like(data_r), cmap="viridis")
            fig.colorbar(img3, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("HSR_rain 10-class (cell level)")
        ax.axis("off")

        fig.tight_layout()
        return fig

    fig = _plot_hsr_maps(HSR_static, HSR_rain_map)
    st.pyplot(fig, use_container_width=True)

    # -----------------------------------------------------
    # Parcel-level 10-class HSR_rain
    # -----------------------------------------------------
    st.subheader("Parcel-level HSR_rain (10-class deciles)")

    # Frequency of each class (1 = lowest storage, 10 = highest)
    class_counts = (
        parcels_hsr["HSR_rain_class_10"]
        .value_counts()
        .sort_index()
        .rename_axis("Class")
        .reset_index(name="Parcels")
    )
    class_counts["Percent"] = class_counts["Parcels"] / len(parcels_hsr) * 100.0
    st.table(class_counts.style.format({"Percent": "{:.1f}"}))

    # Top parcels table
    st.subheader("Top 10 parcels by HSR_rain storage")

    cols_show = [
        "HSR_rain_sum",
        "HSR_rain_mean",
        "HSR_rain_max",
        "HSR_rain_norm",
        "HSR_rain_class_10",
    ]
    existing_cols = [c for c in cols_show if c in parcels_hsr.columns]

    st.dataframe(
        parcels_hsr[existing_cols]
        .sort_values("HSR_rain_sum", ascending=False)
        .head(10)
        .round(3)
    )

    # Download CSV of parcel results
    csv_df = parcels_hsr.drop(columns="geometry").copy()
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download all parcel-level HSR results (CSV)",
        data=csv_bytes,
        file_name=f"HSR_parcels_{int(rainfall_mm)}mm.csv",
        mime="text/csv",
    )

else:
    st.info("Set rainfall and concavity window, then click **Run HSR Analysis**.")

