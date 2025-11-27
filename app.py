import streamlit as st

# ---------- GLOBAL CONFIG ----------
st.set_page_config(
    page_title="Hydrological Analysis Platform",
    page_icon="💧",
    layout="wide",
)

# ---------- STYLES ----------
st.markdown(
    """
    <style>
    .model-card {
        border-radius: 18px;
        padding: 18px 20px;
        background-color: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 15px -3px rgba(15,23,42,0.06);
        transition: all 0.2s ease-in-out;
    }
    .model-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 20px 25px -5px rgba(59,130,246,0.25);
        transform: translateY(-2px);
    }
    .metric-card {
        border-radius: 14px;
        padding: 16px;
        background-color: white;
        box-shadow: 0 8px 12px -4px rgba(15,23,42,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- MODEL METADATA ----------
# IMPORTANT: page paths MUST match actual filenames in your `pages/` folder
models_info = [
    {
        "id": "hsr",
        "title": "Hydrological Storage Role",
        "acronym": "HSR",
        "emoji": "💧",
        "description": "Analyzes water storage capacity and retention patterns across the landscape.",
        "page": "pages/1_HSR_Analysis.py",
    },
    {
        "id": "fci",
        "title": "Flow Corridor Importance",
        "acronym": "FCI",
        "emoji": "📈",
        "description": "Evaluates the significance of water flow pathways and corridor connectivity.",
        "page": "pages/2_FCI_Analysis.py",
    },
    {
        "id": "pec",
        "title": "Parcel Elevation Context",
        "acronym": "PEC",
        "emoji": "⛰️",
        "description": "Assesses elevation-based risk factors and topographical context for each parcel.",
        "page": "pages/3_PEC_Analysis.py",
    },
    {
        "id": "uds",
        "title": "Upstream–Downstream Sensitivity",
        "acronym": "UDS",
        "emoji": "🔁",
        "description": "Measures the interdependence between upstream and downstream areas.",
        "page": "pages/4_UDS_Analysis.py",
    },
    {
        "id": "sei",
        "title": "Surrounding Exposure Index",
        "acronym": "SEI",
        "emoji": "📍",
        "description": "Quantifies exposure levels based on surrounding environmental factors.",
        "page": "pages/5_SEI_Analysis.py",
    },
]

# ---------- HEADER ----------
with st.container():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.markdown("### 💧")
    with col_title:
        st.title("Hydrological Analysis Platform")
        st.write(
            "Run individual models or perform comprehensive analysis to understand flow corridors, "
            "storage roles, parcel sensitivity and exposure across your study area."
        )

st.markdown("---")

# ---------- MODEL GRID ----------
st.subheader("Analysis Models")

cols = st.columns(3)
for i, model in enumerate(models_info):
    col = cols[i % 3]
    with col:
        st.markdown(
            f"""
            <div class="model-card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <div style="font-size:28px;">{model['emoji']}</div>
                    <div>
                        <div style="font-weight:700;font-size:18px;">{model['acronym']}</div>
                        <div style="font-size:14px;color:#6b7280;">{model['title']}</div>
                    </div>
                </div>
                <div style="font-size:13px;color:#4b5563;margin-bottom:12px;">
                    {model['description']}
                </div>
            """,
            unsafe_allow_html=True,
        )

        btn_label = f"Run {model['acronym']} Analysis"
        if st.button(btn_label, key=f"btn_{model['id']}"):
            # Try to switch page programmatically
            if hasattr(st, "switch_page"):
                try:
                    st.switch_page(model["page"])
                except Exception as e:
                    # Friendly fallback: don't crash the whole app
                    st.error(
                        "Navigation failed. "
                        "Please open the page from the left sidebar.\n\n"
                        f"Details: {e}"
                    )
            else:
                st.info(
                    "This Streamlit version does not support `st.switch_page`.\n\n"
                    "Please use the left sidebar to select the page instead."
                )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- COMPREHENSIVE ANALYSIS ----------
st.subheader("Comprehensive Analysis (All Models)")

st.write(
    "Run all five models together to generate an integrated hydrological assessment. "
    "This button will later call HSR, FCI, PEC, UDS, and SEI in sequence once all "
    "models are implemented."
)

if st.button("🚀 Run All Models (coming soon)"):
    st.info(
        "Comprehensive analysis is not yet implemented. Once HSR/PEC/UDS/SEI models "
        "are coded in `models/`, this will run them all and merge outputs."
    )

# ---------- QUICK INFO ----------
st.markdown("### Quick Stats")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="metric-card">
          <div style="font-size:28px;font-weight:700;color:#2563eb;">5</div>
          <div style="color:#4b5563;">Analysis Models</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="metric-card">
          <div style="font-size:24px;font-weight:700;color:#16a34a;">Scalable</div>
          <div style="color:#4b5563;">Streamlit-based processing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="metric-card">
          <div style="font-size:24px;font-weight:700;color:#7c3aed;">Modular</div>
          <div style="color:#4b5563;">Plug-in model architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("© 2024 Hydrological Analysis Platform · Built with Streamlit")
