# app.py
# Main Streamlit entrypoint – Hydrological Analysis Platform
# Uses the built-in multipage system (pages/ folder) and page_link for navigation.
# Currently only the FCI model page (pages/2_FCI_Analysis.py) is implemented.

import streamlit as st

st.set_page_config(
    page_title="Hydrological Analysis Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# Hero / header
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #cbd5f5;
        max-width: 900px;
    }
    .model-card {
        border-radius: 1.1rem;
        padding: 1.2rem 1.4rem;
        background-color: #111827;
        border: 1px solid #1f2937;
    }
    .model-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }
    .model-acronym {
        font-weight: 700;
        font-size: 1.2rem;
    }
    .model-desc {
        font-size: 0.90rem;
        color: #d1d5db;
    }
    .small-tag {
        font-size: 0.7rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>Hydrological Analysis Platform</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Run individual models or perform comprehensive analysis to understand "
    "flow corridors, storage roles, parcel sensitivity and exposure across your study area.</div>",
    unsafe_allow_html=True,
)
st.write("")  # small spacer

# -------------------------------------------------------------------
# Model cards configuration
# -------------------------------------------------------------------
models_info = [
    {
        "id": "hsr",
        "acronym": "HSR",
        "name": "Hydrological Storage Role",
        "emoji": "💧",
        "description": "Analyzes water storage capacity and retention patterns across the landscape.",
        "implemented": False,            # placeholder only
        "page": None,
    },
    {
        "id": "fci",
        "acronym": "FCI",
        "name": "Flow Corridor Importance",
        "emoji": "📈",
        "description": "Evaluates the significance of water flow pathways and corridor connectivity.",
        "implemented": True,             # THIS ONE IS LIVE
        "page": "pages/2_FCI_Analysis.py",  # must match your real file name
    },
    {
        "id": "pec",
        "acronym": "PEC",
        "name": "Parcel Elevation Context",
        "emoji": "⛰️",
        "description": "Assesses elevation-based risk factors and topographical context for each parcel.",
        "implemented": False,
        "page": None,
    },
    {
        "id": "uds",
        "acronym": "UDS",
        "name": "Upstream–Downstream Sensitivity",
        "emoji": "🔁",
        "description": "Measures interdependence between upstream and downstream areas.",
        "implemented": False,
        "page": None,
    },
    {
        "id": "sei",
        "acronym": "SEI",
        "name": "Surrounding Exposure Index",
        "emoji": "📍",
        "description": "Quantifies exposure levels based on surrounding environmental factors.",
        "implemented": True,
        "page": "pages/5_SEI_Analysis.py",
    },
]

st.subheader("Analysis Models")

cols = st.columns(3, gap="large")

for i, model in enumerate(models_info):
    col = cols[i % 3]
    with col:
        st.markdown("<div class='model-card'>", unsafe_allow_html=True)

        # header row
        c1, c2 = st.columns([0.25, 0.75])
        with c1:
            st.markdown(
                f"<div style='font-size: 1.8rem; text-align:center;'>{model['emoji']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='model-acronym' style='text-align:center;'>{model['acronym']}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='model-title'>{model['name']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='model-desc'>{model['description']}</div>",
                unsafe_allow_html=True,
            )

        st.write("")

        if model["implemented"] and model["page"] is not None:
            # Use page_link instead of switch_page – avoids the navigation error
            st.page_link(
                model["page"],
                label=f"Run {model['acronym']} Analysis →",
                icon="🚀",
                use_container_width=True,
            )
            st.markdown(
                "<div class='small-tag'>Available</div>",
                unsafe_allow_html=True,
            )
        else:
            st.button(
                f"{model['acronym']} – Coming soon",
                disabled=True,
                use_container_width=True,
            )
            st.markdown(
                "<div class='small-tag'>Not implemented yet</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.write("---")

# -------------------------------------------------------------------
# Comprehensive analysis section (placeholder for now)
# -------------------------------------------------------------------
st.subheader("Comprehensive Analysis (all models)")
st.info(
    "This section will eventually run **HSR, FCI, PEC, UDS, and SEI** together and "
    "produce an integrated parcel-level hydrological risk summary.\n\n"
    "For now, only the **Flow Corridor Importance (FCI)** model is available "
    "via the card above or the FCI page in the left sidebar."
)

st.button("Run All Models (coming soon)", disabled=True, use_container_width=True)

st.write("")
st.caption("Tip: You can also open the FCI page from the left sidebar navigation.")

