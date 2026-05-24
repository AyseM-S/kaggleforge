"""
KaggleForge — AI-Powered Kaggle Competition Assistant
Main Streamlit application entry point.
No API keys needed. Runs entirely on your local machine.
"""

import streamlit as st
import os

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Page Configuration ──
st.set_page_config(
    page_title="KaggleForge",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.8rem;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.06);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Initialize Session State ──
defaults = {
    "competition_rules": "",
    "train_file": None,
    "test_file": None,
    "sample_sub_file": None,
    "extra_files": [],
    "selected_model": None,
    "pipeline_results": None,
    "chat_messages": [],
    "ai_brain": None,
    "data_summary_text": "",
    "quality_mode": "standard",
    "n_folds": 5,
    "metric": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Render Sidebar ──
from ui.sidebar import render_sidebar

render_sidebar()

# ── Main Content Area ──
tab_solution, tab_chat = st.tabs(["🚀 Solution Generator", "💬 Chat & Discuss"])

with tab_solution:
    from ui.solution_view import render_solution_view
    render_solution_view()

with tab_chat:
    from ui.chat_view import render_chat_view
    render_chat_view()
