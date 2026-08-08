"""Theme management for light and dark modes."""

import streamlit as st

THEME_LIGHT = {
    "primary_color": "#1f77b4",
    "background_color": "#ffffff",
    "secondary_background_color": "#f0f2f6",
    "text_color": "#1f1f1f",
    "font": "sans serif"
}

THEME_DARK = {
    "primary_color": "#00d4ff",
    "background_color": "#0e1117",
    "secondary_background_color": "#161b22",
    "text_color": "#e6e6e6",
    "font": "sans serif"
}

LIGHT_CSS = """
<style>
    body {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #155fa2;
    }
</style>
"""

DARK_CSS = """
<style>
    body {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 8px;
        color: #e6e6e6;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #00b8d4;
    }
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        color: #e6e6e6;
    }
</style>
"""

def apply_theme(theme_name: str):
    """Apply theme styling to the app."""
    if theme_name == "dark":
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

def get_theme_config(theme_name: str) -> dict:
    """Get theme configuration dictionary."""
    return THEME_DARK if theme_name == "dark" else THEME_LIGHT
