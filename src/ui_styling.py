"""Professional theme and CSS styling for light and dark modes."""

import streamlit as st

LIGHT_MODE_CSS = """
<style>
    /* Root color variables */
    :root {
        --primary-color: #4A90E2;
        --primary-dark: #357ABD;
        --secondary-color: #F5F7FA;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --border-color: #E5E7EB;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --bg-color: #FFFFFF;
        --surface-color: #F9FAFB;
    }

    /* Global body styling */
    body {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }

    /* Main app container */
    .stApp {
        background-color: var(--bg-color);
    }

    /* Streamlit elements */
    .stMetric {
        background: linear-gradient(135deg, #F5F7FA 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.1);
        border-color: var(--primary-color);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.25);
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        background-color: var(--surface-color);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: var(--surface-color);
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary-color) !important;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.95rem;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    }

    /* Info, Success, Warning, Error boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
        padding: 1rem;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar */
    .stSidebar {
        background-color: var(--surface-color);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    }

    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: var(--surface-color);
        border-radius: 6px;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--surface-color);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .stMetric {
            padding: 1rem;
        }
        
        body {
            font-size: 0.95rem;
        }
    }
</style>
"""

DARK_MODE_CSS = """
<style>
    /* Root color variables - Dark Mode */
    :root {
        --primary-color: #60A5FA;
        --primary-dark: #3B82F6;
        --secondary-color: #1F2937;
        --text-primary: #F3F4F6;
        --text-secondary: #D1D5DB;
        --border-color: #374151;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --bg-color: #0F172A;
        --surface-color: #1E293B;
    }

    /* Global body styling */
    body {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }

    /* Main app container */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
    }

    /* Text elements - fix dark mode text visibility */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text-primary) !important;
    }

    .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }

    /* Streamlit elements */
    .stMetric {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .stMetric * {
        color: var(--text-primary) !important;
    }

    .stMetric:hover {
        box-shadow: 0 4px 16px rgba(96, 165, 250, 0.15);
        border-color: var(--primary-color);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.3);
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(96, 165, 250, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        background-color: var(--surface-color);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: var(--surface-color);
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        color: var(--primary-color) !important;
        box-shadow: 0 2px 8px rgba(96, 165, 250, 0.2);
        border: 1px solid var(--primary-color);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: var(--surface-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.95rem;
        color: var(--text-primary) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-secondary) !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2) !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    }

    /* Info, Success, Warning, Error boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
        padding: 1rem;
        background-color: var(--surface-color);
    }

    .stAlert * {
        color: var(--text-primary) !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        background-color: var(--surface-color);
    }

    /* Sidebar */
    .stSidebar {
        background-color: var(--surface-color);
    }

    .stSidebar * {
        color: var(--text-primary) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    }

    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: var(--surface-color);
        border-radius: 6px;
        color: var(--text-primary);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--surface-color);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .stMetric {
            padding: 1rem;
        }
        
        body {
            font-size: 0.95rem;
            color: var(--text-primary);
        }
    }
</style>
"""

def apply_theme_css(theme: str):
    """Apply the appropriate theme CSS based on theme selection."""
    if theme == "dark":
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

def get_gradient_background(color1: str, color2: str, angle: int = 135) -> str:
    """Generate a gradient background CSS string.
    
    Args:
        color1: Starting color (hex)
        color2: Ending color (hex)
        angle: Gradient angle (default 135deg)
    
    Returns:
        CSS gradient string
    """
    return f"linear-gradient({angle}deg, {color1} 0%, {color2} 100%)"

def get_theme_colors(theme: str = "light") -> dict:
    """Get color palette for the selected theme."""
    if theme == "dark":
        return {
            "primary": "#60A5FA",
            "primary_dark": "#3B82F6",
            "secondary": "#1F2937",
            "text_primary": "#F3F4F6",
            "text_secondary": "#D1D5DB",
            "border": "#374151",
            "success": "#10B981",
            "warning": "#F59E0B",
            "danger": "#EF4444",
            "background": "#0F172A",
            "surface": "#1E293B"
        }
    else:
        return {
            "primary": "#4A90E2",
            "primary_dark": "#357ABD",
            "secondary": "#F5F7FA",
            "text_primary": "#1F2937",
            "text_secondary": "#6B7280",
            "border": "#E5E7EB",
            "success": "#10B981",
            "warning": "#F59E0B",
            "danger": "#EF4444",
            "background": "#FFFFFF",
            "surface": "#F9FAFB"
        }
