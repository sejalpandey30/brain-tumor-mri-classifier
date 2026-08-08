"""UI Components for consistent styling across the app."""

import streamlit as st
from typing import Optional, Dict, Any

class UIComponents:
    """Reusable UI component library."""
    
    def __init__(self, theme: str = "light"):
        self.theme = theme
        self.colors = self._get_colors()
    
    def _get_colors(self) -> Dict[str, str]:
        """Get color scheme based on theme."""
        if self.theme == "dark":
            return {
                "primary": "#00d4ff",
                "success": "#10b981",
                "danger": "#ef4444",
                "warning": "#f59e0b",
                "info": "#3b82f6",
                "background": "#0e1117",
                "surface": "#161b22",
                "text": "#e6e6e6"
            }
        else:
            return {
                "primary": "#1f77b4",
                "success": "#10b981",
                "danger": "#ef4444",
                "warning": "#f59e0b",
                "info": "#3b82f6",
                "background": "#ffffff",
                "surface": "#f0f2f6",
                "text": "#1f1f1f"
            }
    
    def metric_card(self, title: str, value: str, subtitle: Optional[str] = None):
        """Create a styled metric card."""
        st.metric(title, value, delta=subtitle)
    
    def section_header(self, text: str, level: int = 3):
        """Create a styled section header."""
        st.markdown(f"{'#' * level} {text}")
    
    def info_box(self, text: str, icon: str = "ℹ️"):
        """Create an info box."""
        st.info(f"{icon} {text}")
    
    def success_box(self, text: str, icon: str = "✅"):
        """Create a success box."""
        st.success(f"{icon} {text}")
    
    def warning_box(self, text: str, icon: str = "⚠️"):
        """Create a warning box."""
        st.warning(f"{icon} {text}")
    
    def error_box(self, text: str, icon: str = "❌"):
        """Create an error box."""
        st.error(f"{icon} {text}")
    
    def button_group(self, buttons: Dict[str, str], columns: int = 3):
        """Create a group of buttons."""
        cols = st.columns(columns)
        results = {}
        for idx, (label, key) in enumerate(buttons.items()):
            with cols[idx % columns]:
                results[key] = st.button(label, key=key, use_container_width=True)
        return results
    
    def stat_row(self, stats: Dict[str, str]):
        """Create a row of statistics."""
        cols = st.columns(len(stats))
        for col, (label, value) in zip(cols, stats.items()):
            with col:
                st.metric(label, value)
