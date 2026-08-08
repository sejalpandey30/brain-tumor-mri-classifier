"""Utility functions for the application."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import json

def format_confidence(confidence: float, decimals: int = 2) -> str:
    """Format confidence score."""
    return f"{confidence:.{decimals}f}%"

def get_confidence_level(confidence: float) -> str:
    """Get confidence level description."""
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Medium"
    elif confidence >= 40:
        return "Low"
    else:
        return "Very Low"

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 90:
        return "🟢"
    elif confidence >= 75:
        return "🟡"
    elif confidence >= 60:
        return "🟠"
    else:
        return "🔴"

def calculate_prediction_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from predictions."""
    if not predictions:
        return {}
    
    confidences = [p['confidence'] for p in predictions]
    classes = [p['prediction'] for p in predictions]
    
    return {
        'total': len(predictions),
        'avg_confidence': np.mean(confidences),
        'max_confidence': np.max(confidences),
        'min_confidence': np.min(confidences),
        'std_confidence': np.std(confidences),
        'most_common_class': max(set(classes), key=classes.count),
        'class_distribution': {cls: classes.count(cls) for cls in set(classes)}
    }

def save_prediction_history(history: List[Dict], filepath: str):
    """Save prediction history to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

def load_prediction_history(filepath: str) -> List[Dict]:
    """Load prediction history from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def format_timestamp(dt: datetime) -> str:
    """Format datetime to readable string."""
    return dt.strftime('%Y-%m-%d %H:%M:%S')
