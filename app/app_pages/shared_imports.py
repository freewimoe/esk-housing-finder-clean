"""
Shared imports and utilities for Streamlit app pages
Implements DRY principle by centralizing common imports
"""

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Common Streamlit configurations
def setup_page_config(title="Smart Real Estate Predictor", layout="wide"):
    """Configure common page settings"""
    st.set_page_config(
        page_title=title,
        layout=layout,
        initial_sidebar_state="expanded"
    )

# Common data loading function
def load_sample_data():
    """Load or create sample data for demonstration"""
    sample_data = pd.DataFrame({
        'price': [500000, 300000, 700000, 450000, 600000],
        'bedrooms': [3, 2, 4, 3, 3],
        'bathrooms': [2, 1, 3, 2, 2],
        'sqft': [1500, 1000, 2000, 1300, 1700],
        'year_built': [2000, 1995, 2010, 2005, 2015],
        'latitude': [40.7589, 40.7505, 40.7831, 40.7614, 40.7648],
        'longitude': [-73.9851, -73.9934, -73.9712, -73.9776, -73.9808]
    })
    return sample_data

# Common utility functions
def format_currency(value):
    """Format value as currency"""
    return f"${value:,.0f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1%}"
