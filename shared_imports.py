"""
Shared imports and utilities for Streamlit app pages
Implements DRY principle by centralizing common imports
ENHANCED: Demo-ready ML functionality for Streamlit Share
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

# Demo Model Class for Streamlit Share compatibility
class DemoRealEstateModel:
    """Demo model that works without persistent file storage"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
        
    def train(self, data):
        """Train model with provided data"""
        try:
            X = data[self.feature_names]
            y = data['price']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            self.is_trained = True
            return metrics
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return None
    
    def predict(self, data):
        """Make predictions"""
        if not self.is_trained or self.model is None:
            # Create a simple demo model for immediate use
            self._create_demo_model()
        
        return self.model.predict(data[self.feature_names])
    
    def _create_demo_model(self):
        """Create a pre-trained demo model"""
        # Generate realistic training data
        np.random.seed(42)
        n_samples = 1000
        
        demo_data = pd.DataFrame({
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'sqft': np.random.randint(800, 3000, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples),
            'latitude': np.random.normal(40.7589, 0.1, n_samples),
            'longitude': np.random.normal(-73.9851, 0.1, n_samples)
        })
        
        # Create realistic price based on features
        demo_data['price'] = (
            demo_data['bedrooms'] * 50000 +
            demo_data['bathrooms'] * 30000 +
            demo_data['sqft'] * 200 +
            (2023 - demo_data['year_built']) * -1000 +
            np.random.normal(100000, 50000, n_samples)
        ).clip(100000, 2000000)
        
        self.train(demo_data)

# Global demo model instance
@st.cache_resource
def get_demo_model():
    """Get cached demo model instance"""
    return DemoRealEstateModel()

# Common Streamlit configurations
def setup_page_config(title="Smart Real Estate Predictor", layout="wide"):
    """Configure common page settings"""
    st.set_page_config(
        page_title=title,
        layout=layout,
        initial_sidebar_state="expanded"
    )

# Enhanced data loading function
def load_sample_data():
    """Load or create sample data for demonstration"""
    sample_data = pd.DataFrame({
        'price': [500000, 300000, 700000, 450000, 600000, 820000, 380000, 650000],
        'bedrooms': [3, 2, 4, 3, 3, 4, 2, 3],
        'bathrooms': [2, 1, 3, 2, 2, 3, 1, 2],
        'sqft': [1500, 1000, 2000, 1300, 1700, 2200, 950, 1600],
        'year_built': [2000, 1995, 2010, 2005, 2015, 2018, 1992, 2008],
        'latitude': [40.7589, 40.7505, 40.7831, 40.7614, 40.7648, 40.7580, 40.7520, 40.7600],
        'longitude': [-73.9851, -73.9934, -73.9712, -73.9776, -73.9808, -73.9840, -73.9920, -73.9780]
    })
    return sample_data

# Enhanced utility functions
def format_currency(value):
    """Format value as currency"""
    return f""

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1%}"

def get_model_metrics_summary(model, test_data=None):
    """Get comprehensive model metrics"""
    if test_data is None:
        test_data = load_sample_data()
    
    try:
        predictions = model.predict(test_data)
        actual = test_data['price']
        
        r2 = r2_score(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'quality': 'Excellent' if r2 >= 0.8 else 'Good' if r2 >= 0.6 else 'Fair' if r2 >= 0.4 else 'Poor'
        }
    except:
        return None

def create_model_info_display():
    """Create model information display"""
    st.info(" **Demo Mode Active**: Using pre-trained model for demonstration. Train your own model for custom predictions!")

# Export all commonly used items
__all__ = [
    'st', 'pd', 'np', 'os', 'sys', 'Path',
    'px', 'go', 'make_subplots',
    'train_test_split', 'RandomForestRegressor', 
    'mean_absolute_error', 'r2_score', 'mean_squared_error', 'joblib',
    'DemoRealEstateModel', 'get_demo_model',
    'setup_page_config', 'load_sample_data',
    'format_currency', 'format_percentage', 'get_model_metrics_summary',
    'create_model_info_display'
]
