"""
Smart Real Estate Predictor - Source Code Module

This module contains the core functionality for the Smart Real Estate Predictor,
including data loading, preprocessing, machine learning models, and utilities.
"""

__version__ = "1.0.0"
__author__ = "Smart Real Estate Predictor Team"

# Import main classes for easy access
try:
    from .data_loaders import RealEstateDataLoader, load_real_estate_data
    from .model import RealEstatePricePredictor, load_trained_model
    from .preprocess import RealEstatePreprocessor, preprocess_real_estate_data
    from .config import *
except ImportError:
    # Handle import errors gracefully during development
    pass

__all__ = [
    'RealEstateDataLoader',
    'RealEstatePricePredictor', 
    'RealEstatePreprocessor',
    'load_real_estate_data',
    'load_trained_model',
    'preprocess_real_estate_data'
]
