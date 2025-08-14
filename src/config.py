"""
Configuration settings for the Smart Real Estate Predictor
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

MODELS_DIR = PROJECT_ROOT / "models"
VERSIONED_MODELS_DIR = MODELS_DIR / "versioned"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data files
SAMPLE_DATA_FILE = RAW_DATA_DIR / "sample_real_estate.csv"

# Model settings
DEFAULT_MODEL_VERSION = "v1"
MODEL_FILE_NAME = "latest.joblib"

# Feature columns
REQUIRED_COLUMNS = [
    'price',  # Target variable
    'bedrooms',
    'bathrooms', 
    'sqft',
    'latitude',
    'longitude'
]

NUMERIC_FEATURES = [
    'bedrooms',
    'bathrooms',
    'sqft', 
    'lot_size',
    'year_built',
    'latitude',
    'longitude',
    'school_rating',
    'crime_score',
    'walkability_score',
    'distance_to_transit',
    'distance_to_shopping',
    'distance_to_school',
    'median_income',
    'unemployment_rate',
    'days_on_market'
]

CATEGORICAL_FEATURES = [
    'property_type',
    'neighborhood',
    'zipcode'
]

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Validation settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_FOLDS = 5

# API settings
API_TIMEOUT = 30
MAX_RETRIES = 3

# Streamlit settings
PAGE_CONFIG = {
    'page_title': 'Smart Real Estate Predictor',
    'page_icon': '🏠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
