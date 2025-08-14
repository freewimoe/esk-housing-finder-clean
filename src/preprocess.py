"""
Data preprocessing utilities for real estate datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)

class RealEstatePreprocessor:
    """Preprocessor for real estate data"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        df_clean = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle negative prices
        if 'price' in df_clean.columns:
            negative_prices = df_clean['price'] < 0
            if negative_prices.any():
                logger.warning(f"Found {negative_prices.sum()} negative prices, removing...")
                df_clean = df_clean[~negative_prices]
        
        # Handle zero square feet
        if 'sqft' in df_clean.columns:
            zero_sqft = df_clean['sqft'] <= 0
            if zero_sqft.any():
                logger.warning(f"Found {zero_sqft.sum()} zero/negative square feet")
                df_clean.loc[zero_sqft, 'sqft'] = df_clean['sqft'].median()
        
        # Handle zero bedrooms/bathrooms
        for col in ['bedrooms', 'bathrooms']:
            if col in df_clean.columns:
                zero_values = df_clean[col] < 0
                if zero_values.any():
                    logger.warning(f"Found {zero_values.sum()} negative {col}")
                    df_clean.loc[zero_values, col] = df_clean[col].median()
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Handle missing values with different strategies per column"""
        df_filled = df.copy()
        
        if strategy is None:
            strategy = {
                'numeric': 'median',
                'categorical': 'mode'
            }
        
        # Handle numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                if strategy.get('numeric') == 'median':
                    fill_value = df_filled[col].median()
                elif strategy.get('numeric') == 'mean':
                    fill_value = df_filled[col].mean()
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
                logger.info(f"Filled {col} missing values with {strategy.get('numeric')}: {fill_value}")
        
        # Handle categorical columns
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_filled[col].isnull().any():
                if strategy.get('categorical') == 'mode':
                    fill_value = df_filled[col].mode().iloc[0] if len(df_filled[col].mode()) > 0 else 'Unknown'
                else:
                    fill_value = 'Unknown'
                
                df_filled[col] = df_filled[col].fillna(fill_value)
                logger.info(f"Filled {col} missing values with: {fill_value}")
        
        return df_filled
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers in numeric columns"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_info[col] = {
                        'count': outliers.sum(),
                        'percentage': (outliers.sum() / len(df)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        logger.info(f"Outlier detection completed for {len(columns)} columns")
        return pd.DataFrame(outlier_info).T
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        df_clean = df.copy()
        original_length = len(df_clean)
        
        if columns is None:
            columns = ['price', 'sqft']  # Focus on key columns
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Remove outliers
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & 
                        (df_clean[col] <= upper_bound)
                    ]
        
        removed_count = original_length - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier rows ({removed_count/original_length*100:.1f}%)")
        
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data"""
        df_features = df.copy()
        
        # Price per square foot
        if 'price' in df_features.columns and 'sqft' in df_features.columns:
            df_features['price_per_sqft'] = df_features['price'] / df_features['sqft']
            df_features['price_per_sqft'] = df_features['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
        
        # Property age
        if 'year_built' in df_features.columns:
            current_year = pd.Timestamp.now().year
            df_features['property_age'] = current_year - df_features['year_built']
        
        # Bathrooms per bedroom ratio
        if 'bathrooms' in df_features.columns and 'bedrooms' in df_features.columns:
            df_features['bathroom_bedroom_ratio'] = df_features['bathrooms'] / (df_features['bedrooms'] + 1)
        
        # Total rooms
        if 'bedrooms' in df_features.columns and 'bathrooms' in df_features.columns:
            df_features['total_rooms'] = df_features['bedrooms'] + df_features['bathrooms']
        
        # Lot size per sqft ratio
        if 'lot_size' in df_features.columns and 'sqft' in df_features.columns:
            df_features['lot_sqft_ratio'] = df_features['lot_size'] / df_features['sqft']
            df_features['lot_sqft_ratio'] = df_features['lot_sqft_ratio'].replace([np.inf, -np.inf], np.nan)
        
        logger.info("Feature engineering completed")
        return df_features
    
    def get_preprocessing_summary(self, df_original: pd.DataFrame, 
                                df_processed: pd.DataFrame) -> Dict:
        """Get summary of preprocessing operations"""
        summary = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'rows_removed': df_original.shape[0] - df_processed.shape[0],
            'features_added': df_processed.shape[1] - df_original.shape[1],
            'missing_values_original': df_original.isnull().sum().sum(),
            'missing_values_processed': df_processed.isnull().sum().sum(),
        }
        
        return summary

def preprocess_real_estate_data(df: pd.DataFrame, 
                              clean_data: bool = True,
                              handle_missing: bool = True,
                              remove_outliers: bool = False,
                              create_features: bool = True) -> pd.DataFrame:
    """
    Convenience function for full preprocessing pipeline
    
    Args:
        df: Input dataframe
        clean_data: Whether to perform basic cleaning
        handle_missing: Whether to handle missing values
        remove_outliers: Whether to remove outliers
        create_features: Whether to create additional features
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    preprocessor = RealEstatePreprocessor()
    df_processed = df.copy()
    
    if clean_data:
        df_processed = preprocessor.clean_data(df_processed)
    
    if handle_missing:
        df_processed = preprocessor.handle_missing_values(df_processed)
    
    if remove_outliers:
        df_processed = preprocessor.remove_outliers(df_processed)
    
    if create_features:
        df_processed = preprocessor.create_features(df_processed)
    
    logger.info("Full preprocessing pipeline completed")
    return df_processed
