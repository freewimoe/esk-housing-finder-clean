"""
Data loading utilities for the Smart Real Estate Predictor
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_FILE, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

class RealEstateDataLoader:
    """Data loader for real estate datasets"""
    
    def __init__(self):
        self.data_path = RAW_DATA_DIR
        self.processed_path = PROCESSED_DATA_DIR
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load the sample NYC real estate dataset"""
        try:
            df = pd.read_csv(SAMPLE_DATA_FILE)
            logger.info(f"Loaded sample data: {df.shape[0]} records")
            return df
        except FileNotFoundError:
            logger.error(f"Sample data file not found: {SAMPLE_DATA_FILE}")
            raise
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            raise
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file with basic validation"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV data: {df.shape[0]} records, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, 
                     required_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """Validate that dataframe has required columns"""
        if required_columns is None:
            required_columns = REQUIRED_COLUMNS
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False, missing_columns
        
        logger.info("Data validation passed")
        return True, []
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get basic summary statistics of the dataset"""
        summary = {
            'n_records': len(df),
            'n_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        if 'price' in df.columns:
            summary['price_stats'] = {
                'mean': df['price'].mean(),
                'median': df['price'].median(),
                'min': df['price'].min(),
                'max': df['price'].max(),
                'std': df['price'].std()
            }
        
        logger.info(f"Generated data summary: {summary['n_records']} records")
        return summary
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save processed dataframe to processed data directory"""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_path / filename
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

def load_real_estate_data(source: str = "sample") -> pd.DataFrame:
    """
    Convenience function to load real estate data
    
    Args:
        source: 'sample' for sample data, or path to CSV file
        
    Returns:
        pd.DataFrame: Loaded real estate data
    """
    loader = RealEstateDataLoader()
    
    if source == "sample":
        return loader.load_sample_data()
    else:
        return loader.load_csv(source)

def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Categorize columns by data type
    
    Args:
        df: Input dataframe
        
    Returns:
        dict: Dictionary with 'numeric' and 'categorical' feature lists
    """
    return {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
