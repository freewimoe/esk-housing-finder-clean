"""
Machine learning models for real estate price prediction
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import logging

from config import RANDOM_FOREST_PARAMS, TEST_SIZE, CV_FOLDS, NUMERIC_FEATURES
from preprocess import RealEstatePreprocessor

logger = logging.getLogger(__name__)

class RealEstatePricePredictor:
    """Real estate price prediction model"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.preprocessor = RealEstatePreprocessor()
        self.feature_names = None
        self.is_trained = False
        self.training_metrics = {}
        
    def _get_model(self) -> object:
        """Initialize the ML model based on type"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_columns: List[str] = None) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        if feature_columns is None:
            # Use numeric features that exist in the dataframe
            feature_columns = [col for col in NUMERIC_FEATURES if col in df.columns]
        
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        self.feature_names = feature_columns
        logger.info(f"Prepared features: {len(feature_columns)} columns")
        
        return X
    
    def train(self, df: pd.DataFrame, 
              target_column: str = 'price',
              feature_columns: List[str] = None,
              test_size: float = TEST_SIZE) -> Dict:
        """
        Train the real estate price prediction model
        
        Args:
            df: Training dataframe
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Fraction of data for testing
            
        Returns:
            dict: Training metrics
        """
        logger.info("Starting model training...")
        
        # Prepare features and target
        X = self.prepare_features(df, feature_columns)
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize and train model
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=CV_FOLDS, scoring='r2')
        
        self.training_metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, 
                                         self.model.feature_importances_)),
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }
        
        self.is_trained = True
        
        logger.info(f"Model training completed. Test R²: {test_metrics['r2']:.3f}")
        return self.training_metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make price predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df, self.feature_names)
        predictions = self.model.predict(X)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_single(self, features: Dict) -> float:
        """Make single prediction from feature dictionary"""
        # Convert to dataframe
        df = pd.DataFrame([features])
        prediction = self.predict(df)
        return float(prediction[0])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as dataframe"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from: {filepath}")

def load_trained_model(filepath: str) -> RealEstatePricePredictor:
    """Load a trained model from file"""
    predictor = RealEstatePricePredictor()
    predictor.load_model(filepath)
    return predictor
