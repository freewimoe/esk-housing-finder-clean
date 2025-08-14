"""
Smoke tests for model inference pipeline
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loaders import RealEstateDataLoader
from model import RealEstatePricePredictor
from preprocess import RealEstatePreprocessor

class TestInferenceSmoke(unittest.TestCase):
    """Basic smoke tests to ensure the inference pipeline works"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'price': [500000, 300000, 700000],
            'bedrooms': [3, 2, 4],
            'bathrooms': [2, 1, 3],
            'sqft': [1500, 1000, 2000],
            'year_built': [2000, 1995, 2010],
            'latitude': [40.7589, 40.7505, 40.7831],
            'longitude': [-73.9851, -73.9934, -73.9712]
        })
    
    def test_data_loader_initialization(self):
        """Test that data loader can be initialized"""
        loader = RealEstateDataLoader()
        self.assertIsInstance(loader, RealEstateDataLoader)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        loader = RealEstateDataLoader()
        is_valid, missing = loader.validate_data(self.sample_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)
    
    def test_data_summary(self):
        """Test data summary generation"""
        loader = RealEstateDataLoader()
        summary = loader.get_data_summary(self.sample_data)
        
        self.assertIn('n_records', summary)
        self.assertIn('n_features', summary)
        self.assertIn('price_stats', summary)
        self.assertEqual(summary['n_records'], 3)
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor can be initialized"""
        preprocessor = RealEstatePreprocessor()
        self.assertIsInstance(preprocessor, RealEstatePreprocessor)
    
    def test_data_cleaning(self):
        """Test basic data cleaning"""
        preprocessor = RealEstatePreprocessor()
        cleaned_data = preprocessor.clean_data(self.sample_data)
        
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(len(cleaned_data), len(self.sample_data))
    
    def test_feature_creation(self):
        """Test feature engineering"""
        preprocessor = RealEstatePreprocessor()
        featured_data = preprocessor.create_features(self.sample_data)
        
        self.assertIn('price_per_sqft', featured_data.columns)
        self.assertIn('property_age', featured_data.columns)
        self.assertGreater(len(featured_data.columns), len(self.sample_data.columns))
    
    def test_model_initialization(self):
        """Test that model can be initialized"""
        model = RealEstatePricePredictor()
        self.assertIsInstance(model, RealEstatePricePredictor)
        self.assertFalse(model.is_trained)
    
    def test_model_training(self):
        """Test basic model training"""
        model = RealEstatePricePredictor()
        
        # Use more data for training
        larger_data = pd.concat([self.sample_data] * 10, ignore_index=True)
        larger_data['price'] += np.random.normal(0, 10000, len(larger_data))
        
        metrics = model.train(larger_data)
        
        self.assertTrue(model.is_trained)
        self.assertIn('test_metrics', metrics)
        self.assertIn('r2', metrics['test_metrics'])
    
    def test_model_prediction(self):
        """Test model prediction"""
        model = RealEstatePricePredictor()
        
        # Train with larger dataset
        larger_data = pd.concat([self.sample_data] * 10, ignore_index=True)
        larger_data['price'] += np.random.normal(0, 10000, len(larger_data))
        
        model.train(larger_data)
        
        # Make predictions
        predictions = model.predict(self.sample_data)
        
        self.assertEqual(len(predictions), len(self.sample_data))
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_single_prediction(self):
        """Test single property prediction"""
        model = RealEstatePricePredictor()
        
        # Train model
        larger_data = pd.concat([self.sample_data] * 10, ignore_index=True)
        larger_data['price'] += np.random.normal(0, 10000, len(larger_data))
        model.train(larger_data)
        
        # Single prediction
        features = {
            'bedrooms': 3,
            'bathrooms': 2,
            'sqft': 1500,
            'year_built': 2000,
            'latitude': 40.7589,
            'longitude': -73.9851
        }
        
        prediction = model.predict_single(features)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInferenceSmoke)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
