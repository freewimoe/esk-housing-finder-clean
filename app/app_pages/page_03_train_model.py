import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from shared_imports import (
        train_test_split, RandomForestRegressor, 
        mean_absolute_error, r2_score, joblib,
        load_sample_data, format_currency, format_percentage,
        get_demo_model, create_model_info_display
    )
except ImportError:
    # Fallback imports
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib
    
    def load_sample_data():
        return pd.DataFrame({
            'price': [300000, 400000, 500000],
            'sqft': [1200, 1500, 2000],
            'bedrooms': [2, 3, 4]
        })
    
    def format_currency(amount):
        return f"${amount:,.2f}"
    
    def format_percentage(value):
        return f"{value:.1%}"
    
    def get_demo_model():
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def create_model_info_display():
        return "Model info display"

class TrainModelPage:
    @staticmethod
    def render():
        st.title("Model Training Center")
        st.markdown("### Train ML Models for Real Estate Price Prediction")
        
        # Demo mode info
        create_model_info_display()
        
        # Model configuration
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Depth", 5, 50, 10)
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
        
        # Data loading section
        st.subheader("Training Data")
        data_source = st.radio("Choose data source:", ["Sample Data", "Upload CSV"])
        
        df = None
        if data_source == "Sample Data":
            if st.button("Load Sample NYC Data"):
                df = load_sample_data()
                st.success(f"Loaded {len(df)} sample properties")
                st.dataframe(df.head())
                
                # Show data distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Price", format_currency(df['price'].mean()))
                    st.metric("Price Range", f"{format_currency(df['price'].min())} - {format_currency(df['price'].max())}")
                
                with col2:
                    st.metric("Average Sqft", f"{df['sqft'].mean():.0f}")
                    st.metric("Year Range", f"{df['year_built'].min()} - {df['year_built'].max()}")
        else:
            uploaded_file = st.file_uploader("Upload real estate CSV", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} properties from CSV")
                st.dataframe(df.head())
        
        # Training section
        if df is not None:
            st.subheader("Model Training")
            
            # Training options
            col1, col2 = st.columns(2)
            with col1:
                save_to_session = st.checkbox("Save to Session State", value=True, 
                                            help="Saves model in browser session for immediate use")
            with col2:
                save_to_file = st.checkbox("Save to File", value=False,
                                         help="Only works locally, not on Streamlit Share")
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare features and target
                        feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                        if all(col in df.columns for col in feature_cols + ['price']):
                            X = df[feature_cols]
                            y = df['price']
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=int(random_state)
                            )
                            
                            # Train model
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=int(random_state)
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            
                            # Display results
                            st.success(" Model trained successfully!")
                            
                            # Metrics display
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("R² Score", f"{r2:.3f}")
                            with col2:
                                st.metric("MAE", format_currency(mae))
                            with col3:
                                st.metric("RMSE", format_currency(rmse))
                            with col4:
                                st.metric("MAPE", f"{mape:.1f}%")
                            
                            # Model quality assessment
                            if r2 >= 0.8:
                                quality = " Excellent"
                            elif r2 >= 0.6:
                                quality = " Good"
                            elif r2 >= 0.4:
                                quality = " Fair"
                            else:
                                quality = " Poor"
                            
                            st.info(f"**Model Quality:** {quality}")
                            
                            # Feature importance
                            st.subheader("Feature Importance")
                            feature_importance = pd.DataFrame({
                                'feature': feature_cols,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            # Create bar chart
                            import plotly.express as px
                            fig = px.bar(
                                feature_importance, 
                                x='importance', 
                                y='feature',
                                orientation='h',
                                title="Feature Importance Rankings"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Save model options
                            if save_to_session:
                                st.session_state['trained_model'] = model
                                st.session_state['model_metrics'] = {
                                    'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape
                                }
                                st.success(" Model saved to session state! You can now use it in the Predict section.")
                            
                            if save_to_file:
                                try:
                                    os.makedirs("models/versioned/latest", exist_ok=True)
                                    model_path = "models/versioned/latest/model.joblib"
                                    joblib.dump(model, model_path)
                                    st.success(f" Model saved to {model_path} (local only)")
                                except Exception as e:
                                    st.warning(f" Could not save to file: {str(e)}. This is normal on Streamlit Share.")
                            
                            # Prediction examples
                            st.subheader("Test Predictions")
                            if len(X_test) > 0:
                                test_sample = X_test.iloc[:3]
                                test_predictions = model.predict(test_sample)
                                actual_prices = y_test.iloc[:3]
                                
                                prediction_df = pd.DataFrame({
                                    'Actual Price': actual_prices.apply(format_currency),
                                    'Predicted Price': [format_currency(p) for p in test_predictions],
                                    'Difference': [format_currency(abs(a - p)) for a, p in zip(actual_prices, test_predictions)]
                                })
                                
                                st.dataframe(prediction_df)
                        
                        else:
                            st.error("Required columns missing. Need: bedrooms, bathrooms, sqft, year_built, latitude, longitude, price")
                            
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        # Demo training option
        st.subheader("Quick Demo Training")
        st.markdown("Want to see the training process without uploading data?")
        
        if st.button("Train Demo Model with Synthetic Data"):
            with st.spinner("Training demo model with synthetic NYC data..."):
                try:
                    # Generate larger synthetic dataset
                    np.random.seed(42)
                    n_samples = 200
                    
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
                    
                    st.success(f" Generated {len(demo_data)} synthetic properties")
                    st.dataframe(demo_data.head())
                    
                    # Trigger training with demo data
                    st.session_state['demo_training_data'] = demo_data
                    st.info(" Now click 'Load Sample NYC Data' above and then 'Train Model' to see the full training process!")
                    
                except Exception as e:
                    st.error(f"Demo training failed: {str(e)}")
        
        # Model management
        if 'trained_model' in st.session_state:
            st.subheader("Current Session Model")
            metrics = st.session_state.get('model_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Session Model R²", f"{metrics.get('r2', 0):.3f}")
            with col2:
                st.metric("Session Model MAE", format_currency(metrics.get('mae', 0)))
            with col3:
                if st.button("Clear Session Model"):
                    del st.session_state['trained_model']
                    if 'model_metrics' in st.session_state:
                        del st.session_state['model_metrics']
                    st.success("Session model cleared!")
                    st.rerun()
        
        # Info section
        if df is None:
            st.info(" Load training data above to start the machine learning process!")
            st.markdown("""
            **Training Process:**
            1.  Load your real estate data (CSV with price, bedrooms, bathrooms, sqft, year_built, latitude, longitude)
            2.  Configure model parameters (trees, depth, test size)
            3.  Train the Random Forest model
            4.  View performance metrics and feature importance
            5.  Save model to session for immediate use in Predict section
            """)
