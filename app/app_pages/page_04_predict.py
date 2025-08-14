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
        joblib, load_sample_data, format_currency, get_demo_model, create_model_info_display
    )
except ImportError:
    # Fallback imports
    import joblib
    
    def load_sample_data():
        return pd.DataFrame({
            'price': [300000, 400000, 500000],
            'sqft': [1200, 1500, 2000],
            'bedrooms': [2, 3, 4]
        })
    
    def format_currency(amount):
        return f"${amount:,.2f}"
    
    def get_demo_model():
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def create_model_info_display():
        return "Model info display"

class PredictPage:
    @staticmethod
    def render():
        st.title("Property Price Predictor")
        st.markdown("### Get AI-Powered Real Estate Valuations")
        
        # Demo mode info
        create_model_info_display()

        # Model selection with demo fallback
        st.subheader("Model Selection")
        
        # Try to load file-based model first, fallback to demo model
        model_data = None
        use_demo = st.checkbox("Use Demo Model (Recommended for online version)", value=True)
        
        if use_demo:
            model_data = get_demo_model()
            if not model_data.is_trained:
                with st.spinner("Initializing demo model..."):
                    model_data._create_demo_model()
            st.success(" Demo model ready for predictions!")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                model_path = st.text_input("Model Path", value="models/versioned/latest/model.joblib")
            with col2:
                if st.button("Load Model"):
                    try:
                        if os.path.exists(model_path):
                            model_data = joblib.load(model_path)
                            st.success("Model loaded successfully!")
                            st.session_state['model'] = model_data
                        else:
                            st.error("Model file not found! Using demo model instead.")
                            model_data = get_demo_model()
                            if not model_data.is_trained:
                                model_data._create_demo_model()
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
                        st.info("Switching to demo model...")
                        model_data = get_demo_model()
                        if not model_data.is_trained:
                            model_data._create_demo_model()
        
        # Use model from session state if available
        if 'model' in st.session_state and not use_demo:
            model_data = st.session_state['model']
        
        if model_data is not None:
            if not use_demo:
                st.success(" Model ready for predictions")
            
            # Prediction input section
            st.subheader("Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
                sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1500)
                
            with col2:
                year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
                latitude = st.number_input("Latitude", value=40.7589, format="%.4f")
                longitude = st.number_input("Longitude", value=-73.9851, format="%.4f")
            
            # Location helper
            st.info(" NYC coordinates: Latitude ~40.7, Longitude ~-73.9")
            
            # Prediction button
            if st.button("Predict Price", type="primary"):
                try:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'bedrooms': [bedrooms],
                        'bathrooms': [bathrooms],
                        'sqft': [sqft],
                        'year_built': [year_built],
                        'latitude': [latitude],
                        'longitude': [longitude]
                    })
                    
                    # Make prediction
                    if use_demo:
                        prediction = model_data.predict(input_data)[0]
                    else:
                        prediction = model_data.predict(input_data)[0]
                    
                    # Display results
                    st.success(" Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Price", format_currency(prediction))
                    with col2:
                        price_per_sqft = prediction / sqft
                        st.metric("Price per Sq Ft", format_currency(price_per_sqft))
                    with col3:
                        # Simple confidence indicator based on typical ranges
                        if 200000 <= prediction <= 1000000:
                            confidence = "High"
                            confidence_color = ""
                        elif 100000 <= prediction <= 2000000:
                            confidence = "Medium"
                            confidence_color = ""
                        else:
                            confidence = "Low"
                            confidence_color = ""
                        st.metric("Confidence", f"{confidence_color} {confidence}")
                    
                    # Property summary
                    st.subheader("Property Summary")
                    st.markdown(f"""
                    **Property Details:**
                    -  {bedrooms} bed, {bathrooms} bath
                    -  {sqft:,} square feet
                    -  Built in {year_built}
                    -  Located at ({latitude:.4f}, {longitude:.4f})
                    
                    **Valuation:**
                    -  Estimated Value: **{format_currency(prediction)}**
                    -  Price per Sq Ft: **{format_currency(price_per_sqft)}**
                    """)
                    
                    if use_demo:
                        st.info(" This prediction is generated by a demo model trained on synthetic NYC real estate data.")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
            
            # Quick prediction examples
            st.subheader("Quick Examples")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(" Small Apartment"):
                    st.session_state.update({
                        'ex_bedrooms': 1, 'ex_bathrooms': 1, 'ex_sqft': 800,
                        'ex_year': 2010, 'ex_lat': 40.7505, 'ex_lng': -73.9934
                    })
            
            with col2:
                if st.button(" Family Home"):
                    st.session_state.update({
                        'ex_bedrooms': 3, 'ex_bathrooms': 2, 'ex_sqft': 1800,
                        'ex_year': 2015, 'ex_lat': 40.7589, 'ex_lng': -73.9851
                    })
            
            with col3:
                if st.button(" Luxury Condo"):
                    st.session_state.update({
                        'ex_bedrooms': 4, 'ex_bathrooms': 3, 'ex_sqft': 2500,
                        'ex_year': 2020, 'ex_lat': 40.7831, 'ex_lng': -73.9712
                    })
            
            # Display example prediction if triggered
            if any(key.startswith('ex_') for key in st.session_state.keys()):
                if 'ex_bedrooms' in st.session_state:
                    example_data = pd.DataFrame({
                        'bedrooms': [st.session_state['ex_bedrooms']],
                        'bathrooms': [st.session_state['ex_bathrooms']],
                        'sqft': [st.session_state['ex_sqft']],
                        'year_built': [st.session_state['ex_year']],
                        'latitude': [st.session_state['ex_lat']],
                        'longitude': [st.session_state['ex_lng']]
                    })
                    
                    example_pred = model_data.predict(example_data)[0]
                    st.info(f" Example prediction: **{format_currency(example_pred)}**")
                    
                    # Clear example state
                    for key in list(st.session_state.keys()):
                        if key.startswith('ex_'):
                            del st.session_state[key]
            
            # Batch prediction section (simplified for demo)
            st.subheader("Batch Predictions")
            uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    required_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                    
                    if all(col in batch_df.columns for col in required_cols):
                        predictions = model_data.predict(batch_df)
                        batch_df['predicted_price'] = predictions
                        batch_df['predicted_price_formatted'] = batch_df['predicted_price'].apply(format_currency)
                        
                        st.success(f" Predicted prices for {len(batch_df)} properties")
                        st.dataframe(batch_df)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="property_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Missing required columns: {required_cols}")
                        
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
        else:
            st.warning(" Please select a model option above to make predictions")
