import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

class PredictPage:
    @staticmethod
    def render():
        st.title("Property Price Predictor")
        st.markdown("### Get AI-Powered Real Estate Valuations")
        
        # Load model
        model_data = None
        # Get the correct path from app directory
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(app_dir)
        model_path = os.path.join(project_root, "models", "versioned", "v1", "latest.joblib")
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                st.success("Model loaded successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model R² Score", f"{model_data['r2']:.3f}")
                with col2:
                    st.metric("Mean Absolute Error", f"${model_data['mae']:,.0f}")
                    
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                model_data = None
        elif 'trained_model' in st.session_state:
            model_data = st.session_state['trained_model']
            st.info("Using model from current session")
        else:
            st.warning("⚠️ No trained model found")
            st.markdown("""
            **To make predictions, you need to train a model first:**
            
            1. 📊 Go to the **Train Model** page
            2. 🔄 Load sample data or upload your CSV file  
            3. 🧠 Select features and train a model
            4. 🔙 Return here to make predictions
            
            Once you've trained a model, this page will show prediction inputs.
            """)
            
            if st.button("🧠 Go to Train Model Page", type="primary"):
                st.info("Please use the sidebar navigation to go to the 'Train Model' page.")
            return
        
        if model_data:
            st.subheader("Property Details")
            
            # Create input form based on model features
            features = model_data['features']
            input_data = {}
            
            # Common real estate features with sensible defaults
            feature_defaults = {
                'bedrooms': 2,
                'bathrooms': 1,
                'sqft': 1000,
                'year_built': 2000,
                'latitude': 40.7128,
                'longitude': -74.0060,
                'days_on_market': 30
            }
            
            # Create input widgets
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(features):
                with col1 if i % 2 == 0 else col2:
                    if feature in ['bedrooms', 'bathrooms']:
                        input_data[feature] = st.number_input(
                            f"{feature.title()}",
                            min_value=0,
                            value=feature_defaults.get(feature, 1),
                            step=1
                        )
                    elif feature == 'sqft':
                        input_data[feature] = st.number_input(
                            "Square Feet",
                            min_value=100,
                            value=feature_defaults.get(feature, 1000),
                            step=50
                        )
                    elif feature == 'year_built':
                        input_data[feature] = st.number_input(
                            "Year Built",
                            min_value=1800,
                            max_value=2024,
                            value=feature_defaults.get(feature, 2000),
                            step=1
                        )
                    elif feature in ['latitude', 'longitude']:
                        input_data[feature] = st.number_input(
                            f"{feature.title()}",
                            value=feature_defaults.get(feature, 0.0),
                            format="%.6f"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            value=feature_defaults.get(feature, 0),
                            step=1
                        )
            
            # Prediction button
            if st.button("Predict Price", type="primary"):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    prediction = model_data['model'].predict(input_df)[0]
                    
                    # Display result
                    st.success("Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Price", f"${prediction:,.0f}")
                    with col2:
                        # Confidence interval (rough estimate)
                        mae = model_data['mae']
                        st.metric("Price Range", f"${prediction-mae:,.0f} - ${prediction+mae:,.0f}")
                    with col3:
                        st.metric("Confidence", f"{model_data['r2']*100:.1f}%")
                    
                    # Show input summary
                    st.subheader("Property Summary")
                    summary_df = pd.DataFrame([input_data]).T
                    summary_df.columns = ['Value']
                    st.dataframe(summary_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            
            # Market comparison
            st.subheader("Market Context")
            st.markdown("""
            **How to interpret your prediction:**
            - The predicted price is based on similar properties in the training data
            - Price range shows the typical prediction uncertainty
            - Consider local market conditions and recent sales
            - Use this as a starting point for property valuation
            """)
            
            # Quick presets
            st.subheader("Quick Presets")
            
            col1, col2, col3 = st.columns(3)
            
            presets = {
                "Studio Apartment": {"bedrooms": 0, "bathrooms": 1, "sqft": 500},
                "Family Home": {"bedrooms": 3, "bathrooms": 2, "sqft": 1500},
                "Luxury Condo": {"bedrooms": 2, "bathrooms": 2, "sqft": 1200}
            }
            
            for i, (preset_name, preset_values) in enumerate(presets.items()):
                with [col1, col2, col3][i]:
                    if st.button(preset_name):
                        for key, value in preset_values.items():
                            if key in features:
                                st.session_state[f"preset_{key}"] = value
                        st.rerun()
