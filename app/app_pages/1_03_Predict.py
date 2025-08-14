from shared_imports import (
    st, pd, np, os, joblib,
    load_sample_data, format_currency
)

class PredictPage:
    @staticmethod
    def render():
        st.title("Property Price Predictor")
        st.markdown("### Get AI-Powered Real Estate Valuations")

        # Load model section
        st.subheader("Model Selection")
        model_data = None
        
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
                        st.error("Model file not found!")
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
        
        # Use model from session state if available
        if 'model' in st.session_state:
            model_data = st.session_state['model']
        
        if model_data is not None:
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
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
            
            # Batch prediction section
            st.subheader("Batch Predictions")
            uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    required_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                    
                    if all(col in batch_df.columns for col in required_cols):
                        predictions = model_data.predict(batch_df[required_cols])
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
            st.warning(" Please load a model first to make predictions")
            st.info(" Train a model in the 'Train Model' section or load an existing model file")
