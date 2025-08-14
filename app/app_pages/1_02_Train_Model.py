from shared_imports import (
    st, pd, np, os, 
    train_test_split, RandomForestRegressor, 
    mean_absolute_error, r2_score, joblib,
    load_sample_data, format_currency, format_percentage
)

class TrainModelPage:
    @staticmethod
    def render():
        st.title("Model Training Center")
        st.markdown("### Train ML Models for Real Estate Price Prediction")
        
        # Model selection
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
        else:
            uploaded_file = st.file_uploader("Upload real estate CSV", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} properties from CSV")
                st.dataframe(df.head())
        
        # Training section
        if df is not None and st.button("Train Model", type="primary"):
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
                        
                        # Display results
                        st.success("Model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{r2:.3f}")
                        with col2:
                            st.metric("Mean Absolute Error", format_currency(mae))
                        with col3:
                            st.metric("Accuracy", format_percentage(r2))
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        st.bar_chart(feature_importance.set_index('feature'))
                        
                        # Save model
                        if st.button("Save Model"):
                            os.makedirs("models/versioned/latest", exist_ok=True)
                            model_path = "models/versioned/latest/model.joblib"
                            joblib.dump(model, model_path)
                            st.success(f"Model saved to {model_path}")
                    
                    else:
                        st.error("Required columns missing. Need: bedrooms, bathrooms, sqft, year_built, latitude, longitude, price")
                        
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
        
        # Model info
        if df is None:
            st.info(" Load data above to start training")
