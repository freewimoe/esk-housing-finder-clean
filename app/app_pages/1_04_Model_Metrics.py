from shared_imports import (
    st, pd, np, os, joblib, px, go,
    load_sample_data, format_currency, format_percentage,
    mean_absolute_error, r2_score, mean_squared_error
)

class MetricsPage:
    @staticmethod
    def render():
        st.title("Model Performance Metrics")
        st.markdown("### Evaluate and Monitor Model Performance")

        # Model loading section
        st.subheader("Model Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            model_path = st.text_input("Model Path", value="models/versioned/latest/model.joblib")
        with col2:
            if st.button("Load & Analyze Model"):
                try:
                    if os.path.exists(model_path):
                        model_data = joblib.load(model_path)
                        st.session_state['analysis_model'] = model_data
                        st.success("Model loaded for analysis!")
                    else:
                        st.error("Model file not found!")
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
        
        # Use model from session state if available
        model_data = st.session_state.get('analysis_model', None)
        
        if model_data is not None:
            st.success(" Model ready for analysis")
            
            # Model information
            st.subheader("Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", "Random Forest")
            with col2:
                if hasattr(model_data, 'n_estimators'):
                    st.metric("Number of Trees", model_data.n_estimators)
            with col3:
                if hasattr(model_data, 'max_depth'):
                    st.metric("Max Depth", model_data.max_depth)
            
            # Feature importance analysis
            if hasattr(model_data, 'feature_importances_'):
                st.subheader("Feature Importance Analysis")
                
                feature_names = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_data.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                # Interactive bar chart
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    title="Feature Importance Scores",
                    orientation='h'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top features
                top_features = importance_df.tail(3)
                st.markdown("**Most Important Features:**")
                for _, row in top_features.iterrows():
                    st.write(f" **{row['Feature']}**: {row['Importance']:.3f}")
            
            # Model validation section
            st.subheader("Model Validation")
            
            validation_data_source = st.radio(
                "Choose validation data:", 
                ["Use Sample Data", "Upload Test CSV"]
            )
            
            test_df = None
            if validation_data_source == "Use Sample Data":
                if st.button("Run Validation on Sample Data"):
                    test_df = load_sample_data()
                    st.info(f"Using {len(test_df)} sample properties for validation")
            else:
                uploaded_test_file = st.file_uploader("Upload test CSV", type=['csv'])
                if uploaded_test_file is not None:
                    test_df = pd.read_csv(uploaded_test_file)
                    st.success(f"Loaded {len(test_df)} test properties")
            
            # Perform validation
            if test_df is not None:
                try:
                    feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                    
                    if all(col in test_df.columns for col in feature_cols + ['price']):
                        X_test = test_df[feature_cols]
                        y_true = test_df['price']
                        
                        # Make predictions
                        y_pred = model_data.predict(X_test)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                        
                        # Display metrics
                        st.subheader("Validation Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{r2:.3f}")
                        with col2:
                            st.metric("MAE", format_currency(mae))
                        with col3:
                            st.metric("RMSE", format_currency(rmse))
                        with col4:
                            st.metric("MAPE", f"{mape:.1f}%")
                        
                        # Prediction vs Actual plot
                        st.subheader("Prediction vs Actual")
                        
                        plot_df = pd.DataFrame({
                            'Actual': y_true,
                            'Predicted': y_pred
                        })
                        
                        fig = px.scatter(
                            plot_df, 
                            x='Actual', 
                            y='Predicted',
                            title="Predicted vs Actual Prices",
                            labels={'Actual': 'Actual Price ($)', 'Predicted': 'Predicted Price ($)'}
                        )
                        
                        # Add perfect prediction line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines', 
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residuals analysis
                        st.subheader("Residuals Analysis")
                        residuals = y_true - y_pred
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Residuals histogram
                            fig_hist = px.histogram(
                                x=residuals, 
                                title="Distribution of Residuals",
                                labels={'x': 'Residuals ($)'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Residuals vs predicted
                            fig_resid = px.scatter(
                                x=y_pred, 
                                y=residuals,
                                title="Residuals vs Predicted",
                                labels={'x': 'Predicted Price ($)', 'y': 'Residuals ($)'}
                            )
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_resid, use_container_width=True)
                        
                        # Performance summary
                        st.subheader("Performance Summary")
                        
                        # Determine model quality
                        if r2 >= 0.8:
                            quality = " Excellent"
                        elif r2 >= 0.6:
                            quality = " Good"
                        elif r2 >= 0.4:
                            quality = " Fair"
                        else:
                            quality = " Poor"
                        
                        st.markdown(f"""
                        **Model Quality:** {quality}
                        
                        **Key Metrics:**
                        - **R² Score:** {r2:.3f} (Higher is better, max 1.0)
                        - **Mean Absolute Error:** {format_currency(mae)} (Lower is better)
                        - **Root Mean Square Error:** {format_currency(rmse)} (Lower is better)
                        - **Mean Absolute Percentage Error:** {mape:.1f}% (Lower is better)
                        
                        **Interpretation:**
                        - The model explains {r2*100:.1f}% of the price variance
                        - On average, predictions are off by {format_currency(mae)}
                        - Typical prediction error is {mape:.1f}% of the actual price
                        """)
                        
                    else:
                        st.error("Test data must contain columns: bedrooms, bathrooms, sqft, year_built, latitude, longitude, price")
                        
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
        
        else:
            st.warning(" Please load a model first to view metrics")
            st.info(" Train a model or load an existing model file above")
