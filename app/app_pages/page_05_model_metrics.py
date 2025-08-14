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
        joblib, px, go, load_sample_data, format_currency, format_percentage,
        mean_absolute_error, r2_score, mean_squared_error,
        get_demo_model, create_model_info_display
    )
except ImportError:
    # Fallback imports
    import joblib
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    
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
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def create_model_info_display():
        return "Model info display"

class MetricsPage:
    @staticmethod
    def render():
        st.title("Model Performance Metrics")
        st.markdown("### Evaluate and Monitor Model Performance")
        
        # Demo mode info
        create_model_info_display()

        # Model selection
        st.subheader("Model Analysis")
        
        model_option = st.radio(
            "Select model to analyze:",
            ["Demo Model", "Session Model", "File Model"]
        )
        
        model_data = None
        model_info = {}
        
        if model_option == "Demo Model":
            model_data = get_demo_model()
            if not model_data.is_trained:
                with st.spinner("Initializing demo model..."):
                    model_data._create_demo_model()
            model_info = {
                'type': 'Demo Model',
                'source': 'Pre-trained on synthetic NYC data',
                'status': ' Ready'
            }
            
        elif model_option == "Session Model":
            if 'trained_model' in st.session_state:
                model_data = st.session_state['trained_model']
                metrics = st.session_state.get('model_metrics', {})
                model_info = {
                    'type': 'Session Model',
                    'source': 'Trained in current session',
                    'status': ' Available',
                    'metrics': metrics
                }
            else:
                st.warning(" No model found in session. Please train a model first in the 'Train Model' section.")
                
        else:  # File Model
            col1, col2 = st.columns([2, 1])
            with col1:
                model_path = st.text_input("Model Path", value="models/versioned/latest/model.joblib")
            with col2:
                if st.button("Load & Analyze Model"):
                    try:
                        if os.path.exists(model_path):
                            model_data = joblib.load(model_path)
                            model_info = {
                                'type': 'File Model',
                                'source': model_path,
                                'status': ' Loaded'
                            }
                            st.success("Model loaded for analysis!")
                        else:
                            st.error("Model file not found!")
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
        
        if model_data is not None:
            # Display model information
            st.subheader("Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_info.get('type', 'Unknown'))
            with col2:
                if model_option == "Demo Model" or hasattr(model_data, 'model'):
                    actual_model = model_data.model if hasattr(model_data, 'model') else model_data
                    if hasattr(actual_model, 'n_estimators'):
                        st.metric("Number of Trees", actual_model.n_estimators)
                    else:
                        st.metric("Trees", "N/A")
                else:
                    if hasattr(model_data, 'n_estimators'):
                        st.metric("Number of Trees", model_data.n_estimators)
            with col3:
                status = model_info.get('status', 'Unknown')
                st.metric("Status", status)
            
            # Feature importance analysis
            st.subheader("Feature Importance Analysis")
            
            try:
                # Get feature importance from model
                if model_option == "Demo Model":
                    feature_importance = model_data.model.feature_importances_
                else:
                    feature_importance = model_data.feature_importances_
                
                feature_names = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'latitude', 'longitude']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                # Interactive bar chart
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    title="Feature Importance Scores",
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top features
                top_features = importance_df.tail(3)
                st.markdown("**Most Important Features:**")
                for _, row in top_features.iterrows():
                    st.write(f" **{row['Feature']}**: {row['Importance']:.3f}")
                    
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
            
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
                        if model_option == "Demo Model":
                            y_pred = model_data.predict(X_test)
                        else:
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
                            labels={'Actual': 'Actual Price ($)', 'Predicted': 'Predicted Price ($)'},
                            color_discrete_sequence=['#1f77b4']
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
                                labels={'x': 'Residuals ($)'},
                                nbins=20
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
                        
                        # Model comparison (if session metrics available)
                        if model_option != "Session Model" and 'model_metrics' in st.session_state:
                            st.subheader("Model Comparison")
                            session_metrics = st.session_state['model_metrics']
                            
                            comparison_df = pd.DataFrame({
                                'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE'],
                                'Current Model': [r2, mae, rmse, mape],
                                'Session Model': [
                                    session_metrics.get('r2', 0),
                                    session_metrics.get('mae', 0),
                                    session_metrics.get('rmse', 0),
                                    session_metrics.get('mape', 0)
                                ]
                            })
                            
                            st.dataframe(comparison_df)
                        
                    else:
                        st.error("Test data must contain columns: bedrooms, bathrooms, sqft, year_built, latitude, longitude, price")
                        
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    
            # Quick insights
            if model_option == "Demo Model":
                st.subheader("Demo Model Insights")
                st.info("""
                 **Demo Model Characteristics:**
                - Trained on 1000 synthetic NYC properties
                - Features realistic price patterns based on size, location, and age
                - Optimized for demonstration purposes
                - Performance may vary on real-world data
                """)
        
        else:
            st.warning(" Please select and load a model to view metrics")
            st.info(" Options: Use demo model, train a model, or load an existing model file")
