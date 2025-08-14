import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

class MetricsPage:
    @staticmethod
    def render():
        st.title("Model Performance Metrics")
        st.markdown("### Evaluate and Monitor Model Performance")
        
        # Load model
        model_data = None
        # Get the correct path from app directory
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(app_dir)
        model_path = os.path.join(project_root, "models", "versioned", "v1", "latest.joblib")
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                st.success("Model metrics loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                model_data = None
        elif 'trained_model' in st.session_state:
            model_data = st.session_state['trained_model']
            st.info("Using model from current session")
        else:
            st.warning("⚠️ No trained model found")
            st.markdown("""
            **To view model metrics, you need to train a model first:**
            
            1. 📊 Go to the **Train Model** page
            2. 🔄 Load sample data or upload your CSV file  
            3. 🧠 Select features and train a model
            4. 🔙 Return here to view performance metrics
            
            Once you've trained a model, this page will show detailed performance analysis.
            """)
            
            if st.button("🧠 Go to Train Model Page", type="primary"):
                st.info("Please use the sidebar navigation to go to the 'Train Model' page.")
            return
        
        if model_data:
            # Performance Overview
            st.subheader("Performance Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "R² Score",
                    f"{model_data['r2']:.3f}",
                    help="Coefficient of determination - how well the model explains variance"
                )
            
            with col2:
                st.metric(
                    "Mean Absolute Error",
                    f"${model_data['mae']:,.0f}",
                    help="Average prediction error in dollars"
                )
            
            with col3:
                accuracy_pct = model_data['r2'] * 100
                st.metric(
                    "Model Accuracy",
                    f"{accuracy_pct:.1f}%",
                    help="Overall prediction accuracy"
                )
            
            # Performance Interpretation
            st.subheader("Performance Interpretation")
            
            r2_score = model_data['r2']
            
            if r2_score >= 0.8:
                st.success("🎯 **Excellent Performance** - Model explains 80%+ of price variance")
            elif r2_score >= 0.6:
                st.info("✅ **Good Performance** - Model provides reliable predictions")
            elif r2_score >= 0.4:
                st.warning("⚠️ **Fair Performance** - Model has moderate predictive power")
            else:
                st.error("❌ **Poor Performance** - Model needs improvement")
            
            # Feature Importance
            st.subheader("Feature Importance")
            
            if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': model_data['features'],
                    'Importance': model_data['model'].feature_importances_
                }).sort_values('Importance', ascending=True)
                
                st.bar_chart(importance_df.set_index('Feature'), use_container_width=True)
                
                # Top features
                top_features = importance_df.tail(3)
                st.markdown("**Most Important Features:**")
                for _, row in top_features.iterrows():
                    st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
            
            # Model Details
            st.subheader("Model Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Type**")
                model_type = type(model_data['model']).__name__
                st.write(model_type)
                
                st.markdown("**Features Used**")
                for feature in model_data['features']:
                    st.write(f"• {feature}")
            
            with col2:
                st.markdown("**Training Statistics**")
                st.write(f"Number of features: {len(model_data['features'])}")
                
                if hasattr(model_data['model'], 'n_estimators'):
                    st.write(f"Number of estimators: {model_data['model'].n_estimators}")
                
                if hasattr(model_data['model'], 'max_depth'):
                    max_depth = model_data['model'].max_depth
                    st.write(f"Max depth: {max_depth if max_depth else 'Unlimited'}")
            
            # Performance Guidelines
            st.subheader("Performance Guidelines")
            
            guidelines = {
                "R² Score": {
                    "Excellent (0.8+)": "Model explains most price variance",
                    "Good (0.6-0.8)": "Model provides reliable predictions",
                    "Fair (0.4-0.6)": "Model has moderate accuracy",
                    "Poor (<0.4)": "Model needs improvement"
                },
                "MAE (Mean Absolute Error)": {
                    "Low (<5% of avg price)": "Very accurate predictions",
                    "Moderate (5-15% of avg price)": "Acceptable accuracy",
                    "High (>15% of avg price)": "Consider model improvements"
                }
            }
            
            for metric, ranges in guidelines.items():
                with st.expander(f"Understanding {metric}"):
                    for range_desc, meaning in ranges.items():
                        st.write(f"**{range_desc}**: {meaning}")
            
            # Improvement Suggestions
            st.subheader("Model Improvement Suggestions")
            
            suggestions = []
            
            if model_data['r2'] < 0.6:
                suggestions.append("🔄 Try different algorithms (XGBoost, Linear Regression)")
                suggestions.append("📊 Add more features (location, amenities, market data)")
                suggestions.append("🧹 Improve data quality (handle outliers, missing values)")
            
            if model_data['mae'] > 50000:  # If MAE is high
                suggestions.append("🎯 Focus on feature engineering")
                suggestions.append("📈 Collect more training data")
                suggestions.append("⚙️ Tune hyperparameters")
            
            if len(model_data['features']) < 5:
                suggestions.append("➕ Add more relevant features")
                suggestions.append("🏠 Include property characteristics")
                suggestions.append("📍 Add location-based features")
            
            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)
            else:
                st.success("🎉 Your model is performing well! Consider minor optimizations.")
            
            # Export Model Info
            st.subheader("Export Model Information")
            
            if st.button("Download Model Report"):
                report_data = {
                    'Model Type': type(model_data['model']).__name__,
                    'R² Score': model_data['r2'],
                    'Mean Absolute Error': model_data['mae'],
                    'Features': ', '.join(model_data['features']),
                    'Feature Count': len(model_data['features'])
                }
                
                report_df = pd.DataFrame([report_data])
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name="model_performance_report.csv",
                    mime="text/csv"
                )
