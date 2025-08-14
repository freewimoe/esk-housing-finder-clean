import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class TrainModelPage:
    @staticmethod
    def render():
        st.title("Model Training Center")
        st.markdown("### Train ML Models for Real Estate Price Prediction")
        
        # Load data
        st.subheader("1. Load Training Data")
        
        data_loaded = False
        df = None
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Sample Data"):
                # Get the correct path from app directory
                app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                project_root = os.path.dirname(app_dir)
                data_path = os.path.join(project_root, "data", "raw", "sample_real_estate.csv")
                
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    st.success(f"Sample data loaded: {df.shape[0]} properties")
                    data_loaded = True
                else:
                    st.error(f"Sample data not found at: {data_path}")
        
        with col2:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success(f"Data uploaded: {df.shape[0]} properties")
                data_loaded = True
        
        if data_loaded and df is not None:
            # Data preview
            st.subheader("2. Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Feature selection
            st.subheader("3. Feature Selection")
            
            if 'price' in df.columns:
                # Identify numeric columns for features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'price' in numeric_cols:
                    numeric_cols.remove('price')
                
                if len(numeric_cols) > 0:
                    selected_features = st.multiselect(
                        "Select features for training:",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if selected_features:
                        st.subheader("4. Model Configuration")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                        with col2:
                            random_state = st.number_input("Random State", value=42)
                        
                        # Train button
                        if st.button("🚀 Train Model", type="primary"):
                            with st.spinner("🧠 Training model... This may take a moment..."):
                                try:
                                    # Prepare data
                                    X = df[selected_features].fillna(0)
                                    y = df['price']
                                    
                                    st.info(f"📊 Training with {len(X)} samples and {len(selected_features)} features")
                                    
                                    # Split data
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=test_size, random_state=random_state
                                    )
                                    
                                    # Train model
                                    model = RandomForestRegressor(
                                        n_estimators=100,
                                        max_depth=10,
                                        min_samples_leaf=2,
                                        max_features='sqrt',
                                        random_state=random_state
                                    )
                                    model.fit(X_train, y_train)
                                    
                                    # Make predictions
                                    y_pred = model.predict(X_test)
                                    
                                    # Calculate metrics
                                    mae = mean_absolute_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    # Display results
                                    st.success("🎉 Model trained successfully!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Mean Absolute Error", f"${mae:,.0f}", 
                                                help="Average prediction error in dollars")
                                    with col2:
                                        st.metric("R² Score", f"{r2:.3f}", 
                                                help="How well model explains price variance (0-1)")
                                    with col3:
                                        st.metric("Training Samples", len(X_train),
                                                help="Number of properties used for training")
                                    
                                    # Feature importance
                                    st.subheader("Feature Importance")
                                    feature_importance = pd.DataFrame({
                                        'feature': selected_features,
                                        'importance': model.feature_importances_
                                    }).sort_values('importance', ascending=False)
                                    
                                    st.bar_chart(feature_importance.set_index('feature'))
                                    
                                    # Save model
                                    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                    project_root = os.path.dirname(app_dir)
                                    model_dir = os.path.join(project_root, "models", "versioned", "v1")
                                    os.makedirs(model_dir, exist_ok=True)
                                    model_path = os.path.join(model_dir, "latest.joblib")
                                    
                                    model_data = {
                                        'model': model,
                                        'features': selected_features,
                                        'mae': mae,
                                        'r2': r2
                                    }
                                    
                                    joblib.dump(model_data, model_path)
                                    st.success(f"💾 Model saved to {model_path}")
                                    
                                    # Store in session state
                                    st.session_state['trained_model'] = model_data
                                    
                                    st.balloons()  # Celebration animation
                                    
                                    # Show next steps
                                    st.info("""
                                    🎯 **Next Steps:**
                                    - 📈 Go to **Predict** page to make price predictions
                                    - 📊 Check **Model Metrics** for detailed performance analysis
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Error training model: {str(e)}")
                    else:
                        st.warning("Please select at least one feature for training.")
                else:
                    st.error("No numeric features found for training.")
            else:
                st.error("No 'price' column found in the data.")
        else:
            st.info("Please load data to start training.")
            
            # Show training tips
            st.subheader("Training Tips")
            st.markdown("""
            **For best results:**
            1. Ensure your data has a 'price' column as the target
            2. Include relevant numeric features (bedrooms, bathrooms, sqft, etc.)
            3. Remove outliers and handle missing values
            4. Consider feature engineering (price per sqft, age, etc.)
            5. Use sufficient training data (recommended: 100+ samples)
            """)
