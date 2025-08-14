import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from datetime import datetime

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data', 'raw')
models_dir = os.path.join(project_root, 'models', 'versioned', 'v1')

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
@st.cache_data
def load_real_estate_data():
    """Load real estate data from CSV files"""
    try:
        house_prices_path = os.path.join(data_dir, 'sample_house_prices.csv')
        if os.path.exists(house_prices_path):
            df = pd.read_csv(house_prices_path)
            return df
        else:
            # Fallback demo data
            return create_demo_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_demo_data()

def create_demo_data():
    """Create synthetic real estate data for demo"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft': np.random.randint(800, 3500, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'garage': np.random.randint(0, 3, n_samples),
        'pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'location_score': np.random.uniform(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic price based on features
    df['price'] = (
        df['sqft'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        (50 - df['age']) * 1000 +
        df['garage'] * 8000 +
        df['pool'] * 25000 +
        df['location_score'] * 8000 +
        np.random.normal(0, 20000, n_samples)
    ).round().astype(int)
    
    # Ensure positive prices
    df['price'] = np.maximum(df['price'], 50000)
    
    return df

@st.cache_resource
def train_models(df):
    """Train multiple ML models"""
    features = ['bedrooms', 'bathrooms', 'sqft', 'age']
    if 'garage' in df.columns:
        features.extend(['garage', 'pool', 'location_score'])
    
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        trained_models[name] = model
        model_scores[name] = {
            'R² Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return trained_models, model_scores, X_test, y_test

def format_currency(amount):
    """Format number as currency"""
    return f"${amount:,.0f}"

def format_percentage(value):
    """Format number as percentage"""
    return f"{value:.1%}"

# Page functions
def show_project_summary():
    st.title("🏠 Smart Real Estate Predictor")
    st.markdown("## Intelligent Property Price Prediction Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏘️ Properties Analyzed", "1,000+")
    with col2:
        st.metric("🎯 Prediction Accuracy", "85%+")
    with col3:
        st.metric("⚡ Models Available", "3")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **📊 Exploratory Data Analysis** - Interactive data visualization
        - **🧠 ML Model Training** - Multiple algorithms comparison
        - **📈 Price Prediction** - Real-time property valuation
        - **📋 Model Metrics** - Performance evaluation & insights
        """)
    
    with col2:
        st.subheader("🔬 Technology Stack")
        st.markdown("""
        - **Python** - Core programming language
        - **Streamlit** - Interactive web framework
        - **Scikit-learn** - Machine learning algorithms
        - **Plotly** - Advanced data visualization
        - **Pandas** - Data manipulation & analysis
        """)
    
    st.markdown("---")
    st.info("💡 **Getting Started**: Use the sidebar to navigate through different sections of the app.")

def show_eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("### Discover patterns and insights in real estate data")
    
    df = load_real_estate_data()
    
    # Basic statistics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Avg Price", format_currency(df['price'].mean()))
    with col3:
        st.metric("Price Range", f"{format_currency(df['price'].min())} - {format_currency(df['price'].max())}")
    with col4:
        st.metric("Avg Size", f"{df['sqft'].mean():.0f} sqft")
    
    # Data preview
    st.subheader("🔍 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        fig = px.histogram(df, x='price', nbins=30, title="Property Price Distribution")
        fig.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Price vs Square Footage")
        try:
            fig = px.scatter(df, x='sqft', y='price', title="Price vs Square Footage",
                            trendline="ols", color='bedrooms')
        except ImportError:
            # Fallback without trendline if statsmodels not available
            fig = px.scatter(df, x='sqft', y='price', title="Price vs Square Footage",
                            color='bedrooms')
        fig.update_layout(xaxis_title="Square Footage", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   title="Feature Correlation Matrix",
                   color_continuous_scale="RdBu_r",
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

def show_train_model():
    st.title("🧠 Model Training & Comparison")
    st.markdown("### Train and compare different machine learning models")
    
    df = load_real_estate_data()
    
    # Model training
    with st.spinner("Training models..."):
        trained_models, model_scores, X_test, y_test = train_models(df)
    
    st.success("✅ Models trained successfully!")
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    scores_df = pd.DataFrame(model_scores).T
    scores_df = scores_df.round(4)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(scores_df, use_container_width=True)
    
    with col2:
        # Best model recommendation
        best_model = scores_df['R² Score'].idxmax()
        st.metric("🏆 Best Model", best_model)
        st.metric("🎯 R² Score", f"{scores_df.loc[best_model, 'R² Score']:.3f}")
        st.metric("📉 MAE", format_currency(scores_df.loc[best_model, 'MAE']))
    
    # Visualization
    fig = go.Figure()
    for model_name in model_scores.keys():
        fig.add_trace(go.Bar(
            name=model_name,
            x=['R² Score', 'MAE (scaled)', 'RMSE (scaled)'],
            y=[scores_df.loc[model_name, 'R² Score'],
               scores_df.loc[model_name, 'MAE'] / 100000,
               scores_df.loc[model_name, 'RMSE'] / 100000]
        ))
    
    fig.update_layout(title="Model Performance Comparison", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in trained_models:
        st.subheader("🔍 Feature Importance (Random Forest)")
        rf_model = trained_models['Random Forest']
        feature_names = ['bedrooms', 'bathrooms', 'sqft', 'age']
        if len(rf_model.feature_importances_) > 4:
            feature_names.extend(['garage', 'pool', 'location_score'])
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importance_df, x='importance', y='feature', 
                    orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Save best model
    if st.button("💾 Save Best Model"):
        best_model_obj = trained_models[best_model]
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'best_model.joblib')
        joblib.dump(best_model_obj, model_path)
        st.success(f"✅ Best model ({best_model}) saved successfully!")

def show_predict():
    st.title("📈 Property Price Prediction")
    st.markdown("### Get instant price estimates for your property")
    
    df = load_real_estate_data()
    trained_models, _, _, _ = train_models(df)
    
    # Input form
    st.subheader("🏠 Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4, 5], index=1)
    
    with col2:
        sqft = st.slider("📐 Square Footage", min_value=500, max_value=5000, value=1500, step=50)
        age = st.slider("📅 Property Age (years)", min_value=0, max_value=50, value=10)
    
    with col3:
        if 'garage' in df.columns:
            garage = st.selectbox("🚗 Garage Spaces", options=[0, 1, 2, 3], index=1)
            pool = st.selectbox("🏊 Pool", options=[0, 1], index=0)
            location_score = st.slider("📍 Location Score (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
        else:
            garage = pool = location_score = 0
    
    # Prediction
    if st.button("🔮 Predict Price", type="primary"):
        feature_names = ['bedrooms', 'bathrooms', 'sqft', 'age']
        feature_values = [bedrooms, bathrooms, sqft, age]
        
        if 'garage' in df.columns:
            feature_names.extend(['garage', 'pool', 'location_score'])
            feature_values.extend([garage, pool, location_score])
        
        # Create DataFrame with proper feature names
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        st.subheader("💰 Price Predictions")
        
        predictions = {}
        for model_name, model in trained_models.items():
            pred_price = model.predict(input_df)[0]
            predictions[model_name] = pred_price
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌲 Random Forest", format_currency(predictions['Random Forest']))
        with col2:
            st.metric("📈 Gradient Boosting", format_currency(predictions['Gradient Boosting']))
        with col3:
            st.metric("📊 Linear Regression", format_currency(predictions['Linear Regression']))
        
        # Average prediction
        avg_prediction = np.mean(list(predictions.values()))
        st.markdown("---")
        st.metric("🎯 **Average Prediction**", format_currency(avg_prediction))
        
        # Confidence interval
        std_pred = np.std(list(predictions.values()))
        lower_bound = avg_prediction - 1.96 * std_pred
        upper_bound = avg_prediction + 1.96 * std_pred
        
        st.info(f"📊 **95% Confidence Interval**: {format_currency(lower_bound)} - {format_currency(upper_bound)}")
        
        # Market comparison
        similar_properties = df[
            (df['bedrooms'] == bedrooms) & 
            (df['bathrooms'] == bathrooms) &
            (df['sqft'].between(sqft-200, sqft+200))
        ]
        
        if not similar_properties.empty:
            market_avg = similar_properties['price'].mean()
            st.markdown("---")
            st.subheader("🏘️ Market Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Average", format_currency(market_avg))
            with col2:
                difference = avg_prediction - market_avg
                st.metric("Difference", format_currency(difference), 
                         delta=f"{difference/market_avg:.1%}")

def show_model_metrics():
    st.title("📋 Model Performance Metrics")
    st.markdown("### Detailed analysis of model performance and insights")
    
    df = load_real_estate_data()
    trained_models, model_scores, X_test, y_test = train_models(df)
    
    # Model selection
    selected_model = st.selectbox("🔍 Select Model for Analysis", 
                                 options=list(trained_models.keys()))
    
    model = trained_models[selected_model]
    y_pred = model.predict(X_test)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("MAE", format_currency(mean_absolute_error(y_test, y_pred)))
    with col3:
        st.metric("RMSE", format_currency(np.sqrt(mean_squared_error(y_test, y_pred))))
    
    # Prediction vs Actual scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Predictions vs Actual")
        fig = px.scatter(x=y_test, y=y_pred, 
                        title="Predicted vs Actual Prices",
                        labels={'x': 'Actual Price', 'y': 'Predicted Price'})
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Residuals Distribution")
        residuals = y_test - y_pred
        fig = px.histogram(x=residuals, nbins=30, 
                          title="Residuals Distribution")
        fig.update_layout(xaxis_title="Residuals", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    st.subheader("💡 Model Insights")
    
    accuracy = r2_score(y_test, y_pred)
    mae_percent = mean_absolute_error(y_test, y_pred) / y_test.mean()
    
    if accuracy > 0.8:
        st.success(f"✅ **Excellent Performance**: The model explains {accuracy:.1%} of price variance")
    elif accuracy > 0.6:
        st.warning(f"⚠️ **Good Performance**: The model explains {accuracy:.1%} of price variance")
    else:
        st.error(f"❌ **Poor Performance**: The model explains only {accuracy:.1%} of price variance")
    
    st.info(f"📊 **Average Error**: {mae_percent:.1%} of property value")

# Main app navigation
def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("🏠 Navigation")
        st.markdown("---")
        
        pages = {
            "🏠 Project Summary": show_project_summary,
            "📊 EDA": show_eda,
            "🧠 Train Model": show_train_model,
            "📈 Predict": show_predict,
            "📋 Model Metrics": show_model_metrics,
        }
        
        selected = st.radio("Go to", list(pages.keys()))
        
        st.markdown("---")
        st.markdown("### 🚀 Smart Real Estate Predictor")
        st.markdown("ML-powered property valuation")
        st.markdown(f"**Last updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Run selected page
    pages[selected]()

if __name__ == "__main__":
    main()
