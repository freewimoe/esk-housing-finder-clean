import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from shared_imports import px, go, make_subplots, load_sample_data, format_currency
except ImportError:
    # Fallback imports
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def load_sample_data():
        return pd.DataFrame({
            'price': [300000, 400000, 500000],
            'sqft': [1200, 1500, 2000],
            'bedrooms': [2, 3, 4]
        })
    
    def format_currency(amount):
        return f"${amount:,.2f}"




import os

class EdaPage:
    @staticmethod
    def render():
        st.title("Real Estate Market Explorer")
        st.markdown("### Interactive Property Data Exploration & Analysis")
        
        # Load sample data
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Load NYC Sample Data"):
                st.session_state['load_sample'] = True
        
        with col2:
            data_file = st.file_uploader("Or upload your own real estate CSV", type=["csv"])
        
        # Load data
        df = None
        if st.session_state.get('load_sample', False) or data_file:
            try:
                if st.session_state.get('load_sample', False):
                    # Load sample NYC real estate data
                    # Get the correct path from app directory
                    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    project_root = os.path.dirname(app_dir)
                    data_path = os.path.join(project_root, "data", "raw", "sample_real_estate.csv")
                    
                    st.info(f"🔍 Looking for data at: {data_path}")
                    
                    if os.path.exists(data_path):
                        df = pd.read_csv(data_path)
                        st.success(f"✅ NYC Sample Dataset loaded successfully! ({df.shape[0]} properties)")
                    else:
                        st.error(f"❌ Sample data file not found at: {data_path}")
                        # Try alternative path
                        alt_path = os.path.join(project_root, "sample_real_estate.csv")
                        if os.path.exists(alt_path):
                            df = pd.read_csv(alt_path)
                            st.success(f"✅ Found sample data at alternative location: {alt_path}")
                        else:
                            st.error("Please check if the data file exists in the project directory.")
                            return
                else:
                    df = pd.read_csv(data_file)
                    st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                # Dataset overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Properties", df.shape[0])
                with col2:
                    st.metric("Features", df.shape[1])
                with col3:
                    avg_price = df['price'].mean() if 'price' in df.columns else 0
                    st.metric("Avg Price", f"${avg_price:,.0f}")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Data Quality", f"{100-missing_pct:.1f}%")
                
                # Data preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Quick stats
                if 'price' in df.columns:
                    st.subheader("Price Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Price histogram
                        fig = px.histogram(df, x='price', 
                                         title='Property Price Distribution',
                                         nbins=20)
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Price by neighborhood
                        if 'neighborhood' in df.columns:
                            fig = px.box(df, y='price', x='neighborhood',
                                       title='Price by Neighborhood')
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                st.subheader("Feature Correlations")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  text_auto=True, 
                                  aspect="auto",
                                  title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(df.describe(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.info("Please load sample data or upload a CSV file to begin exploration.")
            
            # Show expected data format
            st.subheader("Expected Data Format")
            st.markdown("""
            Your CSV should contain columns like:
            - **price**: Property price
            - **bedrooms**: Number of bedrooms
            - **bathrooms**: Number of bathrooms
            - **sqft**: Square footage
            - **neighborhood**: Area/location
            - **latitude, longitude**: GPS coordinates (optional)
            """)
            
            sample_data = {
                'price': [450000, 680000, 320000],
                'bedrooms': [2, 3, 1],
                'bathrooms': [1, 2, 1],
                'sqft': [800, 1200, 600],
                'neighborhood': ['Manhattan', 'Brooklyn', 'Queens']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
