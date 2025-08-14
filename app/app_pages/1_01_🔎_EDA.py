import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class EdaPage:
    @staticmethod
    def render():
        st.title("�️ Real Estate Market Explorer")
        st.markdown("### Interactive Property Data Exploration & Geospatial Analysis")
        
        # Load sample data button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🏠 Load NYC Sample Data"):
                st.session_state['load_sample'] = True
        
        with col2:
            data_file = st.file_uploader("Or upload your own real estate CSV", type=["csv"])
        
        # Load data
        df = None
        if st.session_state.get('load_sample', False) or data_file:
            try:
                if st.session_state.get('load_sample', False):
                    # Load sample NYC real estate data
                    df = pd.read_csv('data/raw/sample_real_estate.csv')
                    st.success("✅ NYC Sample Dataset loaded successfully!")
                else:
                    df = pd.read_csv(data_file)
                    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Dataset overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏠 Properties", df.shape[0])
                with col2:
                    st.metric("📊 Features", df.shape[1])
                with col3:
                    avg_price = df['price'].mean() if 'price' in df.columns else 0
                    st.metric("💰 Avg Price", f"${avg_price:,.0f}")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("🔍 Data Quality", f"{100-missing_pct:.1f}%")
                
                # Data preview
                st.subheader("📊 Dataset Preview")
                st.dataframe(df.head(10))
                
                # Column information
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Unique Values': df.nunique(),
                        'Missing Values': df.isnull().sum()
                    })
                    st.dataframe(col_info)
                
                # Statistical summary
                with st.expander("📈 Statistical Summary"):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe())
                    else:
                        st.info("No numeric columns found")
                
                # Visualizations
                st.subheader("📊 Data Visualizations")
                
                # Select column for analysis
                all_cols = df.columns.tolist()
                selected_col = st.selectbox("Select column for detailed analysis:", all_cols)
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution plot
                        if df[selected_col].dtype in ['int64', 'float64']:
                            fig = px.histogram(df, x=selected_col, 
                                             title=f"Distribution of {selected_col}",
                                             marginal="box")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            value_counts = df[selected_col].value_counts().head(10)
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                        title=f"Top 10 values in {selected_col}")
                            fig.update_xaxis(title=selected_col)
                            fig.update_yaxis(title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Missing values visualization
                        if df.isnull().sum().sum() > 0:
                            missing_data = df.isnull().sum().sort_values(ascending=False)
                            missing_data = missing_data[missing_data > 0]
                            
                            if len(missing_data) > 0:
                                fig = px.bar(x=missing_data.index, y=missing_data.values,
                                           title="Missing Values by Column")
                                fig.update_xaxis(title="Columns")
                                fig.update_yaxis(title="Missing Count")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No missing values found!")
                        else:
                            st.success("✅ No missing values in dataset!")
                
                # Correlation analysis
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    with st.expander("🔗 Correlation Analysis"):
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download processed data
                st.subheader("💾 Download Data")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download processed data as CSV",
                    data=csv,
                    file_name='processed_data.csv',
                    mime='text/csv'
                )
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.info("Please upload a CSV file to start exploring your data.")
            
            # Sample data info
            with st.expander("ℹ️ Sample Data Format"):
                st.markdown("""
                **Your CSV should contain:**
                - Headers in the first row
                - Numeric or categorical data
                - UTF-8 encoding (recommended)
                
                **Example structure:**
                ```
                feature1,feature2,target
                1.2,Category A,0
                2.3,Category B,1
                ```
                """)