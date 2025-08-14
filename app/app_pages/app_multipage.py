import streamlit as st

class MultiPage:
    def __init__(self):
        self.pages = []
    
    def add_page(self, title, func):
        self.pages.append({"title": title, "function": func})
    
    def run(self):
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            st.markdown("---")
            
            # Create selection
            page_names = [page["title"] for page in self.pages]
            selected = st.radio("Go to", page_names)
            
            st.markdown("---")
            st.markdown("###  Smart Real Estate Predictor")
            st.markdown("ML-powered property price prediction")
        
        # Run the selected page
        for page in self.pages:
            if page["title"] == selected:
                page["function"]()
                break
