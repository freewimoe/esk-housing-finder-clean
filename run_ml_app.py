import streamlit as st
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import the main ML app
from app.ml_app import main

if __name__ == "__main__":
    main()
