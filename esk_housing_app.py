"""
ESK Housing Finder - Entry Point für Streamlit Cloud
Importiert die Hauptanwendung aus dem app/ Verzeichnis
"""

import sys
import os

# Add app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
sys.path.insert(0, app_dir)

# Import and run the main ESK Housing App
if __name__ == "__main__":
    from app.esk_housing_app import main
    main()
