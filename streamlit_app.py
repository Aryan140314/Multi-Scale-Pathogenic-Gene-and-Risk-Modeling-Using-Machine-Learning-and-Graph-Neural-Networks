"""
Streamlit App Entry Point
Multi-Scale Pathogenic Gene and Risk Modeling Using ML and GNN

This file serves as the main entry point for Streamlit Cloud deployment.
"""

import os
import sys

# Add the app directory to the path
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)

# Import the main app
from app import *  # noqa: F401, F403
