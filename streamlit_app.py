"""
Streamlit App Entry Point
Multi-Scale Pathogenic Gene and Risk Modeling Using ML and GNN

This file serves as the main entry point for Streamlit Cloud deployment.
"""

import os
import sys

# Ensure we're running from the app directory context
os.chdir(os.path.dirname(__file__))

# Import and run the app module
exec(open(os.path.join(os.path.dirname(__file__), 'app', 'app.py')).read())
