import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data

# Setup Device
device = torch.device('cpu') 

# =====================================================
# PATH HELPER (Cloud Optimized)
# =====================================================
# Streamlit Cloud runs from the repository root. 
# We define paths relative to that root.
BASE_DIR = os.getcwd() 
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# GNN MODEL CLASSES
# =====================================================
class GeneSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2)
        self.skip = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        return self.conv2(x1, edge_index)

class GeneGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=8):  
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.ln1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, 2, heads=1, concat=False, dropout=0.3)
        self.skip = torch.nn.Linear(input_dim, hidden_dim * heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        return self.conv2(x1, edge_index)

# =====================================================
# DATA & MODEL LOADING
# =====================================================
@st.cache_data
def load_datasets():
    f_path = os.path.join(DATA_DIR, "final_gene_features.csv")
    e_path = os.path.join(DATA_DIR, "string_interactions.csv")
    
    if not os.path.exists(f_path):
        st.error(f"Missing file: {f_path}")
        st.stop()
        
    features_df = pd.read_csv(f_path)
    
    # Try alternate edge filename if first fails
    if not os.path.exists(e_path):
        e_path = os.path.join(DATA_DIR, "final_edge_list.csv")
        
    edges_df = pd.read_csv(e_path)
    
    if 'pathogenic' not in features_df.columns:
        features_df['pathogenic'] = (features_df['total_variants'] > features_df['total_variants'].median()).astype(int)
        
    return features_df, edges_df

@st.cache_resource
def load_models():
    # Load Scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    
    # Load ML Models
    ml_models = {}
    model_files = {
        "RandomForest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "GradientBoost": "gradient_boost.pkl",
        "SVM": "svm.pkl",
        "LogisticRegression": "logistic_regression.pkl",
        "StackingEnsemble": "stacking_ensemble.pkl"
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            ml_models[name] = joblib.load(path)
            
    # Load GNNs
    sage_model = GeneSAGE(input_dim=38).to(device)
    sage_path = os.path.join(MODELS_DIR, "gene_gnn_model.pt")
    if os.path.exists(sage_path):
        sage_model.load_state_dict(torch.load(sage_path, map_location=device))
    sage_model.eval()

    gat_model = GeneGAT(input_dim=38, hidden_dim=128, heads=8).to(device)
    gat_path = os.path.join(MODELS_DIR, "gene_gat_model.pt")
    if os.path.exists(gat_path):
        gat_model.load_state_dict(torch.load(gat_path, map_location=device))
    gat_model.eval()
    
    return ml_models, scaler, sage_model, gat_model

# =====================================================
# MAIN APP EXECUTION
# =====================================================
st.set_page_config(layout="wide", page_title="PathoGAT Dashboard")
st.title("🧬 PathoGAT AI Predictor")

try:
    features_df, edges_df = load_datasets()
    ml_models, scaler, sage_model, gat_model = load_models()
except Exception as e:
    st.error(f"Detailed Initialization Error: {e}")
    st.stop()

# Get Features for Inference
rf_cols = list(ml_models["RandomForest"].feature_names_in_)
gene_list = features_df["GeneSymbol"].tolist()
gene_to_idx = {g: i for i, g in enumerate(gene_list)}

# Sidebar
selected_gene = st.sidebar.selectbox("Select Gene", sorted(gene_list))
idx = gene_to_idx[selected_gene]

# Basic Inference Logic
X_raw = features_df[rf_cols].values[idx].reshape(1, -1)
X_scaled = scaler.transform(pd.DataFrame(X_raw, columns=rf_cols))
prob = ml_models["StackingEnsemble"].predict_proba(X_scaled)[0][1]

# Display Result
st.metric("Pathogenicity Risk", f"{prob:.3f}")

# (Rest of your UI logic goes here - keep it simple first to ensure it boots!)
