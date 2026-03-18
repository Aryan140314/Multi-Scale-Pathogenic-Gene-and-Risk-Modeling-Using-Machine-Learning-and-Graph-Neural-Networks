import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data

# Setup Device globally
device = torch.device('cpu') 

# =====================================================
# PATH HELPER (Solves FileNotFoundError on Cloud)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# GNN MODEL CLASSES (SAGE & GAT)
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
    features_path = os.path.join(BASE_DIR, "data", "processed", "final_gene_features.csv")
    features_df = pd.read_csv(features_path)
    
    # Try multiple common names for the edges file
    edge_filenames = ["string_interactions.csv", "final_edge_list.csv", "string_interactions_short.csv"]
    edges_df = None
    
    for fname in edge_filenames:
        path = os.path.join(BASE_DIR, "data", "processed", fname)
        if os.path.exists(path):
            edges_df = pd.read_csv(path)
            break
            
    if edges_df is None:
        st.error("Could not find the interaction edges CSV file in data/processed/")
        st.stop()
        
    if 'pathogenic' not in features_df.columns and 'label' in features_df.columns:
        features_df['pathogenic'] = features_df['label']
    elif 'pathogenic' not in features_df.columns:
        features_df['pathogenic'] = (features_df['total_variants'] > features_df['total_variants'].median()).astype(int)
        
    return features_df, edges_df

@st.cache_resource
def load_models():
    models_dir = os.path.join(BASE_DIR, "models")
    
    # Load Scaler
    scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
    
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
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            ml_models[name] = joblib.load(path)
            
    # Load GNNs
    sage_model = GeneSAGE(input_dim=38).to(device)
    sage_path = os.path.join(models_dir, "gene_gnn_model.pt")
    if os.path.exists(sage_path):
        sage_model.load_state_dict(torch.load(sage_path, map_location=device))
    sage_model.eval()

    gat_model = GeneGAT(input_dim=38, hidden_dim=128, heads=8).to(device)
    gat_path = os.path.join(models_dir, "gene_gat_model.pt")
    if os.path.exists(gat_path):
        gat_model.load_state_dict(torch.load(gat_path, map_location=device))
    gat_model.eval()
    
    return ml_models, scaler, sage_model, gat_model

# INITIALIZE
st.set_page_config(layout="wide", page_title="PathoGAT AI Dashboard", page_icon="🧬")
st.title("🧬 PathoGAT: Multi-Scale Gene Pathogenicity Predictor")

features_df, edges_df = load_datasets()
ml_models, scaler, sage_model, gat_model = load_models()

# =====================================================
# PIPELINE & GRAPH
# =====================================================
@st.cache_data
def build_graph_and_pipelines(_features_df, _edges_df, _rf_model, _scaler):
    all_cols = list(_scaler.feature_names_in_)  
    for col in all_cols:
        if col not in _features_df.columns: _features_df[col] = 0.0
            
    X_raw = _features_df[all_cols].values
    X_all_scaled = _scaler.transform(_features_df[all_cols])
    
    graph_cols_to_drop = ["neighbor_pathogenic_ratio", "mutation_network_score", "rare_network_score", "gene_degree", "clustering_coefficient", "pagerank", "betweenness_centrality"]
    gnn_indices = [i for i, col in enumerate(all_cols) if col not in graph_cols_to_drop]
    gnn_cols = [all_cols[i] for i in gnn_indices]
    
    X_gnn_scaled = X_all_scaled[:, gnn_indices]
    
    gene_list = _features_df["GeneSymbol"].tolist() if "GeneSymbol" in _features_df.columns else _features_df.index.astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
    valid_edges = []
    for _, row in _edges_df.iterrows():
        if row['gene1'] in gene_to_idx and row['gene2'] in gene_to_idx:
            valid_edges.append([gene_to_idx[row['gene1']], gene_to_idx[row['gene2']]])
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    
    return X_raw, X_gnn_scaled, all_cols, gnn_cols, gnn_indices, gene_to_idx, edge_index, gene_list

X_raw, X_gnn_scaled, rf_cols, gnn_cols, gnn_indices, gene_to_idx, edge_index, gene_list = build_graph_and_pipelines(
    features_df, edges_df, ml_models.get("RandomForest", list(ml_models.values())[0]), scaler
)

# =====================================================
# SIDEBAR (DYNAMIC FEATURE EDITOR)
# =====================================================
st.sidebar.header("🎛️ Gene Configuration")
selected_gene = st.sidebar.selectbox("Select Target Gene", sorted(gene_list))
selected_idx = gene_to_idx[selected_gene]

st.sidebar.markdown("---")
st.sidebar.header("✏️ Edit Features dynamically")

available_numeric_features = sorted([col for col in rf_cols if pd.api.types.is_numeric_dtype(features_df[col])])
default_edits = [f for f in ["total_variants", "benign_variants", "rare_variants", "gene_degree", "clustering_coefficient"] if f in available_numeric_features]

editable_features = st.sidebar.multiselect("Select features to modify:", available_numeric_features, default=default_edits)

user_inputs = {}
for feat in editable_features:
    val = float(features_df.iloc[selected_idx][feat])
    user_inputs[feat] = st.sidebar.number_input(label=feat, value=val, format="%.4f")

# =====================================================
# INFERENCE ENGINE
# =====================================================
node_raw_features = X_raw[selected_idx].copy()
for feat, val in user_inputs.items():
    if feat in rf_cols: 
        node_raw_features[rf_cols.index(feat)] = val

X_in_df = pd.DataFrame([node_raw_features], columns=rf_cols)
X_in_scaled = scaler.transform(X_in_df)
X_in_scaled_df = pd.DataFrame(X_in_scaled, columns=rf_cols)

individual_ml_probs = {}
for name, m in ml_models.items():
    if name == "StackingEnsemble": continue 
    if hasattr(m, 'predict_proba'):
        individual_ml_probs[name] = m.predict_proba(X_in_scaled_df)[0][1]
    else:
        individual_ml_probs[name] = 1 / (1 + np.exp(-m.decision_function(X_in_scaled_df)[0]))

ml_ensemble_prob = ml_models["StackingEnsemble"].predict_proba(X_in_scaled_df)[0][1] if "StackingEnsemble" in ml_models else np.mean(list(individual_ml_probs.values()))

X_gnn_scaled_live = X_gnn_scaled.copy()
X_gnn_scaled_live[selected_idx] = X_in_scaled[0, gnn_indices]

with torch.no_grad():
    graph_data = Data(x=torch.tensor(X_gnn_scaled_live, dtype=torch.float), edge_index=edge_index).to(device)
    
    sage_out = sage_model(graph_data)
    sage_prob = torch.softmax(sage_out, dim=1)[selected_idx][1].item()
    
    gat_out = gat_model(graph_data)
    gat_prob = torch.softmax(gat_out, dim=1)[selected_idx][1].item()

gnn_ensemble_prob = (sage_prob + gat_prob) / 2
final_ensemble_prob = (ml_ensemble_prob * 0.4) + (gnn_ensemble_prob * 0.6)

# =====================================================
# DASHBOARD LAYOUT
# =====================================================
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("🌟 FINAL CONSENSUS RISK", f"{final_ensemble_prob:.3f}")
m_col2.metric("📊 ML Stacking Ensemble", f"{ml_ensemble_prob:.3f}")
m_col3.metric("🕸️ GNN Consensus", f"{gnn_ensemble_prob:.3f}")

st.markdown("---")
st.markdown("### 🤖 Individual Model Predictions")
i_cols = st.columns(len(individual_ml_probs) + 2)
for idx, (m_name, m_prob) in enumerate(individual_ml_probs.items()):
    i_cols[idx].metric(m_name, f"{m_prob:.3f}")
i_cols[-2].metric("GraphSAGE", f"{sage_prob:.3f}")
i_cols[-1].metric("PathoGAT (GAT)", f"{gat_prob:.3f}")

st.markdown("---")
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("🎯 Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=final_ensemble_prob, gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#ff4b4b" if final_ensemble_prob > 0.5 else "#00fa9a"}}))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, width='stretch')

with col_right:
    st.subheader("📊 All Models Comparison")
    all_names = list(individual_ml_probs.keys()) + ["ML Ensemble", "GraphSAGE", "PathoGAT", "GNN Consensus", "FINAL CONSENSUS"]
    all_probs = list(individual_ml_probs.values()) + [ml_ensemble_prob, sage_prob, gat_prob, gnn_ensemble_prob, final_ensemble_prob]
    colors = ['#1f77b4']*len(individual_ml_probs) + ['#ff7f0e', '#9467bd', '#9467bd', '#f1c40f', '#00fa9a' if final_ensemble_prob < 0.5 else '#ff4b4b']
    
    all_names.reverse()
    all_probs.reverse()
    colors.reverse()
    
    fig_bar = go.Figure(go.Bar(x=all_probs, y=all_names, orientation='h', marker_color=colors, text=[f"{p:.3f}" for p in all_probs], textposition='auto'))
    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Pathogenic Cutoff")
    fig_bar.update_layout(xaxis=dict(range=[0, 1]), height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bar, width='stretch')

st.markdown("---")
dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    st.subheader("🧪 Feature Population Analysis")
    if editable_features:
        feat_to_plot = st.selectbox("Compare Feature Dist:", editable_features)
        fig_dist = px.histogram(features_df, x=feat_to_plot, nbins=50, title=f"Global {feat_to_plot} Distribution")
        fig_dist.add_vline(x=user_inputs[feat_to_plot], line_dash="dash", line_color="red", annotation_text="Selected Gene")
        st.plotly_chart(fig_dist, width='stretch')

with dist_col2:
    st.subheader("🔬 Feature Sensitivity Analysis")
    if len(rf_cols) > 0:
        feature_sens = st.selectbox("Sensitivity Target:", rf_cols[:10])
        test_range = np.linspace(features_df[feature_sens].min(), features_df[feature_sens].max(), 20)
        sens_probs = []
        for v in test_range:
            t_row = node_raw_features.copy()
            t_row[rf_cols.index(feature_sens)] = v
            t_row_scaled_df = pd.DataFrame(scaler.transform(pd.DataFrame([t_row], columns=rf_cols)), columns=rf_cols)
            model_to_test = ml_models.get("XGBoost", list(ml_models.values())[0])
            if hasattr(model_to_test, 'predict_proba'):
                sens_probs.append(model_to_test.predict_proba(t_row_scaled_df)[0][1])
            else:
                sens_probs.append(1 / (1 + np.exp(-model_to_test.decision_function(t_row_scaled_df)[0])))
        fig_sens = px.line(x=test_range, y=sens_probs, labels={'x': feature_sens, 'y': 'Pathogenicity Risk'})
        st.plotly_chart(fig_sens, width='stretch')

st.markdown("---")
st.subheader("🌐 Interactive Protein Network")
theme = st.radio("Network Theme", ["Dark", "Light"], horizontal=True)
bg_color = "#0e1117" if theme == "Dark" else "#ffffff"
text_color = "white" if theme == "Dark" else "black"

local_edges = edges_df[(edges_df["gene1"] == selected_gene) | (edges_df["gene2"] == selected_gene)].head(60)

if not local_edges.empty:
    net = Network(height="600px", width="100%", bgcolor=bg_color, font_color=text_color, notebook=False)
    net.barnes_hut(gravity=-8000, spring_length=200)
    net.set_edge_smooth("dynamic")
    genes_in_network = set(local_edges['gene1']).union(set(local_edges['gene2']))
    for gene in genes_in_network:
        net.add_node(gene, label=gene, color="#ff4b4b" if gene == selected_gene else "#1f77b4", size=40 if gene == selected_gene else 20)
    for _, r in local_edges.iterrows():
        net.add_edge(r['gene1'], r['gene2'], color="#888888")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=615)

st.markdown("---")
st.subheader("🧠 Node2Vec Global Embedding Space")
if "node2vec_0" in features_df.columns:
    sample = features_df.sample(min(800, len(features_df)))
    fig_embed = px.scatter(sample, x="node2vec_0", y="node2vec_1", color="pathogenic", opacity=0.4)
    fig_embed.add_trace(go.Scatter(x=[features_df.iloc[selected_idx]["node2vec_0"]], y=[features_df.iloc[selected_idx]["node2vec_1"]], mode='markers', marker=dict(size=15, color="red", symbol="star")))
    st.plotly_chart(fig_embed, width='stretch')
