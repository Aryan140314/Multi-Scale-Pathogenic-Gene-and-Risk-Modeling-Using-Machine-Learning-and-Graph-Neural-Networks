import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & PATHS
# ==========================================
BASE_DATA_DIR = r"C:\mutation\data\processed"
BASE_MODEL_DIR = r"C:\mutation\models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================
class GeneSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.ln1   = torch.nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2)
        self.skip  = torch.nn.Linear(input_dim, hidden_dim)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x1 = self.conv1(x, ei) + self.skip(x)
        return self.conv2(F.elu(self.ln1(x1)), ei)

class GeneGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.ln1   = torch.nn.LayerNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, 2, heads=1, concat=False)
        self.skip  = torch.nn.Linear(input_dim, hidden_dim * heads)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x1 = self.conv1(x, ei) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        x1 = F.dropout(x1, p=0.4, training=self.training)
        return self.conv2(x1, ei)

# ==========================================
# 3. DATA LOADING & PREPARATION
# ==========================================
print("Loading data...")
features_df = pd.read_csv(f"{BASE_DATA_DIR}/final_gene_features.csv")
edges_df = pd.read_csv(f"{BASE_DATA_DIR}/final_edge_list.csv")

# Extract Labels
y_full = features_df['label'].values if 'label' in features_df.columns else features_df['pathogenic'].values

# Stratified Train/Test Split (Matching Training)
train_idx, test_idx, y_train, y_test = train_test_split(
    np.arange(len(y_full)), y_full, test_size=0.2, random_state=42, stratify=y_full
)

# 🔥 MUST MATCH TRAINING
GNN_DROP = {
    "GeneSymbol","description","label",
    "pathogenic_variants","pathogenic",
    "neighbor_pathogenic_ratio","mutation_network_score",
    "rare_network_score","gene_degree",
    "clustering_coefficient","pagerank","betweenness_centrality"
}
gnn_cols = [c for c in features_df.columns if c not in GNN_DROP and pd.api.types.is_numeric_dtype(features_df[c])]

# Scale Features
gnn_scaler = StandardScaler()
X_gnn_scaled = gnn_scaler.fit_transform(features_df[gnn_cols].values.astype(np.float32))

# Build Graph
genes = features_df["GeneSymbol"].tolist()
g2i = {g: i for i, g in enumerate(genes)}
valid_edges = [[g2i[r.gene1], g2i[r.gene2]] for r in edges_df.itertuples() if r.gene1 in g2i and r.gene2 in g2i]

edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(genes))

graph_data = Data(x=torch.tensor(X_gnn_scaled, dtype=torch.float), edge_index=edge_index).to(device)

# ==========================================
# 4. LOAD MODELS & RUN INFERENCE
# ==========================================
print("Running inference...")
input_dim = len(gnn_cols)

# Load GraphSAGE
sage_model = GeneSAGE(input_dim=input_dim).to(device)
sage_model.load_state_dict(torch.load(rf"{BASE_MODEL_DIR}\gene_gnn_model.pt", map_location=device))
sage_model.eval()

# Load GATv2
gat_model = GeneGAT(input_dim=input_dim, hidden_dim=128, heads=8).to(device)
gat_model.load_state_dict(torch.load(rf"{BASE_MODEL_DIR}\gene_gat_model.pt", map_location=device))
gat_model.eval()

# Get Probabilities
with torch.no_grad():
    sage_probs_full = torch.softmax(sage_model(graph_data), dim=1)[:, 1].cpu().numpy()
    gat_probs_full = torch.softmax(gat_model(graph_data), dim=1)[:, 1].cpu().numpy()

# Isolate Test Set Predictions
sage_test_probs = sage_probs_full[test_idx]
gat_test_probs = gat_probs_full[test_idx]
cons_test_probs = (sage_test_probs + gat_test_probs) / 2

# ==========================================
# 5. CALCULATE & DISPLAY METRICS
# ==========================================
def evaluate_model(name, y_true_test, y_probs_test):
    # ROC-AUC
    auc_val = roc_auc_score(y_true_test, y_probs_test)
    
    # Calculate Optimal Threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true_test, y_probs_test)
    f1_scores = [f1_score(y_true_test, (y_probs_test >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Generate binary predictions
    y_pred = (y_probs_test >= best_threshold).astype(int)
    
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true_test, y_pred), 4),
        "Balanced Acc": round(balanced_accuracy_score(y_true_test, y_pred), 4),
        "Precision": round(precision_score(y_true_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true_test, y_pred), 4),
        "F1-Score": round(f1_score(y_true_test, y_pred), 4),
        "ROC-AUC": round(auc_val, 4)
    }

results = [
    evaluate_model("GraphSAGE (GNN)", y_test, sage_test_probs),
    evaluate_model("PathoGAT (GAT)", y_test, gat_test_probs),
    evaluate_model("Hybrid Consensus", y_test, cons_test_probs)
]

results_df = pd.DataFrame(results)

print("\n" + "="*85)
print("🧬 GNN TESTING METRICS (Test Set Only)")
print("="*85)
print(results_df.to_string(index=False))
print("="*85 + "\n")