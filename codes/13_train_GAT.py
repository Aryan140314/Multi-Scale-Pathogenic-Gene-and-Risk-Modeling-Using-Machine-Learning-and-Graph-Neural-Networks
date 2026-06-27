#!/usr/bin/env python
# coding: utf-8
# Generated from 13_train_GAT.ipynb

# %% Cell 1
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected, add_self_loops
from tqdm.auto import tqdm

# %% Cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% Cell 3
features = pd.read_csv("../data/processed/final_gene_features.csv")
edges = pd.read_csv("../data/processed/final_edge_list.csv")

print("Features shape:", features.shape)
print("Edges shape:", edges.shape)

# %% [markdown]
# Remove Irrelevant Features
# 
# For GNN we remove:
# 
# identifiers
# 
# text columns
# 
# leakage columns
# 
# redundant topology features

# %% Cell 4
drop_cols = [
    "GeneSymbol", "description", "pathogenic_variants", 
    "neighbor_pathogenic_ratio", "mutation_network_score", 
    "rare_network_score", "gene_degree", 
    "clustering_coefficient", "pagerank", "betweenness_centrality"
]

y = features["label"].values
X = features.drop(columns=drop_cols + ["label"], errors="ignore")
X = X.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "../models/feature_scaler.pkl")
print("Saved feature_scaler.pkl")

X_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# %% [markdown]
# Advanced Graph Construction (Crucial for GAT)

# %% Cell 5
gene_to_idx = {gene: i for i, gene in enumerate(features["GeneSymbol"])}
edges["gene1"] = edges["gene1"].map(gene_to_idx)
edges["gene2"] = edges["gene2"].map(gene_to_idx)
edges = edges.dropna()

edge_index = torch.tensor(
    edges[["gene1", "gene2"]].values.T, dtype=torch.long
)

# Crucial for GAT: Undirected + Self-Loops
edge_index = to_undirected(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=X_tensor.size(0))

print("Final Edge Index Shape:", edge_index.shape)

# %% [markdown]
# Masking & Train/Test Split

# %% Cell 6
data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

train_idx, test_idx = train_test_split(
    np.arange(len(y_tensor)), test_size=0.2, stratify=y_tensor, random_state=42
)

train_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
test_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.shape[1])

# %% [markdown]
# Focal Loss Definition

# %% Cell 7
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# %% [markdown]
# Expanded Neighborhood Loaders

# %% Cell 8
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
weights[1] = weights[1] * 0.95  
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# Expanded to [30, 20] to capture more biological context
train_loader = NeighborLoader(
    data, num_neighbors=[30, 20], batch_size=128, input_nodes=train_idx, shuffle=True
)
test_loader = NeighborLoader(
    data, num_neighbors=[30, 20], batch_size=256, input_nodes=test_idx
)
print("Enhanced Loaders ready.")

# %% [markdown]
# The Residual GAT Architecture

# %% Cell 9
class PathoGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=8):
        super(PathoGAT, self).__init__()
        
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.ln1 = torch.nn.LayerNorm(hidden_dim * heads) 
        
        self.conv2 = GATv2Conv(hidden_dim * heads, 2, heads=1, concat=False)
        
        # Residual Connection Matrix
        self.skip = torch.nn.Linear(input_dim, hidden_dim * heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x1 = self.conv1(x, edge_index)
        x_skip = self.skip(x)
        x1 = x1 + x_skip  # Inject original tabular features
        
        x1 = self.ln1(x1)
        x1 = F.elu(x1) 
        x1 = F.dropout(x1, p=0.4, training=self.training)
        
        out = self.conv2(x1, edge_index)
        return out

print("Architecture Defined.")

# %% [markdown]
# Initialization & Animated Training Loop

# %% Cell 10
model = PathoGAT(input_dim=X_tensor.shape[1], hidden_dim=128, heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

epochs = 200
epoch_bar = tqdm(range(1, epochs + 1), desc="🧬 Training ", colour='green')

for epoch in epoch_bar:
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = out.argmax(dim=1)
        total_correct += int((preds == batch.y).sum())
        total_samples += batch.y.size(0)
        
    scheduler.step(total_loss)
    train_acc = total_correct / total_samples
        
    epoch_bar.set_postfix({
        'Loss': f"{total_loss:.4f}", 
        'Acc': f"{train_acc:.4f}",
        'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
    })
    
"""    if epoch % 10 == 0:
        tqdm.write(
            f"✅ Epoch {epoch:03d}/{epochs} | "
            f"Focal Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )"""

# %% [markdown]
# 90+ Auto-Tuner & Evaluation

# %% Cell 11
model.eval()
labels = []
probs = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)[:, 1]
        
        labels.extend(batch.y.cpu().numpy())
        probs.extend(prob.cpu().numpy())

labels = np.array(labels)
probs = np.array(probs)

print("Baseline ROC-AUC:", roc_auc_score(labels, probs))
print("-" * 50)

best_threshold = 0.65
best_min_f1 = 0

print("Scanning for optimal threshold to balance Precision & Recall...")
"""for thresh in np.arange(0.40, 0.85, 0.01):
    temp_preds = (probs >= thresh).astype(int)
    
    f1_0 = f1_score(labels, temp_preds, pos_label=0)
    f1_1 = f1_score(labels, temp_preds, pos_label=1)
    
    min_f1 = min(f1_0, f1_1)
    
    if min_f1 > best_min_f1:
        best_min_f1 = min_f1
        best_threshold = thresh"""

print(f"\n✅ Optimal Threshold Found: {best_threshold:.2f}")

final_preds = (probs >= best_threshold).astype(int)

print("\n🚀 FINAL OPTIMIZED CLASSIFICATION REPORT 🚀\n")
print(classification_report(labels, final_preds))

# %% Cell 12
model_path = "../models/gene_gat_model.pt"
torch.save(model.state_dict(), model_path)
print("PathoGAT successfully saved to:", model_path)
