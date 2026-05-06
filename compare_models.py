import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import add_self_loops, to_undirected
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------------------------------------------------------
# Feature groups
# -----------------------------------------------------------------------------
ML_DROP = {
    "GeneSymbol",
    "description",
    "label",
    "pathogenic_variants",
    "pathogenic",
}
GNN_EXTRA_DROP = {
    "neighbor_pathogenic_ratio",
    "mutation_network_score",
    "rare_network_score",
    "gene_degree",
    "clustering_coefficient",
    "pagerank",
    "betweenness_centrality",
}
GNN_DROP = ML_DROP | GNN_EXTRA_DROP

# -----------------------------------------------------------------------------
# Graph neural network models
# -----------------------------------------------------------------------------
class GeneSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2)
        self.skip = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        return self.conv2(x1, edge_index)


class GeneGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.ln1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, 2, heads=1, concat=False)
        self.skip = torch.nn.Linear(input_dim, hidden_dim * heads)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        x1 = F.dropout(x1, p=0.4, training=self.training)
        return self.conv2(x1, edge_index)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_data():
    features_path = os.path.join(DATA_DIR, "final_gene_features.csv")
    edges_path = os.path.join(DATA_DIR, "final_edge_list.csv")

    df = pd.read_csv(features_path)
    edges = pd.read_csv(edges_path)
    return df, edges


def build_feature_sets(features_df):
    ml_cols = [c for c in features_df.columns if c not in ML_DROP and pd.api.types.is_numeric_dtype(features_df[c])]
    gnn_cols = [c for c in features_df.columns if c not in GNN_DROP and pd.api.types.is_numeric_dtype(features_df[c])]
    return ml_cols, gnn_cols


def load_scaler():
    for pname in ["ml_feature_scaler.pkl", "feature_scaler.pkl"]:
        candidate = os.path.join(MODEL_DIR, pname)
        if os.path.exists(candidate):
            return joblib.load(candidate)
    return StandardScaler()


def load_sklearn_models():
    model_files = {
        "LogisticRegression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "GradientBoost": "gradient_boost.pkl",
        "SVM": "svm.pkl",
        "ML Ensemble": "stacking_ensemble.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")
        models[name] = joblib.load(path)
    return models


def load_gnn_models(input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sage = GeneSAGE(input_dim=input_dim).to(device)
    gat = GeneGAT(input_dim=input_dim).to(device)

    sage_path = os.path.join(MODEL_DIR, "gene_gnn_model.pt")
    gat_path = os.path.join(MODEL_DIR, "gene_gat_model.pt")
    if not os.path.exists(sage_path) or not os.path.exists(gat_path):
        raise FileNotFoundError("Missing one of the GNN model checkpoint files.")

    sage.load_state_dict(torch.load(sage_path, map_location=device))
    sage.eval()
    gat.load_state_dict(torch.load(gat_path, map_location=device))
    gat.eval()
    return sage, gat, device


def sklearn_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        return 1 / (1 + np.exp(-logits))
    return model.predict(X)


def best_threshold_f1(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    f1_vals = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1_vals)] if len(thresholds) else 0.5


def evaluate_metrics(name, y_true, y_probs):
    thresh = best_threshold_f1(y_true, y_probs)
    y_pred = (y_probs >= thresh).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probs),
    }


def build_graph_data(features_df, edges_df, gnn_cols):
    genes = features_df["GeneSymbol"].tolist()
    g2i = {g: i for i, g in enumerate(genes)}
    valid_edges = [[g2i[r.gene1], g2i[r.gene2]] for r in edges_df.itertuples() if r.gene1 in g2i and r.gene2 in g2i]
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(genes))
    return edge_index


def plot_model_comparison(df, output_path="model_comparison.png"):
    melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="tab10")
    plt.title("Model Performance Comparison: ML vs GNN")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    features_df, edges_df = load_data()
    label_col = "label" if "label" in features_df.columns else "pathogenic"
    y = features_df[label_col].astype(int).values

    ml_cols, gnn_cols = build_feature_sets(features_df)
    X_ml = features_df[ml_cols].values.astype(np.float32)
    X_gnn = features_df[gnn_cols].values.astype(np.float32)

    train_idx, test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = load_scaler()
    if hasattr(scaler, "transform"):
        scaler.fit(X_ml[train_idx])
        X_ml_scaled = scaler.transform(X_ml)
    else:
        scaler = StandardScaler().fit(X_ml[train_idx])
        X_ml_scaled = scaler.transform(X_ml)

    ml_models = load_sklearn_models()
    results = []
    for name, model in ml_models.items():
        probs = sklearn_predict_proba(model, X_ml_scaled[test_idx])
        results.append(evaluate_metrics(name, y_test, probs))

    sage_model, gat_model, device = load_gnn_models(input_dim=len(gnn_cols))
    edge_index = build_graph_data(features_df, edges_df, gnn_cols)

    data = Data(x=torch.tensor(StandardScaler().fit_transform(X_gnn).astype(np.float32), dtype=torch.float), edge_index=edge_index).to(device)
    with torch.no_grad():
        sage_probs = torch.softmax(sage_model(data), dim=1)[:, 1].cpu().numpy()
        gat_probs = torch.softmax(gat_model(data), dim=1)[:, 1].cpu().numpy()

    results.append(evaluate_metrics("GraphSAGE", y_test, sage_probs[test_idx]))
    results.append(evaluate_metrics("PathoGAT (GAT)", y_test, gat_probs[test_idx]))
    ensemble_probs = (sage_probs + gat_probs) / 2
    results.append(evaluate_metrics("GNN Ensemble", y_test, ensemble_probs[test_idx]))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    plot_model_comparison(results_df)
