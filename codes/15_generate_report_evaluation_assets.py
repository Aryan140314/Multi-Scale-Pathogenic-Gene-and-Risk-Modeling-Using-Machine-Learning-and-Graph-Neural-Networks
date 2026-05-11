import math
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import add_self_loops, to_undirected


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "codes" / "report_outputs" / "evaluation_assets"

ML_DROP = {"GeneSymbol", "description", "label", "pathogenic_variants", "pathogenic"}
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2)
        self.skip = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        return self.conv2(F.elu(self.ln1(x1)), edge_index)


class GeneGAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 8):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.ln1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, 2, heads=1, concat=False)
        self.skip = torch.nn.Linear(input_dim, hidden_dim * heads)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index) + self.skip(x)
        x1 = F.elu(self.ln1(x1))
        x1 = F.dropout(x1, p=0.4, training=self.training)
        return self.conv2(x1, edge_index)


def make_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        OUTPUT_DIR / "classification_reports",
        OUTPUT_DIR / "confusion_matrices",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    keep = []
    for char in name.lower():
        if char.isalnum():
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def load_features() -> pd.DataFrame:
    features_df = pd.read_csv(DATA_DIR / "final_gene_features.csv")
    if "label" not in features_df.columns and "pathogenic" in features_df.columns:
        features_df["label"] = features_df["pathogenic"]
    if "pathogenic" not in features_df.columns and "label" in features_df.columns:
        features_df["pathogenic"] = features_df["label"]
    return features_df


def load_edges() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "final_edge_list.csv")


def get_split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return train_idx, test_idx


def get_ml_columns(features_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features_df.columns
        if col not in ML_DROP and pd.api.types.is_numeric_dtype(features_df[col])
    ]


def get_gnn_columns(features_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features_df.columns
        if col not in GNN_DROP and pd.api.types.is_numeric_dtype(features_df[col])
    ]


def get_ml_test_frame(
    features_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    ml_cols = get_ml_columns(features_df)
    x_raw = features_df[ml_cols].values.astype(np.float32)
    y = features_df["label"].values.astype(int)

    scaler = StandardScaler()
    scaler.fit(x_raw[train_idx])
    x_test_scaled = scaler.transform(x_raw[test_idx])

    return pd.DataFrame(x_test_scaled, columns=ml_cols), y[test_idx]


def maybe_load_matching_scaler(feature_count: int):
    for scaler_name in ["gnn_feature_scaler.pkl", "feature_scaler.pkl"]:
        scaler_path = MODEL_DIR / scaler_name
        if not scaler_path.exists():
            continue
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", None) == feature_count:
            return scaler
    return None


def build_gnn_graph(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> tuple[Data, np.ndarray, np.ndarray]:
    gnn_cols = get_gnn_columns(features_df)
    x_raw = features_df[gnn_cols].values.astype(np.float32)
    y = features_df["label"].values.astype(int)
    _, test_idx = get_split_indices(y)

    scaler = maybe_load_matching_scaler(len(gnn_cols))
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x_raw)
    x_scaled = scaler.transform(x_raw)

    genes = features_df["GeneSymbol"].tolist()
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    valid_edges = [
        [gene_to_index[row.gene1], gene_to_index[row.gene2]]
        for row in edges_df.itertuples()
        if row.gene1 in gene_to_index and row.gene2 in gene_to_index
    ]

    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(genes))

    graph_data = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
    ).to(DEVICE)

    return graph_data, y[test_idx], test_idx


def load_ml_models() -> dict[str, object]:
    return {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
        "SVM": joblib.load(MODEL_DIR / "svm.pkl"),
        "Random Forest": joblib.load(MODEL_DIR / "random_forest.pkl"),
        "Gradient Boost": joblib.load(MODEL_DIR / "gradient_boost.pkl"),
        "XGBoost": joblib.load(MODEL_DIR / "xgboost.pkl"),
        "Stacking Ensemble": joblib.load(MODEL_DIR / "stacking_ensemble.pkl"),
    }


def load_gnn_models(input_dim: int) -> tuple[GeneSAGE, GeneGAT]:
    sage_model = GeneSAGE(input_dim=input_dim).to(DEVICE)
    sage_model.load_state_dict(torch.load(MODEL_DIR / "gene_gnn_model.pt", map_location=DEVICE))
    sage_model.eval()

    gat_model = GeneGAT(input_dim=input_dim, hidden_dim=128, heads=8).to(DEVICE)
    gat_model.load_state_dict(torch.load(MODEL_DIR / "gene_gat_model.pt", map_location=DEVICE))
    gat_model.eval()

    return sage_model, gat_model


def probability_scores(model, x_test_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test_df)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x_test_df)
        return 1.0 / (1.0 + np.exp(-decision))
    raise ValueError("Model does not expose probability-like scores.")


def optimal_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_model(
    name: str,
    family: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    color: str,
):
    threshold = optimal_f1_threshold(y_true, probs)
    preds = (probs >= threshold).astype(int)

    cm_raw = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm_raw.ravel()
    cm_norm = confusion_matrix(y_true, preds, labels=[0, 1], normalize="true")

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    fpr, tpr, _ = roc_curve(y_true, probs)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    metrics_row = {
        "Model": name,
        "Family": family,
        "Threshold": round(threshold, 4),
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "Balanced Acc": round(balanced_accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "Specificity": round(specificity, 4),
        "NPV": round(npv, 4),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, preds), 4),
        "ROC-AUC": round(roc_auc_score(y_true, probs), 4),
        "PR-AUC": round(average_precision_score(y_true, probs), 4),
        "Brier Score": round(brier_score_loss(y_true, probs), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    report_df = pd.DataFrame(
        classification_report(y_true, preds, output_dict=True, zero_division=0)
    ).transpose()

    return {
        "name": name,
        "family": family,
        "color": color,
        "threshold": threshold,
        "preds": preds,
        "probs": probs,
        "metrics": metrics_row,
        "report_df": report_df,
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "roc": (fpr, tpr),
        "pr": (recall, precision),
        "pr_thresholds": pr_thresholds,
        "calibration": (prob_pred, prob_true),
    }


def save_classification_outputs(results: list[dict]) -> None:
    for result in results:
        stem = sanitize_filename(result["name"])
        result["report_df"].to_csv(
            OUTPUT_DIR / "classification_reports" / f"{stem}_classification_report.csv"
        )
        pd.DataFrame(
            result["cm_raw"],
            index=["Actual Benign", "Actual Pathogenic"],
            columns=["Pred Benign", "Pred Pathogenic"],
        ).to_csv(OUTPUT_DIR / "confusion_matrices" / f"{stem}_confusion_matrix_raw.csv")
        pd.DataFrame(
            result["cm_norm"],
            index=["Actual Benign", "Actual Pathogenic"],
            columns=["Pred Benign", "Pred Pathogenic"],
        ).to_csv(OUTPUT_DIR / "confusion_matrices" / f"{stem}_confusion_matrix_normalized.csv")


def save_metrics_tables(results: list[dict]) -> pd.DataFrame:
    summary_df = pd.DataFrame([result["metrics"] for result in results])
    summary_df = summary_df.sort_values(
        by=["F1-Score", "ROC-AUC", "PR-AUC"], ascending=False
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "all_model_metrics_report.csv", index=False)
    summary_df[summary_df["Family"] == "ML"].to_csv(
        OUTPUT_DIR / "ml_metrics_report.csv",
        index=False,
    )
    summary_df[summary_df["Family"] == "GNN"].to_csv(
        OUTPUT_DIR / "gnn_metrics_report.csv",
        index=False,
    )

    best_by_family = (
        summary_df.sort_values(by=["Family", "F1-Score", "ROC-AUC"], ascending=[True, False, False])
        .groupby("Family", as_index=False)
        .first()
    )
    best_by_family.to_csv(OUTPUT_DIR / "best_model_by_family.csv", index=False)

    return summary_df


def finish_axis_grid(fig, axes, used_count: int) -> None:
    for index in range(used_count, len(axes)):
        fig.delaxes(axes[index])


def save_confusion_grid(results: list[dict], normalized: bool, filename: str, title: str) -> None:
    sns.set_style("white")
    cols = 3
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 4.6))
    axes = np.atleast_1d(axes).flatten()

    for idx, result in enumerate(results):
        data = result["cm_norm"] if normalized else result["cm_raw"]
        fmt = ".2f" if normalized else "d"
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Benign", "Pathogenic"],
            yticklabels=["Benign", "Pathogenic"],
            annot_kws={"size": 11, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{result['name']}\nThreshold={result['threshold']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    finish_axis_grid(fig, axes, len(results))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_curve_plot(
    results: list[dict],
    filename: str,
    title: str,
    kind: str,
) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    for result in results:
        if kind == "roc":
            x_vals, y_vals = result["roc"]
            score = result["metrics"]["ROC-AUC"]
            label = f"{result['name']} (AUC={score:.3f})"
            ax.plot(x_vals, y_vals, lw=2.4, color=result["color"], label=label)
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1)
            ax.set_xlabel("False Positive Rate", fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontweight="bold")
        elif kind == "pr":
            x_vals, y_vals = result["pr"]
            score = result["metrics"]["PR-AUC"]
            label = f"{result['name']} (AP={score:.3f})"
            ax.plot(x_vals, y_vals, lw=2.4, color=result["color"], label=label)
            ax.set_xlabel("Recall", fontweight="bold")
            ax.set_ylabel("Precision", fontweight="bold")
        elif kind == "calibration":
            x_vals, y_vals = result["calibration"]
            score = result["metrics"]["Brier Score"]
            label = f"{result['name']} (Brier={score:.3f})"
            ax.plot(x_vals, y_vals, marker="o", lw=2, color=result["color"], label=label)
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, label="Perfect calibration")
            ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
            ax.set_ylabel("Observed Fraction Positive", fontweight="bold")
        else:
            raise ValueError(f"Unsupported curve kind: {kind}")

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_leaderboards(summary_df: pd.DataFrame) -> None:
    metrics_to_plot = ["F1-Score", "ROC-AUC", "PR-AUC", "Balanced Acc", "MCC"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        ordered = summary_df.sort_values(metric_name, ascending=False)
        sns.barplot(
            data=ordered,
            x=metric_name,
            y="Model",
            hue="Family",
            dodge=False,
            palette={"ML": "#1f77b4", "GNN": "#d62728"},
            ax=ax,
        )
        ax.set_title(metric_name, fontweight="bold")
        ax.set_ylabel("")
        ax.legend_.remove()
        xmax = max(ordered[metric_name].max() + 0.05, 1.0)
        ax.set_xlim(0, min(xmax, 1.05))
        for row_idx, value in enumerate(ordered[metric_name]):
            ax.text(value + 0.01, row_idx, f"{value:.3f}", va="center", fontsize=9)

    axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=8),
        plt.Line2D([0], [0], color="#d62728", lw=8),
    ]
    axes[-1].legend(handles, ["ML", "GNN"], loc="center", frameon=False, fontsize=12)

    fig.suptitle("Model Evaluation Leaderboards", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_metric_leaderboards.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_report_outline() -> None:
    outline = """5.6 Results, Evaluation and Visualization
5.6.1 Experimental Setup and Test Split
5.6.2 Core Performance Metrics
5.6.3 Confusion Matrix Analysis
5.6.4 ROC Curve Analysis
5.6.5 Precision-Recall Curve Analysis
5.6.6 Calibration and Reliability Analysis
5.6.7 Comparative Analysis of ML and GNN Models
5.6.8 Feature and Representation Visualization
5.6.9 Explainability and Biological Interpretation

Recommended figures and tables:
- Table: all_model_metrics_report.csv
- Table: best_model_by_family.csv
- Figure: ml_confusion_matrices_raw.png
- Figure: gnn_confusion_matrices_raw.png
- Figure: all_models_roc_curve.png
- Figure: all_models_pr_curve.png
- Figure: all_models_calibration_curve.png
- Figure: model_metric_leaderboards.png
- Notebook reference: 14a_evaluate_tabular_ml.ipynb for correlation, density, DCA and bootstrap visuals
- Notebook reference: 14b_evaluate_tabular_gnn.ipynb for t-SNE, lift/gains, DCA and bootstrap visuals
- Notebook reference: GNN_explainer.ipynb for interpretability visuals
"""
    (OUTPUT_DIR / "report_section_5_6_outline.md").write_text(outline, encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    make_dirs()

    features_df = load_features()
    edges_df = load_edges()
    y = features_df["label"].values.astype(int)
    train_idx, test_idx = get_split_indices(y)

    x_test_df, y_test_ml = get_ml_test_frame(features_df, train_idx, test_idx)
    ml_models = load_ml_models()
    ml_specs = [
        ("Logistic Regression", "#4C78A8"),
        ("SVM", "#72B7B2"),
        ("Random Forest", "#54A24B"),
        ("Gradient Boost", "#E45756"),
        ("XGBoost", "#F58518"),
        ("Stacking Ensemble", "#B279A2"),
    ]

    ml_results = []
    for model_name, color in ml_specs:
        probs = probability_scores(ml_models[model_name], x_test_df)
        ml_results.append(evaluate_model(model_name, "ML", y_test_ml, probs, color))

    graph_data, y_test_gnn, gnn_test_idx = build_gnn_graph(features_df, edges_df)
    sage_model, gat_model = load_gnn_models(len(get_gnn_columns(features_df)))

    with torch.no_grad():
        sage_probs_full = torch.softmax(sage_model(graph_data), dim=1)[:, 1].cpu().numpy()
        gat_probs_full = torch.softmax(gat_model(graph_data), dim=1)[:, 1].cpu().numpy()

    sage_probs = sage_probs_full[gnn_test_idx]
    gat_probs = gat_probs_full[gnn_test_idx]
    ensemble_probs = (sage_probs + gat_probs) / 2.0

    gnn_results = [
        evaluate_model("GraphSAGE", "GNN", y_test_gnn, sage_probs, "#FF9800"),
        evaluate_model("GATv2", "GNN", y_test_gnn, gat_probs, "#F44336"),
        evaluate_model("GNN Hybrid Ensemble", "GNN", y_test_gnn, ensemble_probs, "#B71C1C"),
    ]

    all_results = ml_results + gnn_results
    save_classification_outputs(all_results)
    summary_df = save_metrics_tables(all_results)

    save_confusion_grid(
        ml_results,
        normalized=False,
        filename="ml_confusion_matrices_raw.png",
        title="ML Confusion Matrices",
    )
    save_confusion_grid(
        gnn_results,
        normalized=False,
        filename="gnn_confusion_matrices_raw.png",
        title="GNN Confusion Matrices",
    )
    save_confusion_grid(
        ml_results,
        normalized=True,
        filename="ml_confusion_matrices_normalized.png",
        title="ML Confusion Matrices (Row-Normalized)",
    )
    save_confusion_grid(
        gnn_results,
        normalized=True,
        filename="gnn_confusion_matrices_normalized.png",
        title="GNN Confusion Matrices (Row-Normalized)",
    )

    save_curve_plot(
        all_results,
        filename="all_models_roc_curve.png",
        title="ROC Curve Comparison Across ML and GNN Models",
        kind="roc",
    )
    save_curve_plot(
        all_results,
        filename="all_models_pr_curve.png",
        title="Precision-Recall Curve Comparison Across ML and GNN Models",
        kind="pr",
    )
    save_curve_plot(
        all_results,
        filename="all_models_calibration_curve.png",
        title="Calibration Curve Comparison Across ML and GNN Models",
        kind="calibration",
    )

    save_metric_leaderboards(summary_df)
    save_report_outline()

    print(f"Evaluation assets saved to: {OUTPUT_DIR}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
