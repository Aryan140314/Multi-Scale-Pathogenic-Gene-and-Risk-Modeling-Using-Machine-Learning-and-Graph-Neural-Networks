"""
PathoGAT — Advanced Gene Pathogenicity Dashboard  (v3)
=======================================================
NEW in v3:
  • Gene Description Card  — full NCBI description for every gene
  • Variant Breakdown Tab  — per-gene SNV / Deletion / Insertion breakdown
                             from clinvar_filtered.csv
  • AI Explainer Mode      — Claude/Gemini API analyses what the risk score means,
                             what causes the risk, what changing a feature
                             does biologically, and what the gene does
  • All sklearn warnings   — suppressed via safe_transform helper
  • use_container_width    — replaced with width='stretch'
"""

import os, warnings
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
from scipy.spatial.distance import cdist

from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.preprocessing import StandardScaler as _SS

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_transform(scaler, X_numpy):
    """Pass numpy → scaler and suppress sklearn feature-name warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(X_numpy, pd.DataFrame):
            X_numpy = X_numpy.values
        return scaler.transform(X_numpy)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
ML_DROP = {
    "GeneSymbol", "description", "label",
    "pathogenic_variants", "pathogenic",
}
GNN_EXTRA_DROP = {
    "neighbor_pathogenic_ratio", "mutation_network_score", "rare_network_score",
    "gene_degree", "clustering_coefficient", "pagerank", "betweenness_centrality",
}
GNN_DROP = ML_DROP | GNN_EXTRA_DROP

# ─────────────────────────────────────────────────────────────────────────────
# GNN MODEL CLASSES  (match training notebooks exactly)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PathoGAT AI Dashboard", page_icon="🧬")
st.markdown(
    "<h1 style='margin-bottom:2px'>🧬 PathoGAT: Multi-Scale Gene Pathogenicity Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:grey;margin-top:0'>5-Model ML Ensemble · GraphSAGE · GATv2 · "
    f"Node2Vec · PPI Network · AI Explainer &nbsp;|&nbsp; Compute: <code>{device}</code></p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_datasets():
    base  = os.path.dirname(os.path.abspath(__file__))
    proc  = os.path.join(base, "..", "data", "processed")
    feat  = pd.read_csv(os.path.join(proc, "final_gene_features.csv"))
    edges = pd.read_csv(os.path.join(proc, "final_edge_list.csv"))
    if "pathogenic" not in feat.columns:
        feat["pathogenic"] = feat["label"]
    return feat, edges

@st.cache_data
def load_clinvar():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "data", "processed", "clinvar_filtered.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_gene_info():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "data", "processed", "gene_info_filtered.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Symbol": "GeneSymbol"}, inplace=True)
    df["GeneSymbol"] = df["GeneSymbol"].str.upper()
    return df

@st.cache_data
def build_clinvar_lookup(_clinvar_df):
    """Per-gene variant breakdown from ClinVar."""
    if _clinvar_df.empty:
        return pd.DataFrame()
    g = _clinvar_df.groupby("GeneSymbol").agg(
        total_variants=("VariationID", "count"),
        pathogenic=("ClinicalSignificance",
                    lambda x: x.str.contains("Pathogenic", na=False).sum()),
        likely_pathogenic=("ClinicalSignificance",
                           lambda x: x.str.contains("Likely pathogenic", na=False).sum()),
        benign=("ClinicalSignificance",
                lambda x: x.str.contains("Benign", na=False, regex=False).sum()),
        likely_benign=("ClinicalSignificance",
                       lambda x: x.str.contains("Likely benign", na=False).sum()),
        snv=("Type", lambda x: (x == "single nucleotide variant").sum()),
        deletion=("Type", lambda x: (x == "Deletion").sum()),
        insertion=("Type", lambda x: (x == "Insertion").sum()),
        duplication=("Type", lambda x: (x == "Duplication").sum()),
        indel=("Type", lambda x: (x == "Indel").sum()),
        n_chromosomes=("Chromosome", "nunique"),
    ).reset_index()
    return g

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(gnn_dim: int):
    base = os.path.dirname(os.path.abspath(__file__))
    mdir = os.path.join(base, "..", "models")
    mdls, miss = {}, []

    for name, fname in {
        "RandomForest":       "random_forest.pkl",
        "XGBoost":            "xgboost.pkl",
        "GradientBoost":      "gradient_boost.pkl",
        "SVM":                "svm.pkl",
        "LogisticRegression": "logistic_regression.pkl",
        "StackingEnsemble":   "stacking_ensemble.pkl",
    }.items():
        p = os.path.join(mdir, fname)
        if os.path.exists(p):
            mdls[name] = joblib.load(p)
        else:
            miss.append(fname)

    def _sc(names):
        for n in names:
            p = os.path.join(mdir, n)
            if os.path.exists(p):
                return joblib.load(p)
        return None

    ml_sc  = _sc(["ml_feature_scaler.pkl", "feature_scaler.pkl"])
    gnn_sc = _sc(["gnn_feature_scaler.pkl", "feature_scaler.pkl"])
    if ml_sc is None:
        miss.append("feature_scaler.pkl")

    sage = GeneSAGE(input_dim=gnn_dim).to(device)
    sp   = os.path.join(mdir, "gene_gnn_model.pt")
    if os.path.exists(sp):
        sage.load_state_dict(torch.load(sp, map_location=device))
    else:
        miss.append("gene_gnn_model.pt")
    sage.eval()

    gat  = GeneGAT(input_dim=gnn_dim, hidden_dim=128, heads=8).to(device)
    gp   = os.path.join(mdir, "gene_gat_model.pt")
    if os.path.exists(gp):
        gat.load_state_dict(torch.load(gp, map_location=device))
    else:
        miss.append("gene_gat_model.pt")
    gat.eval()

    return mdls, ml_sc, gnn_sc, sage, gat, miss

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_pipeline(_fdf, _edf):
    ml_c  = [c for c in _fdf.columns if c not in ML_DROP
             and pd.api.types.is_numeric_dtype(_fdf[c])]
    gnn_c = [c for c in ml_c if c not in GNN_EXTRA_DROP]

    X_ml  = _fdf[ml_c].values.astype(np.float32)
    X_gnn = _fdf[gnn_c].values.astype(np.float32)

    gl  = _fdf["GeneSymbol"].tolist()
    g2i = {g: i for i, g in enumerate(gl)}

    valid = [[g2i[r.gene1], g2i[r.gene2]]
             for r in _edf.itertuples()
             if r.gene1 in g2i and r.gene2 in g2i]
    ei = torch.tensor(valid, dtype=torch.long).t().contiguous()
    ei = to_undirected(ei)
    ei, _ = add_self_loops(ei, num_nodes=len(gl))
    return X_ml, X_gnn, ml_c, gnn_c, gl, g2i, ei

# ─────────────────────────────────────────────────────────────────────────────
# SCALER OPTIMIZATION (Prevents UI Lag)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_scaler(sc, X):
    """Force a fresh scaler to guarantee compatibility with Sklearn 1.8.0 and prevent ValueErrors"""
    s = _SS()
    if isinstance(X, pd.DataFrame):
        s.fit(X.values)
    else:
        s.fit(X)
    return s

@st.cache_resource
def get_scaled_data(_ml_sc, _gnn_sc, _X_ml_raw, _X_gnn_raw):
    """Caches the heavy scaling operations so it doesn't run on every slider change."""
    m_use = _safe_scaler(_ml_sc, _X_ml_raw)
    g_use = _safe_scaler(_gnn_sc, _X_gnn_raw)
    return m_use, g_use, safe_transform(m_use, _X_ml_raw), safe_transform(g_use, _X_gnn_raw)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
try:
    features_df, edges_df = load_datasets()
    clinvar_df             = load_clinvar()
    gene_info_df           = load_gene_info()
    clinvar_lookup         = build_clinvar_lookup(clinvar_df)

    X_ml_raw, X_gnn_raw, ml_cols, gnn_cols, gene_list, gene_to_idx, edge_index = build_pipeline(
        features_df, edges_df
    )
    ml_models, ml_sc, gnn_sc, sage_model, gat_model, missing_files = load_models(
        gnn_dim=len(gnn_cols)
    )

    # Fetch pre-calculated scaled arrays to fix UI lag
    ml_sc_use, gnn_sc_use, X_ml_scaled_all, X_gnn_scaled_all = get_scaled_data(
        ml_sc, gnn_sc, X_ml_raw, X_gnn_raw
    )

    if missing_files:
        st.warning("⚠️ Missing model files in `../models/`: " +
                   ", ".join(f"`{f}`" for f in missing_files))
except Exception as e:
    st.error("🚨 Startup failed while loading data/models. Please check the deployment logs.")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(idx, user_inputs):
    ml_raw = X_ml_raw[idx].copy()
    for f, v in user_inputs.items():
        if f in ml_cols:
            ml_raw[ml_cols.index(f)] = v
    ml_sc_arr = safe_transform(ml_sc_use, ml_raw.reshape(1, -1))
    ml_df     = pd.DataFrame(ml_sc_arr, columns=ml_cols)

    ind = {}
    for name, m in ml_models.items():
        if name == "StackingEnsemble": continue
        try:
            ind[name] = (float(m.predict_proba(ml_df)[0][1]) if hasattr(m, "predict_proba")
                         else float(1 / (1 + np.exp(-m.decision_function(ml_df)[0]))))
        except Exception:
            ind[name] = 0.5

    ml_ens = (float(ml_models["StackingEnsemble"].predict_proba(ml_df)[0][1])
              if "StackingEnsemble" in ml_models
              else float(np.mean(list(ind.values()))) if ind else 0.5)

    gnn_raw = X_gnn_raw[idx].copy()
    for f, v in user_inputs.items():
        if f in gnn_cols:
            gnn_raw[gnn_cols.index(f)] = v
    gnn_node = safe_transform(gnn_sc_use, gnn_raw.reshape(1, -1))[0]
    X_live   = X_gnn_scaled_all.copy()
    X_live[idx] = gnn_node

    with torch.no_grad():
        gd     = Data(x=torch.tensor(X_live, dtype=torch.float),
                      edge_index=edge_index).to(device)
        sage_p = float(torch.softmax(sage_model(gd), 1)[idx][1])
        gat_p  = float(torch.softmax(gat_model(gd),  1)[idx][1])

    gnn_ens = (sage_p + gat_p) / 2
    final   = ml_ens * 0.4 + gnn_ens * 0.6
    return ind, ml_ens, sage_p, gat_p, gnn_ens, final, ml_df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Gene Configuration")
selected_gene = st.sidebar.selectbox("Select Gene", sorted(gene_list))
selected_idx  = gene_to_idx[selected_gene]
gene_row      = features_df.iloc[selected_idx]
gene_label    = int(gene_row["label"])

# Gene description from gene_info_filtered.csv
gene_desc = "Description not available."
if not gene_info_df.empty:
    match = gene_info_df[gene_info_df["GeneSymbol"] == selected_gene.upper()]
    if not match.empty:
        gene_desc = str(match.iloc[0].get("description", "N/A"))
# Also check final_gene_features description column
if "description" in features_df.columns:
    d = str(gene_row.get("description", ""))
    if d and d != "nan" and len(d) > 3:
        gene_desc = d

st.sidebar.markdown(
    f"<div style='background:#1e2a3a;padding:10px;border-radius:8px;margin-bottom:8px'>"
    f"<b style='color:#90caf9'>{selected_gene}</b><br>"
    f"<span style='color:#cfd8dc;font-size:12px'>{gene_desc}</span>"
    f"</div>",
    unsafe_allow_html=True
)
st.sidebar.info(
    f"**Label:** {'🔴 Pathogenic' if gene_label==1 else '🟢 Benign'}\n\n"
    f"**Total variants:** {int(gene_row.get('total_variants', 0))}\n\n"
    f"**Degree:** {gene_row.get('gene_degree', 'N/A')}"
)

st.sidebar.markdown("---")
st.sidebar.header("✏️ What-If Editor")
editable = st.sidebar.multiselect(
    "Features to modify:", sorted(ml_cols),
    default=[f for f in ["total_variants", "benign_variants", "rare_variants",
                         "gene_degree", "clustering_coefficient"] if f in ml_cols]
)
user_inputs = {f: st.sidebar.number_input(f, value=float(gene_row[f]), format="%.4f")
               for f in editable}

# ─────────────────────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
ind_probs, ml_ens, sage_p, gat_p, gnn_ens, final_prob, ml_df_node = run_inference(
    selected_idx, user_inputs
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRIC BAR
# ─────────────────────────────────────────────────────────────────────────────
risk_icon = "🔴" if final_prob > 0.5 else "🟢"
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{risk_icon} Final Consensus", f"{final_prob:.3f}")
c2.metric("📊 ML Ensemble",               f"{ml_ens:.3f}")
c3.metric("🕸️ GNN Consensus",            f"{gnn_ens:.3f}")
c4.metric("🌿 GraphSAGE",                f"{sage_p:.3f}")
c5.metric("⚡ PathoGAT (GAT)",           f"{gat_p:.3f}")
st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🤖 AI Explainer",
    "🧬 Gene Profile",
    "🔬 Variant Breakdown",
    "🌐 PPI Network",
    "🧠 Explainability",
    "🔭 Embedding Space",
    "📈 Network Topology",
    "⚖️ Gene Comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🎯 Risk Gauge")
        bar_col = "#c62828" if final_prob > 0.5 else "#2e7d32"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=final_prob,
            number={"font": {"size": 44}, "valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar":  {"color": bar_col},
                "steps": [{"range": [0.0, 0.3], "color": "#e8f5e9"},
                           {"range": [0.3, 0.7], "color": "#fff9c4"},
                           {"range": [0.7, 1.0], "color": "#ffebee"}],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 0.5},
            }
        ))
        fig_g.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g, width="stretch")

    with col_r:
        st.subheader("📊 Model Comparison")
        bnames = list(ind_probs.keys()) + ["ML Ens.", "GraphSAGE", "PathoGAT", "GNN Avg", "▶ FINAL"]
        bprobs = list(ind_probs.values()) + [ml_ens, sage_p, gat_p, gnn_ens, final_prob]
        bcols  = (["#1565C0"] * len(ind_probs) +
                  ["#E65100", "#6A1B9A", "#6A1B9A", "#F9A825",
                   "#c62828" if final_prob > 0.5 else "#2e7d32"])
        fig_b = go.Figure(go.Bar(
            x=bprobs[::-1], y=bnames[::-1], orientation="h",
            marker_color=bcols[::-1],
            text=[f"{p:.3f}" for p in bprobs[::-1]], textposition="auto"
        ))
        fig_b.add_vline(x=0.5, line_dash="dash", line_color="red",
                        annotation_text="0.5 threshold")
        fig_b.update_layout(xaxis=dict(range=[0, 1]), height=360,
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_b, width="stretch")

    st.markdown("---")
    st.subheader("📋 Gene Feature Summary")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Mutation Burden**")
        mc = ["total_variants","benign_variants","rare_variants",
              "variant_type_diversity","chromosome_diversity","unique_variant_count"]
        st.dataframe(
            pd.DataFrame({c: [round(float(gene_row.get(c,0)), 4)]
                          for c in mc if c in features_df.columns}).T.rename(columns={0:"Value"}),
            width="stretch"
        )
    with p2:
        st.markdown("**Network Topology**")
        tc = ["gene_degree","clustering_coefficient","pagerank",
              "betweenness_centrality","neighbor_pathogenic_ratio",
              "mutation_network_score","rare_network_score"]
        st.dataframe(
            pd.DataFrame({c: [round(float(gene_row.get(c, 0)), 6)]
                          for c in tc if c in features_df.columns}).T.rename(columns={0:"Value"}),
            width="stretch"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🤖 AI Gene Explainer")
    st.markdown(
        "This mode uses the **Anthropic Claude API** to explain your gene's risk score "
        "in plain language — what the gene does, why it is flagged as high/low risk, "
        "and what happens when features change."
    )

    # Collect SHAP values for context
    shap_context = ""
    xgb_m = ml_models.get("XGBoost")
    if xgb_m and hasattr(xgb_m, "predict_proba"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp  = shap.TreeExplainer(xgb_m)
                svs  = exp.shap_values(ml_df_node)
            sv = svs[1][0] if isinstance(svs, list) else (svs[0] if svs.ndim == 2 else svs)
            shap_df = (pd.DataFrame({"Feature": ml_cols, "SHAP": sv})
                       .assign(Abs=lambda d: d.SHAP.abs())
                       .sort_values("Abs", ascending=False).head(10))
            lines = []
            for _, r in shap_df.iterrows():
                direction = "↑ increases" if r.SHAP > 0 else "↓ decreases"
                lines.append(f"  - {r.Feature} = {float(gene_row.get(r.Feature, 0)):.4f} "
                             f"({direction} pathogenicity risk by {abs(r.SHAP):.4f})")
            shap_context = "\n".join(lines)
        except Exception:
            shap_context = "  (SHAP not available)"

    # Neighbour context
    local_e = edges_df[
        (edges_df["gene1"] == selected_gene) | (edges_df["gene2"] == selected_gene)
    ].head(50)
    nbr_genes  = (set(local_e["gene1"]) | set(local_e["gene2"])) - {selected_gene}
    nbr_sub    = features_df[features_df["GeneSymbol"].isin(nbr_genes)]
    n_path_nbr = int((nbr_sub["label"] == 1).sum()) if len(nbr_sub) > 0 else 0

    # clinvar context
    cl_row = (clinvar_lookup[clinvar_lookup["GeneSymbol"] == selected_gene].iloc[0]
              if not clinvar_lookup.empty and selected_gene in clinvar_lookup["GeneSymbol"].values
              else None)
    clinvar_ctx = ""
    if cl_row is not None:
        clinvar_ctx = (
            f"ClinVar records for this gene: {int(cl_row['total_variants'])} total variants "
            f"({int(cl_row['pathogenic'])} pathogenic, {int(cl_row['likely_pathogenic'])} likely pathogenic, "
            f"{int(cl_row['benign'])} benign). "
            f"Variant types: {int(cl_row['snv'])} SNVs, {int(cl_row['deletion'])} deletions, "
            f"{int(cl_row['insertion'])} insertions."
        )

    # What-if context
    whatif_ctx = ""
    if user_inputs:
        changes = []
        for f, v in user_inputs.items():
            orig = float(gene_row.get(f, 0))
            delta = v - orig
            if abs(delta) > 1e-6:
                changes.append(f"{f}: {orig:.2f} → {v:.2f} (change: {delta:+.2f})")
        if changes:
            whatif_ctx = "User has modified features:\n" + "\n".join(f"  - {c}" for c in changes)

    # Prompt assembly
    prompt_template = f"""You are a senior computational biologist and genomics expert analysing a gene pathogenicity prediction system called PathoGAT.

Gene being analysed: {selected_gene}
Gene description (NCBI): {gene_desc}
Ground-truth clinical label: {"PATHOGENIC" if gene_label == 1 else "BENIGN"}

PathoGAT Risk Scores:
  - Final Consensus Risk: {final_prob:.3f} ({"HIGH RISK" if final_prob > 0.5 else "LOW RISK"})
  - ML Ensemble: {ml_ens:.3f}
  - GraphSAGE: {sage_p:.3f}
  - PathoGAT (GAT): {gat_p:.3f}
  - GNN Average: {gnn_ens:.3f}

Top SHAP feature contributions (XGBoost):
{shap_context}

Protein-protein interaction context:
  - {selected_gene} has {len(nbr_genes)} direct PPI neighbours
  - {n_path_nbr} of those neighbours are labelled Pathogenic

ClinVar variant data:
  {clinvar_ctx}

{whatif_ctx}
"""

    explain_mode = st.selectbox(
        "What do you want explained?",
        [
            "🔍 Why is this gene flagged as high/low risk?",
            "🧬 What does this gene do biologically?",
            "📊 What do the SHAP features mean for this gene?",
            "🔄 What happens biologically when I change the selected features?",
            "🌐 What does the gene's network neighbourhood tell us?",
            "⚗️ What are the clinical implications of this risk score?",
            "📝 Full Summary Report",
        ],
        key="explain_mode"
    )

    mode_instructions = {
        "🔍 Why is this gene flagged as high/low risk?":
            "Explain in 3–5 paragraphs why PathoGAT gives this gene a risk score of "
            f"{final_prob:.3f}. Focus on the most important contributing features, "
            "the network neighbourhood, and how they relate to pathogenicity. "
            "Compare to the ground-truth label.",

        "🧬 What does this gene do biologically?":
            f"Based on the gene description '{gene_desc}', explain in clear language "
            f"what {selected_gene} does in the human body. Describe its biological pathway, "
            "known disease associations, and why mutations in this gene can cause disease.",

        "📊 What do the SHAP features mean for this gene?":
            "Explain what each top SHAP feature means biologically for this gene. "
            "For each feature, explain: what it measures, why it drives the score up or down, "
            "and what a researcher should pay attention to.",

        "🔄 What happens biologically when I change the selected features?":
            f"The user has modified gene features. {whatif_ctx if whatif_ctx else 'No features changed yet — explain what each editable feature represents biologically.'} "
            "Explain what each change means biologically: why increasing/decreasing this feature "
            "would affect pathogenicity, what cellular processes are involved.",

        "🌐 What does the gene's network neighbourhood tell us?":
            f"{selected_gene} has {len(nbr_genes)} PPI neighbours of which {n_path_nbr} are pathogenic. "
            "Explain the 'guilt by association' principle in genomics. What does it mean when a gene "
            "is highly connected to pathogenic genes? How does the GNN use this to compute risk?",

        "⚗️ What are the clinical implications of this risk score?":
            f"Risk score: {final_prob:.3f}. Ground truth: {'Pathogenic' if gene_label==1 else 'Benign'}. "
            "Explain the clinical implications of this score. What should a clinician do with this information? "
            "What are the limitations? How should this AI prediction be used alongside traditional diagnostics?",

        "📝 Full Summary Report":
            f"Write a comprehensive 6-section genomics report about {selected_gene}: "
            "(1) Gene function and biology, (2) Risk assessment rationale, "
            "(3) Key driving features, (4) Network context, "
            "(5) ClinVar variant landscape, (6) Clinical implications and caveats. "
            "Use professional but accessible language suitable for a clinical researcher.",
    }

    if st.button("🚀 Generate AI Explanation", type="primary", key="ai_btn"):
        full_prompt = prompt_template + "\n\nTask:\n" + mode_instructions[explain_mode]

        try:
            import google.generativeai as genai
            import os
            
            # Retrieve API key securely from Streamlit secrets or environment
            api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
            
            if not api_key:
                st.error("⚠️ Gemini API Key missing! Please add GEMINI_API_KEY to `.streamlit/secrets.toml`")
            else:
                genai.configure(api_key=api_key)
                # Using Gemini 1.5 Pro for advanced biological reasoning
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                with st.spinner("Gemini is analysing your gene..."):
                    response = model.generate_content(full_prompt)
                    explanation = response.text
                
                st.markdown(
                    f"<div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;"
                    f"padding:20px;line-height:1.7;color:#e0e0e0'>{explanation}</div>",
                    unsafe_allow_html=True
                )
        except ImportError:
            st.error("Install the Google Generative AI SDK: `pip install google-generativeai`")
        except Exception as e:
            st.error(f"API error: {e}")
            # Fallback: show the rich prompt so user can paste it manually
            with st.expander("📋 Copy this prompt to any AI chatbot"):
                st.text_area("Prompt", full_prompt + "\n\nTask:\n" + mode_instructions[explain_mode],
                             height=300, key="prompt_copy")

    # Always show the context being used
    with st.expander("📋 See the full context sent to AI"):
        st.text(prompt_template)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENE PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader(f"🧬 Gene Profile: {selected_gene}")

    # Description card
    st.markdown(
        f"<div style='background:#1a237e;border-radius:8px;padding:16px;margin-bottom:16px'>"
        f"<h3 style='color:#e8eaf6;margin:0'>{selected_gene}</h3>"
        f"<p style='color:#c5cae9;margin:6px 0 0'>{gene_desc}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Risk Score",     f"{final_prob:.3f}")
    gc2.metric("Ground Truth",   "Pathogenic" if gene_label == 1 else "Benign")
    deg = gene_row.get("gene_degree", 0)
    gc3.metric("PPI Degree",     f"{int(deg)}")

    st.markdown("---")
    st.subheader("📊 Feature Profile vs Population")

    compare_features = [c for c in
                        ["total_variants", "benign_variants", "rare_variants",
                         "gene_degree", "clustering_coefficient", "pagerank",
                         "betweenness_centrality", "neighbor_pathogenic_ratio",
                         "mutation_network_score", "rare_network_score"]
                        if c in features_df.columns]

    pct_data = []
    for f in compare_features:
        v   = float(gene_row.get(f, 0))
        col = features_df[f].dropna()
        pct = float((col < v).mean() * 100)
        med = float(col.median())
        pct_data.append({"Feature": f, "Gene Value": round(v, 4),
                         "Population Median": round(med, 4),
                         "Percentile": round(pct, 1)})
    pct_df = pd.DataFrame(pct_data)
    st.dataframe(pct_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("🎚️ Percentile Radar")
    if len(compare_features) >= 4:
        n_vals = [float((features_df[f].dropna() < float(gene_row.get(f, 0))).mean())
                  for f in compare_features]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=n_vals + [n_vals[0]],
            theta=compare_features + [compare_features[0]],
            fill="toself", fillcolor="rgba(21,101,192,0.2)",
            line=dict(color="#1565C0"), name=selected_gene
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[0.5] * (len(compare_features) + 1),
            theta=compare_features + [compare_features[0]],
            mode="lines", line=dict(color="#888", dash="dot"), name="50th pct"
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=420, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_r, width="stretch")

    st.markdown("---")
    st.subheader("🔍 Similar Genes")
    n2v_c = [c for c in gnn_cols if c.startswith("node2vec_")]
    if len(n2v_c) >= 2:
        k = st.slider("Number of similar genes", 5, 25, 10, key="sim_k")
        emb_m   = features_df[n2v_c].values.astype(np.float32)
        dists   = cdist(emb_m[selected_idx].reshape(1, -1), emb_m, metric="cosine")[0]
        dists[selected_idx] = 999
        sim_idx = np.argsort(dists)[:k]
        sim_df  = features_df.iloc[sim_idx][["GeneSymbol", "label", "gene_degree",
                                               "total_variants", "description"]].copy()
        sim_df["Cosine Sim"] = (1 - dists[sim_idx]).round(4)
        sim_df["Type"]       = sim_df["label"].map({0: "Benign", 1: "Pathogenic"})
        fig_sim = px.bar(sim_df, x="GeneSymbol", y="Cosine Sim", color="Type",
                         color_discrete_map={"Pathogenic": "#c62828", "Benign": "#1565C0"},
                         title=f"Top {k} similar genes to {selected_gene}")
        fig_sim.update_layout(height=300, margin=dict(t=40, b=60), xaxis_tickangle=-30)
        st.plotly_chart(fig_sim, width="stretch")
        st.dataframe(sim_df.drop(columns=["label"]), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VARIANT BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader(f"🔬 ClinVar Variant Breakdown: {selected_gene}")

    if not clinvar_lookup.empty and selected_gene in clinvar_lookup["GeneSymbol"].values:
        cr = clinvar_lookup[clinvar_lookup["GeneSymbol"] == selected_gene].iloc[0]

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Total Variants",       int(cr["total_variants"]))
        v2.metric("Pathogenic",           int(cr["pathogenic"]))
        v3.metric("Likely Pathogenic",    int(cr["likely_pathogenic"]))
        v4.metric("Benign + Likely Benign", int(cr["benign"]) + int(cr["likely_benign"]))

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Clinical Significance Distribution**")
            clin_labels = ["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"]
            clin_vals   = [int(cr["pathogenic"]), int(cr["likely_pathogenic"]),
                           int(cr["benign"]), int(cr["likely_benign"])]
            fig_cs = go.Figure(go.Pie(
                labels=clin_labels, values=clin_vals, hole=0.4,
                marker_colors=["#c62828", "#EF5350", "#1565C0", "#42A5F5"],
                textinfo="percent+label"
            ))
            fig_cs.update_layout(height=320, margin=dict(t=20, b=10))
            st.plotly_chart(fig_cs, width="stretch")

        with col_b:
            st.markdown("**Variant Type Distribution**")
            vtype_labels = ["SNV", "Deletion", "Insertion", "Duplication", "Indel"]
            vtype_vals   = [int(cr["snv"]), int(cr["deletion"]), int(cr["insertion"]),
                            int(cr["duplication"]), int(cr["indel"])]
            fig_vt = go.Figure(go.Bar(
                x=vtype_labels, y=vtype_vals,
                marker_color=["#1565C0", "#c62828", "#2e7d32", "#F57C00", "#7B1FA2"],
                text=vtype_vals, textposition="auto"
            ))
            fig_vt.update_layout(height=320, margin=dict(t=20, b=10))
            st.plotly_chart(fig_vt, width="stretch")

        st.markdown("---")
        # Actual variants table for this gene
        st.subheader("📋 Individual Variant Records")
        if not clinvar_df.empty:
            gene_variants = clinvar_df[clinvar_df["GeneSymbol"] == selected_gene].copy()
            gene_variants = gene_variants[["VariationID", "Chromosome", "Start",
                                           "ClinicalSignificance", "Type"]].head(200)
            gene_variants.columns = gene_variants.columns.str.strip()

            # Filter controls
            fc1, fc2 = st.columns(2)
            sig_filter  = fc1.multiselect(
                "Filter by significance:",
                options=["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"],
                default=["Pathogenic", "Likely pathogenic"],
                key="sig_filter"
            )
            type_filter = fc2.multiselect(
                "Filter by type:",
                options=list(clinvar_df["Type"].dropna().unique()),
                default=[],
                key="type_filter"
            )
            if sig_filter:
                gene_variants = gene_variants[
                    gene_variants["ClinicalSignificance"].isin(sig_filter)
                ]
            if type_filter:
                gene_variants = gene_variants[gene_variants["Type"].isin(type_filter)]

            st.dataframe(gene_variants, width="stretch", hide_index=True)
            st.caption(f"Showing {len(gene_variants)} variants (max 200 loaded).")

        # Compare vs population
        st.markdown("---")
        st.subheader("📊 Pathogenic Variant Ratio vs Population")
        if "pathogenic_variants" in features_df.columns:
            feat_ratio = (features_df["pathogenic_variants"].fillna(0) /
                          (features_df["total_variants"].fillna(1) + 1e-6))
            sel_ratio  = (float(gene_row.get("pathogenic_variants", 0)) /
                          (float(gene_row.get("total_variants", 1)) + 1e-6))
            pct_r = float((feat_ratio < sel_ratio).mean() * 100)
            fig_pr = px.histogram(
                x=feat_ratio, nbins=60, color_discrete_sequence=["#1565C0"],
                labels={"x": "Pathogenic variant ratio"},
                title=f"Population distribution — {selected_gene} at {pct_r:.1f}th pct"
            )
            fig_pr.add_vline(x=sel_ratio, line_dash="dash", line_color="orange",
                             annotation_text=f"{selected_gene}: {sel_ratio:.3f}")
            fig_pr.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pr, width="stretch")
    else:
        st.info(f"No ClinVar records found for **{selected_gene}** in the processed dataset.")
        if clinvar_df.empty:
            st.warning("clinvar_filtered.csv not found at `../data/processed/clinvar_filtered.csv`.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PPI NETWORK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🌐 Protein-Protein Interaction Network")
    nc1, nc2, nc3, nc4 = st.columns(4)
    ppi_theme   = nc1.radio("Theme",    ["Dark", "Light"], horizontal=True, key="ppi_t")
    ppi_lim     = nc2.slider("Max edges", 10, 200, 80, key="ppi_l")
    ppi_phys    = nc3.toggle("Physics", True, key="ppi_p")
    ppi_risk_cl = nc4.toggle("Colour by risk", False, key="ppi_r")

    bg_c  = "#0e1117" if ppi_theme == "Dark" else "#ffffff"
    txt_c = "white"   if ppi_theme == "Dark" else "black"

    local_e   = edges_df[(edges_df["gene1"]==selected_gene)|(edges_df["gene2"]==selected_gene)].head(ppi_lim)
    nbr_genes = (set(local_e["gene1"]) | set(local_e["gene2"])) - {selected_gene}
    nbr_df    = features_df[features_df["GeneSymbol"].isin(nbr_genes)]

    if not local_e.empty:
        n_path_n = int((nbr_df["label"]==1).sum())
        avg_deg  = float(nbr_df["gene_degree"].mean()) if "gene_degree" in nbr_df.columns else 0
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Nodes", len(nbr_genes)+1)
        m2.metric("Edges", len(local_e))
        m3.metric("Pathogenic neighbours", n_path_n)
        m4.metric("Avg neighbour degree", f"{avg_deg:.1f}")

        risk_lk = {}
        if ppi_risk_cl and ml_models.get("XGBoost"):
            vis_idx = [gene_to_idx[g] for g in list(nbr_genes)+[selected_gene] if g in gene_to_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _p = ml_models["XGBoost"].predict_proba(
                    pd.DataFrame(X_ml_scaled_all[vis_idx], columns=ml_cols)
                )[:, 1]
            for g, p in zip([gene_list[i] for i in vis_idx], _p):
                risk_lk[g] = float(p)

        net = Network(height="640px", width="100%", bgcolor=bg_c,
                      font_color=txt_c, notebook=False)
        if ppi_phys:
            net.barnes_hut(gravity=-8000, spring_length=180)
        net.set_edge_smooth("dynamic")

        for gene in list(nbr_genes) + [selected_gene]:
            grow    = features_df[features_df["GeneSymbol"] == gene]
            is_sel  = gene == selected_gene
            is_path = bool(grow["label"].values[0]) if len(grow) > 0 else False
            rv      = risk_lk.get(gene, 0.5)
            desc_g  = str(grow["description"].values[0]) if len(grow) > 0 and "description" in grow.columns else ""

            if is_sel:
                color, sz, bw = "#ff4b4b", 50, 4
            elif ppi_risk_cl:
                r = int(min(255, 57 + 198 * rv))
                b = int(min(255, 101 + 50 * (1 - rv)))
                color = f"rgb({r},50,{b})"
                sz, bw = 22, 1
            elif is_path:
                color, sz, bw = "#FF6F00", 24, 2
            else:
                color, sz, bw = "#1565C0", 18, 1

            deg_g = float(grow["gene_degree"].values[0]) if len(grow)>0 and "gene_degree" in grow.columns else "?"
            ttip  = (f"<b>{gene}</b><br>{'Pathogenic' if is_path else 'Benign'}<br>"
                     f"Degree: {deg_g}<br>Risk: {rv:.3f}<br>"
                     f"<i>{desc_g[:60]}{'...' if len(desc_g)>60 else ''}</i>")
            net.add_node(gene, label=gene, color=color, size=sz, borderWidth=bw, title=ttip)

        for _, r in local_e.iterrows():
            net.add_edge(r["gene1"], r["gene2"], color="#444", width=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
            net.save_graph(tmp.name)
            html_c = open(tmp.name, "r", encoding="utf-8").read()
        components.html(html_c, height=650)
        st.caption("🔴 Selected gene · 🟠 Pathogenic neighbour · 🔵 Benign neighbour")

        st.markdown("---")
        st.subheader("📊 Neighbourhood Analysis")
        na1, na2 = st.columns(2)
        with na1:
            fig_pie = go.Figure(go.Pie(
                labels=["Pathogenic", "Benign"],
                values=[n_path_n, len(nbr_genes)-n_path_n],
                marker_colors=["#c62828", "#1565C0"], hole=0.45
            ))
            fig_pie.update_layout(title="Neighbour label split", height=300,
                                  margin=dict(t=40,b=10))
            st.plotly_chart(fig_pie, width="stretch")
        with na2:
            if "gene_degree" in nbr_df.columns:
                top10 = nbr_df.nlargest(10, "gene_degree")[
                    ["GeneSymbol", "gene_degree", "label", "description"]].copy()
                top10["Type"] = top10["label"].map({0:"Benign",1:"Pathogenic"})
                fig_nb = px.bar(top10, x="gene_degree", y="GeneSymbol", orientation="h",
                                color="Type",
                                color_discrete_map={"Pathogenic":"#c62828","Benign":"#1565C0"},
                                title="Top 10 neighbours by degree")
                fig_nb.update_layout(height=300, margin=dict(t=40,b=10))
                st.plotly_chart(fig_nb, width="stretch")
    else:
        st.info(f"No PPI interactions found for **{selected_gene}**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🧠 SHAP Feature Importance (XGBoost)")
    if xgb_m and hasattr(xgb_m, "predict_proba"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp  = shap.TreeExplainer(xgb_m)
                svs  = exp.shap_values(ml_df_node)
            sv = svs[1][0] if isinstance(svs, list) else (svs[0] if svs.ndim==2 else svs)
            sd = (pd.DataFrame({"Feature": ml_cols, "SHAP": sv})
                  .assign(Abs=lambda d: d.SHAP.abs())
                  .sort_values("Abs", ascending=False).head(20))
            fig_sh = go.Figure(go.Bar(
                x=sd["SHAP"], y=sd["Feature"], orientation="h",
                marker_color=["#c62828" if v>0 else "#1565C0" for v in sd["SHAP"]],
                text=[f"{v:+.4f}" for v in sd["SHAP"]], textposition="auto"
            ))
            fig_sh.add_vline(x=0, line_color="black", line_width=1)
            fig_sh.update_layout(
                title=f"SHAP for {selected_gene} — red = pushes toward Pathogenic",
                height=520, margin=dict(l=10,r=10,t=50,b=10)
            )
            st.plotly_chart(fig_sh, width="stretch")
        except Exception as e:
            st.warning(f"SHAP failed: {e}")
    else:
        st.info("XGBoost model required for SHAP.")

    st.markdown("---")
    st.subheader("🤝 Model Agreement Matrix")
    amd = {**ind_probs, "ML Ens.":ml_ens, "SAGE":sage_p, "GAT":gat_p,
           "GNN Avg":gnn_ens, "Final":final_prob}
    amn = list(amd.keys())
    amv = np.array(list(amd.values()))
    ag  = np.array([[1-abs(amv[i]-amv[j]) for j in range(len(amv))] for i in range(len(amv))])
    fig_hm = px.imshow(ag, x=amn, y=amn, color_continuous_scale="RdYlGn",
                       zmin=0, zmax=1, text_auto=".2f",
                       title="Model agreement (1.0 = identical)")
    fig_hm.update_layout(height=400, margin=dict(t=50,b=10))
    st.plotly_chart(fig_hm, width="stretch")

    st.markdown("---")
    st.subheader("📊 Sensitivity Analysis")
    sens_feats = [c for c in ml_cols[:20] if c not in {"variant_type_diversity"}]
    feat_s = st.selectbox("Sweep:", sens_feats, key="shap_sens")
    sweep_rng = np.linspace(float(features_df[feat_s].min()),
                             float(features_df[feat_s].max()), 30)
    sp_list = []
    if xgb_m:
        for v in sweep_rng:
            t = X_ml_raw[selected_idx].copy(); t[ml_cols.index(feat_s)] = v
            ts_df = pd.DataFrame(safe_transform(ml_sc_use, t.reshape(1,-1)), columns=ml_cols)
            sp_list.append(float(xgb_m.predict_proba(ts_df)[0][1]) if hasattr(xgb_m,"predict_proba")
                           else float(1/(1+np.exp(-xgb_m.decision_function(ts_df)[0]))))
        cur_v = float(gene_row[feat_s])
        fig_sv = go.Figure()
        fig_sv.add_trace(go.Scatter(x=sweep_rng, y=sp_list, mode="lines",
                                    line=dict(color="#1565C0", width=2.5)))
        fig_sv.add_vline(x=cur_v, line_dash="dot", line_color="orange",
                         annotation_text=f"Current: {cur_v:.2f}")
        fig_sv.add_hline(y=0.5, line_dash="dash", line_color="red")
        fig_sv.add_hrect(y0=0.5, y1=1.0, fillcolor="red", opacity=0.04, line_width=0)
        fig_sv.update_layout(xaxis_title=feat_s, yaxis_title="Risk",
                              yaxis=dict(range=[0,1]), height=340,
                              margin=dict(t=10,b=20))
        st.plotly_chart(fig_sv, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EMBEDDING SPACE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    emb_cols = [c for c in gnn_cols if c.startswith("node2vec_")]
    if len(emb_cols) >= 2:
        ec1, ec2 = st.columns([3, 1])
        with ec2:
            n_bg     = st.slider("Background genes", 300, 3000, 1000, key="eb_n")
            dim_3    = st.toggle("3D view", False, key="eb_3d")
            show_lbl = st.toggle("Labels (slow)", False, key="eb_lb")

        sdf = features_df.sample(min(n_bg, len(features_df)), random_state=42)
        sdf = sdf.assign(Label=sdf["label"].map({0:"Benign",1:"Pathogenic"}))
        sr  = features_df.iloc[selected_idx]

        with ec1:
            if dim_3 and len(emb_cols) >= 3:
                fig_e = px.scatter_3d(sdf, x="node2vec_0", y="node2vec_1", z="node2vec_2",
                                       color="Label", opacity=0.3,
                                       color_discrete_map={"Pathogenic":"#c62828","Benign":"#1565C0"},
                                       text="GeneSymbol" if show_lbl else None)
                fig_e.add_trace(go.Scatter3d(
                    x=[sr["node2vec_0"]], y=[sr["node2vec_1"]], z=[sr["node2vec_2"]],
                    mode="markers+text",
                    marker=dict(size=10, color="gold", symbol="diamond"),
                    text=[selected_gene], textposition="top center", name=selected_gene
                ))
            else:
                fig_e = px.scatter(sdf, x="node2vec_0", y="node2vec_1", color="Label",
                                   opacity=0.3,
                                   color_discrete_map={"Pathogenic":"#c62828","Benign":"#1565C0"},
                                   text="GeneSymbol" if show_lbl else None)
                fig_e.add_trace(go.Scatter(
                    x=[sr["node2vec_0"]], y=[sr["node2vec_1"]],
                    mode="markers+text",
                    marker=dict(size=16, color="gold", symbol="star",
                                line=dict(color="black", width=1)),
                    text=[selected_gene], textposition="top center", name=selected_gene
                ))
            fig_e.update_layout(height=500, margin=dict(t=40, b=10))
            st.plotly_chart(fig_e, width="stretch")
    else:
        st.info("Node2Vec columns not found.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — NETWORK TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    topo_f = [c for c in ["gene_degree","clustering_coefficient","pagerank",
                           "betweenness_centrality"] if c in features_df.columns]
    if topo_f:
        st.subheader("📊 Centrality Percentile Card")
        pc = st.columns(len(topo_f))
        for i, f in enumerate(topo_f):
            v   = float(gene_row.get(f, 0))
            pct = float((features_df[f] < v).mean() * 100)
            pc[i].metric(f, f"{pct:.1f}th pct", f"{v:.4f}")

        st.markdown("---")
        st.subheader("📈 Centrality Distributions")
        for i in range(0, len(topo_f), 2):
            cols_t = st.columns(min(2, len(topo_f)-i))
            for j, feat in enumerate(topo_f[i:i+2]):
                with cols_t[j]:
                    v   = float(gene_row.get(feat, 0))
                    pct = float((features_df[feat] < v).mean() * 100)
                    fig_t = px.histogram(features_df, x=feat, nbins=60,
                                         color="pathogenic",
                                         color_discrete_map={0:"#1565C0",1:"#c62828"},
                                         barmode="overlay", opacity=0.65, title=feat)
                    fig_t.add_vline(x=v, line_dash="dash", line_color="orange",
                                    annotation_text=f"{selected_gene} ({pct:.0f}th pct)")
                    fig_t.update_layout(height=280, margin=dict(t=40,b=10))
                    st.plotly_chart(fig_t, width="stretch")

        st.markdown("---")
        st.subheader("🔷 Degree vs Neighbour Pathogenic Ratio")
        if "gene_degree" in features_df.columns and "neighbor_pathogenic_ratio" in features_df.columns:
            samp = features_df.sample(min(3000, len(features_df)), random_state=0)
            fig_dg = px.scatter(samp, x="gene_degree", y="neighbor_pathogenic_ratio",
                                color="pathogenic", opacity=0.35,
                                color_discrete_map={0:"#1565C0",1:"#c62828"},
                                title="Network hub score vs neighbourhood pathogenicity")
            fig_dg.add_trace(go.Scatter(
                x=[float(gene_row.get("gene_degree",0))],
                y=[float(gene_row.get("neighbor_pathogenic_ratio",0))],
                mode="markers+text",
                marker=dict(size=16, color="gold", symbol="star",
                            line=dict(color="black",width=1)),
                text=[selected_gene], textposition="top center", name=selected_gene
            ))
            fig_dg.update_layout(height=420, margin=dict(t=50,b=10))
            st.plotly_chart(fig_dg, width="stretch")

        st.markdown("---")
        st.subheader("🧮 Population Topology Statistics")
        rows = []
        pct_d = {}
        for f in topo_f:
            v = float(gene_row.get(f,0))
            col = features_df[f].dropna()
            p = float((col < v).mean()*100)
            pct_d[f] = p
            rows.append({"Feature":f, "Gene":round(v,6),
                          "Min":round(col.min(),6), "25th":round(col.quantile(.25),6),
                          "Median":round(col.median(),6), "75th":round(col.quantile(.75),6),
                          "Max":round(col.max(),6), "Percentile":f"{p:.1f}th"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — GENE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("⚖️ Side-by-Side Gene Comparison")
    cg    = st.selectbox("Compare with:", [g for g in sorted(gene_list) if g != selected_gene],
                         key="cmp_g")
    ci    = gene_to_idx[cg]
    crow  = features_df.iloc[ci]
    cg_desc = "N/A"
    if "description" in features_df.columns:
        d = str(crow.get("description",""))
        if d and d != "nan": cg_desc = d

    cc1, cc2 = st.columns(2)
    cc1.markdown(
        f"<div style='background:#1a237e;border-radius:8px;padding:12px'>"
        f"<b style='color:#e8eaf6'>{selected_gene}</b><br>"
        f"<span style='color:#c5cae9;font-size:12px'>{gene_desc}</span></div>",
        unsafe_allow_html=True
    )
    cc2.markdown(
        f"<div style='background:#4a0000;border-radius:8px;padding:12px'>"
        f"<b style='color:#ffcdd2'>{cg}</b><br>"
        f"<span style='color:#ffcdd2;font-size:12px;opacity:.8'>{cg_desc}</span></div>",
        unsafe_allow_html=True
    )

    _, _, _, _, _, cmp_final, _ = run_inference(ci, {})
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(f"{selected_gene} Risk",  f"{final_prob:.3f}",
               "Pathogenic" if final_prob>0.5 else "Benign")
    mc2.metric(f"{cg} Risk", f"{cmp_final:.3f}",
               "Pathogenic" if cmp_final>0.5 else "Benign")
    mc3.metric(f"{selected_gene} Truth", "Pathogenic" if gene_label==1 else "Benign")
    mc4.metric(f"{cg} Truth", "Pathogenic" if int(crow["label"])==1 else "Benign")

    compare_c = [c for c in ["total_variants","benign_variants","rare_variants",
                              "gene_degree","clustering_coefficient","pagerank",
                              "betweenness_centrality","neighbor_pathogenic_ratio",
                              "mutation_network_score"] if c in features_df.columns]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name=selected_gene, x=compare_c,
                              y=[float(gene_row.get(c,0)) for c in compare_c],
                              marker_color="#1565C0"))
    fig_cmp.add_trace(go.Bar(name=cg, x=compare_c,
                              y=[float(crow.get(c,0)) for c in compare_c],
                              marker_color="#c62828"))
    fig_cmp.update_layout(barmode="group", height=360,
                           margin=dict(t=20,b=60), xaxis_tickangle=-25)
    st.plotly_chart(fig_cmp, width="stretch")

    # ClinVar comparison
    if not clinvar_lookup.empty:
        for g_name, g_lbl in [(selected_gene, "Selected"), (cg, "Comparison")]:
            row = clinvar_lookup[clinvar_lookup["GeneSymbol"]==g_name]
            if not row.empty:
                r = row.iloc[0]
                st.markdown(f"**{g_name} ClinVar:** {int(r['total_variants'])} variants — "
                            f"{int(r['pathogenic'])} pathogenic, {int(r['benign'])} benign, "
                            f"{int(r['snv'])} SNVs, {int(r['deletion'])} deletions")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin:20px 0 8px'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:grey;font-size:11px'>"
    f"PathoGAT v3 &nbsp;·&nbsp; "
    f"ML: RandomForest · XGBoost · GradBoost · SVM · LogReg · Stacking &nbsp;·&nbsp; "
    f"GNN: GraphSAGE · GATv2 &nbsp;·&nbsp; "
    f"Genes: {len(features_df):,} &nbsp;·&nbsp; "
    f"PPI Edges: {len(edges_df):,} &nbsp;·&nbsp; "
    f"ClinVar Variants: {len(clinvar_df):,} &nbsp;·&nbsp; "
    f"Compute: {device}"
    f"</div>",
    unsafe_allow_html=True
)