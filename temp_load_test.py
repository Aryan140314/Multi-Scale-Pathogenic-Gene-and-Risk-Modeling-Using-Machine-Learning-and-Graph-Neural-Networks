import pandas as pd
import torch
from app.app import load_datasets, load_clinvar, load_gene_info, build_clinvar_lookup, build_pipeline, load_models, get_scaled_data
print('Starting import and load test')
feat, edges = load_datasets()
print('loaded datasets', feat.shape, edges.shape)
clin = load_clinvar()
print('clinvar', clin.shape)
ginfo = load_gene_info()
print('gene_info', ginfo.shape)
clin_l = build_clinvar_lookup(clin)
print('clinvar lookup', clin_l.shape)
X_ml_raw, X_gnn_raw, ml_cols, gnn_cols, gene_list, gene_to_idx, edge_index = build_pipeline(feat, edges)
print('build_pipeline ok', X_ml_raw.shape, X_gnn_raw.shape, len(ml_cols), len(gnn_cols), len(gene_list), edge_index.shape)
ml_models, ml_sc, gnn_sc, sage_model, gat_model, missing = load_models(gnn_dim=len(gnn_cols))
print('models', missing, list(ml_models.keys()), ml_sc is not None, gnn_sc is not None)
ml_sc_use, gnn_sc_use, X_ml_scaled_all, X_gnn_scaled_all = get_scaled_data(ml_sc, gnn_sc, X_ml_raw, X_gnn_raw)
print('scaling ok', X_ml_scaled_all.shape, X_gnn_scaled_all.shape)
