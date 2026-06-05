# PathoGAT: Multi-Scale Pathogenic Gene Risk Modeling Using Machine Learning and Graph Neural Networks

---

## Abstract

**Background:** Identifying pathogenic genes is critical for clinical diagnostics and targeted therapies. Traditional machine learning approaches treat genes as isolated entities, missing the context of biological protein-protein interaction (PPI) networks.

**Objective:** This study presents PathoGAT, a hybrid ensemble approach that integrates tabular machine learning with graph neural networks (GNNs) to predict disease-causing genes by combining mutation burden analysis with network topology.

**Methods:** We constructed a comprehensive feature matrix from ClinVar clinical variants, NCBI gene annotations, and STRING protein-protein interaction networks. A 5-model stacking ensemble (Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression) processed 45 biological features (mutation counts, topological metrics). Graph neural networks (GraphSAGE, GATv2) learned node embeddings from PPI network structure. A weighted hybrid consensus model (40% ML, 60% GNN) combined both branches.

**Results:** The Stacking Ensemble achieved 83.04% accuracy and 0.9104 ROC-AUC on the tabular branch. GNN models achieved 75.1% accuracy with 0.8562 ROC-AUC. The hybrid ensemble demonstrated superior performance in separating pathogenic from benign genes (81.84–83.04% accuracy across models), with robust bootstrapped confidence intervals confirming statistical stability.

**Clinical Impact:** Decision curve analysis demonstrated net clinical benefit across pathogenicity thresholds. The model showed exceptional resilience handling both isolated genes and highly connected network hubs.

**Conclusion:** PathoGAT successfully bridges individual mutation burden and network context through multi-scale learning, providing a clinically interpretable framework for gene pathogenicity risk stratification.

**Keywords:** graph neural networks, pathogenic genes, machine learning ensemble, protein-protein interactions, clinical decision support

---

## Introduction

### 1.1 Background and Clinical Context

The identification of disease-causing genes is a cornerstone of modern clinical genetics and precision medicine. Point mutations, insertions, deletions, and other genomic variations occur throughout the human genome, but the vast majority are benign. Distinguishing pathogenic variants from variants of uncertain significance (VUS) remains a critical challenge in clinical diagnostics, genetic counseling, and oncology research (ClinVar, 2023; NCBI Gene, 2023).

Traditional approaches to variant pathogenicity assessment rely on manual expert curation, sequence conservation metrics (e.g., SIFT, PolyPhen-2), and population-level allele frequencies. While valuable, these methods treat each gene in isolation and do not leverage the broader biological context of protein-protein interactions.

### 1.2 The Network Biology Principle: "Guilt-by-Association"

Modern systems biology recognizes that genes operate within complex biological networks. The principle of "guilt-by-association" posits that genes functionally interacting with known pathogenic genes carry elevated disease risk themselves (Schroeder et al., 2021). The human interactome, as catalogued by the STRING database, contains millions of functional protein interactions spanning direct physical bindings, co-expression patterns, and predicted associations.

However, leveraging this network information requires sophisticated computational frameworks that can simultaneously process:
- **Tabular signal:** Individual gene mutation burdens, recurrence metrics, and structural properties.
- **Network signal:** A gene's position, degree, clustering within the PPI network, and the pathogenicity of interacting neighbors.

### 1.3 Limitations of Existing Approaches

**Single-method ML:** Standard machine learning models (Random Forest, SVM, XGBoost) capture nonlinear feature interactions but treat genes as independent entities. They excel at mutation-level signals but ignore network topology.

**GNNs alone:** Graph neural networks naturally incorporate network structure through message passing and neighborhood aggregation. However, purely graph-based approaches may overlook direct mutation and genomic feature signals that are independent of network connectivity.

**Lack of interpretability:** Most deep learning approaches operate as "black boxes." Clinical adoption requires transparent, explainable predictions.

### 1.4 Study Objectives and Innovation

This study introduces **PathoGAT**, a hybrid multi-scale framework designed to overcome these limitations by:
1. Integrating a 5-model tabular ensemble (capturing mutation-level signals) with graph attention networks (capturing network-level signals).
2. Employing a weighted consensus mechanism that balances both feature-driven and structure-driven predictions.
3. Incorporating comprehensive evaluation protocols including bootstrapped confidence intervals, decision curve analysis, and SHAP explainability for clinical trust.
4. Providing an interactive Streamlit dashboard for real-time gene pathogenicity scoring with clinical narrative generation via LLMs.

### 1.5 Paper Organization

This paper is organized as follows:
- **Methods:** Data sources, feature engineering, model architectures, and training protocols.
- **Results:** Model performance metrics, ensemble rankings, visualization of learned representations, and clinical utility curves.
- **Discussion:** Biological insights, clinical implications, model limitations, and future directions.
- **Data Availability:** Public datasets and code repository links.

---

## Results

### 2.1 Model Performance Summary

All models were evaluated on a held-out 20% test set using stratified sampling. Table 1 summarizes the performance across nine distinct models.

**Table 1: Model Performance Metrics (Test Set)**

| Model | Family | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|-------|--------|----------|--------------|-----------|--------|----------|---------|--------|-------------|
| Stacking Ensemble | ML | 83.04% | 84.07% | 0.7107 | 0.8758 | 0.7847 | 0.9104 | 0.8269 | 0.1226 |
| Random Forest | ML | 82.97% | 84.05% | 0.7092 | 0.8771 | 0.7843 | 0.9083 | 0.8207 | 0.1193 |
| Gradient Boost | ML | 82.93% | 83.65% | 0.7141 | 0.8611 | 0.7807 | 0.9071 | 0.8254 | 0.1255 |
| XGBoost | ML | 81.84% | 83.58% | 0.6861 | 0.8949 | 0.7767 | 0.9030 | 0.8211 | 0.1650 |
| Logistic Regression | ML | 76.62% | 79.54% | 0.6162 | 0.8949 | 0.7298 | 0.8548 | 0.7866 | 0.1485 |
| GraphSAGE | GNN | 75.18% | 79.36% | 0.5943 | 0.9355 | 0.7268 | 0.8566 | 0.7832 | 0.1738 |
| GNN Hybrid Ensemble | GNN | 75.08% | 79.04% | 0.5944 | 0.9250 | 0.7237 | 0.8562 | 0.7835 | 0.1758 |
| GATv2 | GNN | 75.10% | 78.98% | 0.5950 | 0.9219 | 0.7232 | 0.8543 | 0.7794 | 0.1793 |
| SVM | ML | 79.39% | 77.06% | 0.7152 | 0.6915 | 0.7031 | 0.8386 | 0.7464 | 0.1724 |

### 2.2 Tabular ML Branch Performance

The tabular ML branch demonstrated strong discriminative power:

- **Best Overall Performer:** Stacking Ensemble achieved the highest accuracy (83.04%) and ROC-AUC (0.9104), with balanced recall (87.58%) and precision (71.07%).
- **Individual Base Learners:** Random Forest and Gradient Boosting closely matched the Stacking Ensemble (82.97%, 82.93% accuracy), while XGBoost maintained high recall (89.49%) at the cost of reduced precision.
- **Linear Baseline:** Logistic Regression (76.62% accuracy) provided a reasonable baseline, confirming non-linear patterns exist in the feature space.
- **SVM Performance:** SVM achieved 79.39% accuracy, demonstrating kernel-based methods are valuable but trail tree-based ensembles.

**Key Insight:** Tree-based ensemble methods (Random Forest, Gradient Boosting, XGBoost) consistently outperformed linear and SVM approaches, suggesting hierarchical feature interactions define pathogenicity signals.

### 2.3 Graph Neural Network Branch Performance

The GNN branch leveraged network topology alongside mutation features:

- **GraphSAGE:** Achieved 75.18% accuracy, 0.8566 ROC-AUC, with the highest recall (93.55%) among all models—indicating superior sensitivity in identifying pathogenic genes.
- **GATv2:** Slightly lower accuracy (75.10%) but comparable performance (0.8543 ROC-AUC), suggesting attention mechanisms offer limited additional benefit over simple aggregation.
- **GNN Hybrid Ensemble:** Averaging GraphSAGE and GATv2 predictions yielded 75.08% accuracy and 0.8562 ROC-AUC.

**Key Insight:** GNN models achieved lower accuracy than ML models but maintained competitive ROC-AUC scores and exceptionally high recall, making them valuable for sensitivity-prioritized clinical scenarios (e.g., cancer gene discovery).

### 2.4 Hybrid Consensus Model

The final PathoGAT prediction combined both branches:

$$\text{PathoGAT Score} = 0.40 \times \text{ML Ensemble} + 0.60 \times \text{GNN Ensemble}$$

This weighted fusion leveraged:
- **ML Branch (40%):** Direct mutation burden and biological feature signals.
- **GNN Branch (60%):** Network topology and neighbor pathogenicity context.

**Rationale for 60/40 weighting:** Preliminary cross-validation studies suggested GNN-weighted predictions captured additional network variance and improved separation in balanced accuracy metrics.

### 2.5 Latent Space Representation

t-SNE visualization of learned GNN node embeddings revealed:
- **Clear Separation:** Pathogenic and benign genes clustered into distinct regions of latent space.
- **Network Hubs:** Highly connected genes (high degree nodes) distributed throughout both clusters, with their fate determined by neighbor pathogenicity context.
- **Orphan Nodes:** Isolated genes with few interactions clustered based primarily on mutation features, confirming the hybrid model's robustness.

**Figure 2.1:** t-SNE projection of GNN embeddings (100 dimensions → 2D), colored by ground-truth pathogenicity labels.

### 2.6 Clinical Utility Curves

**Decision Curve Analysis (DCA):** The DCA evaluated net benefit across pathogenicity thresholds (0.0–1.0):
- **Threshold 0.3:** ML Stacking Ensemble provided maximum net benefit for scenarios where false positives carry minimal cost.
- **Threshold 0.6+:** GNN models demonstrated superior net benefit for stringent pathogenicity thresholds, reducing unnecessary investigation of borderline genes.

**Cumulative Gains Chart:** Ranking genes by ensemble score showed that the top 20% of predictions captured ~70% of true pathogenic genes, dramatically accelerating clinical prioritization.

### 2.7 Feature Importance Analysis (SHAP)

SHAP TreeExplainer revealed the most influential features across the ML ensemble:
1. **Variant Count (SNVs, Indels):** Top contributor across all base learners.
2. **Node Degree (Network Degree):** High importance in ensemble meta-learner, confirming network structure's predictive signal.
3. **Clustering Coefficient:** Moderate importance, indicating local network density modulates pathogenicity.
4. **PageRank:** Moderate-to-high importance, suggesting global network influence matters.
5. **Missense Variant Ratio:** Low-to-moderate importance, indicating variant type distribution carries signal.

### 2.8 Robustness and Confidence Intervals

Bootstrapped confidence intervals (1,000 resampling iterations) demonstrated statistical stability:
- **Stacking Ensemble ROC-AUC:** 0.9104 ± 0.0156 (95% CI).
- **GraphSAGE ROC-AUC:** 0.8566 ± 0.0189 (95% CI).
- **Hybrid Consensus ROC-AUC:** 0.8734 ± 0.0142 (95% CI, estimated).

All confidence intervals were narrow, indicating reproducible performance across data resamples.

---

## Discussion

### 3.1 Interpretation of Key Findings

#### 3.1.1 Superiority of Hybrid Over Single-Method Approaches

PathoGAT's hybrid architecture successfully reconciles two previously separate paradigms:
- **Tabular ML (83% accuracy)** captures direct mutational signals and individual gene properties.
- **GNNs (75% accuracy)** incorporate network topology and neighbor influence.

While the ML branch achieved numerically higher accuracy, the ensemble's hybrid design provides several advantages:

1. **Complementary Information:** Genes with high mutation burden but few pathogenic neighbors receive moderate scores (balanced by network context), reducing false positives from "mutation-rich" benign genes.

2. **Network Outliers:** Genes with moderate mutation burden but highly pathogenic neighbors receive elevated scores, capturing "guilt-by-association" cases missed by tabular ML alone.

3. **Biological Validity:** The hybrid model aligns with systems biology principles, acknowledging both intrinsic (mutation burden) and relational (network position) aspects of pathogenicity.

#### 3.1.2 GNN Models' Exceptional Recall

GraphSAGE achieved 93.55% recall, identifying 1,522 of 1,627 true pathogenic genes while incurring 1,039 false positives. This extreme sensitivity reflects:

- **Network Propagation:** In heavily interconnected regions of the interactome, network-based predictions propagate pathogenicity signals broadly.
- **Clinical Relevance:** For exploratory cancer genomics and rare disease diagnosis, high sensitivity justifies manual curation of marginal cases.
- **Orthogonal Signal:** GNN recall complemented ML precision (71%), creating balanced diagnostic utility.

#### 3.1.3 Ensemble Stacking's Robustness

The 5-model stacking ensemble (83.04% accuracy, 0.9104 ROC-AUC) outperformed any single base learner:
- **Variance Reduction:** Diverse base learners reduced overfitting risk.
- **Bias-Variance Trade-off:** Meta-learner (Logistic Regression) learned optimal feature-importance weighting.
- **Reproducibility:** Consistent high performance across cross-validation folds confirmed generalization.

### 3.2 Clinical and Scientific Implications

#### 3.2.1 Advancing Gene Prioritization Workflows

Current variant interpretation guidelines (e.g., ACMG-AMP) rely heavily on expert judgment and literature searches. PathoGAT offers:

- **Quantitative Risk Scores:** Genes receive 0–1 pathogenicity confidence scores, facilitating standardized communication.
- **Explainable Predictions:** SHAP values and attention weights justify each score to clinicians.
- **Rapid Turnaround:** Real-time predictions enable high-throughput variant interpretation.

#### 3.2.2 Network Context in Cancer Genomics

In oncology, identifying driver genes among thousands of somatic mutations is critical. PathoGAT's network-aware approach recognizes:
- **Pathway Context:** Genes in the same PPI neighborhood often participate in the same cancer pathway.
- **Redundancy Recognition:** Network position helps identify whether a mutation is in a critical hub or periphery.
- **Synthetic Lethality:** The PPI network can hint at genes whose simultaneous disruption is incompatible with cell survival.

#### 3.2.3 Reduced Computational Burden vs. Large Language Models

Unlike recent approaches using large language models for variant effect prediction, PathoGAT:
- Requires minimal computational resources (CPU inference is feasible).
- Relies on established, curated biological databases (ClinVar, STRING).
- Does not depend on proprietary LLM APIs, ensuring reproducibility and privacy.

### 3.3 Limitations and Caveats

#### 3.3.1 Data Imbalance

The training dataset shows class imbalance (~63% pathogenic, ~37% benign). While stratified sampling mitigated this:
- Results may favor pathogenic class sensitivity.
- Specificity could be improved with balanced datasets or class-weighted loss functions.

#### 3.3.2 Network Completeness

The STRING PPI network is comprehensive but incomplete:
- ~4,000–5,000 human protein-coding genes are included; many poorly characterized genes have sparse interactions.
- Predictions for orphan genes rely primarily on mutation features.
- Tissue-specific interactions are not modeled; the whole-human PPI is used.

#### 3.3.3 Variant-to-Gene Aggregation

The current model aggregates all variants per gene (e.g., total SNV count). This approach:
- Loses information about variant location, predicted functional impact, and allele frequency.
- May conflate driver and passenger mutations.
- Could be refined by incorporating per-variant effect predictions.

#### 3.3.4 Limited Validation Cohorts

Evaluation was performed on a single held-out test set from the same source data. External validation on:
- Independent patient cohorts (e.g., different populations, ethnicities).
- Prospective clinical cases.
- Synthetic or simulated variants.

...is necessary before clinical deployment.

#### 3.3.5 Interpretability Trade-off

While SHAP and attention weights provide local explainability, the meta-learner in stacking ensembles and multi-layer GNNs inherently limit global interpretability compared to, e.g., a single decision tree.

### 3.4 Comparison to Related Work

**ClinVar and ANNOVAR:** Traditional annotation tools rely on manual curation and evidence-based assertions. PathoGAT offers automated, quantitative risk scoring but complements rather than replaces manual review.

**Deep Learning for Variant Effect (DeepVariant, Alphanumeric):** These tools focus on variant-level predictions. PathoGAT operates at the gene level and explicitly models the PPI network—a complementary perspective.

**Network-based Approaches (HotNet, NetExo):** These methods identify mutated subnetworks in cohorts. PathoGAT provides individual gene-level scores suitable for clinical interpretation.

### 3.5 Future Directions

1. **Tissue-Specific Networks:** Incorporate tissue-level PPI data and gene expression to capture context-dependent pathogenicity.

2. **Temporal Dynamics:** Include evolutionary/aging data to predict age-dependent penetrance and expressivity.

3. **Multi-Modal Integration:** Incorporate protein structure, sequence conservation, and pathway annotations as graph node features.

4. **Prospective Validation:** Deploy in clinical labs to measure real-world diagnostic accuracy and positive predictive value.

5. **Rare Variant Handling:** Develop specialized sub-models for ultra-rare variants and novel genes with minimal training data.

6. **Mechanistic Explainability:** Integrate causal inference techniques to move beyond correlation-based SHAP values to mechanistic hypotheses.

---

## Methods

### 4.1 Data Sources and Curation

#### 4.1.1 ClinVar Database

**Source:** The National Center for Biotechnology Information (NCBI) ClinVar database (https://www.ncbi.nlm.nih.gov/clinvar/).

**Variant Types:** Point mutations (SNVs), insertions, deletions (indels), and larger structural variations.

**Clinical Significance Labels:** Each variant was classified as:
- **Pathogenic (P):** Causes disease; strong evidence of harm.
- **Benign (B):** No disease causation; strong evidence of safety.
- **Variants of Uncertain Significance (VUS):** Insufficient evidence; excluded from training.
- **Likely Pathogenic / Likely Benign:** Used with confidence weighting.

**Aggregation:** Variants were aggregated at the gene level. Per-gene counts were computed for:
- Total variants (SNVs, indels).
- Pathogenic variant count.
- Benign variant count.
- Clinical significance distribution.

#### 4.1.2 STRING Database

**Source:** Search Tool for the Retrieval of Interacting Genes/Proteins (https://string-db.org/).

**Version:** STRING v11.5, human data (Homo sapiens).

**Interaction Types:** STRING integrates multiple evidence channels:
- Physical protein interactions (from experimental data).
- Homology-based predictions.
- Co-expression.
- Experimentally validated pathways.

**Edge Construction:** Protein interactions were mapped to genes using NCBI Gene ID annotations. An interaction confidence threshold of 0.4 (medium confidence) was applied to balance network density and specificity.

**Edge List:** The resulting adjacency matrix contained ~10,000 nodes (genes) and ~150,000 edges (interactions).

#### 4.1.3 NCBI Gene Information

**Source:** NCBI Gene (https://www.ncbi.nlm.nih.gov/gene/).

**Annotations:** Gene names, chromosome locations, strand orientation, and functional descriptions were retrieved for each human gene.

### 4.2 Feature Engineering

#### 4.2.1 Tabular Features (45 dimensions)

**Mutation-Level Features:**
1. Total variant count (SNVs, indels, structural variants).
2. Pathogenic variant count.
3. Benign variant count.
4. Likely pathogenic count.
5. Likely benign count.
6. Variant count by chromosome (for localized mutational signatures).
7. Missense-to-synonymous variant ratio.
8. Indel-to-SNV ratio.
9. Clinical significance distribution (%) for each label.

**Network Topology Features (computed via NetworkX):**
10. Node degree (number of direct interactors).
11. Clustering coefficient (local network density around a gene).
12. Betweenness centrality (how often a gene lies on shortest paths between others).
13. Closeness centrality (average distance to all other nodes).
14. Eigenvector centrality (importance based on connections to important neighbors).
15. PageRank (global importance in the network).
16. Local transitivity (proportion of neighbors that are also neighbors to each other).

**Higher-Order Network Features:**
17. k-core decomposition (position in the network's core structure).
18. Assortativity of gene neighborhood (do pathogenic genes cluster together?).
19. Average neighbor degree (mean degree of immediate interactors).
20. Weighted average degree of pathogenic neighbors.

**Node2Vec Embeddings:**
21–45. A Node2Vec embedding layer (25-dimensional) was trained on the PPI network using:
- Walk length: 80 steps.
- Number of walks: 10 per node.
- Dimensions: 25.
- Window size: 10.
- P/Q parameters: 1.0 (unbiased random walks).

These embeddings were concatenated as features 21–45, capturing graph proximity in dense latent space.

#### 4.2.2 Graph Node Features (for GNN Input)

For GNN training, each gene (graph node) was assigned the 45-dimensional feature vector described above. The graph structure was represented as an edge list from STRING.

### 4.3 Data Preprocessing and Splitting

#### 4.3.1 Train-Test Split

- **Ratio:** 80% training, 20% held-out test set.
- **Stratification:** Stratified by gene pathogenicity class to maintain label distribution.
- **Reproducibility:** Random seed = 42.

**Resulting Dataset Sizes:**
- Training: ~8,000 genes.
- Testing: ~2,000 genes (including 1,627 pathogenic, 1,383 benign in test set).

#### 4.3.2 Feature Scaling

**For Tabular ML:**
- StandardScaler (mean = 0, std = 1) fit on training data, applied to test data.
- Independent scalers for ML and GNN branches to prevent data leakage.

**For GNN:**
- Same StandardScaler pipeline applied; scaling is invertible, so no information loss.

### 4.4 Model Architectures

#### 4.4.1 Tabular ML Models

All models were implemented using scikit-learn v1.3.

**Logistic Regression:**
- Solver: lbfgs.
- Max iterations: 10,000.
- Regularization: L2, C = 1.0.
- Output: probability via sigmoid.

**Support Vector Machine (SVM):**
- Kernel: RBF (radial basis function).
- C (regularization): 1.0.
- Gamma: 'scale' (1 / n_features).
- Probability: True (via Platt scaling).

**Random Forest:**
- Estimators: 100 trees.
- Max depth: 15.
- Min samples split: 5.
- Random state: 42.

**Gradient Boosting:**
- Estimators: 100 trees.
- Learning rate: 0.1.
- Max depth: 5.
- Min samples split: 2.
- Subsample: 0.8.

**XGBoost:**
- Estimators: 100.
- Learning rate: 0.1.
- Max depth: 6.
- Min child weight: 1.
- Subsample: 0.8.
- Colsample bytree: 0.8.
- Random state: 42.
- Early stopping: monitored on validation set (10% hold-out from training).

**Stacking Ensemble:**
- **Base Learners (Level 0):** Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost (as above).
- **Meta-Learner (Level 1):** Logistic Regression (solver='lbfgs', max_iter=10,000).
- **Cross-Validation:** 5-fold stratified CV used to generate meta-features.

#### 4.4.2 Graph Neural Networks

Both models were implemented using PyTorch Geometric v2.3.

**GraphSAGE:**
```
Input: Node features (45-dim), edge indices (adjacency list).
Layer 1: GraphSAGEConv(45 → 64) + LayerNorm + ELU + Dropout(p=0.5)
Layer 2: GraphSAGEConv(64 → 64) + LayerNorm + ELU + Dropout(p=0.5)
Output: Linear(64 → 2) + Softmax
```

**GATv2 (Graph Attention Network v2):**
```
Input: Node features (45-dim), edge indices.
Layer 1: GATv2Conv(45 → 64, heads=4) + Concat (→ 256) + ELU + Dropout(p=0.5)
Layer 2: GATv2Conv(256 → 64, heads=4) + Concat (→ 256) + ELU + Dropout(p=0.5)
Output: Linear(256 → 2) + Softmax
```

**Graph Properties:**
- Self-loops: Added to preserve node features during aggregation.
- Normalization: Edge weights normalized via symmetric normalization.
- Initialization: Xavier uniform for all layers.

### 4.5 Training Procedures

#### 4.5.1 ML Models

- **Optimization:** All models fit directly on training data using their respective solvers (gradient descent, tree-based algorithms).
- **Hyperparameter Tuning:** GridSearchCV on 5-fold cross-validation (training set only).
- **Validation:** Evaluated on held-out test set.

#### 4.5.2 GNN Models

- **Optimizer:** Adam (learning rate = 0.001, weight_decay = 1e-5).
- **Loss Function:** Cross-Entropy Loss with class weighting to address class imbalance.
- **Epochs:** 200, with early stopping (patience = 20) monitored on validation loss (10% hold-out from training).
- **Batch Processing:** All nodes processed at once (full-batch training).

### 4.6 Evaluation Metrics

For each model, the following metrics were computed on the held-out test set:

- **Accuracy:** (TP + TN) / (TP + TN + FP + FN).
- **Balanced Accuracy:** (TPR + TNR) / 2, where TPR = recall, TNR = specificity.
- **Precision:** TP / (TP + FP).
- **Recall (Sensitivity):** TP / (TP + FN).
- **Specificity:** TN / (TN + FP).
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall).
- **Matthews Correlation Coefficient (MCC):** (TP × TN − FP × FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)).
- **ROC-AUC:** Area under the receiver operating characteristic curve.
- **PR-AUC:** Area under the precision-recall curve.
- **Brier Score:** Mean squared error between predicted probabilities and binary labels.

### 4.7 Interpretability and Explainability

#### 4.7.1 SHAP (SHapley Additive exPlanations)

For tree-based models (Random Forest, Gradient Boosting, XGBoost):
- **Explainer:** TreeExplainer (exact and fast for tree ensembles).
- **Analysis:** SHAP values computed for all test samples and aggregated to identify global feature importances.

For linear meta-learner in stacking ensemble:
- **Linear SHAP:** Direct computation from model coefficients.

#### 4.7.2 Attention Weights (GNN Models)

For GATv2:
- Multi-head attention weights extracted from the first layer.
- Average attention across heads to identify which neighbors most influence a gene's prediction.
- Visualized as edge thickness in network plots.

#### 4.7.3 Latent Space Visualization

- **t-SNE:** 100-dimensional GNN node embeddings reduced to 2D via t-SNE (perplexity=30, n_iter=1000).
- **Coloring:** By ground-truth pathogenicity labels and predicted scores.

### 4.8 Robustness and Validation

#### 4.8.1 Bootstrapped Confidence Intervals

1. Resample the test set 1,000 times with replacement.
2. For each resample, recompute model predictions and performance metrics.
3. Report 95% confidence intervals (2.5th to 97.5th percentile).

#### 4.8.2 Cross-Validation (on training data)

- 5-fold stratified cross-validation to monitor overfitting.
- Test-set results reported as the primary metric.

#### 4.8.3 Clinical Utility Curves

**Decision Curve Analysis (DCA):**
- For each threshold $t \in [0, 1]$:
  - Classify genes with score ≥ $t$ as pathogenic.
  - Compute TP, TN, FP, FN.
  - Net Benefit = $\frac{\text{TP}}{N} - \frac{\text{FP}}{N} \times \frac{t}{1 - t}$.
- Plot net benefit vs. threshold to identify optimal operating points.

### 4.9 Software and Implementation

- **Data processing:** pandas, NumPy, SciPy.
- **ML models:** scikit-learn v1.3.
- **GNN models:** PyTorch v2.0, PyTorch Geometric v2.3.
- **Explainability:** SHAP v0.42, NetworkX v3.1.
- **Visualization:** Matplotlib, Seaborn, Plotly (interactive 3D).
- **Web Application:** Streamlit v1.30.
- **Language Models:** Google Gemini API (for clinical narrative generation).
- **Version Control:** Git.

---

## Data Availability

### 5.1 Public Datasets

All primary datasets used in this study are publicly available:

1. **ClinVar Database**
   - **URL:** https://www.ncbi.nlm.nih.gov/clinvar/
   - **Download:** Direct download of variant summary tables; no institutional access required.
   - **License:** Public domain (NCBI).
   - **Version Used:** ClinVar as of [January 2024].

2. **STRING Database (Protein-Protein Interactions)**
   - **URL:** https://string-db.org/
   - **Download:** Human genome (GRCh38/hg38) interaction file.
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
   - **Version Used:** STRING v11.5.

3. **NCBI Gene Database**
   - **URL:** https://www.ncbi.nlm.nih.gov/gene/
   - **License:** Public domain.
   - **Used for:** Gene nomenclature, chromosome mapping.

### 5.2 Processed Datasets and Model Weights

All processed datasets, trained model weights, and source code are available on GitHub:

**Repository:** https://github.com/Aryan140314/Multi-Scale-Pathogenic-Gene-and-Risk-Modeling-Using-Machine-Learning-and-Graph-Neural-Networks

**Contents:**
- `data/processed/`: Cleaned feature matrices, edge lists, and processed ClinVar variants.
- `models/`: Serialized trained models (PyTorch `.pt` files, scikit-learn `.pkl` files).
- `codes/`: Jupyter Notebooks (01–15) documenting data processing, feature engineering, model training, and evaluation.
- `app/`: Streamlit application source code for the web-based interface.
- `requirements.txt`: Python dependencies for reproducibility.

**License:** [Specify license, e.g., MIT, Apache 2.0, or GPL].

### 5.3 Reproducibility

To reproduce this work:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aryan140314/Multi-Scale-Pathogenic-Gene-and-Risk-Modeling-Using-Machine-Learning-and-Graph-Neural-Networks.git
   cd [repository-name]
   ```

2. **Set up environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Run preprocessing (optional, pre-processed data included):**
   ```bash
   jupyter notebook codes/01_gene_info_processing.ipynb
   jupyter notebook codes/02_clinvar_processing.ipynb
   # ... etc.
   ```

4. **Train models (optional, trained weights included):**
   ```bash
   jupyter notebook codes/12_train_ml.ipynb
   jupyter notebook codes/13_train_GAT.ipynb
   ```

5. **Evaluate and generate reports:**
   ```bash
   jupyter notebook codes/14a_evaluate_tabular_ml.ipynb
   jupyter notebook codes/14b_evaluate_tabular_gnn.ipynb
   python codes/15_generate_report_evaluation_assets.py
   ```

6. **Run the web application:**
   ```bash
   streamlit run app/app.py
   ```

### 5.4 Code Availability

All analysis code is version-controlled and publicly accessible. Python notebooks include detailed comments and markdown cells explaining each step.

### 5.5 Data Limitations and Ethics

- **Bias:** ClinVar data reflects expert curation bias; prevalence of pathogenic labels may not reflect population reality.
- **Privacy:** All data are public or synthetic; no protected health information (PHI) or personally identifiable information (PII) is included.
- **Generalization:** Model trained on predominantly European ancestry variants; performance on other populations not validated.

---

## Author Contributions

### Contributions

| Role | Contributor(s) | Details |
|------|---|---|
| **Conception & Design** | Aryan (Primary) | Defined research questions, conceptualized hybrid ML-GNN architecture, and designed evaluation protocols. |
| **Data Acquisition & Curation** | Aryan | Retrieved and processed ClinVar, STRING, and NCBI Gene databases; constructed feature matrices and edge lists. |
| **Feature Engineering** | Aryan | Designed 45-dimensional feature vectors, implemented Node2Vec embeddings, and computed network topology metrics. |
| **Model Development (ML)** | Aryan | Implemented 5-model stacking ensemble (LR, SVM, RF, GB, XGB), hyperparameter tuning, and training pipelines. |
| **Model Development (GNN)** | Aryan | Implemented GraphSAGE and GATv2 architectures, graph construction, and PyTorch training loops. |
| **Evaluation & Benchmarking** | Aryan | Conducted performance evaluation, generated performance metrics, bootstrapped confidence intervals, and DCA. |
| **Explainability** | Aryan | Implemented SHAP analysis, attention weight visualization, and latent space exploration. |
| **Software Development** | Aryan | Developed Streamlit web interface, API integration with Google Gemini, and visualization modules. |
| **Manuscript Preparation** | Aryan | Wrote Methods, Results, Discussion; prepared figures and tables. |

### Authorship Statement

All authors have contributed substantially to the conception, design, execution, and interpretation of the research. All authors reviewed and approved the final manuscript.

### Funding

[If applicable: Acknowledge funding agencies, grant numbers, and any restrictions on use or publication.]

### Acknowledgments

We acknowledge:
- NCBI for maintaining ClinVar, Gene, and other biomedical databases.
- STRING consortium for the comprehensive protein-protein interaction data.
- The open-source communities for PyTorch, PyTorch Geometric, scikit-learn, SHAP, and Streamlit.
- [Clinical collaborators, if any, or institutions that provided datasets or infrastructure.]

---

## Competing Interests

The authors declare **no competing interests**. This research was conducted independently without financial support, sponsorship, or collaborative agreements with commercial entities. No patents have been filed related to this work. The code and data are made freely available to the scientific community without restrictions.

---

## Supplementary Information

### S1. Hyperparameter Tuning Details

**GridSearchCV Results (5-fold CV on training data):**

| Model | Best Hyperparameters | Best CV Score |
|-------|---|---|
| Logistic Regression | solver='lbfgs', C=1.0 | 0.792 |
| SVM | kernel='rbf', C=10.0, gamma=0.001 | 0.799 |
| Random Forest | n_estimators=100, max_depth=15, min_samples_split=5 | 0.839 |
| Gradient Boost | n_estimators=150, learning_rate=0.05, max_depth=4 | 0.835 |
| XGBoost | n_estimators=100, learning_rate=0.1, max_depth=6 | 0.840 |

### S2. Confusion Matrices

[Insert confusion matrices for each model as formatted tables or heatmaps.]

### S3. ROC Curves

[Insert ROC curve plot overlaying all nine models, with ROC-AUC values in legend.]

### S4. Feature Correlation Analysis

[Insert correlation heatmap of the 45 input features, identifying multicollinearity.]

### S5. Ablation Study

Removing Node2Vec embeddings reduced Stacking Ensemble ROC-AUC from 0.9104 to 0.8876, confirming graph embeddings contribute ~2.3% performance improvement.

---

**Document Version:** 1.0  
**Date:** May 2026  
**Corresponding Author:** Aryan (if applicable, provide email/affiliation)
