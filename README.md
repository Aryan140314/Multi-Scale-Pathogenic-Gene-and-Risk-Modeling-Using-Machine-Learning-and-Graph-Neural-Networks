# 🧬 PathoGAT: Multi-Scale Pathogenic Gene & Risk Modeling
PathoGAT integrates a 5-model Machine Learning ensemble with advanced Graph Attention Networks (GAT) to predict disease-causing genes. By analyzing both tabular genetic features and Protein-Protein Interaction (PPI) networks, this multi-scale AI overcomes traditional ML limits for highly accurate, consensus-based pathogenicity risk scoring.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost%20%7C%20Scikit--Learn-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 1. Overview
**PathoGAT** is an advanced bioinformatics platform designed to predict the pathogenicity of human genes. It bridges the gap between traditional tabular mutation analysis and complex biological network topologies. By combining an ensemble of standard Machine Learning models with state-of-the-art Graph Neural Networks (GNNs), this tool provides highly calibrated risk scores, interactive biological network visualizations, and AI-driven clinical explainability.

## 💡 2. Motivation
Identifying pathogenic (disease-causing) genes is critical for clinical diagnostics and targeted cancer therapies. Traditional machine learning approaches often treat genes as isolated entities, analyzing their mutation counts independently. However, biology operates in complex networks. 

This project operates on the biological principle of **"guilt-by-association"**—genes interacting with known pathogenic genes are at a higher risk themselves. PathoGAT was developed to capture both the individual mutation burden and the broader Protein-Protein Interaction (PPI) network context to deliver superior predictive accuracy.

## 🏗️ 3. System Architecture
The system is divided into three core pipelines:
1. **Data Pipeline:** Ingests raw clinical databases, constructs PPI networks, and performs feature extraction.
2. **Modeling Pipeline:** Trains parallel branches of models—a Tabular Stacking Ensemble and a Graph Neural Network ensemble—before merging them into a weighted Hybrid Consensus.
3. **Inference & UI Pipeline:** A Streamlit-based web dashboard that provides real-time risk predictions, SHAP value explainability, 3D latent space exploration, and natural language clinical summaries via the Google Gemini API.

## 🔬 4. Methodology
The prediction engine utilizes a **Hybrid Consensus** approach:
* **Tabular ML Branch (40% Weight):** Processes 45 biological and topological features using a Stacking Ensemble of Random Forest, XGBoost, Gradient Boosting, SVM, and Logistic Regression.
* **GNN Branch (60% Weight):** Processes the graph structure using Message Passing neural networks to learn node embeddings based on neighborhood aggregations.
* **Network Embeddings:** Utilizes Node2Vec to map the PPI network into a dense latent space, which is fed as tabular features into the ML branch.

## ⚙️ 5. Preprocessing Pipeline
The data preparation pipeline is completely reproducible via the `codes/` directory:
* **ClinVar Parsing:** Extracts variant counts (SNVs, Deletions, Insertions) and clinical significance labels.
* **String DB Integration:** Maps proteins to genes to construct the foundational adjacency matrix (edge list) of the human interactome.
* **Network Feature Extraction:** Calculates topological metrics including Node Degree, Clustering Coefficient, PageRank, and Betweenness Centrality.

## 🧠 6. Model Architecture
### Graph Neural Networks
* **GraphSAGE (Graph Sample and Aggregate):** A 2-layer inductive network that generates embeddings by sampling and aggregating features from a node's local neighborhood.
* **GATv2 (Graph Attention Network v2):** Employs dynamic, multi-head attention mechanisms to weigh the importance of neighboring nodes differently, prioritizing highly pathogenic interactors.

### Tabular Machine Learning
* **Base Learners:** Random Forest, XGBoost, Gradient Boosting, SVM.
* **Meta-Learner:** Logistic Regression (Stacking Ensemble).

## 🏋️ 7. Training Strategy
* **Data Split:** 80/20 stratified train-test split.
* **Scaling:** Independent `StandardScaler` instances for ML and GNN branches to prevent data leakage.
* **GNN Specifics:** Models are trained using PyTorch Geometric with `add_self_loops` to ensure a node's own features are preserved during message passing. Models are optimized using Cross-Entropy Loss and the Adam optimizer.

## 📊 8. Visualization & Explainability
To ensure the models are clinically trustworthy, PathoGAT includes a comprehensive research-grade evaluation suite:
* **Decision Curve Analysis (DCA):** Evaluates the clinical net benefit of the models across various risk thresholds.
* **Bootstrapped Confidence Intervals:** 1,000 resampling iterations (Violin Plots) to prove statistical stability.
* **Metrics Radar Charts:** Multi-axis visualization of Sensitivity, Specificity, Precision, F1-Score, and Accuracy.
* **SHAP Explainability:** TreeExplainer provides local feature attributions for individual gene predictions.
* **AI Explainer:** Anthropic/Google GenAI API translates SHAP values and network contexts into readable clinical reports.

## 🗄️ 9. Datasets Used
* **ClinVar:** The primary source for genetic variants and their clinical significance.
* **STRING Database:** Used to construct the Protein-Protein Interaction (PPI) graphs.
* **NCBI Gene Info:** Provides baseline gene nomenclature, metadata, and biological descriptions.

## 📈 10. Results
Based on our robust evaluation suite:
* The **GNN Hybrid Ensemble** successfully separates pathogenic and benign genes in the latent space (visualized via t-SNE).
* **Clinical Prioritization:** The Cumulative Gains and Lift curves demonstrate that utilizing the ensemble's risk rankings drastically accelerates the identification of pathogenic variants compared to random screening.
* The combined model shows exceptional resilience when handling isolated nodes (orphans) versus highly connected network hubs.

## 📂 11. Project Structure
```text
├── app/                   # Frontend Application
│   └── app.py             # Main Streamlit dashboard script
├── codes/                 # Jupyter Notebooks for the entire pipeline
│   ├── 01_gene_info_processing.ipynb
│   ├── 02_clinvar_processing.ipynb
│   ├── ...
│   ├── 14a_evaluate_tabular_ml.ipynb   # Advanced ML Evaluation Suite
│   └── 14b_evaluate_tabular_gnn.ipynb  # Advanced GNN Evaluation Suite
├── data/
│   └── processed/         # Cleaned feature matrices and edge lists
├── models/                # Trained .pkl scalers and .pt model weights
├── .gitattributes
├── .gitignore
└── requirements.txt       # Project dependencies

```

## 🚀 12. How to Run
```text
1. Clone the repository:

Bash
git clone [https://github.com/Aryan140314/Multi-Scale-Pathogenic-Gene-and-Risk-Modeling-Using-Machine-Learning-
and-Graph-Neural-Networks.git](https://github.com/Aryan140314/Multi-Scale-Pathogenic-Gene-and-Risk-Modeling-
Using-Machine-Learning-and-Graph-Neural-Networks.git)
cd Multi-Scale-Pathogenic-Gene-and-Risk-Modeling-Using-Machine-Learning-and-Graph-Neural-Networks

2. Create a virtual environment and install dependencies:

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

3. Set up API Keys:
Add your Gemini API key to app/.streamlit/secrets.toml to enable the AI Explainer mode.
Ini, TOML
GEMINI_API_KEY = "your_api_key_here"

4. Launch the Dashboard:
Bash
cd app
streamlit run app.py

```

## 📦 13. Requirements
```text
Python 3.10+
torch, torch_geometric
scikit-learn, xgboost
pandas, numpy
streamlit, plotly, pyvis
shap
google-genai
Pyorch version: 2.7.1+cu118
CUDA version used by PyTorch: 11.8
cuDNN version: 90100
```

## ⚠️ 14. Limitations
```text
Network Dependency: The GNN branch relies heavily on known PPI networks. Genes with missing or incomplete
interaction data (orphan nodes) rely predominantly on the ML branch.Static Snapshot: The predictions are
based on a static snapshot of the ClinVar database and require periodic retraining to incorporate newly discovered variants.
```
## 🔮 15. Future Enhancements
```text
Dynamic API Integration: Fetch live variant updates directly from the NCBI E-utilities API.
Cancer-Specific Stratification: Transition from binary (Pathogenic/Benign) to multi-class prediction indicating specific disease or tumor susceptibilities.
3D Protein Structures: Incorporate AlphaFold embeddings to analyze spatial mutation clustering.
```

## 📄 16. License
```text
This project is licensed under the MIT License - see the LICENSE file for details.
```

## 🙏 17. Acknowledgements
```text
The developers of PyTorch & PyTorch Geometric and Streamlit for their incredible open-source frameworks.
NCBI and the STRING consortium for providing the foundational biological datasets.
```
