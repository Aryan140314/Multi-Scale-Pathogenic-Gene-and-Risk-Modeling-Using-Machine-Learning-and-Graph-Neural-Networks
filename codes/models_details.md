# 5.5 Performance Analysis of Individual Models

## 5.5.1 Overview
This section summarizes the architecture of every model used in the PathoGAT system, including the tabular ML branch and the graph-based GNN branch. The goal is to clearly explain how each model processes input features and contributes to a final pathogenicity prediction.

## 5.5.2 Tabular ML Models

### Logistic Regression
- Input: gene-level numeric features from `final_gene_features.csv`.
- Architecture: linear combination of features, followed by a sigmoid activation.
- Output: probability of pathogenic vs benign.
- Role: provides a strong linear baseline and contributes to the ensemble.

### Support Vector Machine (SVM)
- Input: same gene-level numeric features.
- Architecture: hyperplane learned in feature space, with support vectors defining the margin.
- Output: class probability computed from decision function or calibrated probabilities.
- Role: captures nonlinear boundaries when paired with kernels, improving robustness.

### Random Forest
- Input: gene-level numeric features.
- Architecture: ensemble of decision trees trained on bootstrapped samples of data.
- Prediction: individual tree votes are averaged, producing a final pathogenicity score.
- Role: reduces variance and models feature interactions effectively.

### Gradient Boosting
- Input: gene-level numeric features.
- Architecture: sequence of decision trees where each new tree fits the residual error of the previous ensemble.
- Prediction: outputs are summed across trees to form the final score.
- Role: captures complex nonlinear patterns through weighted residual correction.

### XGBoost
- Input: gene-level numeric features.
- Architecture: highly optimized gradient boosting implementation with regularization (L1/L2), column subsampling, and learning-rate control.
- Prediction: tree ensemble output with strong generalization and faster training.
- Role: provides one of the strongest tabular predictors in the ML branch.

### Stacking Ensemble
- Input: predictions from base learners (Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting).
- Architecture:
  - Level 0: base learners each produce a probability estimate.
  - Level 1: meta learner (Logistic Regression) receives base predictions and learns the optimal combination.
- Output: final ensemble probability.
- Role: consolidates strengths of all tabular models and improves overall performance.

## 5.5.3 Graph Neural Network Models

### GraphSAGE
- Input: node features from genes + gene interaction graph edges.
- Architecture:
  - Neighborhood sampling and feature aggregation from each gene's immediate neighbors.
  - Two-layer GraphSAGE network with LayerNorm and ELU activation.
  - Output layer predicts pathogenic vs benign for each gene node.
- Role: learns gene risk from local graph structure and mutation features.

### Graph Attention Network (GAT)
- Input: same node features and graph structure.
- Architecture:
  - Attention mechanism computes importance weights for each neighbor.
  - Multi-head attention is used on the first layer, followed by ELU and dropout.
  - Final attention layer produces node-level logits.
- Output: probability of pathogenicity per gene node.
- Role: assigns learnable attention to interactions so important neighbor relationships weigh more.

## 5.5.4 Hybrid and Consensus Prediction
- The final PathoGAT prediction is derived by combining the ML ensemble and the GNN ensemble.
- In the main application, the ML branch prediction and the averaged GNN branch prediction are blended to produce a consensus risk score.
- This hybrid approach ensures both mutation-level tabular signals and network-context signals are integrated.

## 5.5.5 Visual Summary of Model Architectures
Use the following conceptual diagram structure to visualize how each model works in the report:

1. **Logistic Regression / SVM / Random Forest / Gradient Boosting / XGBoost**
   - Input Features -> Base Model -> Prediction
   - For Random Forest: multiple decision trees -> Majority vote
   - For Gradient Boosting / XGBoost: sequence of trees -> Residual fitting -> Final score

2. **Stacking Ensemble**
   - Input Features -> Base Learners (LR, SVM, RF, XGB, GB) -> Predictions
   - Base predictions -> Meta Learner (LR) -> Final Ensemble Prediction

3. **GraphSAGE**
   - Gene Graph + Node Features -> Neighborhood Aggregation -> Node Representation -> Output Prediction

4. **GAT**
   - Gene Graph + Node Features -> Attention Mechanism -> Weighted Aggregation -> Output Prediction

### Suggested Figure Captions
- **Figure 5.5a:** Tabular ML branch architectures, showing base learners and stacking ensemble flow.
- **Figure 5.5b:** GraphSAGE architecture with neighborhood aggregation and node update.
- **Figure 5.5c:** GAT architecture with attention weights and multi-head aggregation.

## 5.5.6 Notes for Your Own Work
- The diagrams in this project should reflect the actual naming and structure used in the code: `LogisticRegression`, `SVM`, `RandomForest`, `GradientBoost`, `XGBoost`, `StackingEnsemble`, `GraphSAGE`, `PathoGAT (GAT)`.
- Emphasize that the ML branch is feature-driven, while the GNN branch is graph-structure-driven.
- Mention the final consensus score as the combined output used for risk ranking.
