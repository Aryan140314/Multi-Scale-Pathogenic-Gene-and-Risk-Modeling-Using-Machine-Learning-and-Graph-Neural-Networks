import pako
import base64

def encode_mermaid(markup):
    compressed = pako.compress(markup.encode('utf-8'))
    encoded = base64.b64encode(compressed).decode('utf-8')
    return f"https://mermaid.live/view#pako:{encoded}"

# List of markups
markups = {
    "logistic_regression": """graph TD
    A[Input Features] --> B[Linear Combination: w·x + b]
    B --> C[Sigmoid Function: σ(z)]
    C --> D[Probability Output: P(y=1|x)]
    D --> E[Binary Classification: Threshold at 0.5]""",
    "svm": """graph TD
    A[Input Features] --> B[Map to Higher Dimension: φ(x)]
    B --> C[Find Optimal Hyperplane]
    C --> D[Support Vectors: Critical Points]
    D --> E[Decision Function: sign(w·φ(x) + b)]
    E --> F[Classification]""",
    "random_forest": """graph TD
    A[Training Data] --> B[Build Multiple Decision Trees]
    B --> C[Random Subset of Features]
    C --> D[Random Bootstrap Samples]
    D --> E[Ensemble of Trees]
    E --> F[Voting/Averaging]
    F --> G[Final Prediction]""",
    "gradient_boosting": """graph TD
    A[Initial Prediction] --> B[Compute Residuals]
    B --> C[Fit Weak Learner to Residuals]
    C --> D[Update Prediction: F_m = F_{m-1} + α_m * h_m]
    D --> E[Repeat for M Iterations]
    E --> F[Final Model: Sum of Weak Learners]""",
    "xgboost": """graph TD
    A[Input Data] --> B[Objective Function: Loss + Regularization]
    B --> C[Gradient Boosting Framework]
    C --> D[Tree Pruning: max_depth, min_child_weight]
    D --> E[Regularization: λ, γ]
    E --> F[Parallel Processing]
    F --> G[Optimized Predictions]""",
    "stacking_ensemble": """graph TD
    A[Training Data] --> B[Train Base Learners]
    B --> C[Logistic Regression]
    B --> D[SVM]
    B --> E[Random Forest]
    B --> F[XGBoost]
    C --> G[Generate Predictions]
    D --> G
    E --> G
    F --> G
    G --> H[Meta Learner: Logistic Regression]
    H --> I[Final Prediction]""",
    "graphsage": """graph TD
    A[Graph Nodes & Edges] --> B[Neighborhood Sampling]
    B --> C[Aggregate Neighbor Features]
    C --> D[Mean Pooling: h_v = σ(W · MEAN({h_u | u ∈ N(v)} ∪ h_v))]
    D --> E[Update Node Embeddings]
    E --> F[Multiple Layers]
    F --> G[Final Node Representations]""",
    "gat": """graph TD
    A[Graph Nodes & Edges] --> B[Compute Attention Scores]
    B --> C[Attention Mechanism: α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))]
    C --> D[Weighted Aggregation: h_i' = σ(∑_j α_ij W h_j)]
    D --> E[Multi-Head Attention]
    E --> F[Concatenate Heads]
    F --> G[Final Node Embeddings]"""
}

for name, markup in markups.items():
    url = encode_mermaid(markup)
    print(f"{name}: {url}")