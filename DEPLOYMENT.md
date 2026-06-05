# Streamlit Cloud Deployment Guide

## Quick Start - Deploy to Streamlit Cloud

### Prerequisites
- GitHub account with this repository pushed
- Streamlit Community Cloud account (free at https://streamlit.io/cloud)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Select branch: `main`
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configuration**
   The `.streamlit/config.toml` file contains:
   - Custom color theme (medical/genomics aesthetic)
   - Security settings (XSRF protection, headless mode)
   - Client settings (error details, upload limits)

### Project Structure for Deployment

```
mutation/
├── streamlit_app.py           ← Entry point for Streamlit Cloud
├── requirements.txt           ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Streamlit configuration
├── README.md
├── app/
│   ├── app.py                ← Main Streamlit application
│   └── lib/                  ← JavaScript libraries & CSS
├── data/
│   └── processed/            ← Feature CSVs and edge lists
├── models/                   ← Trained ML & GNN models
└── codes/                    ← Jupyter notebooks (optional)
```

### Data & Models

Ensure the following files are in the repository:
- **Data:** `data/processed/*.csv`
- **Models:** 
  - ML models: `models/*.pkl` (Random Forest, XGBoost, SVM, etc.)
  - GNN models: `models/gene_gnn_model.pt`, `models/gene_gat_model.pt`
  - Scalers: `models/*_scaler.pkl`

### Running Locally (Testing Before Deploy)

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### Troubleshooting

**Issue:** "ModuleNotFoundError: No module named 'app'"
- Solution: Ensure `streamlit_app.py` is at the root and imports correctly

**Issue:** Models/data files not found
- Solution: Verify file paths in `app/app.py` use relative paths
- Ensure all files are committed to Git (not in .gitignore)

**Issue:** GPU/CUDA not available on Cloud
- Solution: App automatically falls back to CPU (already configured in code)

### Performance Notes

- First run may take 2-3 minutes (model loading, caching)
- Streamlit Cloud has limited memory (~1GB for free tier)
- Caching is enabled (`@st.cache_data`, `@st.cache_resource`) for optimal performance

### Monitoring

Once deployed, you can:
- View logs in the Streamlit Cloud dashboard
- Check app health and usage metrics
- Configure auto-deploy on push to main branch

---

For more info: https://docs.streamlit.io/streamlit-cloud
