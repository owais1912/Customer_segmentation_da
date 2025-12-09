# Customer Segmentation Dashboard

AI-Powered Customer Intelligence Platform for analyzing customer segments, generating marketing campaigns, and predicting customer behavior.

## Features

- 📊 Executive Dashboard with key metrics
- 👤 Customer Intelligence with CLV predictions
- ✉️ AI-Powered Email Campaign Studio
- 🔮 Predictive Analytics (Churn Risk, A/B Testing)
- 📈 Segment Trends Analysis
- 🚨 Anomaly Detection
- 📋 Campaign Management
- 📥 Advanced Data Analyzer

## Live Demo

[Deploy on Streamlit Cloud](https://streamlit.io/)

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Data Requirements

The app expects the following files in the `outputs/` directory:
- `master_segments_kmeans.csv` - Main customer segmentation data
- `cluster_kpi_table.csv` - KPI metrics by cluster
- `segment_drift.csv` - Historical segment data (optional)
- `master_segments_with_anomaly.csv` - Anomaly detection results (optional)

Models should be in the `models/` directory:
- `scaler.joblib`
- `kmeans.joblib`
- `isolation_forest.joblib`

## Environment Variables

Optional:
- `GEMINI_API_KEY` - For AI-powered email generation
- `SMARTSEG_BASE` - Base directory path (defaults to current directory)

## Theme

The dashboard uses a butter yellow (#FFEFB3) and dark green (#013E37) color scheme for a fresh, professional look.
