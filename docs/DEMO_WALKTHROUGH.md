# AI AutoML Intelligence Platform - Demo Walkthrough

**Phase:** 6B.3E  
**Date:** 2026-05-23  
**Scope:** Local demo walkthrough only. Not a production deployment.  
**App version:** Next.js 15.2.1 frontend + FastAPI 0.110.0 backend  
**Run method:** Docker Compose v5.1.0 (verified local interface)

---

## 1. Demo Scope

This walkthrough documents the verified local interface of the AI AutoML
Intelligence Platform as it runs on a local machine via Docker Compose. It
covers the full pipeline from data upload through model training, evaluation,
time-series forecasting, AI-generated insights, RAG query, and report
generation.

**What this walkthrough proves:**
- The platform ships a working full-stack ML pipeline UI
- All 10 frontend routes render and are reachable
- The backend FastAPI API serves 16 routes across 10 pipeline modules
- The pipeline can be run end-to-end on a local machine
- Generated report artifacts (SHAP plots, forecast charts, heatmaps) are real
  outputs produced by the system

**What this walkthrough does not claim:**
- Production or cloud deployment
- Customer usage evidence
- Benchmark-speed real-time inference
- Any external API keys required for core functionality

---

## 2. Prerequisites

| Requirement               | Version          | Notes                                    |
|---------------------------|------------------|------------------------------------------|
| Docker Desktop            | 29.3.0+          | WSL2 backend on Windows                  |
| Docker Compose            | v5.1.0+          | Bundled with Docker Desktop              |
| Ollama (optional)         | any              | For AI Insights and RAG endpoints only   |
| Port 8000                 | free             | Backend FastAPI                          |
| Port 3000                 | free             | Frontend Next.js                         |

Ollama is optional. All pipeline stages (preprocessing, EDA, feature
engineering, model training, evaluation, forecasting, reports) work without
Ollama. Only the `/ai-insights/generate/` and `/rag/query-summarize` endpoints
require a running Ollama instance on the host.

---

## 3. Startup Commands

```bash
# From the project root
docker compose up -d

# Verify both containers are running and backend is healthy
docker compose ps

# Expected output:
# NAME              IMAGE                   STATUS
# automl-backend    automl-backend:local    Up (healthy)   0.0.0.0:8000->8000/tcp
# automl-frontend   automl-frontend:local   Up             0.0.0.0:3000->3000/tcp

# Quick API check
curl http://localhost:8000/
# {"message":"AI AutoML Backend is running. This is the RAG + Agentic AI pipeline."}

# Open frontend
# Navigate to http://localhost:3000 in a browser
```

To stop:

```bash
docker compose down
```

---

## 4. Dataset Used

The platform is dataset-agnostic. For this walkthrough, a retail sales CSV was
used (available in the project's `original_data/` folder). It includes
columns: `Date`, `Product`, `Region`, `Category`, `Sales`, `Units Sold`,
`Discount`.

Any CSV file with numeric and categorical columns can be used in place of this
dataset. The preprocessing pipeline handles missing values, outlier detection,
encoding, and scaling automatically.

---

## 5. Step 1: Open Platform Homepage

**URL:** `http://localhost:3000`  
**Screenshot:** [01_homepage.png](evidence/screenshots/01_homepage.png)

The homepage displays:
- Platform hero section with project title and description
- Quick-access navigation cards for each pipeline stage
- Live area chart showing sample training metrics (epochs vs accuracy/loss)
- Pie chart showing sample class distribution
- Status badges for each pipeline module

The homepage is a static dashboard view; no API calls are required to render
it.

---

## 6. Step 2: Upload and Preprocess Dataset

**URL:** `http://localhost:3000/preprocessing`  
**Screenshot:** [02_preprocessing.png](evidence/screenshots/02_preprocessing.png)

The preprocessing page provides:
- CSV file upload control
- Preview table showing original data (first rows)
- Pipeline configuration options:
  - Numeric imputation method: KNN, mean, median, most_frequent
  - KNN neighbors count
  - Outlier removal checkbox (IQR method)
  - Low-variance feature removal
  - Date column handling
  - Encoding strategy for categorical features
  - Scaling method: StandardScaler, MinMaxScaler, RobustScaler
  - Target column selection for stratification

**API called:** `POST /preprocessing/run/`

After preprocessing, the platform stores a `dataset_id` in browser
`localStorage` which subsequent steps use to retrieve the processed dataset
without re-uploading.

---

## 7. Step 3: Run Exploratory Data Analysis

**URL:** `http://localhost:3000/eda`  
**Screenshot:** [03_eda.png](evidence/screenshots/03_eda.png)

The EDA page:
- Accepts a CSV file upload or reads the preprocessed dataset by `dataset_id`
- Sends data to `POST /eda/run/`
- Returns:
  - Summary statistics table (count, mean, std, min, max per column)
  - Missing values report
  - Feature distributions (histograms, countplots)
  - Pairplot
  - Correlation heatmap

Generated EDA artifacts are saved to `reports/figures/`. Example generated
outputs included as evidence:

- [14_correlation_heatmap.png](evidence/screenshots/14_correlation_heatmap.png)
  - Correlation matrix across numeric features

---

## 8. Step 4: Feature Engineering

**URL:** `http://localhost:3000/feature-engineering`  
**Screenshot:** [04_feature_engineering.png](evidence/screenshots/04_feature_engineering.png)

The feature engineering page:
- Uses the preprocessed dataset by `dataset_id`
- Calls `POST /feature-engineering/suggest/`
- Returns automated feature engineering suggestions (polynomial features,
  interaction terms, date decomposition)
- Suggestions are displayed as a text report

Generated output is saved to `reports/feature_engineering_suggestions.txt`.

---

## 9. Step 5: Train Model

**URL:** `http://localhost:3000/model-training`  
**Screenshot:** [05_model_training.png](evidence/screenshots/05_model_training.png)

The model training page supports:
- Dataset source: original upload or preprocessed session data
- Target column selection
- Task type: regression or classification
- Model selection (multi-select):
  - Regression: LinearRegression, RandomForestRegressor,
    GradientBoostingRegressor, SVR, XGBRegressor
  - Classification: LogisticRegression, RandomForestClassifier,
    GradientBoostingClassifier, SVC, XGBClassifier
- Evaluation metric selection: RMSE, R2, MAE, Accuracy, F1, Precision, Recall
- Test split ratio

**API called:** `POST /model-training/train/`

Training results include:
- Per-model metric scores
- SHAP summary plots saved to `reports/`
- Pred vs actual plots
- Residual plots

Generated SHAP evidence:

| Plot | File |
|------|------|
| Random Forest SHAP summary | [11_shap_random_forest.png](evidence/screenshots/11_shap_random_forest.png) |
| XGBoost SHAP summary | [12_shap_xgboost.png](evidence/screenshots/12_shap_xgboost.png) |
| Feature importance | [15_feature_importance.png](evidence/screenshots/15_feature_importance.png) |

---

## 10. Step 6: Review Evaluation Outputs

**URL:** `http://localhost:3000/evaluate`  
**Screenshot:** [06_evaluate.png](evidence/screenshots/06_evaluate.png)

The evaluate page:
- Lists trained models available in the current session
- Accepts a dataset for evaluation (original upload or preprocessed)
- Calls `POST /evaluation/run/`
- Returns evaluation metrics for the selected model on the provided dataset
- Optionally shows per-class breakdown for classification tasks

---

## 11. Step 7: Time-Series Forecasting

**URL:** `http://localhost:3000/forecasting`  
**Screenshot:** [07_forecasting.png](evidence/screenshots/07_forecasting.png)

The forecasting page:
- Accepts a time-series CSV or uses the preprocessed dataset
- Model choice: Prophet or ARIMA
- Target column: configurable (default `Sales`)
- Forecast period: configurable (default 30 periods)

**API called:** `POST /forecasting/run/`

Returns:
- Forecast CSV results
- Forecast plot saved to `reports/`

Generated forecast evidence:

- [13_prophet_forecast.png](evidence/screenshots/13_prophet_forecast.png)
  - Prophet 30-period forecast on retail Sales data

---

## 12. Step 8: Generate AI Insights (Ollama Required)

**URL:** `http://localhost:3000/insights`  
**Screenshot:** [08_insights.png](evidence/screenshots/08_insights.png)

The AI insights page:
- Accepts manual EDA and model summaries, or reads from session data
- Model choice: Ollama model (default `mistral`)
- Calls `POST /ai-insights/generate/`

**Requires Ollama** running on host at `http://localhost:11434`. Without
Ollama, this endpoint returns an error; all other pipeline stages work normally.

The platform passes EDA and training summaries to the Ollama LLM and returns
narrative data science insights.

---

## 13. Step 9: RAG Query (Ollama Required)

**URL:** `http://localhost:3000/rag`  
**Screenshot:** [09_rag.png](evidence/screenshots/09_rag.png)

The RAG interface:
- Accepts a free-text query
- Top-K document retrieval setting
- Calls `POST /rag/query-summarize`
- Uses ChromaDB + sentence-transformers (all-MiniLM-L6-v2) for vector search
- Passes retrieved context to Ollama for answer generation

**Requires Ollama** on host. Without Ollama, query returns an error.

---

## 14. Step 10: Generate Report

**URL:** `http://localhost:3000/reports`  
**Screenshot:** [10_reports.png](evidence/screenshots/10_reports.png)

The reports page:
- Calls `POST /reports/generate/` using the current session data
- Generates a comprehensive pipeline report in multiple formats:
  - `reports/full_report.txt`
  - `reports/full_report.html`
  - `reports/full_report.pdf`

---

## 15. Evidence Screenshots

All screenshots were captured using Playwright headless Chromium 1.60.0 against
the locally running Docker Compose stack on 2026-05-23.

| File | Route | Source |
|------|-------|--------|
| [01_homepage.png](evidence/screenshots/01_homepage.png) | `/` | Live Playwright capture |
| [02_preprocessing.png](evidence/screenshots/02_preprocessing.png) | `/preprocessing` | Live Playwright capture |
| [03_eda.png](evidence/screenshots/03_eda.png) | `/eda` | Live Playwright capture |
| [04_feature_engineering.png](evidence/screenshots/04_feature_engineering.png) | `/feature-engineering` | Live Playwright capture |
| [05_model_training.png](evidence/screenshots/05_model_training.png) | `/model-training` | Live Playwright capture |
| [06_evaluate.png](evidence/screenshots/06_evaluate.png) | `/evaluate` | Live Playwright capture |
| [07_forecasting.png](evidence/screenshots/07_forecasting.png) | `/forecasting` | Live Playwright capture |
| [08_insights.png](evidence/screenshots/08_insights.png) | `/insights` | Live Playwright capture |
| [09_rag.png](evidence/screenshots/09_rag.png) | `/rag` | Live Playwright capture |
| [10_reports.png](evidence/screenshots/10_reports.png) | `/reports` | Live Playwright capture |
| [11_shap_random_forest.png](evidence/screenshots/11_shap_random_forest.png) | `reports/` | Generated artifact |
| [12_shap_xgboost.png](evidence/screenshots/12_shap_xgboost.png) | `reports/` | Generated artifact |
| [13_prophet_forecast.png](evidence/screenshots/13_prophet_forecast.png) | `reports/` | Generated artifact |
| [14_correlation_heatmap.png](evidence/screenshots/14_correlation_heatmap.png) | `reports/` | Generated artifact |
| [15_feature_importance.png](evidence/screenshots/15_feature_importance.png) | `reports/` | Generated artifact |

---

## 16. Known Limitations

| Limitation | Detail |
|------------|--------|
| Ollama not containerized | AI Insights and RAG require Ollama on host; those UI screens show empty state without it |
| No authentication | Platform has no login system; suited for local/demo use |
| Session state in localStorage | Session data (`dataset_id`, `session_id`) is stored in browser localStorage; clearing it resets pipeline state |
| Large model images | Docker images include full ML stack (~17.6 GB backend); first build is slow |
| No persistent volume | Uploaded data and generated reports are not persisted across `docker compose down` |

---

## 17. Claim-Safe Conclusion

This walkthrough demonstrates a verified local interface for a full-stack ML
automation pipeline. All 10 frontend routes were reachable and rendered
correctly in the local Docker Compose environment. Generated report artifacts
(SHAP plots, forecast charts, correlation heatmaps) are real pipeline outputs
produced by the system during prior test runs.

This is not a production deployment, not a cloud deployment, and not customer
usage evidence. It is a local portfolio demonstration of a working AutoML
system.
