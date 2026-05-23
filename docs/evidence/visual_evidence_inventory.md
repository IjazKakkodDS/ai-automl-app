# Visual Evidence Inventory

**Phase:** 6B.3E  
**Date:** 2026-05-23  
**Capture method:** Playwright headless Chromium 1.60.0 (UI routes);
direct copy from `reports/` (generated artifacts)  
**App state:** Docker Compose stack, `automl-backend (healthy)` + `automl-frontend Up`

---

## UI Route Screenshots

| File | Route | What It Proves | Size (bytes) | README-ready | Portfolio-card-ready |
|------|-------|----------------|-------------|--------------|---------------------|
| `01_homepage.png` | `/` | Platform hero, nav cards, sample charts render in browser | 114,834 | Yes | Yes |
| `02_preprocessing.png` | `/preprocessing` | CSV upload interface and pipeline config controls exist | 66,610 | Yes | No (too form-heavy) |
| `03_eda.png` | `/eda` | EDA route renders with dataset input and result area | 65,217 | No | No |
| `04_feature_engineering.png` | `/feature-engineering` | Feature suggestion interface renders | 54,050 | No | No |
| `05_model_training.png` | `/model-training` | Multi-model selector and training controls render | 220,613 | Yes | Yes |
| `06_evaluate.png` | `/evaluate` | Evaluation interface renders with model list area | 287,149 | Yes | No |
| `07_forecasting.png` | `/forecasting` | Forecast model selector and period config render | 247,984 | Yes | No |
| `08_insights.png` | `/insights` | AI insights text input and Ollama model selector | 98,749 | Yes | Yes |
| `09_rag.png` | `/rag` | RAG query input and top-K selector render | 93,253 | Yes | Yes |
| `10_reports.png` | `/reports` | Report generation interface renders | 44,431 | No | No |

---

## Generated Report Artifacts

These files are real pipeline outputs produced by previous system runs.
They were copied from `reports/` into `docs/evidence/screenshots/`.

| File | Source Path | What It Proves | Size (bytes) | README-ready | Portfolio-card-ready |
|------|-------------|----------------|-------------|--------------|---------------------|
| `11_shap_random_forest.png` | `reports/random_forest_shap_summary.png` | SHAP beeswarm plot generated for Random Forest model | 17,995 | Yes | Yes |
| `12_shap_xgboost.png` | `reports/xgboost_shap_summary.png` | SHAP beeswarm plot generated for XGBoost model | 17,599 | Yes | Yes |
| `13_prophet_forecast.png` | `reports/prophet_forecast_plot.png` | Prophet 30-period sales forecast chart | 43,109 | Yes | Yes |
| `14_correlation_heatmap.png` | `reports/correlation_heatmap.png` | Feature correlation heatmap from EDA pipeline | 29,334 | Yes | No |
| `15_feature_importance.png` | `reports/feature_importance.png` | Feature importance bar chart from training pipeline | 13,303 | Yes | Yes |

---

## Capture Metadata

```json
{
  "captured": "2026-05-23T16:32:37.741Z",
  "tool": "playwright@1.60.0 headless chromium",
  "viewport": "1440x900",
  "wait_strategy": "networkidle + 2500ms",
  "base_url": "http://localhost:3000",
  "backend_url": "http://localhost:8000",
  "backend_status": "healthy (docker compose healthcheck passed)"
}
```

---

## Priority for README Use

Recommended order for README hero section:

1. `01_homepage.png` - overall platform impression
2. `05_model_training.png` - shows ML model selector richness
3. `11_shap_random_forest.png` - explainability evidence
4. `13_prophet_forecast.png` - time-series capability
5. `08_insights.png` - LLM/AI integration

---

## Notes

- Screenshots captured with app in default state (no prior pipeline run loaded)
- SHAP and forecast artifacts were generated during earlier development runs
  on a retail sales dataset
- No metrics were added to screenshots after capture
- No screenshots from other projects were used
