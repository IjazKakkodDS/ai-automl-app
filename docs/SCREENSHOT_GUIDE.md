# Screenshot Guide

**Project:** AI AutoML Intelligence Platform  
**Last updated:** 2026-05-23

This guide documents how to refresh or add screenshots for portfolio and
documentation use. Follow these steps to capture real screenshots from the
actual running system.

---

## 1. Start the App

### Option A: Docker Compose (recommended, verified)

```bash
# From project root
docker compose up -d
docker compose ps   # wait until automl-backend shows (healthy)
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:3000`

Verify:

```bash
curl http://localhost:8000/
# Expected: {"message":"AI AutoML Backend is running..."}
```

### Option B: Local dev server

If Docker is unavailable:

```bash
# Terminal 1 - backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - frontend
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`

---

## 2. Automated Playwright Capture

A capture script is provided at `scripts/capture_screenshots.js`. It
uses Playwright headless Chromium to snapshot all 10 UI routes.

### Prerequisites

```bash
# Install Playwright in a temporary directory (avoids project-level warnings)
mkdir %TEMP%\pw_capture
cd %TEMP%\pw_capture
npm install playwright
npx playwright install chromium
```

### Run the capture

```bash
cd %TEMP%\pw_capture
copy "path\to\project\scripts\capture_screenshots.js" .
node capture_screenshots.js
```

Screenshots are written to `docs/evidence/screenshots/` in the project root.
A `manifest.json` is written alongside them with capture timestamps and file
sizes.

### Move screenshots to project

If running from outside the project directory:

```bash
# PowerShell
Copy-Item "$env:TEMP\docs\evidence\screenshots\*" `
  "path\to\project\docs\evidence\screenshots\" -Force
```

---

## 3. Screenshot Targets

| File | URL | Description |
|------|-----|-------------|
| `01_homepage.png` | `/` | Platform hero, navigation cards, sample charts |
| `02_preprocessing.png` | `/preprocessing` | CSV upload, pipeline config options |
| `03_eda.png` | `/eda` | EDA form, statistics display area |
| `04_feature_engineering.png` | `/feature-engineering` | Feature suggestion interface |
| `05_model_training.png` | `/model-training` | Model selector, metric selector, training controls |
| `06_evaluate.png` | `/evaluate` | Model evaluation interface |
| `07_forecasting.png` | `/forecasting` | Forecast model selector, period config |
| `08_insights.png` | `/insights` | AI Insights text input and output |
| `09_rag.png` | `/rag` | RAG query text field, top-K selector |
| `10_reports.png` | `/reports` | Report generation interface |

Report artifact screenshots (copy from `reports/` after a pipeline run):

| File | Source path | Description |
|------|-------------|-------------|
| `11_shap_random_forest.png` | `reports/random_forest_shap_summary.png` | SHAP beeswarm |
| `12_shap_xgboost.png` | `reports/xgboost_shap_summary.png` | SHAP beeswarm |
| `13_prophet_forecast.png` | `reports/prophet_forecast_plot.png` | Prophet forecast chart |
| `14_correlation_heatmap.png` | `reports/correlation_heatmap.png` | Feature correlation heatmap |
| `15_feature_importance.png` | `reports/feature_importance.png` | Feature importance bar chart |

---

## 4. Naming Convention

- UI screenshots: `NN_routename.png` (two-digit zero-padded prefix)
- Report artifact screenshots: `NN_artifactname.png`
- All lowercase with underscores, no spaces
- Do not rename or renumber existing screenshots without updating
  `DEMO_WALKTHROUGH.md` and `visual_evidence_inventory.md` links

---

## 5. Viewport and Quality Settings

The Playwright script uses:

```js
viewport: { width: 1440, height: 900 }
waitUntil: 'networkidle'
waitForTimeout: 2500   // ms additional settle time after networkidle
fullPage: false         // viewport-only, not full-scroll
```

For full-page captures (useful for long pages like model training results):

```js
await page.screenshot({ path: outPath, fullPage: true });
```

---

## 6. What Not to Capture

- Do not capture screens with visible local file paths in browser address bar
  unless the path does not reveal sensitive information
- Do not capture screens showing `.env` values, API keys, or tokens
- Do not capture Docker Desktop internal dashboards as evidence for this project
- Do not capture screens from other projects or repos
- Do not use screenshots from staging or test environments of other products
- Do not edit metrics or text into screenshots after capture

---

## 7. Secrets Checklist Before Publishing

Before committing or sharing screenshots:

1. Check that no screenshot shows:
   - Text containing `sk-`, `gsk_`, `OPENAI_API_KEY`, or similar tokens
   - Browser URL bar with `localhost` AND sensitive query parameters
   - Terminal output with printed secrets
2. The `.env` file is in `.dockerignore` and `.gitignore`
3. The `OLLAMA_HOST` value is a safe localhost reference only

---

## 8. Refreshing Screenshots After UI Changes

When the frontend is updated:

1. `docker compose down`
2. `docker compose build` (rebuild frontend image)
3. `docker compose up -d`
4. Wait for `automl-backend` to show `(healthy)` in `docker compose ps`
5. Re-run `scripts/capture_screenshots.js`
6. Copy new PNGs to `docs/evidence/screenshots/`
7. Update `docs/evidence/visual_evidence_inventory.md` with new sizes and dates
8. Commit with message: `Refresh UI screenshots after [description of change]`

---

## 9. Stopping the App After Capture

```bash
docker compose down
```

This removes both containers and the compose network. Images are retained and
the next `docker compose up -d` will start containers without rebuilding.
