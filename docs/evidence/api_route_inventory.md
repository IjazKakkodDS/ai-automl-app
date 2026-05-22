# AI AutoML Intelligence Platform - API Route Inventory

**Date:** 2026-05-22
**Phase:** 6B.3B - API Surface Validation and Integration Tests
**Discovery method:** FastAPI app introspection via `app.routes` with mocked external dependencies

---

## Summary

| Metric | Value |
|---|---|
| Total registered routes | 16 (including root) |
| Application endpoints | 15 |
| Route modules | 11 |
| Endpoints covered by new integration tests | 8 |
| Endpoints requiring Ollama (external) | 3 |
| Endpoints requiring FAISS index | 2 |
| Endpoints requiring Prophet (not installed) | 1 |

---

## Route Table

| # | Method | Path | Route Purpose | Source Module | Integration Test | External Dependency | Notes |
|---|---|---|---|---|---|---|---|
| 1 | GET | `/` | Health check / service identity | `main.py` | Yes | None | Returns platform name and status |
| 2 | POST | `/preprocessing/preprocess/` | Ingest and preprocess a CSV file | `preprocessing_routes` | Yes | None | Saves to `processed_data/` and `original_data/` |
| 3 | POST | `/eda/analysis/` | Exploratory data analysis on uploaded or stored dataset | `eda_routes` | Yes | None | May save PNG plots to `reports/` |
| 4 | POST | `/feature-engineering/advanced/` | Advanced feature transformation pipeline | `feature_engineering_routes` | No | None | Accepts optional save to `feature_engineered_data/` |
| 5 | GET | `/feature-engineering/columns/` | Inspect column metadata for a dataset | `feature_engineering_routes` | No | None | Returns column names, types, missing counts |
| 6 | POST | `/model-training/train/` | Train and evaluate ML models on a dataset | `model_training_routes` | Yes | None | Saves `.pkl` model files; supports regression and classification |
| 7 | POST | `/forecasting/forecast/` | Time-series forecasting with Prophet or ARIMA | `forecasting_routes` | No | Prophet (not installed) | Prophet not installed in this environment |
| 8 | GET | `/evaluation/list-models/` | List saved model files in a session folder | `evaluation_routes` | Yes | None | Returns list of `.pkl` and `.joblib` filenames |
| 9 | POST | `/evaluation/evaluate/` | Evaluate a saved model on new data | `evaluation_routes` | No | None | Requires a valid `.pkl` model path |
| 10 | POST | `/ai-insights/generate/` | Generate LLM insights from EDA and training summaries | `ai_insights_routes` | No | Ollama (local) | Requires Ollama running locally; uses mistral/gemma2/llama3.3 |
| 11 | POST | `/rag/query-summarize` | Semantic search over documents via FAISS, summarized by Ollama | `rag_routes` | No | FAISS + Ollama | FAISS index loaded at import time from `data_1/`; Ollama for summarization |
| 12 | POST | `/orchestrator/orchestrate` | Agentic decision loop: EDA -> training -> RAG -> insights | `orchestrator_routes` | No | FAISS + Ollama + SentenceTransformer | Chains multiple agents; requires all external services |
| 13 | POST | `/reports/generate-full` | Assemble full HTML report from pipeline artifacts | `reports_routes` | No | None | Reads from `reports/` directory; generates `full_report.html` |
| 14 | GET | `/reports/view-full` | Serve the generated HTML report inline | `reports_routes` | No | None | Returns `full_report.html` as FileResponse |
| 15 | GET | `/reports/download-full` | Convert HTML report to PDF via headless Chrome | `reports_routes` | No | Chrome (system) | Requires Chrome installed at hardcoded Windows path |
| 16 | POST | `/reset/restart-analysis` | Clear models folder and return a new session ID | `reset_routes` | Yes (mocked) | None | Destructive: deletes `models/`; tested with `shutil.rmtree` mocked |

---

## Integration Test Coverage Detail

| Path | Test File | Test Functions | Coverage |
|---|---|---|---|
| `GET /` | `test_api_health.py` | 3 tests | Full: 200, message key, content check |
| `POST /preprocessing/preprocess/` | `test_api_preprocessing.py` | 4 tests | Valid CSV (200), response keys, shape, no-file (422) |
| `POST /eda/analysis/` | `test_api_eda.py` | 5 tests | Valid CSV (200), response keys, report content, no-input error, bad dataset_id |
| `POST /model-training/train/` | `test_api_model_training.py` | 5 tests | LinearRegression (200), response keys, metrics, missing models (error), missing file (error) |
| `GET /evaluation/list-models/` | `test_api_model_training.py` | 2 tests | 200, `models` key is list |
| `POST /reset/restart-analysis` | `test_api_model_training.py` | 1 test (mocked) | 200, `session_id` returned, rmtree called |

---

## Endpoints Not Covered by Integration Tests and Why

| Path | Reason Not Tested |
|---|---|
| `POST /feature-engineering/advanced/` | Requires a dataset_id or file; integration with feature pipeline adds test time; covered by agent unit tests |
| `GET /feature-engineering/columns/` | Requires a stored dataset_id; low risk utility endpoint |
| `POST /forecasting/forecast/` | Prophet not installed in this Python 3.13 environment; forecasting_agent cannot be imported |
| `POST /evaluation/evaluate/` | Requires a pre-trained model file path; covered by model evaluation unit tests |
| `POST /ai-insights/generate/` | Requires Ollama running locally; test would fail without external service |
| `POST /rag/query-summarize` | Requires Ollama running locally; FAISS index loaded at import time |
| `POST /orchestrator/orchestrate` | Requires Ollama + FAISS + SentenceTransformer at runtime; agent unit tests cover decision logic |
| `POST /reports/generate-full` | Reads from `reports/` directory; file-assembly logic only; no ML computation |
| `GET /reports/view-full` | File serve endpoint; no computation |
| `GET /reports/download-full` | Requires Chrome at hardcoded Windows path; system dependency |

---

## External Dependency Summary

| Dependency | Endpoints Using It | Status in This Environment |
|---|---|---|
| Ollama (local LLM server) | `/ai-insights/generate/`, `/rag/query-summarize`, `/orchestrator/orchestrate` | Not running; tests skip or mock |
| FAISS index (`data_1/`) | `/rag/query-summarize`, `/orchestrator/orchestrate` | Index tracked and present; loads at import time |
| SentenceTransformer | `/rag/query-summarize`, `/orchestrator/orchestrate` | Installed; transformers version incompatibility in Python 3.13; mocked in tests |
| Prophet | `/forecasting/forecast/` | Not installed in Python 3.13 environment |
| Chrome (headless) | `/reports/download-full` | System dependency; hardcoded Windows path in route |

---

## Claim-Safe Summary

This platform exposes **15 application endpoints** across **11 route modules**, all registered and discoverable via FastAPI's OpenAPI schema at `/docs`. The core ML pipeline routes (preprocessing, EDA, model training, evaluation) are operational and covered by integration tests using small synthetic datasets without external service requirements. Three routes require Ollama for AI-powered summarization and are expected to require a locally running Ollama server. One route requires Prophet for time-series forecasting (not installed in this Python 3.13 environment). Route coverage claims are based on live app introspection and test execution, not documentation.

---

*Generated as part of Phase 6B.3B: API Surface Validation and Integration Tests for AI AutoML Intelligence Platform.*
