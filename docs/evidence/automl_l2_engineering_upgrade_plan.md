# AI AutoML Intelligence Platform - L2 Engineering Upgrade Plan

**Phase:** 6B.3 - Engineering Upgrade Plan
**Date:** 2026-05-22
**Prerequisite commits:**
- `d16d8ed` Add AutoML system audit report
- `f84cc69` Align AutoML platform identity and stack claims
- `025692b` Tighten AutoML README deployment wording

---

## Upgrade Objective

Transform the AI AutoML Intelligence Platform from an internally verified ML application into an evidence-backed, portfolio-grade AI platform with:

- A verified and inventoried API surface (15 endpoints across 11 route modules)
- Benchmarked workflow timings for all major pipeline stages
- FastAPI integration tests using TestClient
- Dataset-size-framed scalability evidence
- UI screenshots and a demo walkthrough document
- An elite-standard README with quantitative snapshot, architecture, and evidence index
- Safe and documented environment configuration
- Docker orchestration evidence (backend smoke test confirmed)
- Claim-safe portfolio readiness across all public-facing assets

---

## Current System Snapshot (Baseline)

| Dimension | Current State |
|---|---|
| API route modules | 11 |
| API endpoints | 15 |
| Agent modules | 8 |
| Frontend pages | 10 (excluding _app.js) |
| Pytest test files | 8 (927 total lines) |
| Test type coverage | Unit only (100-row synthetic data) |
| Integration tests | None |
| Frontend tests | None |
| Benchmark scripts | None |
| Report PNG artifacts | 78 |
| UI screenshots in repo | None |
| Architecture diagram | None |
| docker-compose.yml | Missing |
| Frontend Dockerfile | Missing |
| .env.example | Missing |
| GitHub Actions YAML | Missing (folder exists, empty) |
| Known training times | LinearRegression 0.014s, RF 0.080s, GB 0.057s |
| Known dataset sizes | 71 original CSVs (~18MB), 95 processed CSVs (~393MB) |

---

## Phase Map

```
6B.3A - Safety and repository hygiene
6B.3B - API surface validation and integration tests
6B.3C - Benchmark and scalability evidence
6B.3D - Docker and orchestration evidence
6B.3E - Screenshot and demo walkthrough evidence
6B.3F - README elite documentation redesign
6B.3G - Portfolio case study upgrade
```

Phases must be executed in order. Each phase produces evidence consumed by later phases.

---

## Phase 6B.3A: Safety and Repository Hygiene

### Objective

Eliminate security and hygiene risks before any evidence generation or public portfolio exposure.

### Findings requiring action

1. **No .env.example file.** `python-dotenv` is in requirements. Users have no guidance on what environment variables the system expects. If Ollama is running on a non-default host, or if future API keys are added, there is no documented place for them.

2. **Data CSVs tracked in git.** 71 original and 95 processed CSVs were committed before `.gitignore` rules were added. The files are in git history. They appear to be synthetic/test datasets based on column names seen in audit (Sales, Churn, Category, etc.) but must be reviewed before any public portfolio link is shared.

3. **Generated report artifacts committed.** 78 PNG files and multiple CSV/TXT outputs are tracked in git. These are valuable as portfolio evidence but should be reviewed for any sensitive column names or identifying data patterns.

4. **model_reports CSVs committed.** 8 training report CSVs in `model_reports/` are committed. Review for sensitive content before public exposure.

5. **CORS wildcard.** `allow_origins=["*"]` in `main.py`. Acceptable for local development but should be noted in .env.example as a variable to tighten for any public deployment.

### Deliverables

| File | Action |
|---|---|
| `.env.example` | Create with documented variable slots |
| `.gitignore` | Add `model_reports/` and `reports/*.html` and `reports/*.pdf` to prevent future artifact tracking |
| `docs/evidence/data_safety_review.md` | Short review confirming tracked CSVs contain no PII or sensitive data |

### Files to create

- `.env.example`
- `docs/evidence/data_safety_review.md`

### Files to modify

- `.gitignore` (append new exclusions for generated artifacts)

### Commands to run

```bash
git ls-files original_data/ | head -5
git ls-files processed_data/ | head -5
python -c "import pandas as pd; df = pd.read_csv('original_data/<sample>.csv'); print(df.columns.tolist())"
```

### Expected evidence outputs

- `.env.example` committed
- `data_safety_review.md` confirming dataset safety
- Updated `.gitignore` committed

### Claim-safety check

No claims about data handling until data_safety_review.md is complete.

### Definition of done

- `.env.example` exists with at least: `OLLAMA_HOST`, `CORS_ALLOWED_ORIGINS`
- `data_safety_review.md` confirms no PII in tracked datasets
- `.gitignore` updated to exclude future generated HTML/PDF reports and model_reports CSVs

---

## Phase 6B.3B: API Surface Validation and Integration Tests

### Objective

Verify the complete API surface is functional, inventoried, and tested beyond unit-level synthetic data. Produce a route inventory as a README evidence section.

### Current state

- 11 route modules, 15 endpoints identified
- 8 existing pytest unit test files cover agent logic only
- No TestClient-based integration tests exist
- No test for the root `/` health endpoint
- No test for preprocessing upload flow
- No test for training workflow end-to-end

### Deliverables

| File | Action |
|---|---|
| `tests/test_api_health.py` | Health check and root endpoint test |
| `tests/test_api_preprocessing.py` | Upload and preprocess CSV via TestClient |
| `tests/test_api_eda.py` | EDA endpoint with a stored dataset_id |
| `tests/test_api_model_training.py` | Training endpoint with a small dataset |
| `docs/evidence/api_route_inventory.md` | Full API surface documentation |

### Route inventory to produce

| Prefix | Module | Endpoints | Methods |
|---|---|---|---|
| /preprocessing | preprocessing_routes | 1 | POST |
| /eda | eda_routes | 1 | POST |
| /feature-engineering | feature_engineering_routes | 2 | POST |
| /model-training | model_training_routes | 1 | POST |
| /forecasting | forecasting_routes | 1 | POST |
| /evaluation | evaluation_routes | 2 | POST, GET |
| /ai-insights | ai_insights_routes | 1 | POST |
| /rag | rag_routes | 1 | POST |
| /orchestrator | orchestrator_routes | 1 | POST |
| /reports | reports_routes | 3 | POST, GET, GET |
| /reset | reset_routes | 1 | POST |

**Total: 15 endpoints across 11 route modules**

### Integration test design rules

- Use `fastapi.testclient.TestClient`
- Use small synthetic CSV files (50-100 rows) as test payloads
- Do not require Ollama to be running; mock or skip Ollama-dependent endpoints
- Assert HTTP 200 on success paths
- Assert HTTP 422 or 400 on invalid inputs
- Assert response keys match documented contract

### Files to create

- `tests/test_api_health.py`
- `tests/test_api_preprocessing.py`
- `tests/test_api_eda.py`
- `tests/test_api_model_training.py`
- `docs/evidence/api_route_inventory.md`

### Commands to run

```bash
python -m pytest tests/ -v --tb=short
python -m pytest tests/test_api_health.py -v
python -m pytest tests/ --co -q
```

### Expected evidence outputs

- All integration tests pass
- Test output captured as `docs/evidence/test_run_output.txt`
- `api_route_inventory.md` documents every endpoint with method, parameters, response shape

### Claim-safety check

Do not claim "fully tested" -- claim "15 endpoints, 4 integration test modules, 8 unit test modules, all passing"

### Definition of done

- At least 4 integration test modules created
- `python -m pytest tests/` passes with documented results
- `api_route_inventory.md` exists and is accurate

---

## Phase 6B.3C: Benchmark and Scalability Evidence

### Objective

Produce measurable, reproducible timing evidence for all major pipeline stages. This is the core scalability evidence that transforms the platform claim from aspirational to verified.

### Current gaps

- Training times exist in `model_reports/` (0.014s to 0.080s) but are not documented or surfaced
- No API latency measurements
- No P50/P95/P99 for any endpoint
- No RAG query latency
- No EDA timing on a real dataset
- No report generation timing
- Largest tested dataset is ~955KB (~7,000 rows)

### Benchmark targets

| Stage | Tool | Target dataset | Metric |
|---|---|---|---|
| API health endpoint | requests + time | local server | P50/P95 latency ms |
| Preprocessing upload | requests + time | 1,000 row CSV | P50/P95 latency ms |
| EDA analysis | requests + time | 1,000 row CSV | P50/P95 latency ms |
| Model training (regression) | requests + time | 1,000 row CSV | training time sec per model |
| Forecasting (Prophet) | requests + time | time series CSV | forecast time sec |
| RAG query | requests + time | 5-paper index | query + retrieve + summarize ms |
| Report generation | requests + time | existing dataset_id | generation time sec |

### Deliverables

| File | Action |
|---|---|
| `benchmark/run_benchmark.py` | Local API benchmark script |
| `benchmark/benchmark_results.json` | Captured JSON results |
| `docs/evidence/benchmark_report.md` | Human-readable benchmark summary |

### Benchmark script design

```python
# benchmark/run_benchmark.py
# - Start local FastAPI server assumption
# - Run N=10 requests per endpoint
# - Record min, P50, P95, P99, max in ms
# - Output to benchmark_results.json
# - Print summary table
```

### Dataset context framing

| Label | Size | Rows (approx) | Use |
|---|---|---|---|
| Benchmark CSV | 1,000 rows | ~50KB | timing benchmark target |
| Original data median | ~260KB | ~2,000 rows | typical user upload |
| Original data max | ~955KB | ~7,000 rows | largest tested input |
| Processed data total | ~393MB (95 files) | varies | output scale evidence |

### Expected evidence outputs

- `benchmark_results.json` with P50/P95/P99 per endpoint
- `benchmark_report.md` summarizing: latency by endpoint, training times by model, RAG query time, report generation time
- Dataset size context table

### Claim-safety check

Report numbers exactly as measured. Do not round up or present best-case only. Include both P50 and P95. Note if Ollama was running or skipped.

### Definition of done

- `benchmark/run_benchmark.py` exists and runs without error against local server
- `benchmark_results.json` contains at least: health, preprocessing, EDA, and training latency
- `benchmark_report.md` is committed with honest numbers

---

## Phase 6B.3D: Docker and Orchestration Evidence

### Objective

Confirm the Docker backend build works, document the Ollama dependency, and add a `docker-compose.yml` that orchestrates the full local stack.

### Current state

- Backend `Dockerfile` exists and uses `python:3.12-slim`
- Frontend has no Dockerfile
- No `docker-compose.yml`
- Ollama is required for AI insights but is not containerized or documented
- No smoke test output committed

### Decisions to make

| Item | Decision | Rationale |
|---|---|---|
| Frontend Dockerfile | Create it | Next.js can be containerized with node:18-slim; required for docker-compose |
| docker-compose.yml | Create it | Allows single-command local launch of backend + frontend |
| Ollama in Docker | Document only | Ollama container is large; document as a prerequisite, not inline |
| .env.example | Create in 6B.3A | Docker will consume env vars from .env |

### Deliverables

| File | Action |
|---|---|
| `frontend/Dockerfile` | Create Node.js build + serve container |
| `docker-compose.yml` | Create backend + frontend service definition |
| `docs/evidence/docker_smoke_test.md` | Record of docker build + run smoke result |

### Docker architecture target

```
docker-compose up
  -> backend service  (port 8000, python:3.12-slim, FastAPI)
  -> frontend service (port 3000, node:18-slim, Next.js)

Ollama: run separately on host, OLLAMA_HOST configurable via .env
```

### Commands to run and document

```bash
docker build -t automl-backend . --progress=plain 2>&1 | tail -20
docker run --rm -p 8000:8000 automl-backend &
curl http://localhost:8000/ | python -m json.tool
docker-compose build
docker-compose up -d
curl http://localhost:8000/
curl http://localhost:3000/
docker-compose down
```

### Expected evidence outputs

- `docker-compose.yml` working
- `frontend/Dockerfile` working
- `docker_smoke_test.md` with captured output confirming both services start
- Backend `curl /` response recorded as evidence

### Claim-safety check

Only claim Docker works if the build and smoke test actually passes and is documented. Do not claim Ollama is containerized.

### Definition of done

- `docker build` completes without error
- `docker-compose up` starts both services
- Smoke test curl output committed to `docs/evidence/docker_smoke_test.md`

---

## Phase 6B.3E: Screenshot and Demo Walkthrough Evidence

### Objective

Produce real UI screenshots and a written demo walkthrough to support the platform positioning and portfolio evidence cards.

### Screenshot targets

| Screen | URL path | What it proves |
|---|---|---|
| Platform hero / homepage | `/` | Full platform identity visible |
| Data upload / preprocessing | `/preprocessing` | User-facing upload flow |
| EDA analysis view | `/eda` | Profiling and chart outputs |
| Feature engineering | `/feature-engineering` | Transformation controls |
| Model training results | `/model-training` | Multi-model results table |
| Evaluation view | `/evaluate` | Metrics visualization |
| AI insights / Ollama output | `/insights` | LLM integration visible |
| RAG query interface | `/rag` | FAISS RAG in action |
| Reports / download | `/reports` | Report generation output |

### Additional evidence targets

| Asset | Source | What it proves |
|---|---|---|
| SHAP summary plot | `reports/random_forest_shap_summary.png` | Explainability implemented |
| Forecast plot | `reports/prophet_forecast_plot.png` | Time series working |
| Correlation heatmap | `reports/correlation_heatmap.png` | EDA depth |
| Training report CSV table | `model_reports/training_report_*.csv` | Benchmark output readable |
| Full HTML report | `reports/full_report.html` | End-to-end report generation |

### Deliverables

| File | Action |
|---|---|
| `docs/evidence/screenshots/` | New directory for UI screenshots |
| `docs/DEMO_WALKTHROUGH.md` | Step-by-step demo with screenshot references |
| `docs/SCREENSHOT_GUIDE.md` | Instructions for capturing fresh screenshots |

### Demo walkthrough structure

```
docs/DEMO_WALKTHROUGH.md
1. Upload a CSV dataset via /preprocessing
2. Review EDA output at /eda
3. Apply feature engineering at /feature-engineering
4. Train models at /model-training -- observe results table
5. Evaluate saved model at /evaluate
6. Generate AI insights at /insights
7. Query RAG at /rag with a natural language question
8. Generate full report at /reports -- download HTML
9. View SHAP explainability plot from reports/
```

### Claim-safety check

Screenshots must match the actual running application. No mock screenshots. No screenshots from a different version of the app.

### Definition of done

- At least 6 UI screenshots in `docs/evidence/screenshots/`
- `DEMO_WALKTHROUGH.md` references screenshots with file paths
- `SCREENSHOT_GUIDE.md` documents how screenshots were taken

---

## Phase 6B.3F: README Elite Documentation Redesign

### Objective

Rewrite README to match the elite portfolio documentation standard. Every section must reference actual evidence from earlier phases.

### Current README gaps vs elite standard

| Section | Current | Required |
|---|---|---|
| Badge row | 5 badges (some stale) | Updated badges: Python, FastAPI, Tests, Docker |
| System summary | Generic paragraph | Quantitative snapshot table |
| Quantitative snapshot | Missing | 15 endpoints, 8 agents, 10 pages, 78 plots, 8 test files |
| Architecture diagram | Missing | Mermaid or ASCII diagram |
| Model workflow | Text only | Table with models, metrics, training times |
| RAG/agentic workflow | Text only | Workflow diagram + actual query example |
| API surface | Not documented | Route inventory table |
| Test evidence | Not mentioned | Test count, pass rate, command |
| Benchmark evidence | Missing | P50/P95 table from Phase 6B.3C |
| Docker evidence | Instructions only | Smoke test result, docker-compose note |
| Screenshots | None | At least 3 embedded screenshots |
| System scope | Not stated | Explicit scope: what the system does and does not do |
| Evidence index | Missing | Linked index to all docs/evidence files |

### README target structure

```markdown
# AI AutoML Intelligence Platform

[badges]

## System Snapshot
[quantitative table]

## What This System Does
[2-paragraph system description]

## Architecture
[Mermaid diagram or ASCII]

## Pipeline Stages
[table: stage, agent, route, input, output]

## AutoML Workflow Evidence
[model results table with training times]

## RAG and Agentic Components
[component table + example query/output]

## API Surface
[route inventory table: 15 endpoints]

## Testing
[test summary: 8 unit modules + 4 integration modules, command, results link]

## Benchmark Evidence
[P50/P95 latency table per endpoint]

## Docker and Deployment
[docker-compose command + smoke test result]

## Screenshots
[3-6 embedded screenshots with captions]

## System Scope and Boundaries
[explicit list of what the system does and does not claim]

## Evidence Index
[linked list of all docs/evidence/ files]

## Quick Start
[unchanged from current]

## Limitations and Future Work
[updated from current]

## Author
[unchanged]

## License
[unchanged]
```

### Files to modify

- `README.md`

### Expected evidence inputs required before writing

- Phase 6B.3C: benchmark numbers
- Phase 6B.3D: docker-compose smoke result
- Phase 6B.3E: screenshots
- Phase 6B.3B: route inventory and test results

### Claim-safety check

Every number in the README must trace to a file in `docs/evidence/`. No numbers without a source.

### Definition of done

- README contains quantitative snapshot table
- README contains architecture diagram
- README contains API surface table
- README contains benchmark table (even if P50 only)
- README contains at least 2 embedded screenshots
- README contains evidence index linking all docs/evidence files

---

## Phase 6B.3G: Portfolio Case Study Upgrade

### Objective

Update the portfolio's AutoML case study component to reference only verified evidence from the upgrade cycle.

### Current portfolio state (from audit)

- Portfolio public name already updated to "AI AutoML Intelligence Platform"
- Evidence images: `apps/web/public/evidence/automl-dashboard/evidence-snapshot.png` and `product-interface.png` exist in portfolio repo
- These images may predate the upgrade cycle and may not reflect current UI

### Portfolio evidence card plan

| Card | Thumbnail type | Evidence source | What it proves |
|---|---|---|---|
| Product Interface | UI screenshot | `docs/evidence/screenshots/homepage.png` | Platform exists as a navigable system |
| Architecture Workflow | Mermaid/ASCII diagram | `docs/evidence/architecture_diagram.md` | System architecture is real and documented |
| AutoML Pipeline Evidence | Results table screenshot | `model_reports/training_report_*.csv` or screenshot | Multi-model comparison with real metrics |
| RAG and Agentic Evidence | RAG query screenshot | `docs/evidence/screenshots/rag.png` | FAISS retrieval + Ollama output is real |
| Benchmark Evidence | Latency table screenshot | `docs/evidence/benchmark_report.md` | Scalability verified with timing numbers |
| Report and Explainability Evidence | SHAP plot or full report screenshot | `reports/random_forest_shap_summary.png` | Explainability implemented with real output |

### Deliverables

| File | Action |
|---|---|
| `apps/web/public/evidence/automl-dashboard/` | Update with new screenshots from Phase 6B.3E |
| Portfolio `AutoMLCaseStudy.tsx` | Update metrics and evidence references |
| Portfolio `projects.ts` | Confirm product name, tech stack, description match repo |

### Claim-safety check

Every metric in the portfolio case study must match a number verifiable in this repo. No claims not backed by docs/evidence/.

### Definition of done

- Portfolio case study references at least 4 verified evidence cards
- Evidence images in portfolio reflect current UI after Phase 6B.3E screenshots
- All metrics in portfolio case study traceable to `docs/evidence/` sources

---

## Required Metrics to Produce

The following metrics must be generated, measured, and committed before the README redesign and portfolio update can proceed.

| Metric | Source phase | Target value |
|---|---|---|
| API route count | 6B.3B | 15 endpoints confirmed |
| Agent module count | Audit | 8 confirmed |
| Frontend page count | Audit | 10 confirmed |
| Test file count (unit) | Audit | 8 confirmed |
| Integration test count | 6B.3B | Target: at least 4 new modules |
| Total test assertions | 6B.3B | Measure after integration tests pass |
| API health endpoint P50 ms | 6B.3C | Measure |
| API preprocessing P50 ms | 6B.3C | Measure |
| API EDA P50 ms | 6B.3C | Measure |
| API training P50 ms | 6B.3C | Measure |
| P95 for above endpoints | 6B.3C | Measure |
| RAG query latency ms | 6B.3C | Measure (if Ollama running) |
| LinearRegression training time | Existing | 0.014s on ~7K rows (verified) |
| RandomForest training time | Existing | 0.080s on ~7K rows (verified) |
| Report generation time | 6B.3C | Measure |
| Benchmark CSV size and row count | 6B.3C | Document exactly |
| Docker build time | 6B.3D | Measure |
| Docker smoke test result | 6B.3D | Pass/fail + curl output |
| Report PNG count | Audit | 78 confirmed |
| Original dataset count | Audit | 71 files (~18MB) confirmed |
| Processed dataset count | Audit | 95 files (~393MB) confirmed |

---

## Screenshot Plan (Phase 6B.3E Detail)

| Screenshot file | Page / source | Capture priority |
|---|---|---|
| `screenshots/01_homepage.png` | `/` - platform hero, pipeline stage grid | Priority 1 |
| `screenshots/02_preprocessing.png` | `/preprocessing` - upload form visible | Priority 1 |
| `screenshots/03_eda.png` | `/eda` - EDA results with at least one chart | Priority 1 |
| `screenshots/04_model_training.png` | `/model-training` - results table with metrics | Priority 1 |
| `screenshots/05_rag.png` | `/rag` - query input and response visible | Priority 2 |
| `screenshots/06_insights.png` | `/insights` - AI insights text output visible | Priority 2 |
| `screenshots/07_reports.png` | `/reports` - generated report or download option | Priority 2 |
| `screenshots/08_shap_plot.png` | `reports/random_forest_shap_summary.png` | Priority 1 (already exists) |
| `screenshots/09_forecast_plot.png` | `reports/prophet_forecast_plot.png` | Priority 2 (already exists) |

Minimum required before README redesign: `01`, `03`, `04`, `08` (4 screenshots).

---

## Risk Control Rules

The following must not be claimed in any README, portfolio, or documentation update:

| Prohibited claim | Reason |
|---|---|
| Enterprise deployed | No cloud deployment evidence exists |
| Production users | No usage data |
| Customer usage | No usage data |
| CI/CD operational | `.github/workflows/` folder is empty |
| LangChain implemented | Not wired into codebase |
| ChromaDB implemented | Not wired; FAISS is used |
| AWS deployed | No deployment evidence |
| Fully autonomous AutoML | System requires user-driven workflow steps |
| Guaranteed model quality | Models are trained on user-supplied data |
| Headless Chrome PDF generation confirmed | Chrome path is hardcoded to Windows path; may not work in container |
| OpenTelemetry monitoring configured | In requirements, not configured |

What CAN be claimed, with evidence:

| Accurate claim | Evidence source |
|---|---|
| 15 FastAPI endpoints across 11 route modules | `api_route_inventory.md` |
| 8 specialist agents | Agent directory |
| FAISS RAG with SentenceTransformer and Ollama | `rag_routes.py`, `orchestrator_agent.py` |
| 10 frontend pages | `frontend/pages/` directory |
| 78 generated report artifacts | `reports/` directory count |
| Docker backend containerized | `Dockerfile` + smoke test |
| Training times measured: 0.014s to 0.080s | `model_reports/` CSV files |
| 8 pytest unit test modules | `tests/` directory |
| Prophet + ARIMA time-series forecasting | `forecasting_agent.py` |
| SHAP explainability plots | `reports/*_shap_summary.png` |

---

## Implementation Sequence and Dependencies

```
6B.3A  (Safety hygiene)
   |
6B.3B  (API validation + integration tests)
   |
6B.3C  (Benchmarks)        6B.3D (Docker evidence)
   |                             |
   +------------ 6B.3E (Screenshots) -----------+
                      |
                 6B.3F (README redesign)
                      |
                 6B.3G (Portfolio upgrade)
```

6B.3D can run in parallel with 6B.3C but must complete before 6B.3F.
6B.3E requires the app to be running locally.
6B.3F must not start until 6B.3B, 6B.3C, 6B.3D, and 6B.3E are complete.
6B.3G must not start until 6B.3F is complete.

---

## Evidence Index

After all phases complete, the following files must exist in `docs/evidence/`:

| File | Created in phase |
|---|---|
| `automl_system_audit_report.md` | 6B.1 (exists) |
| `automl_l2_engineering_upgrade_plan.md` | 6B.3 (this file) |
| `data_safety_review.md` | 6B.3A |
| `api_route_inventory.md` | 6B.3B |
| `test_run_output.txt` | 6B.3B |
| `benchmark_report.md` | 6B.3C |
| `benchmark_results.json` | 6B.3C |
| `docker_smoke_test.md` | 6B.3D |
| `screenshots/` (directory) | 6B.3E |
| `DEMO_WALKTHROUGH.md` (in docs/) | 6B.3E |
| `SCREENSHOT_GUIDE.md` (in docs/) | 6B.3E |

---

## Recommended First Implementation Phase

**Start with 6B.3A (Safety and repository hygiene).**

Rationale: Creating `.env.example` and reviewing tracked CSVs is the lowest-risk, fastest phase. It produces a hygiene baseline that makes all subsequent phases cleaner. It requires no running services, no benchmarks, and no screenshot capture. It unblocks every other phase because 6B.3B integration tests and 6B.3D Docker work both benefit from having `.env.example` in place first.

Estimated effort: One focused session.
Files to create: `.env.example`, `docs/evidence/data_safety_review.md`, `.gitignore` update.
Risk: Very low. No logic changes.
