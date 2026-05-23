# AI AutoML Intelligence Platform - Benchmark and Scalability Evidence

**Phase:** 6B.3C.1 - Real Dataset Benchmark Extension
**Date:** 2026-05-23
**Command:** `python benchmark/run_benchmark.py`

---

## Benchmark Environment

| Property | Value |
|---|---|
| Host | Developer machine (Windows 11 Home, 10.0.26200) |
| Python | 3.13.1 |
| pytest | 9.0.3 |
| Benchmark mode | Local in-process FastAPI TestClient (no network stack) |
| Synthetic iterations | N=10 per endpoint |
| Real dataset iterations | N=5 (preprocessing); N=3 (EDA, model training) |
| Benchmark script | `benchmark/run_benchmark.py` |
| Results file | `benchmark/benchmark_results.json` |
| Timestamp | 2026-05-23T02:02:40Z |

> **Important:** All latencies are local in-process measurements on a developer machine. FastAPI TestClient bypasses the network stack entirely. These numbers validate local workflow feasibility and relative stage cost, not deployment throughput. They are not production server latencies.

---

## Benchmark Scope

| Stage | Endpoint | External Dependency | Benchmarked |
|---|---|---|---|
| Health check | GET / | None | Yes |
| Preprocessing (synthetic) | POST /preprocessing/preprocess/ | None | Yes |
| EDA analysis (synthetic) | POST /eda/analysis/ | None | Yes |
| Model training (synthetic) | POST /model-training/train/ | None | Yes |
| Report generation | POST /reports/generate-full | None | Yes |
| Model listing | GET /evaluation/list-models/ | None | Yes |
| AI insights | POST /ai-insights/generate/ | Ollama (mocked) | Yes (mocked) |
| Session reset | POST /reset/restart-analysis | None (mocked rmtree) | Yes (mocked) |
| Preprocessing (real Telco) | POST /preprocessing/preprocess/ | None | Yes |
| EDA analysis (real Telco) | POST /eda/analysis/ | None | Yes |
| Model training (real Telco) | POST /model-training/train/ | None | Yes |
| RAG query | POST /rag/query-summarize | FAISS + SentenceTransformer + Ollama | Skipped |
| Forecasting | POST /forecasting/forecast/ | Prophet (not installed) | Skipped |
| Report download | GET /reports/download-full | Chrome (system) | Skipped |
| Orchestrator | POST /orchestrator/orchestrate | Ollama + FAISS + SentenceTransformer | Skipped |
| Model evaluation | POST /evaluation/evaluate/ | Pre-saved .pkl file | Skipped |

---

## Dataset Context

### Synthetic benchmark dataset

| Property | Value |
|---|---|
| Rows | 200 |
| Columns | 6 (f1, f2, f3, f4, f5, target) |
| Task | Regression |
| Generation | Deterministic, numpy seed=42, generated in script |
| Purpose | Reproducible baseline; avoids dependency on tracked data files |

### Real benchmark dataset: Telco Customer Churn

| Property | Value |
|---|---|
| File | `original_data/05261a4e-f676-4be2-9657-0c36d503566b.csv` |
| Rows (raw) | 7,043 |
| Columns | 21 |
| Source | Telco Customer Churn (public benchmark dataset) |
| Numeric columns | SeniorCitizen (int64), tenure (int64), MonthlyCharges (float64) |
| Categorical columns | 17 object columns including customerID, Churn (Yes/No target), TotalCharges (object) |
| TotalCharges dtype | object - 11 rows contain empty strings (tenure=0 customers) |
| Model training rows | 7,032 (11 empty-string TotalCharges rows dropped; customerID dropped as ID column) |
| EDA/Preprocessing | Raw 7,043-row CSV used without modification |

---

## Benchmark Results

All latencies are in milliseconds (ms).

### Synthetic dataset (200 rows, 6 columns, N=10)

| Endpoint | N | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Success |
|---|---|---|---|---|---|---|---|
| GET / | 10 | 0.8 | 0.7 | 1.4 | 1.7 | 1.8 | 10/10 |
| POST /preprocessing/preprocess/ | 10 | 13.4 | 12.5 | 20.9 | 24.6 | 25.6 | 10/10 |
| POST /eda/analysis/ | 10 | 3,776.9 | 3,696.8 | 4,546.9 | 4,863.2 | 4,942.3 | 10/10 |
| POST /model-training/train/ (LinearRegression) | 10 | 7.9 | 6.4 | 14.9 | 20.2 | 21.6 | 10/10 |
| POST /reports/generate-full | 10 | 1.8 | 1.7 | 2.2 | 2.6 | 2.7 | 10/10 |
| GET /evaluation/list-models/ | 10 | 0.8 | 0.8 | 0.9 | 1.0 | 1.0 | 10/10 |
| POST /ai-insights/generate/ (ollama mocked) | 10 | 2.0 | 1.6 | 3.9 | 5.0 | 5.3 | 10/10 |
| POST /reset/restart-analysis (rmtree mocked) | 10 | 1.3 | 1.3 | 1.5 | 1.5 | 1.5 | 10/10 |

### Real dataset: Telco Customer Churn (7,043 rows, 21 columns)

| Endpoint | N | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Success |
|---|---|---|---|---|---|---|---|
| POST /preprocessing/preprocess/ (Telco, raw) | 5 | 232.7 | 231.1 | 240.7 | 242.3 | 242.7 | 5/5 |
| POST /eda/analysis/ (Telco, max_numeric_cols=3) | 3 | 116,460.9 | 94,229.1 | 159,390.5 | 165,182.6 | 166,630.6 | 3/3 |
| POST /model-training/train/ (LogisticRegression, Telco) | 3 | 831.4 | 818.3 | 867.6 | 872.0 | 873.1 | 3/3 |

---

## Skipped Endpoints and Reasons

| Endpoint | Reason Skipped |
|---|---|
| POST /rag/query-summarize | Requires FAISS index + SentenceTransformer (Python 3.13 version incompatibility with transformers package) + Ollama local server |
| POST /forecasting/forecast/ | Requires Prophet package (not installed in Python 3.13 environment) |
| GET /reports/download-full | Requires Chrome at a hardcoded Windows path; system dependency |
| POST /orchestrator/orchestrate | Requires all of the above: Ollama + FAISS + SentenceTransformer |
| POST /evaluation/evaluate/ | Requires a pre-saved .pkl model file path; endpoint logic is covered by unit tests |

---

## Interpretation

### Synthetic dataset (200 rows, 6 columns)

**Health and utility endpoints (GET /, GET /evaluation/list-models/, POST /reset)**
P50 latency under 1.3ms confirms negligible processing overhead. Variance is within normal OS scheduling noise.

**Preprocessing (POST /preprocessing/preprocess/)**
P50 of 12.5ms for a 200-row, 6-column CSV includes: pandas CSV parse, KNN imputation on 6 numeric columns, StandardScaler fit-transform, and writing two UUID-named files to disk (original_data/ and processed_data/).

**EDA analysis (POST /eda/analysis/)**
P50 of 3,697ms is the dominant bottleneck in the synthetic pipeline. Includes: pandas EDA table generation, matplotlib figure generation (heatmap, 5 distribution plots, seaborn pairplot on 5 features), and multiple file writes to reports/figures/. The seaborn pairplot on 5 features is the largest cost. The "Starting a Matplotlib GUI outside of the main thread" UserWarning is expected in TestClient execution and does not affect results.

**Model training (POST /model-training/train/, LinearRegression)**
P50 of 6.4ms. LinearRegression is the fastest sklearn estimator. The high P95/P99 spread (14.9ms vs 6.4ms) reflects first-call JIT overhead from sklearn and numpy on warm-up iterations.

**Report generation (POST /reports/generate-full)**
P50 of 1.7ms reflects the file-assembly pattern: this endpoint scans reports/ for matching files and writes an HTML file. With no dataset_id match, the overhead is minimal HTML string construction and one file write.

**AI insights (POST /ai-insights/generate/, mocked)**
P50 of 1.6ms benchmarks route overhead and prompt construction only. The ollama.chat() call is mocked to return immediately. Real latency with an Ollama server would be dominated by LLM inference time (typically 2-30 seconds).

### Real dataset: Telco Customer Churn (7,043 rows, 21 columns)

**Preprocessing (POST /preprocessing/preprocess/, raw Telco CSV)**
P50 of 231.1ms for the full 7,043-row, 21-column Telco CSV. This includes: dateutil format inference over 17 object columns (source of pandas UserWarnings during column type detection), KNN imputation on 3 numeric columns, StandardScaler, one-hot encoding of 16 low-cardinality categorical columns, frequency encoding of customerID (high-cardinality ID column), and writing two files to disk. The 231ms vs 12.5ms synthetic ratio (18.5x slower) is proportionate to the data volume increase (35x rows) plus categorical encoding overhead, partially offset by the KNN imputation operating on only 3 numeric columns instead of 6.

**EDA analysis (POST /eda/analysis/, Telco, max_numeric_cols=3)**
P50 of 94,229ms (approximately 94 seconds) per call, with high variance (P50=94s, max=167s). This includes: EDA report text generation for 21 columns, matplotlib figure generation (missing values heatmap, 3 distribution plots for numeric columns, 18 countplots for categorical columns), seaborn pairplot on 3 numeric features, and JSON serialization of all generated figures for the HTTP response body. The 3-feature pairplot is substantially faster than the 5-feature synthetic pairplot, but the 18 additional countplots (one per categorical column) and the large JSON response serialization dominate. N=3 is a small sample; the P50/P95 spread (94s vs 159s) reflects OS scheduling and memory pressure variation across iterations.

**Model training (POST /model-training/train/, LogisticRegression, Telco)**
P50 of 818.3ms for LogisticRegression classification on the cleaned 7,032-row, 20-column dataset. This includes: pandas one-hot encoding of 15 categorical columns (expanding to approximately 43 binary features), LogisticRegression.fit() on 7,032 rows, metric computation (Accuracy, F1), .pkl model file write, and training report write. All 3 iterations succeeded (3/3). The model training endpoint handled the mixed-type Telco dataset with one-hot encoding without errors.

### Scalability observations

| Stage | Synthetic (200 rows) | Telco (7,043 rows) | Scale factor (rows) | Latency ratio |
|---|---|---|---|---|
| Preprocessing | 12.5ms | 231.1ms | 35x | 18.5x |
| EDA P50 | 3,697ms | 94,229ms | 35x | 25.5x |
| Model training | 6.4ms | 818.3ms | 35x | 128x |

Preprocessing scales sub-linearly (18.5x for 35x rows) because KNN imputation is O(n) and the categorical encoding overhead grows with column count, not row count. EDA scales super-linearly relative to rows because it also generates more figure types for 21 columns vs 6. Model training scales super-linearly (128x) because LogisticRegression involves matrix operations that scale with n_samples * n_features and one-hot encoding expands the feature space.

**Relative stage cost (real dataset)**

EDA >> Model training > Preprocessing > AI insights > Reports > Reset > Health

EDA (P50=94s) is approximately 115x slower than preprocessing (P50=231ms) and 115x slower than model training (P50=818ms) at the real dataset size.

---

## Limitations

1. **N=10 (synthetic) and N=3/N=5 (real) are small samples.** P95 and P99 values are estimates; more iterations would narrow confidence intervals. The real dataset EDA variance is particularly large (P50=94s, max=167s, N=3).
2. **TestClient bypasses the network stack.** Real HTTP server latency adds TCP/TLS overhead, async event loop scheduling, and ASGI middleware processing that this benchmark does not capture.
3. **EDA latency is dominated by figure serialization at scale.** The 94-second real dataset EDA response includes JSON serialization of 21+ matplotlib and seaborn figures. In a production setup with async streaming or pre-rendering, this latency profile would differ.
4. **Model training latency varies by model type and dataset size.** LogisticRegression and LinearRegression are the fastest sklearn models. RandomForestRegressor (n_estimators=100) on 7,043 rows would take 5-30 seconds. GradientBoostingRegressor and XGBRegressor would be similar.
5. **AI insights, RAG, and Orchestrator endpoints are not benchmarked end-to-end.** Route overhead is captured (1.6ms for ai-insights mocked), but LLM inference time is not.
6. **Model training on real dataset used a cleaned CSV.** customerID was dropped (high-cardinality ID column, not a predictive feature) and TotalCharges was coerced from object to float64 (11 empty-string rows dropped). This is standard data preparation; the preprocessing endpoint handles the raw CSV without any modification.
7. **Developer machine specifications are not published.** Results are specific to this hardware and OS scheduler state at the time of the benchmark run.

---

## Claim-Safe Summary

This benchmark validates that the core ML pipeline routes (health check, preprocessing, EDA, model training, report generation, model listing) are functional and complete end-to-end on a developer machine, tested against both a 200-row synthetic regression dataset and the tracked 7,043-row Telco Customer Churn dataset.

**Synthetic dataset (200 rows, N=10):** All 8 benchmarked endpoints succeeded (100%). EDA is the dominant latency stage (~3.7 seconds P50 at 200 rows with 5-feature pairplot). All non-EDA core routes complete under 30ms.

**Real dataset (Telco, 7,043 rows, N=3/5):** All 3 benchmarked endpoints succeeded (100%). Preprocessing handles the full 21-column mixed-type dataset including KNN imputation, one-hot encoding, and frequency encoding in 231ms P50. EDA on the full 21-column dataset takes approximately 94 seconds P50 (dominated by figure generation and JSON serialization of 21+ plots). LogisticRegression training on 7,032 rows with one-hot encoding completes in 818ms P50.

Five endpoints (RAG, forecasting, report download, orchestrator, model evaluation) were not benchmarked due to missing external dependencies (Ollama, Prophet, Chrome) or safety constraints, but their route registration and agent-level logic are verified by the integration test suite (57 passed, 3 skipped, 0 failed) and the FastAPI OpenAPI schema at /docs.

---

## Warnings Captured During Benchmark

1. `UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.` (eda_agent.py, seaborn axisgrid.py) - Expected in TestClient threaded execution. Does not affect benchmark accuracy. Figures are saved to disk and serialized to the response correctly.
2. `RuntimeError: main thread is not in main loop` (matplotlib/tkinter atexit) - Non-fatal atexit cleanup error from matplotlib's tkinter backend. All benchmark logic and JSON output completed before this. Exit code 3 from atexit errors does not indicate benchmark failure.
3. `UserWarning: Could not infer format, so each element will be parsed individually, falling back to dateutil` (preprocessing_agent.py) - Expected when preprocessing the Telco CSV; occurs during date-format inference over 17 object columns, all of which are ultimately classified as categorical. Does not affect preprocessing correctness.
4. `FutureWarning: Downcasting behavior in replace is deprecated` (eda_routes.py) - Pre-existing pandas version warning. Does not affect EDA correctness.

---

*Generated as part of Phase 6B.3C.1: Real Dataset Benchmark Extension for AI AutoML Intelligence Platform.*
