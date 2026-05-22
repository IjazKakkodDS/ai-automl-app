"""
benchmark/run_benchmark.py

Local in-process benchmark for the AI AutoML Intelligence Platform.
Uses FastAPI TestClient - no live server required, no network overhead.

Usage:
    python benchmark/run_benchmark.py

Output:
    benchmark/benchmark_results.json   machine-readable results
    stdout                             readable summary table

NOTE: All latencies are in-process measurements on a developer machine.
They are not production server latencies. FastAPI TestClient bypasses
the network stack entirely. These numbers validate local workflow
feasibility and relative stage cost, not deployment throughput.
"""

import sys
import os
import json
import io
import time
import platform
import statistics
import datetime
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock unavailable packages before any app code is imported.
# Matches the approach used in tests/conftest.py.
# ---------------------------------------------------------------------------
if "ollama" not in sys.modules:
    _ollama_mock = MagicMock()
    _ollama_mock.chat.return_value = {"message": {"content": "mocked ollama response"}}
    sys.modules["ollama"] = _ollama_mock

if "sentence_transformers" not in sys.modules:
    _st_instance = MagicMock()
    _st_instance.encode.return_value = [[0.1] * 384]
    _st_mock = MagicMock()
    _st_mock.SentenceTransformer.return_value = _st_instance
    sys.modules["sentence_transformers"] = _st_mock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal benchmark app (excludes rag_routes / orchestrator_routes which load
# FAISS and SentenceTransformer at import time)
# ---------------------------------------------------------------------------
def _build_benchmark_app() -> FastAPI:
    from backend.app.api import (
        preprocessing_routes,
        eda_routes,
        model_training_routes,
        evaluation_routes,
        feature_engineering_routes,
        reset_routes,
        reports_routes,
        ai_insights_routes,
    )

    app = FastAPI(title="AutoML Benchmark App")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"message": "AI AutoML Backend is running. This is the RAG + Agentic AI pipeline."}

    app.include_router(preprocessing_routes.router, prefix="/preprocessing")
    app.include_router(eda_routes.router, prefix="/eda")
    app.include_router(model_training_routes.router, prefix="/model-training")
    app.include_router(evaluation_routes.router, prefix="/evaluation")
    app.include_router(feature_engineering_routes.router, prefix="/feature-engineering")
    app.include_router(reset_routes.router, prefix="/reset")
    app.include_router(reports_routes.router, prefix="/reports")
    app.include_router(ai_insights_routes.router, prefix="/ai-insights")
    return app


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
def make_regression_csv(rows: int = 200) -> bytes:
    """Deterministic synthetic regression CSV (seed=42, 5 features + target)."""
    import numpy as np
    rng = np.random.default_rng(42)
    lines = ["f1,f2,f3,f4,f5,target"]
    f1v = rng.standard_normal(rows)
    f2v = rng.random(rows) * 10
    f3v = rng.random(rows) * 5
    f4v = rng.standard_normal(rows) * 2
    f5v = rng.random(rows)
    noise = rng.standard_normal(rows) * 0.1
    targetv = f1v * 1.5 + f2v * 0.8 + f3v * 0.3 + noise
    for i in range(rows):
        lines.append(
            f"{round(float(f1v[i]),4)},{round(float(f2v[i]),4)},"
            f"{round(float(f3v[i]),4)},{round(float(f4v[i]),4)},"
            f"{round(float(f5v[i]),4)},{round(float(targetv[i]),4)}"
        )
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
def _percentile(sorted_data: list, p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def measure(fn, n: int) -> dict:
    """Run fn n times; return timing stats in milliseconds."""
    times_ms = []
    success = 0
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            result = fn()
            elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed)
            sc = getattr(result, "status_code", 200)
            if sc < 400:
                success += 1
            else:
                errors += 1
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed)
            errors += 1
    s = sorted(times_ms)
    return {
        "count": n,
        "success": success,
        "errors": errors,
        "min_ms": round(s[0], 2),
        "mean_ms": round(statistics.mean(times_ms), 2),
        "p50_ms": round(_percentile(s, 50), 2),
        "p95_ms": round(_percentile(s, 95), 2),
        "p99_ms": round(_percentile(s, 99), 2),
        "max_ms": round(s[-1], 2),
    }


# ---------------------------------------------------------------------------
# Artifact cleanup helpers
# ---------------------------------------------------------------------------
def _snapshot(path: str) -> set:
    """Return set of filenames in path (top-level only)."""
    return set(os.listdir(path)) if os.path.isdir(path) else set()


def _delete_new(path: str, before: set) -> list:
    """Delete files that appeared in path since before snapshot; return deleted paths."""
    deleted = []
    if not os.path.isdir(path):
        return deleted
    for name in set(os.listdir(path)) - before:
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
                deleted.append(fp)
            except Exception:
                pass
    return deleted


def _git_restore(paths: list) -> list:
    """Restore tracked files to HEAD. Returns list of successfully restored paths."""
    restored = []
    for p in paths:
        r = subprocess.run(["git", "restore", p], capture_output=True, text=True)
        if r.returncode == 0:
            restored.append(p)
    return restored


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    N = 10
    csv_bytes = make_regression_csv(200)

    # Snapshot watched directories before the benchmark run
    watched = {
        "processed_data": _snapshot("processed_data"),
        "original_data":  _snapshot("original_data"),
        "models":         _snapshot("models"),
        "reports/figures": _snapshot("reports/figures"),
        "ai_cache":       _snapshot("ai_cache"),
    }

    results: dict = {}
    skipped: dict = {}

    print()
    print("AI AutoML Intelligence Platform - Local In-Process Benchmark")
    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Mode:     FastAPI TestClient (in-process, no network stack)")
    print(f"N:        {N} iterations per endpoint")
    print()

    with TestClient(_build_benchmark_app()) as client:

        # 1. Health / root
        print("  Benchmarking GET / ...")
        results["health_root"] = measure(
            lambda: client.get("/"),
            N,
        )

        # 2. Preprocessing (synthetic CSV upload)
        print("  Benchmarking POST /preprocessing/preprocess/ ...")
        results["preprocessing_preprocess"] = measure(
            lambda: client.post(
                "/preprocessing/preprocess/",
                files={"file": ("bench.csv", io.BytesIO(csv_bytes), "text/csv")},
            ),
            N,
        )

        # 3. EDA (file upload; interactive=false to skip Plotly; max_numeric_cols=5)
        print("  Benchmarking POST /eda/analysis/ ...")
        results["eda_analysis"] = measure(
            lambda: client.post(
                "/eda/analysis/",
                files={"file": ("bench.csv", io.BytesIO(csv_bytes), "text/csv")},
                data={"interactive": "false", "sample_size": "200", "max_numeric_cols": "5"},
            ),
            N,
        )

        # 4. Model training: LinearRegression only, 200 rows (fastest configuration)
        print("  Benchmarking POST /model-training/train/ (LinearRegression) ...")
        results["model_training_linear"] = measure(
            lambda: client.post(
                "/model-training/train/",
                files={"file": ("bench.csv", io.BytesIO(csv_bytes), "text/csv")},
                params={
                    "target_col": "target",
                    "task_type": "regression",
                    "selected_models[]": ["LinearRegression"],
                    "selected_metrics[]": ["RMSE", "R2"],
                    "dataset_source": "original",
                },
            ),
            N,
        )

        # 5. Reports generate-full (placeholder dataset_id; missing files handled gracefully)
        print("  Benchmarking POST /reports/generate-full ...")
        results["reports_generate_full"] = measure(
            lambda: client.post(
                "/reports/generate-full",
                params={"dataset_id": "benchmark_placeholder"},
            ),
            N,
        )

        # 6. Evaluation list-models
        print("  Benchmarking GET /evaluation/list-models/ ...")
        results["evaluation_list_models"] = measure(
            lambda: client.get("/evaluation/list-models/"),
            N,
        )

        # 7. AI insights (mocked: is_model_available patched True; ollama.chat mocked)
        print("  Benchmarking POST /ai-insights/generate/ (mocked ollama) ...")

        def _bench_ai():
            with patch(
                "backend.app.agents.ai_insights_agent.is_model_available",
                return_value=True,
            ):
                return client.post(
                    "/ai-insights/generate/",
                    data={
                        "eda_summary": "Benchmark EDA: 200 rows, 5 numeric features, no missing values.",
                        "model_summary": "LinearRegression RMSE=0.10 R2=0.99.",
                        "model_choice": "mistral",
                        "force_regenerate": "true",
                    },
                )

        results["ai_insights_generate_mocked"] = measure(_bench_ai, N)

        # 8. Reset (mocked shutil.rmtree to prevent destructive deletion)
        print("  Benchmarking POST /reset/restart-analysis (mocked rmtree) ...")

        def _bench_reset():
            with patch("backend.app.api.reset_routes.shutil.rmtree"):
                return client.post("/reset/restart-analysis")

        results["reset_restart_mocked"] = measure(_bench_reset, N)

        # Skipped
        skipped["rag_query_summarize"] = (
            "Requires FAISS index + SentenceTransformer (Python 3.13 version incompatibility) + Ollama server"
        )
        skipped["forecasting_forecast"] = (
            "Requires Prophet package (not installed in Python 3.13 environment)"
        )
        skipped["reports_download_full"] = (
            "Requires Chrome at hardcoded Windows path; system dependency not present in CI"
        )
        skipped["orchestrator_orchestrate"] = (
            "Requires Ollama + FAISS + SentenceTransformer at runtime; all unavailable"
        )
        skipped["evaluation_evaluate"] = (
            "Requires a pre-saved .pkl model file path; covered by unit tests"
        )

    # -------------------------------------------------------------------------
    # Cleanup: delete new files created by benchmark runs
    # -------------------------------------------------------------------------
    all_deleted = []
    for path, before_snap in watched.items():
        all_deleted.extend(_delete_new(path, before_snap))

    # Restore tracked files that EDA benchmark may have overwritten
    tracked_files = [
        "reports/eda_report.txt",
        "reports/figures/feature1_distribution.png",
        "reports/figures/missing_values_heatmap.png",
        "reports/figures/pairplot.png",
    ]
    restored = _git_restore(tracked_files)

    # -------------------------------------------------------------------------
    # Build JSON output
    # -------------------------------------------------------------------------
    output = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "benchmark_mode": "in-process FastAPI TestClient",
            "iterations_per_endpoint": N,
            "synthetic_dataset": {
                "rows": 200,
                "cols": 6,
                "feature_cols": ["f1", "f2", "f3", "f4", "f5"],
                "target_col": "target",
                "task": "regression",
                "description": "Deterministic synthetic regression CSV, numpy seed=42",
            },
            "real_dataset_reference": {
                "path": "original_data/05261a4e-f676-4be2-9657-0c36d503566b.csv",
                "rows": 7043,
                "cols": 21,
                "source": "Telco Customer Churn (public benchmark dataset)",
                "note": "Referenced for documentation; not used in this benchmark run.",
            },
            "external_dependencies_skipped": list(skipped.keys()),
            "warnings": [
                "All latencies are local in-process measurements on a developer machine.",
                "FastAPI TestClient bypasses the network stack; real HTTP latency will be higher.",
                "EDA latency includes matplotlib figure generation and file I/O to reports/figures/.",
                "Model training latency includes sklearn fit, evaluation, and .pkl file write to disk.",
                "AI insights benchmark uses mocked ollama.chat (is_model_available patched True).",
                "Reset benchmark uses mocked shutil.rmtree; does not reflect actual filesystem cost.",
            ],
        },
        "results": results,
        "skipped": skipped,
        "artifacts_cleaned": all_deleted,
        "tracked_files_restored": restored,
    }

    os.makedirs("benchmark", exist_ok=True)
    out_path = os.path.join("benchmark", "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    labels = {
        "health_root":                 "GET /",
        "preprocessing_preprocess":    "POST /preprocessing/preprocess/",
        "eda_analysis":                "POST /eda/analysis/",
        "model_training_linear":       "POST /model-training/train/ (LR)",
        "reports_generate_full":       "POST /reports/generate-full",
        "evaluation_list_models":      "GET /evaluation/list-models/",
        "ai_insights_generate_mocked": "POST /ai-insights/generate/ (mocked)",
        "reset_restart_mocked":        "POST /reset/restart-analysis (mocked)",
    }

    print()
    print("=" * 82)
    print("  AI AutoML Intelligence Platform  -  Local In-Process Benchmark Results")
    print("=" * 82)
    print(f"  Mode:    FastAPI TestClient (in-process, no network)")
    print(f"  N:       {N} iterations per endpoint")
    print(f"  Dataset: synthetic 200-row regression CSV (seed=42, 5 features)")
    print()
    hdr = (
        f"  {'Endpoint':<46}"
        f"{'N':>3}  {'Mean':>7}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'Max':>7}  {'OK':>5}"
    )
    print(hdr)
    print("  " + "-" * 78)
    for key, label in labels.items():
        if key in results:
            r = results[key]
            ok_str = f"{r['success']}/{r['count']}"
            print(
                f"  {label:<46}"
                f"{r['count']:>3}"
                f"  {r['mean_ms']:>7.1f}"
                f"  {r['p50_ms']:>7.1f}"
                f"  {r['p95_ms']:>7.1f}"
                f"  {r['p99_ms']:>7.1f}"
                f"  {r['max_ms']:>7.1f}"
                f"  {ok_str:>5}"
            )
    print()
    print("  All times in milliseconds (ms).")
    print()
    print("  Skipped (external service or package unavailable):")
    for k, reason in skipped.items():
        print(f"    {k}")
        print(f"      {reason[:78]}")
    print()
    print(f"  Results saved to: {out_path}")
    print("  NOTE: These are local in-process measurements on a developer machine,")
    print("        not production server latencies.")
    print("=" * 82)


if __name__ == "__main__":
    main()
