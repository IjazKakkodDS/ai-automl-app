import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path for all test imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the working directory is the project root.
# Backend routes use relative paths (processed_data/, reports/, models/, data_1/).
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Patch unavailable or incompatible packages before any test module is imported.
# These patches are applied at conftest load time (before pytest collection)
# so that test files that import these packages can be collected and run.
#
# 1. ollama: not installed in this environment. ai_insights_agent.py imports it
#    at module level. The mock allows import; tests that need real Ollama are
#    either skipped or use monkeypatch to override the mock.
#
# 2. sentence_transformers: installed but triggers a TFPreTrainedModel
#    ImportError in the transformers package (version incompatibility).
#    orchestrator_agent.py calls SentenceTransformer() at module level.
#    The mock prevents the error; the encode() call returns a plausible shape.
# ---------------------------------------------------------------------------
if 'ollama' not in sys.modules:
    _ollama_mock = MagicMock()
    _ollama_mock.chat.return_value = {"message": {"content": "mocked ollama response"}}
    sys.modules['ollama'] = _ollama_mock

if 'sentence_transformers' not in sys.modules:
    _st_instance = MagicMock()
    _st_instance.encode.return_value = [[0.1] * 384]
    _st_mock = MagicMock()
    _st_mock.SentenceTransformer.return_value = _st_instance
    sys.modules['sentence_transformers'] = _st_mock

# ---------------------------------------------------------------------------
# Exclude test files that cannot be collected due to missing runtime packages.
# test_forecasting_agent.py requires 'prophet' which is not installed in this
# Python 3.13 environment. The test collection error is pre-existing.
# ---------------------------------------------------------------------------
collect_ignore = ["test_forecasting_agent.py"]

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


def _build_minimal_app() -> FastAPI:
    """
    Return a FastAPI app with only the routes that do not require
    Ollama or SentenceTransformer at runtime.
    This keeps integration tests independent of external service availability.
    """
    from backend.app.api import (
        preprocessing_routes,
        eda_routes,
        model_training_routes,
        evaluation_routes,
        feature_engineering_routes,
        reset_routes,
    )

    app = FastAPI(title="AutoML Integration Test App")
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

    return app


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient for the minimal integration test app."""
    with TestClient(_build_minimal_app()) as c:
        yield c
