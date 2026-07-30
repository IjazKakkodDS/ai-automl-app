import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to sys.path so modules can be imported.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the OrchestratorAgent from your module.
from backend.app.agents.orchestrator_agent import OrchestratorAgent

# Dummy implementations to override actual calls in tests.
def dummy_generate_eda(data, interactive, exclude_date_features, correlation_method, sample_size):
    return ("Dummy EDA Report", {"Overview": pd.DataFrame()}, {"dummy_fig": "base64string"})

def dummy_train_and_evaluate_models(**kwargs):
    # Return a dummy training output with R2 = 0.7 (below threshold) for example.
    return {"results": [{"Model": "DummyModel", "R2": 0.7}], "trained_models": {"DummyModel": "dummy_path.pkl"}}

def dummy_generate_ai_insights(eda_summary, model_summary, model_choice, force_regenerate, enable_cot):
    # The orchestrator must request a real Ollama-resolvable model. "gpt-4" is
    # never listed by `ollama list`, so requesting it always raised inside
    # generate_ai_insights before this fix; this assertion pins the corrected
    # call site so a regression back to "gpt-4" fails this test immediately.
    assert model_choice == "mistral", (
        f"Orchestrator must request a real Ollama-resolvable model, got '{model_choice}'."
    )
    return "Dummy AI Insights"

def dummy_retrieve_context(query, top_k):
    return "Dummy retrieved context."

# --- Test OrchestratorAgent decision logic ---
@pytest.fixture(autouse=True)
def patch_orchestrator(monkeypatch):
    # Patch the functions called by the orchestrator
    from backend.app.agents import orchestrator_agent as orch
    monkeypatch.setattr(orch, "generate_eda", dummy_generate_eda)
    monkeypatch.setattr(orch, "train_and_evaluate_models", dummy_train_and_evaluate_models)
    monkeypatch.setattr(orch, "generate_ai_insights", dummy_generate_ai_insights)
    monkeypatch.setattr(orch, "retrieve_context", dummy_retrieve_context)

def test_orchestrator_low_model_performance():
    # Create a dummy DataFrame.
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "feature1": [10, 20, 30, 40, 50],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]
    })

    agent = OrchestratorAgent()
    # Set thresholds to force a "tune hyperparameters" decision.
    agent.eda_quality_threshold = 0.8
    agent.model_performance_threshold = 0.75  # Our dummy R2=0.7 is below threshold.

    decision = agent.decide_next_steps(df, target_col="target")
    # Assert that decision contains the expected keys and action.
    assert "eda_report" in decision
    assert "model_results" in decision
    assert decision["next_action"] == "Tune hyperparameters or try alternative models."
    assert "ai_insights" in decision

def test_orchestrator_low_data_quality():
    # Construct a DataFrame with enough missing values that real completeness
    # (1 - missing_cells/total_cells) falls below eda_quality_threshold, so the
    # "Perform advanced feature engineering" branch is genuinely reachable
    # rather than a permanently-unreachable hardcoded stub.
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "feature1": [10, np.nan, np.nan, 40, np.nan],
        "feature2": [0.1, np.nan, np.nan, 0.4, np.nan],
    })
    # 15 cells total, 6 missing -> completeness = 1 - 6/15 = 0.6, below 0.8.

    agent = OrchestratorAgent()
    agent.eda_quality_threshold = 0.8

    decision = agent.decide_next_steps(df, target_col="target")

    assert decision["next_action"] == "Perform advanced feature engineering."
    assert "model_results" not in decision

def test_orchestrator_good_performance():
    # Modify dummy train function to return a high R2.
    def high_r2_train(**kwargs):
        return {"results": [{"Model": "GoodModel", "R2": 0.9}], "trained_models": {"GoodModel": "good_model.pkl"}}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("backend.app.agents.orchestrator_agent.train_and_evaluate_models", high_r2_train)

    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "feature1": [10, 20, 30, 40, 50],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]
    })

    agent = OrchestratorAgent()
    agent.eda_quality_threshold = 0.8
    agent.model_performance_threshold = 0.75  # Now dummy R2=0.9 is above threshold.

    decision = agent.decide_next_steps(df, target_col="target")
    assert decision["next_action"] == "Proceed to forecasting or deployment."
    monkeypatch.undo()

def test_orchestrator_eda_failure(monkeypatch):
    # Force generate_eda to raise an exception.
    monkeypatch.setattr("backend.app.agents.orchestrator_agent.generate_eda", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("EDA error")))
    
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "feature1": [10, 20, 30]
    })

    agent = OrchestratorAgent()
    decision = agent.decide_next_steps(df, target_col="target")
    assert "error" in decision
    assert decision["error"] == "EDA failed"

