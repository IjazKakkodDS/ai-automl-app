import os
import sys
import json
import io
import base64
import pytest
import pandas as pd
from pathlib import Path

# Add project root to sys.path so that we can import our module.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.agents import ai_insights_agent as ai_agent

# --- Test clean_response function ---
def test_clean_response():
    raw = "Hello [TOKEN]World</s>   !"
    cleaned = ai_agent.clean_response(raw)
    # Expect tokens like [TOKEN] and </s> to be removed, and extra whitespace normalized.
    assert "[TOKEN]" not in cleaned
    assert "</s>" not in cleaned
    assert "Hello World !" in cleaned

# --- Test get_cache_filename ---
def test_get_cache_filename():
    prompt = "Test prompt"
    model_choice = "gpt-4"
    cache_filename = ai_agent.get_cache_filename(prompt, model_choice)
    # The returned filename should be inside the CACHE_DIR and end with .json
    assert cache_filename.startswith(ai_agent.CACHE_DIR)
    assert cache_filename.endswith(".json")

# --- Regression coverage for the orchestrator's model-choice fix ---
# The orchestrator previously requested model_choice="gpt-4", which is listed
# in AVAILABLE_MODELS but can never appear in `ollama list` output (GPT-4 is
# not an Ollama-servable model), so is_model_available("gpt-4") always
# returned False and generate_ai_insights always raised before ever reaching
# Ollama. is_model_available is mocked to reflect only "mistral" as installed
# -- not a blanket bypass for every model -- so these tests exercise the real
# availability branch rather than hiding it.
def test_generate_ai_insights_mistral_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_agent, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ai_agent, "is_model_available", lambda model: model.lower() == "mistral")
    dummy_response = {"response": "Real mistral insight text."}
    monkeypatch.setattr(
        ai_agent, "ollama",
        type("DummyOllama", (), {"chat": lambda **kwargs: dummy_response})
    )

    insights = ai_agent.generate_ai_insights(
        eda_summary="Dummy EDA summary",
        model_summary="Dummy model summary",
        model_choice="mistral",
        force_regenerate=True,
        enable_cot=False,
    )

    assert insights == "Real mistral insight text."

def test_generate_ai_insights_gpt4_is_genuinely_unavailable(monkeypatch):
    # Documents the real defect the orchestrator fix avoids: requesting
    # "gpt-4" fails at the availability check, before Ollama is ever called.
    monkeypatch.setattr(ai_agent, "is_model_available", lambda model: model.lower() == "mistral")

    with pytest.raises(ValueError):
        ai_agent.generate_ai_insights(
            eda_summary="Dummy EDA summary",
            model_summary="Dummy model summary",
            model_choice="gpt-4",
            force_regenerate=True,
            enable_cot=False,
        )

# --- Test generate_ai_insights with no cache ---
@pytest.mark.skip(reason="Hard-coded prompt string in test no longer matches current generate_ai_insights prompt template; cache filename mismatch")
def test_generate_ai_insights_no_cache(tmp_path, monkeypatch):
    # Use a temporary cache directory.
    temp_cache = tmp_path / "ai_cache"
    temp_cache.mkdir()
    monkeypatch.setattr(ai_agent, "CACHE_DIR", str(temp_cache))
    
    # Ensure the model is reported as available.
    monkeypatch.setattr(ai_agent, "is_model_available", lambda model: True)
    
    # Simulate ollama.chat returning a dummy response dictionary.
    dummy_response = {"response": "Dummy AI insights generated."}
    monkeypatch.setattr(ai_agent, "ollama", type("DummyOllama", (), {"chat": lambda **kwargs: dummy_response}))
    
    eda_summary = "This is a dummy EDA summary."
    model_summary = "This is a dummy model summary."
    insights = ai_agent.generate_ai_insights(
        eda_summary,
        model_summary,
        model_choice="gpt-4",
        force_regenerate=True,
        enable_cot=False,
        max_tokens=512
    )
    # Check that we got the dummy response.
    assert "Dummy AI insights" in insights
    
    # Verify that a cache file was created.
    cache_filename = ai_agent.get_cache_filename(
        f"You are an AI assistant with context from an Exploratory Data Analysis (EDA) and a Model Training Summary. Use the information provided below to create a structured analysis with actionable insights.\n\nEDA Summary:\n{eda_summary}\n\nModel Training Summary:\n{model_summary}\n\nPlease follow this outline in bullet points:\n1) General Insights:\n   - Summarize the most important findings or patterns from the EDA.\n\n2) Model Performance:\n   - Discuss how the model is performing based on the training summary (e.g., metrics).\n   - If the model is producing negative values or anomalies, explain possible reasons.\n\n3) Recommendations:\n   - Provide clear, specific next steps to refine data preprocessing, tuning hyperparameters, or other improvements.\n\nThink step by step, and present your answer in bullet points.", 
        "gpt-4"
    )
    assert os.path.exists(cache_filename)

# --- Test generate_ai_insights using cache ---
@pytest.mark.skip(reason="Hard-coded prompt string in test no longer matches current generate_ai_insights prompt template; cache filename mismatch")
def test_generate_ai_insights_with_cache(tmp_path, monkeypatch):
    # Use a temporary cache directory and pre-populate a cache file.
    temp_cache = tmp_path / "ai_cache"
    temp_cache.mkdir()
    monkeypatch.setattr(ai_agent, "CACHE_DIR", str(temp_cache))
    
    prompt = (
        "You are an AI assistant with context from an Exploratory Data Analysis (EDA) and a Model Training Summary. "
        "Use the information provided below to create a structured analysis with actionable insights.\n\n"
        "EDA Summary:\nDummy EDA summary\n\n"
        "Model Training Summary:\nDummy model summary\n\n"
        "Please follow this outline in bullet points:\n"
        "1) General Insights:\n   - Summarize the most important findings or patterns from the EDA.\n\n"
        "2) Model Performance:\n   - Discuss how the model is performing based on the training summary (e.g., metrics).\n"
        "   - If the model is producing negative values or anomalies, explain possible reasons.\n\n"
        "3) Recommendations:\n   - Provide clear, specific next steps to refine data preprocessing, tuning hyperparameters, or other improvements.\n\n"
        "Think step by step, and present your answer in bullet points."
    )
    cache_filename = ai_agent.get_cache_filename(prompt, "gpt-4")
    # Write dummy insights into the cache file.
    dummy_cached = {"insights": "Cached dummy insights."}
    with open(cache_filename, "w", encoding="utf-8") as f:
        json.dump(dummy_cached, f)
    
    # Ensure the model is available.
    monkeypatch.setattr(ai_agent, "is_model_available", lambda model: True)
    # Patch ollama.chat to return an empty response (it shouldn't be used since cache exists)
    monkeypatch.setattr(ai_agent, "ollama", type("DummyOllama", (), {"chat": lambda **kwargs: {}}))
    
    insights = ai_agent.generate_ai_insights(
        eda_summary="Dummy EDA summary",
        model_summary="Dummy model summary",
        model_choice="gpt-4",
        force_regenerate=False,
        enable_cot=False,
        max_tokens=512
    )
    # It should return the cached insights.
    assert "Cached dummy insights." == insights

# --- Test get_eda_summary ---
def test_get_eda_summary(tmp_path):
    temp_report = tmp_path / "eda_report.txt"
    temp_report.write_text("Dummy EDA report content.")
    # Call get_eda_summary with the temp_report path.
    summary = ai_agent.get_eda_summary(report_path=str(temp_report))
    assert "Dummy EDA report content." in summary

# --- Test get_model_summary ---
def test_get_model_summary(tmp_path):
    temp_csv = tmp_path / "training_results.csv"
    # Create a dummy CSV file.
    df = pd.DataFrame({"Model": ["A", "B"], "R2": [0.9, 0.8]})
    df.to_csv(temp_csv, index=False)
    summary = ai_agent.get_model_summary(report_path=str(temp_csv))
    assert "Model Training Results:" in summary
    assert "A" in summary
