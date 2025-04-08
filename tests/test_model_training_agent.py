import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import time
import pandas as pd
import numpy as np
import pytest
import joblib

# Import the training function from your module.
# Adjust the import path if needed; here we assume the file is named model_training_agent.py
from backend.app.agents import model_training_agent as mt_agent

# --- Helper Functions to Create Sample Datasets ---

def create_regression_df():
    """Creates a small synthetic regression dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
         "feature1": np.random.rand(100),
         "feature2": np.random.rand(100),
         "target": np.random.rand(100) * 100,
    })
    return df

def create_classification_df():
    """Creates a small synthetic classification dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
         "feature1": np.random.rand(100),
         "feature2": np.random.rand(100),
         "target": np.random.choice(["class0", "class1"], 100)
    })
    return df

# --- Test for Regression Training ---

def test_regression_training(tmp_path):
    df = create_regression_df()
    # Use a temporary directory for saving models.
    model_folder = str(tmp_path / "models_reg")
    result = mt_agent.train_and_evaluate_models(
        df=df,
        target_col="target",
        task_type="regression",
        selected_models=["LinearRegression", "RandomForestRegressor"],
        selected_metrics=["RMSE", "R2", "MAE"],
        sample_data=True,
        sample_size=50,  # Force sampling if dataset is larger than sample_size.
        hyperparameter_tuning=False,
        model_folder=model_folder,
        apply_encoding=False,
        apply_scaling=True,
    )
    # Check that the returned dictionary has expected keys.
    assert "results" in result and "trained_models" in result
    # Check that at least one model was trained.
    assert len(result["results"]) > 0
    # Verify that each trained model file exists.
    for model_name, model_path in result["trained_models"].items():
        assert os.path.exists(model_path)

# --- Test for Classification Training ---

def test_classification_training(tmp_path):
    df = create_classification_df()
    model_folder = str(tmp_path / "models_clf")
    result = mt_agent.train_and_evaluate_models(
        df=df,
        target_col="target",
        task_type="classification",
        selected_models=["LogisticRegression", "RandomForestClassifier"],
        selected_metrics=["Accuracy", "F1", "Precision", "Recall"],
        sample_data=True,
        sample_size=50,
        hyperparameter_tuning=False,
        model_folder=model_folder,
        apply_encoding=True,
        encoding_method="one-hot",
        apply_scaling=True,
        scaling_method="standard",
    )
    # Check that the returned dictionary has expected keys.
    assert "results" in result and "trained_models" in result
    assert len(result["results"]) > 0
    for model_name, model_path in result["trained_models"].items():
        assert os.path.exists(model_path)

# --- Test Error Handling: Missing Target Column ---

def test_missing_target_column():
    df = pd.DataFrame({
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10)
    })
    with pytest.raises(ValueError, match="Target column 'target' not found in DataFrame"):
        mt_agent.train_and_evaluate_models(
            df=df,
            target_col="target",
            task_type="regression",
            selected_models=["LinearRegression"],
            selected_metrics=["RMSE"],
        )

# --- Test Error Handling: Empty DataFrame ---

def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty. Cannot train models."):
        mt_agent.train_and_evaluate_models(
            df=df,
            target_col="target",
            task_type="regression",
            selected_models=["LinearRegression"],
            selected_metrics=["RMSE"],
        )
