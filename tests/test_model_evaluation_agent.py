import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
# Add the project root to sys.path so that we can import our module.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import evaluate_model from your module.
from backend.app.agents.evaluation_agent import evaluate_model

# --- Dummy Models for Testing ---

class DummyClassifier:
    """A dummy classifier that simulates classification behavior."""
    def __init__(self, training_columns, constant=1):
        self._training_columns = training_columns
        self.constant = constant

    def predict(self, X):
        # Always returns the constant value
        return np.array([self.constant] * len(X))

    def predict_proba(self, X):
        # Simulate probability output for binary classification
        preds = self.predict(X)
        prob = []
        for p in preds:
            if p == 1:
                prob.append([0.0, 1.0])
            else:
                prob.append([1.0, 0.0])
        return np.array(prob)

class DummyRegressor:
    """A dummy regressor that simulates regression behavior."""
    def __init__(self, training_columns, constant=10.0):
        self._training_columns = training_columns
        self.constant = constant

    def predict(self, X):
        return np.array([self.constant] * len(X))

# --- Tests for evaluate_model function ---

def test_evaluate_model_classification(tmp_path):
    # Create a dummy DataFrame with two features and a target column.
    df = pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3],
        "feat2": [1.0, 2.0, 3.0],
        "target": [1, 1, 1]
    })
    # Dummy classifier was "trained" on ['feat1', 'feat2'].
    dummy_clf = DummyClassifier(training_columns=["feat1", "feat2"], constant=1)
    model_path = os.path.join(tmp_path, "dummy_classifier.pkl")
    joblib.dump(dummy_clf, model_path)

    results = evaluate_model(df, model_path, target_col="target")
    # Check predictions (all should be 1)
    assert "predictions" in results
    assert all(p == 1 for p in results["predictions"])
    # Check that metrics are computed for classification (Accuracy, F1)
    assert "metrics" in results
    metrics = results["metrics"]
    # With perfect predictions, Accuracy and F1 should equal 1.0.
    assert np.isclose(metrics.get("Accuracy", 0), 1.0)
    assert np.isclose(metrics.get("F1", 0), 1.0)
    # Check that the actual target values are included.
    assert "actual" in results
    assert results["actual"] == [1, 1, 1]

def test_evaluate_model_regression(tmp_path):
    # Create a dummy DataFrame with two features and a numeric target.
    df = pd.DataFrame({
        "feat1": [0.5, 0.6, 0.7],
        "feat2": [10, 20, 30],
        "target": [15, 15, 15]
    })
    dummy_reg = DummyRegressor(training_columns=["feat1", "feat2"], constant=15.0)
    model_path = os.path.join(tmp_path, "dummy_regressor.pkl")
    joblib.dump(dummy_reg, model_path)

    results = evaluate_model(df, model_path, target_col="target")
    preds = results["predictions"]
    # Since dummy regressor always returns 15, predictions should be 15.
    assert all(np.isclose(p, 15.0) for p in preds)
    # Regression metrics should be computed: RMSE should be 0, and R2 should be 1.
    metrics = results.get("metrics", {})
    assert np.isclose(metrics.get("RMSE", 1e-6), 0.0)
    assert np.isclose(metrics.get("R2", 0), 1.0)
    # Verify that the actual values are returned.
    assert results["actual"] == [15, 15, 15]

def test_evaluate_model_missing_training_columns(tmp_path):
    # Create a DataFrame missing a required feature.
    df = pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3],
        "target": [1, 1, 1]
    })
    dummy_clf = DummyClassifier(training_columns=["feat1", "feat2"], constant=1)
    model_path = os.path.join(tmp_path, "dummy_classifier.pkl")
    joblib.dump(dummy_clf, model_path)
    # The function should raise a ValueError because 'feat2' is missing.
    with pytest.raises(ValueError, match="missing columns the model expects"):
        evaluate_model(df, model_path, target_col="target")

def test_evaluate_model_unseen_columns(tmp_path):
    # Create a DataFrame that has extra columns not seen during training.
    df = pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3],
        "feat2": [1.0, 2.0, 3.0],
        "extra": [999, 999, 999],
        "target": [1, 1, 1]
    })
    dummy_clf = DummyClassifier(training_columns=["feat1", "feat2"], constant=1)
    model_path = os.path.join(tmp_path, "dummy_classifier.pkl")
    joblib.dump(dummy_clf, model_path)
    # The evaluation should drop the unseen "extra" column and work.
    results = evaluate_model(df, model_path, target_col="target")
    # Predictions should be correct.
    assert all(p == 1 for p in results["predictions"])
