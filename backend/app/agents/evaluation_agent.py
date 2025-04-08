import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.base import is_classifier, is_regressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

def evaluate_model(new_df: pd.DataFrame, model_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a saved model from 'model_path' and evaluate it on new_df.
    If a target column is provided and exists in new_df, compute standard metrics.
    
    Returns a dictionary with predictions, evaluation metrics, and actual values (if available).
    Also includes information about any column mismatches.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise e

    df = new_df.copy()
    y_true = None
    col_info = {}
    if target_col:
        if target_col in df.columns:
            y_true = df[target_col].copy()
            df.drop(columns=[target_col], inplace=True)
        else:
            logger.warning(f"Target column '{target_col}' not found in evaluation data; metrics computation will be skipped.")
            col_info["missing_target"] = target_col

    # Align evaluation data with training columns
    if hasattr(model, "_training_columns"):
        training_cols = set(model._training_columns)
        current_cols = set(df.columns)
        extra_cols = list(current_cols - training_cols)
        missing_cols = list(training_cols - current_cols)
        if extra_cols:
            logger.warning(f"Extra columns detected in evaluation data that will be dropped: {extra_cols}")
            col_info["extra_columns_dropped"] = extra_cols
            df.drop(columns=extra_cols, inplace=True)
        if missing_cols:
            msg = f"Missing expected columns from training: {missing_cols}. Please check your dataset."
            logger.error(msg)
            raise ValueError(msg)
        # Reorder columns to match training
        df = df[list(model._training_columns)]
        col_info["used_columns"] = list(model._training_columns)

    try:
        predictions = model.predict(df)
        logger.info(f"Prediction completed successfully on {len(predictions)} samples.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

    results = {"predictions": predictions.tolist(), "column_info": col_info}

    if y_true is not None:
        metrics = {}
        if is_classifier(model):
            try:
                metrics["Accuracy"] = accuracy_score(y_true, predictions)
                metrics["F1"] = f1_score(y_true, predictions, average="weighted")
            except Exception as e:
                logger.error(f"Error computing classification metrics: {e}")
        elif is_regressor(model):
            try:
                metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_true, predictions)))
                metrics["R2"] = r2_score(y_true, predictions)
            except Exception as e:
                logger.error(f"Error computing regression metrics: {e}")
        else:
            logger.warning("Unknown model type; skipping metric computation.")
        results["metrics"] = metrics
        results["actual"] = y_true.tolist()

    logger.info("Evaluation completed successfully.")
    return results
