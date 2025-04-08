# model_training_agent.py

import os
import time
import logging
import sys
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Callable
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.base import is_classifier, is_regressor

try:
    from xgboost import XGBRegressor, XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

def get_models(task_type: str) -> Dict[str, object]:
    if task_type.lower() == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "SVR": SVR(),
        }
        if xgboost_available:
            models["XGBRegressor"] = XGBRegressor(n_estimators=50, random_state=42)
        return models
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "SVC": SVC(probability=True),
        }
        if xgboost_available:
            models["XGBClassifier"] = XGBClassifier(n_estimators=50, random_state=42)
        return models

def get_metrics(task_type: str) -> Dict[str, Callable]:
    if task_type.lower() == "regression":
        return {
            "RMSE": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score,
            "MAE": mean_absolute_error,
        }
    else:
        return {
            "Accuracy": accuracy_score,
            "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
            "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }

def get_default_param_grid(model_name: str, task_type: str) -> dict:
    if task_type.lower() == "regression":
        grids = {
            "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
            "GradientBoostingRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            "XGBRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        }
    else:
        grids = {
            "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
            "GradientBoostingClassifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "SVC": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            "XGBClassifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        }
    return grids.get(model_name, {})

def apply_minimal_encoding_scaling(df: pd.DataFrame, apply_encoding: bool, encoding_method: str, apply_scaling: bool, scaling_method: str, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if exclude_columns is None:
        exclude_columns = []
    if apply_encoding:
        cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude_columns]
        if encoding_method == "label":
            for c in cat_cols:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
        else:
            if len(cat_cols) > 0:
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded = ohe.fit_transform(df[cat_cols].astype(str))
                new_cols = ohe.get_feature_names_out(cat_cols)
                enc_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                df.drop(columns=cat_cols, inplace=True)
                df = pd.concat([df, enc_df], axis=1)
    if apply_scaling:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler() if scaling_method == "minmax" else StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    selected_models: List[str],
    selected_metrics: List[str],
    selected_features: Optional[List[str]] = None,
    enable_cross_validation: bool = False,
    cv_folds: int = 3,
    sample_data: bool = False,
    sample_size: int = 100000,
    hyperparameter_tuning: bool = False,
    hyperparameter_search_method: str = "grid",
    random_search_iter: int = 10,
    model_folder: str = "models",
    session_id: Optional[str] = None,  # NEW parameter for session-specific folder
    apply_encoding: bool = False,
    encoding_method: str = "one-hot",
    apply_scaling: bool = False,
    scaling_method: str = "standard",
    dataset_id: Optional[str] = None
) -> Dict[str, any]:
    # If a session_id is provided, update the model_folder accordingly.
    if session_id:
        model_folder = os.path.join("models", session_id)
    os.makedirs(model_folder, exist_ok=True)
    
    if df.empty:
        raise ValueError("Dataset is empty. Cannot train models.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if apply_encoding or apply_scaling:
        df = apply_minimal_encoding_scaling(df, apply_encoding, encoding_method, apply_scaling, scaling_method, exclude_columns=[target_col])
    if selected_features and len(selected_features) > 0:
        missing_feats = [f for f in selected_features if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing selected features: {missing_feats}")
        df = df[selected_features + [target_col]]
    if sample_data and df.shape[0] > sample_size:
        logger.info(f"Sampling {sample_size} rows from {df.shape[0]}.")
        df = df.sample(n=sample_size, random_state=42)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if task_type.lower() == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_dict = get_models(task_type)
    metric_dict = get_metrics(task_type)
    models_to_train = {m: model_dict[m] for m in selected_models if m in model_dict}
    metrics_to_evaluate = {m: metric_dict[m] for m in selected_metrics if m in metric_dict}
    logger.info(f"Training models: {list(models_to_train.keys())}")
    logger.info(f"Evaluating metrics: {list(metrics_to_evaluate.keys())}")
    results = []
    trained_models = {}
    for model_name, model_obj in models_to_train.items():
        try:
            logger.info(f"Training {model_name}...")
            param_grid = get_default_param_grid(model_name, task_type)
            if hyperparameter_tuning and param_grid:
                scoring_method = "accuracy" if task_type.lower() == "classification" else "neg_mean_squared_error"
                if hyperparameter_search_method == "random":
                    search = RandomizedSearchCV(model_obj, param_grid, n_iter=random_search_iter, cv=cv_folds, scoring=scoring_method, random_state=42)
                else:
                    search = GridSearchCV(model_obj, param_grid, cv=cv_folds, scoring=scoring_method)
                search.fit(X_train, y_train)
                model_obj = search.best_estimator_
                logger.info(f"Best params for {model_name}: {search.best_params_}")
            start_time = time.time()
            model_obj.fit(X_train, y_train)
            training_time = time.time() - start_time
            model_obj._training_columns = X_train.columns.tolist()
            y_pred = model_obj.predict(X_test)
            row = {"Model": model_name, "Training_Time_sec": round(training_time, 3)}
            for metric_name, metric_func in metrics_to_evaluate.items():
                val = metric_func(y_test, y_pred)
                row[metric_name] = round(float(val), 4)
            if enable_cross_validation:
                try:
                    if task_type.lower() == "classification" and "F1" in selected_metrics:
                        cv_scores = cross_val_score(model_obj, X, y, scoring='f1_weighted', cv=cv_folds)
                        row["CV_F1_Avg"] = round(float(np.mean(cv_scores)), 4)
                    elif task_type.lower() == "regression" and "R2" in selected_metrics:
                        cv_scores = cross_val_score(model_obj, X, y, scoring='r2', cv=cv_folds)
                        row["CV_R2_Avg"] = round(float(np.mean(cv_scores)), 4)
                except Exception as cv_error:
                    logger.error(f"CV failed for {model_name}: {cv_error}")
            timestamp = int(time.time())
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = os.path.join(model_folder, model_filename)
            joblib.dump(model_obj, model_path)
            logger.info(f"Saved model {model_name} at {model_path}")
            # Store just the filename so that evaluation endpoints can use the session folder.
            trained_models[model_name] = model_filename
            results.append(row)
        except Exception as e:
            logger.exception(f"Training failed for {model_name}")
            continue
    if results:
        try:
            if dataset_id:
                report_path = os.path.join("reports", f"training_results_{dataset_id}.csv")
            else:
                report_folder = "model_reports"
                os.makedirs(report_folder, exist_ok=True)
                report_filename = f"training_report_{int(time.time())}.csv"
                report_path = os.path.join(report_folder, report_filename)
            pd.DataFrame(results).to_csv(report_path, index=False)
            logger.info(f"Training report saved at {report_path}")
        except Exception as e:
            logger.error("Failed to save training report: " + str(e))
    return {"results": results, "trained_models": trained_models}
