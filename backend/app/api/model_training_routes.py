from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import pandas as pd
import logging
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from backend.app.agents.model_training_agent import train_and_evaluate_models
from backend.app.utils.file_utils import load_dataset
import os

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("/train/")
async def train_models(
    file: UploadFile = File(None),
    dataset_id: Optional[str] = Query(None),
    feature_engineered_id: Optional[str] = Query(None),
    dataset_source: str = Query("original"),
    target_col: str = Query(...),
    task_type: str = Query("regression"),
    selected_models: Optional[List[str]] = Query(None, alias="selected_models[]"),
    selected_metrics: Optional[List[str]] = Query(None, alias="selected_metrics[]"),
    selected_features: Optional[List[str]] = Query(None, alias="selected_features[]"),
    enable_cross_validation: bool = Query(False),
    cv_folds: int = Query(3),
    sample_data: bool = Query(False),
    sample_size: int = Query(100000),
    hyperparameter_tuning: bool = Query(False),
    hyperparameter_search_method: str = Query("grid"),
    random_search_iter: int = Query(10),
    apply_encoding: bool = Query(False),
    encoding_method: str = Query("one-hot"),
    apply_scaling: bool = Query(False),
    scaling_method: str = Query("standard")
):
    logger.info(f"[DEBUG] train_models called: dataset_source={dataset_source}, dataset_id={dataset_id}, feature_engineered_id={feature_engineered_id}")
    try:
        df = None
        if dataset_source == "original":
            # For the original dataset, if a file is uploaded, use it.
            if file:
                df = load_dataset(file.file)
            elif dataset_id:
                # Load from original_data folder using stored original ID
                data_path = f"original_data/{dataset_id}.csv"
                if not os.path.exists(data_path):
                    raise HTTPException(status_code=400, detail=f"No original file found for dataset_id='{dataset_id}'.")
                df = pd.read_csv(data_path)
            else:
                raise HTTPException(status_code=400, detail="No file or dataset_id provided for original dataset.")
        elif dataset_source == "preprocessed":
            if not dataset_id:
                raise HTTPException(status_code=400, detail="No dataset_id provided for preprocessed dataset.")
            data_path = f"processed_data/{dataset_id}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(status_code=400, detail=f"No processed file found for dataset_id='{dataset_id}'.")
            df = pd.read_csv(data_path)
        elif dataset_source == "feature_engineered":
            if not feature_engineered_id:
                raise HTTPException(status_code=400, detail="No feature_engineered_id provided.")
            data_path = f"feature_engineered_data/{feature_engineered_id}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(status_code=400, detail=f"No feature-engineered file for id='{feature_engineered_id}'.")
            df = pd.read_csv(data_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown dataset_source='{dataset_source}'.")
        if df is None or df.empty:
            raise ValueError("Dataset is empty or None.")
        if not selected_models or len(selected_models) == 0:
            raise HTTPException(status_code=400, detail="No models selected.")
        if not selected_metrics or len(selected_metrics) == 0:
            raise HTTPException(status_code=400, detail="No metrics selected.")
        output = train_and_evaluate_models(
            df=df,
            target_col=target_col,
            task_type=task_type,
            selected_models=selected_models,
            selected_metrics=selected_metrics,
            selected_features=selected_features,
            enable_cross_validation=enable_cross_validation,
            cv_folds=cv_folds,
            sample_data=sample_data,
            sample_size=sample_size,
            hyperparameter_tuning=hyperparameter_tuning,
            hyperparameter_search_method=hyperparameter_search_method,
            random_search_iter=random_search_iter,
            apply_encoding=apply_encoding if dataset_source == "original" else False,
            encoding_method=encoding_method,
            apply_scaling=apply_scaling if dataset_source == "original" else False,
            scaling_method=scaling_method,
            dataset_id=dataset_id
        )
        return {
            "status": "success",
            "results": jsonable_encoder(output["results"]),
            "trained_models": output["trained_models"]
        }
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
