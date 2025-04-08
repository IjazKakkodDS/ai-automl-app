from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import json
import logging
from typing import Optional
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np

from backend.app.agents.feature_engineering_agent import run_advanced_feature_engineering
from backend.app.utils.file_utils import load_dataset, save_processed_data

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("/advanced/")
async def advanced_feature_engineering(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    folder: str = Form("processed_data"),
    impute_plan: Optional[str] = Form(None),
    knn_neighbors: int = Form(3),
    cat_impute_value: str = Form("Unknown"),
    outlier_plan: Optional[str] = Form(None),
    date_plan: Optional[str] = Form(None),
    convert_plan: Optional[str] = Form(None),
    encoding_method: str = Form("one-hot"),
    scaling_method: str = Form("standard"),
    apply_poly: bool = Form(False),
    poly_degree: int = Form(2),
    interaction_only: bool = Form(False),
    include_bias: bool = Form(False),
    apply_log: bool = Form(False),
    save_result: bool = Form(False),
    high_card_threshold: int = Form(10),
    high_card_option: str = Form("frequency")
):
    try:
        if dataset_id:
            csv_path = os.path.join(folder, f"{dataset_id}.csv")
            if not os.path.exists(csv_path):
                # Try alternate folder if not found
                alt_folder = "original_data" if folder == "processed_data" else "processed_data"
                alt_path = os.path.join(alt_folder, f"{dataset_id}.csv")
                if not os.path.exists(alt_path):
                    raise ValueError(f"No file found for dataset_id={dataset_id} in {folder} or {alt_folder}.")
                csv_path = alt_path
            df = load_dataset(csv_path)
        elif file:
            df = load_dataset(file.file)
        else:
            raise ValueError("No dataset provided. Please upload a file or provide a dataset_id.")
        if df.empty:
            raise ValueError("Dataset is empty or failed to load.")

        def parse_dict(json_str: Optional[str]) -> dict:
            return json.loads(json_str) if json_str else {}

        impute_dict = parse_dict(impute_plan)
        outlier_dict = parse_dict(outlier_plan)
        date_dict = parse_dict(date_plan)
        convert_dict = parse_dict(convert_plan)

        df_fe = run_advanced_feature_engineering(
            df=df,
            impute_plan=impute_dict,
            knn_neighbors=knn_neighbors,
            cat_impute_value=cat_impute_value,
            outlier_plan=outlier_dict,
            date_plan=date_dict,
            convert_plan=convert_dict,
            encoding_method=encoding_method,
            scaling_method=scaling_method,
            apply_poly=apply_poly,
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            apply_log=apply_log,
            high_card_threshold=high_card_threshold,
            high_card_option=high_card_option
        )

        processed_file_path = None
        if save_result:
            processed_file_path = save_processed_data(df_fe, folder="feature_engineered_data")

        df_fe = df_fe.replace([np.inf, -np.inf], np.nan).replace({np.nan: None, pd.NaT: None})
        preview_count = min(len(df_fe), 1000)
        records = df_fe.head(preview_count).to_dict(orient="records")

        return {
            "status": "success",
            "shape": [len(df_fe), df_fe.shape[1]],
            "processed_file_path": processed_file_path,
            "sample_data": jsonable_encoder(records)
        }

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

@router.get("/columns/")
async def get_dataset_columns(dataset_id: str, folder: str = "processed_data"):
    """
    Return basic column information for a given dataset.
    """
    try:
        csv_path = os.path.join(folder, f"{dataset_id}.csv")
        if not os.path.exists(csv_path):
            alt_path = os.path.join("original_data", f"{dataset_id}.csv")
            if not os.path.exists(alt_path):
                raise ValueError(f"No file found in {folder} or original_data for dataset_id={dataset_id}.")
            csv_path = alt_path

        df = load_dataset(csv_path)
        if df.empty:
            raise ValueError("Dataset is empty or failed to load.")

        columns = list(df.columns)
        missing_counts = {col: int(df[col].isnull().sum()) for col in columns}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return {
            "status": "success",
            "columns": columns,
            "missing_counts": missing_counts,
            "numeric_cols": numeric_cols,
            "object_cols": object_cols
        }

    except Exception as e:
        logger.error(f"Failed to get columns for dataset_id={dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Column inspection failed: {str(e)}")
