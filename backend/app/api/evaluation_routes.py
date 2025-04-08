from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
import os
import pandas as pd
import logging
from typing import Optional
from fastapi.encoders import jsonable_encoder
from backend.app.agents.evaluation_agent import evaluate_model
from backend.app.utils.file_utils import load_dataset

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.get("/list-models/")
def list_models(session_id: Optional[str] = None):
    """
    Return a list of model files (.pkl or .joblib) in the session-specific folder.
    """
    model_dir = os.path.join("models", session_id) if session_id else "models"
    if not os.path.exists(model_dir):
        return {"status": "success", "models": []}
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") or f.endswith(".joblib")]
    return {"status": "success", "models": model_files}

@router.post("/evaluate/")
async def evaluate(
    file: UploadFile = File(None),
    dataset_id: Optional[str] = Query(None),
    model_file: str = Form(...),
    has_target: bool = Form(False),
    target_col: str = Form(""),
    session_id: Optional[str] = Form(None)
):
    """
    Evaluate a saved model from the session-specific folder.
    """
    try:
        if dataset_id:
            data_path = f"processed_data/{dataset_id}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(status_code=400, detail=f"No processed file for dataset_id='{dataset_id}'.")
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError(f"File at {data_path} is empty or invalid.")
        else:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded and no dataset_id provided.")
            df = load_dataset(file.file)
            if df.empty:
                raise ValueError("Uploaded file is empty or failed to load.")
        model_dir = os.path.join("models", session_id) if session_id else "models"
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_file} not found in '{model_dir}' directory.")
        target = target_col if has_target and target_col.strip() != "" else None
        results = evaluate_model(new_df=df, model_path=model_path, target_col=target)
        return {"status": "success", "results": jsonable_encoder(results)}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
