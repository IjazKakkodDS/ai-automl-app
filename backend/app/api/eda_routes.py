from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import os
import logging
import pandas as pd
import io
import base64
from fastapi.encoders import jsonable_encoder
import numpy as np

from backend.app.agents.eda_agent import generate_eda
from backend.app.utils.file_utils import load_dataset

router = APIRouter()
logging.basicConfig(level=logging.INFO)

@router.post("/analysis/")
async def perform_eda(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    folder: str = Form("processed_data"),
    exclude_date_features: bool = Form(True),
    interactive: bool = Form(True),
    correlation_method: str = Form("pearson"),
    sample_size: int = Form(100000),
    max_numeric_cols: int = Form(8)
):
    try:
        # PRIORITIZE an uploaded file over a stored dataset_id.
        if file:
            df = load_dataset(file.file)
            logging.info("Using uploaded file for EDA.")
        elif dataset_id:
            csv_path = os.path.join(folder, f"{dataset_id}.csv")
            if not os.path.exists(csv_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"No file found for dataset_id={dataset_id} in folder={folder}."
                )
            df = load_dataset(csv_path)
            logging.info(f"Using stored dataset_id {dataset_id} from folder {folder} for EDA.")
        else:
            raise HTTPException(
                status_code=400,
                detail="No dataset provided. Please upload a file or provide a dataset_id."
            )
        if df.empty:
            raise ValueError("Dataset is empty or failed to load.")

        report_text, eda_tables, eda_figures = generate_eda(
            df=df,
            dataset_id=dataset_id,
            exclude_date_features=exclude_date_features,
            correlation_method=correlation_method,
            interactive=interactive,
            sample_size=sample_size,
            max_numeric_cols=max_numeric_cols
        )
        tables_json = {}
        for name, table_df in eda_tables.items():
            # Replace infs and NaNs for safe JSON serialization
            table_df = table_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
            records = table_df.to_dict(orient="records")
            tables_json[name] = jsonable_encoder(records)
        figures_json = {}
        for fig_name, fig_obj in eda_figures.items():
            if fig_obj is None:
                continue
            if hasattr(fig_obj, "to_json"):
                figures_json[fig_name] = fig_obj.to_json()
            else:
                buf = io.BytesIO()
                fig_obj.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                figures_json[fig_name] = f"data:image/png;base64,{img_base64}"
        return {
            "status": "success",
            "eda_report": report_text,
            "tables": tables_json,
            "figures": figures_json
        }
    except Exception as e:
        logging.error(f"EDA failed: {e}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")
