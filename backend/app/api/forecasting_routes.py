from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import pandas as pd
import numpy as np
import logging
import os
from typing import Optional
from fastapi.encoders import jsonable_encoder

from backend.app.agents.forecasting_agent import train_prophet, train_arima
from backend.app.utils.file_utils import load_dataset

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("/forecast/")
async def forecast(
    file: UploadFile = File(None),
    dataset_id: Optional[str] = Query(None),
    forecast_model: str = Query("prophet"),
    target_col: str = Query("Sales"),
    forecast_period: int = Query(30),
    max_history: int = Query(10000),
    resample_freq: Optional[str] = Query(None),
    p: int = Query(5),
    d: int = Query(1),
    q: int = Query(0),
    folder: str = Query("processed_data")
):
    """
    Forecast endpoint that accepts either an uploaded CSV or a preprocessed/original dataset.
    """
    try:
        if dataset_id:
            data_path = f"{folder}/{dataset_id}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(status_code=400, detail=f"No file found for dataset_id='{dataset_id}' in folder '{folder}'.")
            df = pd.read_csv(data_path)
        else:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded and no dataset_id provided.")
            df = load_dataset(file.file)
        if df.empty:
            raise ValueError("Dataset is empty or invalid.")
        if forecast_model.lower() == "prophet":
            forecast_output = train_prophet(
                df=df,
                target_col=target_col,
                forecast_period=forecast_period,
                max_history=max_history,
                resample_freq=resample_freq
            )
            model_name = "Prophet"
        elif forecast_model.lower() == "arima":
            forecast_output = train_arima(
                df=df,
                target_col=target_col,
                forecast_period=forecast_period,
                order=(p, d, q),
                max_history=max_history
            )
            model_name = "ARIMA"
        else:
            raise ValueError(f"Unknown forecast_model '{forecast_model}'.")
        forecast_results_df = forecast_output["results"].replace([np.inf, -np.inf], np.nan)
        forecast_results_df = forecast_results_df.replace({np.nan: None, pd.NaT: None})
        records = forecast_results_df.to_dict(orient="records")
        forecast_plot = forecast_output["plot"]
        return {
            "status": "success",
            "model": model_name,
            "forecast_period": forecast_period,
            "results": jsonable_encoder(records),
            "forecast_plot": forecast_plot
        }
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")
