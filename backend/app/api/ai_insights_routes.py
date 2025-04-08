from fastapi import APIRouter, HTTPException, Form, Query
import logging
from typing import Optional
from backend.app.agents.ai_insights_agent import (
    generate_ai_insights,
    get_eda_summary,
    get_model_summary,
    AVAILABLE_MODELS
)
import os

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("/generate/")
async def generate_insights(
    eda_summary: Optional[str] = Form(None),
    model_summary: Optional[str] = Form(None),
    model_choice: str = Form("mistral"),
    chunk_threshold: int = Form(1500),
    force_regenerate: bool = Form(False),
    dataset_id: Optional[str] = Query(None),
    enable_cot: bool = Form(False)
):
    try:
        if model_choice not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model '{model_choice}'. Choose from {AVAILABLE_MODELS}.")
        if dataset_id:
            eda_path = f"reports/eda_report_{dataset_id}.txt"
            model_path = f"reports/training_results_{dataset_id}.csv"
            if os.path.exists(eda_path):
                eda_summary_text = get_eda_summary(eda_path)
            else:
                eda_summary_text = "No EDA file found for this dataset_id."
            if os.path.exists(model_path):
                model_summary_text = get_model_summary(model_path)
            else:
                model_summary_text = ""
        else:
            if not eda_summary:
                raise ValueError("No EDA summary provided. Please include EDA text or a dataset_id.")
            eda_summary_text = eda_summary
            model_summary_text = model_summary if model_summary else ""
        insights = generate_ai_insights(
            eda_summary=eda_summary_text,
            model_summary=model_summary_text,
            model_choice=model_choice,
            chunk_threshold=chunk_threshold,
            force_regenerate=force_regenerate,
            enable_cot=enable_cot
        )
        return {
            "status": "success",
            "model": model_choice,
            "insights": insights
        }
    except Exception as e:
        logger.error("Error generating AI insights: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
