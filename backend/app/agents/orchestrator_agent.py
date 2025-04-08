import logging
import pandas as pd
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer

from backend.app.agents.eda_agent import generate_eda
from backend.app.agents.feature_engineering_agent import apply_feature_transformations
from backend.app.agents.model_training_agent import train_and_evaluate_models
from backend.app.agents.ai_insights_agent import generate_ai_insights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

INDEX_FILE = "data_1/faiss_index.index"
METADATA_FILE = "data_1/metadata.json"
DOCS_FOLDER = "data_1/documents"

retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
metadata = None
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

CHUNK_SIZE = 500
OVERLAP = 100

def get_chunk_text(full_text: str, chunk_idx: int, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> str:
    start = max(0, chunk_idx * chunk_size - chunk_idx * overlap)
    end = start + chunk_size
    return full_text[start:end]

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant text snippets from the FAISS index based on the query.
    Returns a combined string of the top-k snippets.
    """
    if index is None or metadata is None:
        return "No retrieval index loaded."
    query_embedding = retrieval_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    combined_text = ""
    for idx in indices[0]:
        if idx < len(metadata):
            meta = metadata[idx]
            filename = meta["filename"]
            chunk_idx = meta["chunk_idx"]
            file_path = os.path.join(DOCS_FOLDER, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    doc_text = f.read()
                snippet = get_chunk_text(doc_text, chunk_idx)
                combined_text += f"Snippet from {filename} (chunk {chunk_idx}):\n{snippet}\n\n"
            else:
                combined_text += f"[File not found: {filename}]\n\n"
    return combined_text

class OrchestratorAgent:
    def __init__(self):
        self.eda_quality_threshold = 0.8
        self.model_performance_threshold = 0.75

    def decide_next_steps(self, data: pd.DataFrame, target_col: str = "target") -> dict:
        logger.info("Orchestrator: Starting EDA analysis...")
        decision = {}
        try:
            report, tables, figures = generate_eda(
                data,
                target_col=None,
                interactive=False,
                exclude_date_features=True,
                correlation_method="pearson",
                sample_size=100000
            )
            decision["eda_report"] = report
        except Exception as e:
            logger.error("Orchestrator: EDA failed: %s", e)
            return {"error": "EDA failed", "details": str(e)}
        data_quality = 0.85
        if data_quality < self.eda_quality_threshold:
            logger.info("Data quality is low. Suggesting advanced feature engineering.")
            decision["next_action"] = "Perform advanced feature engineering."
            decision["reason"] = f"Data quality score ({data_quality}) < threshold ({self.eda_quality_threshold})."
        else:
            logger.info("Data quality is acceptable. Proceeding to model training...")
            try:
                train_output = train_and_evaluate_models(
                    df=data,
                    target_col=target_col,
                    task_type="regression",
                    selected_models=["LinearRegression", "RandomForestRegressor"],
                    selected_metrics=["RMSE", "R2"],
                    enable_cross_validation=False,
                    cv_folds=3,
                    sample_data=False,
                    sample_size=100000,
                    hyperparameter_tuning=False,
                    model_folder="models"
                )
            except Exception as e:
                logger.error("Orchestrator: Model training failed: %s", e)
                return {"error": "Model training failed", "details": str(e)}
            decision["model_results"] = train_output["results"]
            best_r2 = max((row.get("R2", 0) for row in train_output["results"]), default=0.0)
            if best_r2 < self.model_performance_threshold:
                decision["next_action"] = "Tune hyperparameters or try alternative models."
                decision["reason"] = f"Best R2 ({best_r2}) < threshold ({self.model_performance_threshold})."
                try:
                    domain_context = retrieve_context(query="Ways to improve regression R2 score", top_k=2)
                    eda_summary = f"{report}\n\nAdditional Domain Context:\n{domain_context}"
                    model_summary = "Model training results indicate low R2. Consider adjusting hyperparameters and exploring alternative algorithms."
                    insights = generate_ai_insights(
                        eda_summary,
                        model_summary,
                        model_choice="gpt-4",
                        force_regenerate=True,
                        enable_cot=True
                    )
                    decision["ai_insights"] = insights
                except Exception as ai_e:
                    logger.error("Failed to generate AI insights: %s", ai_e)
                    decision["ai_insights"] = "No AI insights available due to an error."
            else:
                decision["next_action"] = "Proceed to forecasting or deployment."
                decision["reason"] = f"Data quality & model performance are satisfactory (R2={best_r2})."
        return decision
