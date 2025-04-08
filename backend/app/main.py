import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

from backend.app.api import (
    preprocessing_routes,
    eda_routes,
    feature_engineering_routes,
    model_training_routes,
    forecasting_routes,
    evaluation_routes,
    ai_insights_routes,
    rag_routes,
    orchestrator_routes,
    reports_routes
)
# Import the new reset routes
from backend.app.api import reset_routes

app = FastAPI(
    title="AI AutoML Backend",
    description="Serious RAG + Agentic AI pipeline for enterprise ML workflows.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(preprocessing_routes.router, prefix="/preprocessing")
app.include_router(eda_routes.router, prefix="/eda")
app.include_router(feature_engineering_routes.router, prefix="/feature-engineering")
app.include_router(model_training_routes.router, prefix="/model-training")
app.include_router(forecasting_routes.router, prefix="/forecasting")
app.include_router(evaluation_routes.router, prefix="/evaluation", tags=["Evaluation"])
app.include_router(ai_insights_routes.router, prefix="/ai-insights", tags=["AI Insights"])
app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"])
app.include_router(orchestrator_routes.router, prefix="/orchestrator", tags=["Orchestrator"])
app.include_router(reports_routes.router, prefix="/reports", tags=["Reports"])

# Include the new reset endpoint
app.include_router(reset_routes.router, prefix="/reset", tags=["Reset"])

@app.get("/")
def read_root():
    return {
        "message": "AI AutoML Backend is running. This is the RAG + Agentic AI pipeline."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
