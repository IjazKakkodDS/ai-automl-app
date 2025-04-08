from fastapi import APIRouter, HTTPException
import logging
import pandas as pd
from io import StringIO
from fastapi.encoders import jsonable_encoder
import math
from backend.app.agents.orchestrator_agent import OrchestratorAgent

router = APIRouter()
logger = logging.getLogger(__name__)
orchestrator = OrchestratorAgent()

def convert_nan(obj):
    if isinstance(obj, dict):
        return {k: convert_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan(item) for item in obj]
    elif isinstance(obj, float):
        return None if math.isnan(obj) else obj
    else:
        return obj

@router.post("/orchestrate")
async def orchestrate(data: dict):
    try:
        if "csv_data" in data:
            csv_str = data["csv_data"]
            df = pd.read_csv(StringIO(csv_str))
        else:
            df = pd.DataFrame(data)
        results = orchestrator.decide_next_steps(df)
        encoded = jsonable_encoder(results)
        safe_response = convert_nan(encoded)
        return safe_response
    except Exception as e:
        logger.exception("Error during orchestration")
        raise HTTPException(status_code=500, detail=str(e))
