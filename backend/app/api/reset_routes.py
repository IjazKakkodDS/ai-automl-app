# reset_routes.py

import os
import shutil
import uuid
import stat
import logging
from fastapi import APIRouter, HTTPException

router = APIRouter()
logging.basicConfig(level=logging.INFO)

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If a file cannot be removed due to permission issues, change its mode and try again.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logging.error(f"Failed to remove {path}: {e}")

@router.post("/restart-analysis")
def restart_analysis():
    """
    Clears all saved models (and optionally caches) and returns a new session ID.
    WARNING: This operation is irreversible.
    """
    base_model_dir = "models"
    try:
        if os.path.exists(base_model_dir):
            # Remove the entire models folder using our error handler.
            shutil.rmtree(base_model_dir, onerror=on_rm_error)
        # Generate a new session ID using UUID.
        new_session_id = str(uuid.uuid4())
        session_folder = os.path.join(base_model_dir, new_session_id)
        os.makedirs(session_folder, exist_ok=True)
        logging.info(f"Restarted analysis. New session id: {new_session_id}")
        return {"status": "success", "session_id": new_session_id}
    except Exception as e:
        logging.error(f"Error in restart_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
