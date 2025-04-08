import os
import logging
import subprocess
import hashlib
import json
import re
import ollama
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

AVAILABLE_MODELS = ["mistral", "gemma2", "llama3.3", "llama2", "gpt-4"]
CACHE_DIR = "ai_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(prompt: str, model_choice: str) -> str:
    try:
        key = f"{model_choice}:{prompt}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{hash_key}.json")
    except Exception as e:
        logger.exception("Error generating cache filename")
        raise e

def is_model_available(model_name: str) -> bool:
    try:
        output = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available_models = output.stdout.lower()
        return model_name.lower() in available_models
    except Exception as e:
        logger.exception("Error checking model availability")
        return False

def clean_response(raw_response: str) -> str:
    try:
        cleaned = re.sub(r'\[/?[A-Z_]+\]', '', raw_response)
        cleaned = re.sub(r'</s>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    except Exception as e:
        logger.exception("Error cleaning AI model response")
        raise e

def generate_ai_insights(
    eda_summary: str,
    model_summary: str = "",
    model_choice: str = "mistral",
    chunk_threshold: int = 1500,
    force_regenerate: bool = False,
    enable_cot: bool = False,
    max_tokens: int = 512,
    clear_cache: bool = False
) -> str:
    """
    Generate AI insights based on provided EDA and model training summaries.
    If force_regenerate or clear_cache is True, any existing cache file is removed.
    """
    try:
        if model_choice not in AVAILABLE_MODELS:
            error_msg = f"Invalid model '{model_choice}'. Choose from {AVAILABLE_MODELS}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not is_model_available(model_choice):
            error_msg = f"Selected model '{model_choice}' is not available in Ollama."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        combined_summary = (
            f"EDA Summary:\n{eda_summary}\n\n"
            f"Model Training Summary:\n{model_summary}\n\n"
        )
        if len(combined_summary) > chunk_threshold:
            logger.info("Combined summary exceeds threshold; truncating.")
            combined_summary = combined_summary[:chunk_threshold] + "\n\n[TRUNCATED FOR LENGTH]\n"
        
        cot_instruction = ""
        if enable_cot:
            cot_instruction = "Also, please include your chain-of-thought in a section labeled 'Chain-of-Thought'. "

        prompt = (
            "You are an AI assistant with context from an Exploratory Data Analysis (EDA) and a Model Training Summary. "
            "Use the information provided below to create structured, actionable insights in bullet points.\n\n"
            f"{combined_summary}"
            "Outline:\n1) General Insights\n2) Model Performance\n3) Recommendations\n"
            f"{cot_instruction}"
            "Think step by step."
        )
        
        logger.info("Generated prompt: %s", prompt)
        
        cache_file = get_cache_filename(prompt, model_choice)
        if (force_regenerate or clear_cache) and os.path.exists(cache_file):
            logger.info("Force regenerate/clear cache enabled. Removing cache file.")
            os.remove(cache_file)
        if not force_regenerate and os.path.exists(cache_file):
            logger.info("Loading AI insights from cache.")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_response = json.load(f)
                    cached_insights = cached_response.get("insights", "").strip()
                    if cached_insights:
                        logger.info("Using cached insights.")
                        return cached_insights
                    else:
                        logger.info("Cached insights empty; regenerating.")
            except Exception as cache_e:
                logger.error("Error reading cache file: %s", cache_e)
        
        try:
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            if "requires more system memory" in str(e):
                logger.warning("Model %s requires more memory; falling back to mistral.", model_choice)
                fallback_model = "mistral"
                try:
                    response = ollama.chat(
                        model=fallback_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    model_choice = fallback_model
                except Exception as fallback_e:
                    logger.exception("Fallback to lighter model also failed.")
                    raise fallback_e
            elif model_choice.lower() == "gpt-4":
                logger.warning("Error with GPT-4; falling back to mistral.")
                try:
                    response = ollama.chat(
                        model="mistral",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    model_choice = "mistral"
                except Exception as fallback_e:
                    logger.exception("Fallback to mistral also failed.")
                    raise fallback_e
            else:
                logger.exception("Error calling Ollama with model=%s", model_choice)
                raise e
        
        if isinstance(response, dict):
            if "response" in response:
                insights = response["response"]
            elif "message" in response and "content" in response["message"]:
                insights = response["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {response}")
        elif isinstance(response, str):
            insights = clean_response(response)
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            insights = response.message.content
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        insights = insights.strip()
        if insights:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"insights": insights}, f)
                logger.info("Cached new AI insights.")
            except Exception as save_cache_e:
                logger.error("Error saving cache file: %s", save_cache_e)
        else:
            logger.warning("AI model returned empty insights; not caching.")
        
        return insights
    except Exception as e:
        logger.exception("Error generating AI insights")
        raise e

def get_eda_summary(report_path: str) -> str:
    try:
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"No EDA report found at {report_path}.")
        with open(report_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.exception("Error retrieving EDA summary")
        raise e

def get_model_summary(report_path: str) -> str:
    try:
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"No model training report found at {report_path}.")
        df = pd.read_csv(report_path)
        if df.empty:
            raise ValueError(f"Training results CSV is empty at {report_path}.")
        summary = df.head().to_string(index=False)
        return f"Model Training Results:\n\n{summary}"
    except Exception as e:
        logger.exception("Error reading model training summary")
        raise e
