import os
import logging
import pandas as pd
import tempfile
from datetime import datetime
from typing import Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FILE_SIZE_THRESHOLD = 25 * 1024 * 1024  # 25 MB
CHUNK_SIZE = 100000

def load_dataset(file: Union[str, object]) -> pd.DataFrame:
    """
    Load a dataset from a file path or file-like object using pandas.
    Supports chunked reading for large files.
    """
    try:
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError(f"File '{file}' does not exist.")
            file_size = os.path.getsize(file)
            file_obj = None
        else:
            file_size = getattr(file, "size", None)
            file_obj = file

        if file_size is not None and file_size > FILE_SIZE_THRESHOLD:
            logging.info(f"Large file detected (size: {file_size} bytes). Using chunked reading...")
            return _chunked_read_csv(file, file_obj)
        else:
            if file_obj:
                file_obj.seek(0)
                df = pd.read_csv(file_obj)
            else:
                df = pd.read_csv(file)
            logging.info("Dataset loaded successfully with pandas read_csv.")
            return df
    except Exception as e:
        logging.exception("Unexpected error loading dataset")
        raise

def _chunked_read_csv(file: Union[str, os.PathLike], file_obj: object = None) -> pd.DataFrame:
    """
    Read a large CSV file in chunks and combine them into a single DataFrame.
    """
    chunks = []
    try:
        if file_obj:
            file_obj.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            try:
                for chunk in pd.read_csv(tmp_path, chunksize=CHUNK_SIZE):
                    chunks.append(chunk)
            finally:
                os.remove(tmp_path)
        else:
            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logging.info("Dataset loaded with chunked reading.")
        return df
    except Exception as e:
        logging.exception("Error during chunked CSV reading")
        raise

def save_processed_data(df: pd.DataFrame, folder: str = "processed_data") -> str:
    """
    Save a DataFrame to CSV with a timestamp in the specified folder.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"processed_{timestamp}.csv"
        file_path = os.path.join(folder, file_name)
        df.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}")
        raise
