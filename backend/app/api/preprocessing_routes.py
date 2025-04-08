from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from backend.app.agents.preprocessing_agent import run_preprocessing_pipeline

router = APIRouter()

@router.post("/preprocess/")
async def preprocess_data(
    file: UploadFile = File(...),
    knn_neighbors: int = 3,
    impute_numeric: str = 'knn',
    remove_outliers: bool = False,
    outlier_factor: float = 1.5,
    drop_original_date_cols: bool = True,
    date_detection_threshold: float = 0.9,
    explicit_datetime_cols: Optional[str] = None,
    high_card_threshold: int = 10,
    high_card_option: str = 'frequency'
):
    """
    Full-featured preprocessing endpoint.
    """
    try:
        # Parse explicit datetime columns if provided (comma-separated string)
        dt_cols_list = None
        if explicit_datetime_cols:
            dt_cols_list = [col.strip() for col in explicit_datetime_cols.split(",") if col.strip()]

        df, processed_file_path, dataset_id, raw_dataset_id = run_preprocessing_pipeline(
            file=file.file,
            knn_neighbors=knn_neighbors,
            impute_numeric=impute_numeric,
            remove_outliers=remove_outliers,
            outlier_factor=outlier_factor,
            drop_original_date_cols=drop_original_date_cols,
            date_detection_threshold=date_detection_threshold,
            explicit_datetime_cols=dt_cols_list,
            high_card_threshold=high_card_threshold,
            high_card_option=high_card_option
        )

        sample_data = df.head(10).to_dict(orient="records")
        return {
            "status": "success",
            "shape": [df.shape[0], df.shape[1]],
            "processed_file_path": processed_file_path,
            "dataset_id": dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "sample_data": sample_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
