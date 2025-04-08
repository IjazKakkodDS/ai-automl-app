import sys
from pathlib import Path

# Append the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import pandas as pd
import numpy as np
import tempfile
import uuid

from backend.app.agents.preprocessing_agent import (
    identify_column_types,
    remove_outliers_iqr,
    process_date_columns,
    run_preprocessing_pipeline,
)

def test_identify_column_types():
    # Create a DataFrame with known column types
    df = pd.DataFrame({
        'num': [1, 2, 3],
        'cat': ['a', 'b', 'c'],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03']
    })
    # Use explicit_datetime_cols to mark the 'date' column
    numerical, categorical, datetime_cols = identify_column_types(df, explicit_datetime_cols=['date'])
    
    assert 'num' in numerical, "Numeric column 'num' was not identified."
    assert 'cat' in categorical, "Categorical column 'cat' was not identified."
    assert 'date' in datetime_cols, "Datetime column 'date' was not identified."

def test_remove_outliers_iqr():
    # Create a DataFrame with an obvious outlier
    df = pd.DataFrame({'values': [10, 12, 12, 13, 15, 100]})
    df_clean = remove_outliers_iqr(df, 'values', factor=1.5)
    
    # Check that the outlier is removed
    assert 100 not in df_clean['values'].values, "Outlier 100 was not removed."

def test_process_date_columns():
    df = pd.DataFrame({'date': ['2020-01-01', '2020-02-15', '2020-03-10']})
    df_processed = process_date_columns(df.copy(), datetime_features=['date'], drop_original_date_cols=True)
    
    # Check that new date columns are created and original is dropped
    for suffix in ['year', 'month', 'day', 'dayofweek']:
        assert f"date_{suffix}" in df_processed.columns, f"Column 'date_{suffix}' not found."
    assert 'date' not in df_processed.columns, "Original 'date' column was not dropped."

def test_run_preprocessing_pipeline():
    # Create temporary CSV file with sample data
    sample_data = """A,B,C,D
1,apple,2020-01-01,10.5
2,banana,2020-02-01,20.5
3,,2020-03-01,30.5
,orange,not_a_date,40.5
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as temp_file:
        temp_file.write(sample_data)
        temp_file_path = temp_file.name

    try:
        # Run the pipeline using the temporary CSV file
        df_processed, final_path, dataset_id, raw_dataset_id = run_preprocessing_pipeline(
            temp_file_path,
            explicit_datetime_cols=["C"],
            remove_outliers=False  # Disable outlier removal for this test
        )
        
        # Check if the processed DataFrame has expected transformations
        assert os.path.exists(final_path), "Processed file was not saved."
        assert "C_year" in df_processed.columns, "Datetime processing did not create 'C_year' column."
        # You can add more assertions to validate imputation, scaling, and encoding.

    finally:
        # Cleanup: Remove the temporary file
        os.remove(temp_file_path)

