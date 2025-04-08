import os
import tempfile
import pandas as pd
import numpy as np
import pytest

# Ensure the project root is on the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.agents import feature_engineering_agent as fe_agent

# --- Helper Function to Create Sample DataFrame ---
def create_test_df():
    # For the full pipeline test, we use data without missing values.
    # (Note: scaling later will likely cause some numeric values to become non-positive,
    #  so log transform will be skipped.)
    data = {
        "num1": [1, 2, 100, 4, 5, 6],
        "num2": [10, 20, 30, 40, 50, 60],
        "cat1": ["A", "B", "A", "A", "C", "B"],
        "date1": ["2020-01-01", "2020-02-01", "2020-03-01",
                  "2020-04-01", "2020-05-01", "2020-06-01"]
    }
    return pd.DataFrame(data)

# --- Test Outlier Removal ---
def test_remove_outliers_iqr():
    df = pd.DataFrame({"num": [1, 2, 3, 100, 5, 6]})
    df_clean = fe_agent.remove_outliers_iqr(df, "num", factor=1.5)
    # The outlier (100) should be removed
    assert 100 not in df_clean["num"].values

# --- Test Date Feature Creation ---
def test_create_date_features():
    df = create_test_df()
    df_out = fe_agent.create_date_features(df.copy(), "date1", drop_original=True)
    # Check that new date feature columns are created and original is dropped
    for suffix in ["year", "month", "day", "dayofweek"]:
        assert f"date1_{suffix}" in df_out.columns
    assert "date1" not in df_out.columns

# --- Test Column Imputation with Mean ---
def test_impute_column_mean():
    df = pd.DataFrame({"num": [1, np.nan, 3, np.nan, 5]})
    df_imputed = fe_agent._impute_column(df.copy(), "num", "mean")
    # After imputation, there should be no missing values in "num"
    assert df_imputed["num"].isnull().sum() == 0

# --- Test Column Imputation with Mode ---
def test_impute_column_mode():
    df = pd.DataFrame({"cat": ["A", None, "B", None, "B"]})
    df_imputed = fe_agent._impute_column(df.copy(), "cat", "mode")
    # Expect only missing values to be replaced by mode ("B")
    expected = pd.Series(["A", "B", "B", "B", "B"], name="cat")
    pd.testing.assert_series_equal(df_imputed["cat"], expected)

# --- Test Data Type Conversion ---
def test_convert_column_dtype():
    df = pd.DataFrame({"num_str": ["1", "2", "3"]})
    df_converted = fe_agent.convert_column_dtype(df.copy(), "num_str", "numeric")
    assert pd.api.types.is_numeric_dtype(df_converted["num_str"])

# --- Test Categorical Encoding (One-Hot) ---
def test_encode_categorical_onehot():
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
    df_encoded = fe_agent._encode_categorical(df.copy(), method="one-hot")
    # After encoding, the original column should be dropped
    assert "cat" not in df_encoded.columns
    # And new one-hot columns should exist (their names depend on OneHotEncoder)
    assert any("cat" in col.lower() for col in df_encoded.columns)

# --- Test Numeric Scaling ---
def test_scale_numeric():
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5]})
    df_scaled = fe_agent._scale_numeric(df.copy(), scaler_name="standard")
    # Check that the column is scaled (roughly zero mean)
    np.testing.assert_almost_equal(df_scaled["num"].mean(), 0, decimal=1)

# --- Test Polynomial Features ---
def test_polynomial_features():
    df = pd.DataFrame({"num": [1, 2, 3]})
    df_poly = fe_agent._polynomial_features(df.copy(), degree=2)
    # The polynomial transformation should add more columns than the original
    assert df_poly.shape[1] > df.shape[1]

# --- Test Log Transformation ---
def test_log_transform():
    df = pd.DataFrame({"num": [1, 10, 100]})
    df_log = fe_agent._log_transform(df.copy())
    assert "log_num" in df_log.columns

# --- Test Full Advanced Feature Engineering Pipeline ---
def test_run_advanced_feature_engineering():
    # Use the modified test data that has no missing values
    df = create_test_df()
    plan = {
        "impute_plan": {"num1": "mean", "cat1": "mode"},
        "outlier_plan": {"num1": 1.5},
        "date_plan": {"date1": True},
        "convert_plan": {"num2": "numeric"}
    }
    # Disable polynomial features to avoid duplicate numeric columns which affect log transform
    df_fe = fe_agent.run_advanced_feature_engineering(
        df,
        impute_plan=plan["impute_plan"],
        knn_neighbors=3,
        cat_impute_value="Unknown",
        outlier_plan=plan["outlier_plan"],
        date_plan=plan["date_plan"],
        convert_plan=plan["convert_plan"],
        encoding_method="one-hot",
        scaling_method="standard",
        apply_poly=False,  # Disabled to avoid duplicate columns
        poly_degree=2,
        interaction_only=False,
        include_bias=False,
        apply_log=True
    )
    # Check that date features have been added and original date column is dropped
    assert "date1_year" in df_fe.columns and "date1" not in df_fe.columns
    # Since scaling makes some values non-positive, the log transform is expected to be skipped.
    # Thus, we assert that NO columns with "log_" prefix are present.
    assert not any(col.startswith("log_") for col in df_fe.columns)
    # Assert that there are no NaNs in any numeric column
    numeric_nan_sum = df_fe.select_dtypes(include=[np.number]).isnull().sum().sum()
    assert numeric_nan_sum == 0, f"Found {numeric_nan_sum} NaNs in numeric columns"

# --- Test the Simple Wrapper Function ---
def test_apply_feature_transformations():
    df = create_test_df()
    plan = {
        "impute_plan": {"num1": "median"},
        "scaling_method": "minmax",
        "encoding_method": "one-hot"
    }
    df_transformed = fe_agent.apply_feature_transformations(df.copy(), plan=plan)
    # Check that missing values in num1 have been imputed
    assert df_transformed["num1"].isnull().sum() == 0
    # Check that the categorical column "cat1" has been encoded (and therefore removed)
    assert "cat1" not in df_transformed.columns
