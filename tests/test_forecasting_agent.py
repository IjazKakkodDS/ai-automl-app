import os
import io
import base64
import pandas as pd
import numpy as np
import pytest
from datetime import timedelta

# Add project root to sys.path so that the module can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the forecasting functions from your module.
from backend.app.agents import forecasting_agent as fc_agent

# --- Test detect_datetime_columns ---
def test_detect_datetime_columns():
    # Create a DataFrame with a date column and a non-date column.
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10),
        "value": np.random.rand(10)
    })
    datetime_cols, df_out = fc_agent.detect_datetime_columns(df)
    # 'date' should be detected based on dtype.
    assert "date" in datetime_cols

def test_detect_datetime_reconstruction():
    # Create a DataFrame with separate year/month/day columns.
    df = pd.DataFrame({
        "date_year": [2020]*5,
        "date_month": [1, 1, 1, 1, 1],
        "date_day": [1, 2, 3, 4, 5],
        "value": np.random.rand(5)
    })
    datetime_cols, df_out = fc_agent.detect_datetime_columns(df)
    # Check that 'Reconstructed_Date' was added.
    assert "Reconstructed_Date" in datetime_cols
    # And that the new column is in the DataFrame.
    assert "Reconstructed_Date" in df_out.columns

# --- Test plot_forecast ---
def test_plot_forecast():
    # Create a simple forecast DataFrame.
    forecast_results = pd.DataFrame({
        "ds": pd.date_range("2020-02-01", periods=5),
        "yhat": np.arange(5),
        "yhat_lower": np.arange(5) - 0.5,
        "yhat_upper": np.arange(5) + 0.5
    })
    # Create a dummy historical DataFrame.
    historical = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=10),
        "y": np.random.rand(10)
    })
    image_base64 = fc_agent.plot_forecast(forecast_results, "Test Model", historical=historical)
    # The function should return a non-empty base64 string.
    assert isinstance(image_base64, str)
    assert len(image_base64) > 0
    # Optionally, decode it to ensure it's valid base64.
    decoded = base64.b64decode(image_base64)
    assert isinstance(decoded, bytes)

# --- Test naive_forecast ---
def test_naive_forecast():
    # Create a DataFrame with columns ['ds', 'y']
    df = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=10),
        "y": np.arange(10)
    })
    forecast_period = 3
    forecast_df = fc_agent.naive_forecast(df, forecast_period)
    # The naive forecast should repeat the last value of 'y'
    last_value = df["y"].iloc[-1]
    assert all(forecast_df["yhat"] == last_value)
    # Check that the forecast DataFrame has the expected number of rows
    assert len(forecast_df) == forecast_period

# --- Test save_forecasting_log ---
def test_save_forecasting_log(tmp_path):
    # Use the temporary directory as the reports folder
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    # Monkey-patch os.path.join so that "reports" resolves to our tmp_path directory
    original_join = os.path.join
    def fake_join(*args):
        args = list(args)
        if args[0] == "reports":
            args[0] = str(reports_dir)
        return original_join(*args)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(os.path, "join", fake_join)
    fc_agent.save_forecasting_log("TestModel", 7)
    log_file = reports_dir / "forecasting_log.txt"
    assert log_file.exists()
    content = log_file.read_text()
    assert "TestModel" in content
    monkeypatch.undo()

# --- Test train_prophet ---
@pytest.mark.skipif(not hasattr(fc_agent, "Prophet"), reason="Prophet not installed")
def test_train_prophet(tmp_path):
    # Create a simple time series DataFrame
    dates = pd.date_range("2020-01-01", periods=60)
    df = pd.DataFrame({
        "date": dates,
        "target": np.sin(np.linspace(0, 3.14, 60)) + np.random.rand(60)*0.1
    })
    result = fc_agent.train_prophet(df, target_col="target", forecast_period=5, resample_freq=None)
    assert "results" in result
    assert "plot" in result
    assert isinstance(result["results"], pd.DataFrame)
    assert isinstance(result["plot"], str)

# --- Test train_arima ---
def test_train_arima():
    # Create a simple time series DataFrame
    dates = pd.date_range("2020-01-01", periods=60)
    df = pd.DataFrame({
        "date": dates,
        "target": np.random.rand(60)
    })
    result = fc_agent.train_arima(df, target_col="target", forecast_period=5, order=(1,1,0))
    assert "results" in result
    assert "plot" in result
    assert isinstance(result["results"], pd.DataFrame)
    assert isinstance(result["plot"], str)
