import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Optional, Tuple, List, Dict

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# Create a module-specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("reports", exist_ok=True)

def detect_datetime_columns(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Detect potential datetime columns in a DataFrame based on dtype or column name.
    Also attempts to reconstruct a date from separate year/month/day columns.
    """
    datetime_cols = [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in col.lower()
        or "time" in col.lower()
    ]
    lower_cols = [c.lower() for c in df.columns]
    if {"date_year", "date_month", "date_day"}.issubset(lower_cols):
        try:
            year_col = [col for col in df.columns if col.lower() == "date_year"][0]
            month_col = [col for col in df.columns if col.lower() == "date_month"][0]
            day_col = [col for col in df.columns if col.lower() == "date_day"][0]
            df["Reconstructed_Date"] = pd.to_datetime(
                df[[year_col, month_col, day_col]].rename(columns={
                    year_col: "year",
                    month_col: "month",
                    day_col: "day"
                }),
                errors="coerce"
            )
            datetime_cols.append("Reconstructed_Date")
            logger.info("Created 'Reconstructed_Date' from year/month/day.")
        except Exception as e:
            logger.exception("Error reconstructing date from year/month/day")
    return datetime_cols, df

def plot_forecast(forecast_results: pd.DataFrame, model_name: str, historical: Optional[pd.DataFrame] = None) -> str:
    """
    Plot forecast results (including yhat and its lower/upper bounds) and optionally historical data.
    Returns a base64 encoded PNG image string.
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_results["ds"], forecast_results["yhat"], label="Predicted", color="blue")
        if "yhat_lower" in forecast_results.columns and "yhat_upper" in forecast_results.columns:
            plt.fill_between(
                forecast_results["ds"],
                forecast_results["yhat_lower"],
                forecast_results["yhat_upper"],
                color="blue", alpha=0.2
            )
        if historical is not None and {"ds", "y"}.issubset(historical.columns):
            plt.plot(historical["ds"], historical["y"], label="Historical", color="black", linestyle="--")
        plt.title(f"{model_name} Forecast")
        plt.xlabel("Date")
        plt.ylabel("Forecasted Value")
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        logger.exception("Error plotting forecast")
        raise

def naive_forecast(df: pd.DataFrame, forecast_period: int) -> pd.DataFrame:
    """
    Returns a naive forecast by repeating the last observed value.
    Expects df to have columns ['ds', 'y'].
    """
    last_date = df["ds"].max()
    last_value = df["y"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period, freq="D")
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": [last_value] * forecast_period,
        "yhat_lower": [last_value] * forecast_period,
        "yhat_upper": [last_value] * forecast_period,
    })
    logger.info("Using naive forecast (repeat last value).")
    return forecast_df

def save_forecasting_log(model_name: str, forecast_period: int) -> None:
    """
    Append a forecasting log entry to 'reports/forecasting_log.txt'.
    """
    log_path = os.path.join("reports", "forecasting_log.txt")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\nModel: {model_name}\nForecast Horizon: {forecast_period} days\n")
    logger.info(f"Forecasting log updated at: {log_path}")

def train_prophet(
    df: pd.DataFrame,
    target_col: str,
    forecast_period: int,
    max_history: int = 10000,
    resample_freq: Optional[str] = None
) -> Dict[str, any]:
    """
    Train a Prophet model on the data.
    Expects df to have at least one datetime column.
    Returns a dictionary with the forecast DataFrame and a base64-encoded forecast plot.
    """
    datetime_cols, df = detect_datetime_columns(df)
    if not datetime_cols:
        raise ValueError("No valid date column found for Prophet forecasting.")
    date_col = datetime_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[[date_col, target_col]].dropna()
    if df.shape[0] < 10:
        raise ValueError("Not enough data points (<10) for Prophet.")
    df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(df["y"].median())
    if resample_freq:
        df = df.set_index("ds").resample(resample_freq).mean().reset_index()
    if df.shape[0] > max_history:
        df = df.tail(max_history)
    try:
        if df.shape[0] < 50:
            model = Prophet(
                growth='flat',
                changepoint_prior_scale=0.01,
                n_changepoints=0,
                changepoint_range=0.0,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )
        else:
            model = Prophet(changepoint_prior_scale=0.01)
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}. Using naive fallback.")
        forecast = naive_forecast(df, forecast_period)
    for col in ["yhat_lower", "yhat_upper"]:
        if col not in forecast.columns:
            forecast[col] = forecast["yhat"]
    forecast_results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    save_forecasting_log("Prophet", forecast_period)
    forecast_plot_base64 = plot_forecast(forecast_results, "Prophet", historical=df)
    return {"results": forecast_results, "plot": forecast_plot_base64}

def train_arima(
    df: pd.DataFrame,
    target_col: str,
    forecast_period: int,
    order: Tuple[int, int, int] = (5, 1, 0),
    max_history: int = 1000
) -> Dict[str, any]:
    """
    Train an ARIMA model on the data.
    Returns a dictionary with the forecast DataFrame and a base64-encoded forecast plot.
    """
    datetime_cols, df = detect_datetime_columns(df)
    if not datetime_cols:
        raise ValueError("No valid date column found for ARIMA forecasting.")
    date_col = datetime_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[[date_col, target_col]].dropna()
    if df.shape[0] < 10:
        raise ValueError("ARIMA requires at least 10 data points.")
    if df.shape[0] > max_history:
        df = df.tail(max_history)
    df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(df["y"].median())
    model = ARIMA(df["y"], order=order)
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_period)
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period, freq="D")
    forecast_results = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast_values,
    })
    forecast_results["yhat_lower"] = forecast_results["yhat"]
    forecast_results["yhat_upper"] = forecast_results["yhat"]
    save_forecasting_log("ARIMA", forecast_period)
    forecast_plot_base64 = plot_forecast(forecast_results, "ARIMA", historical=df)
    return {"results": forecast_results, "plot": forecast_plot_base64}
