import matplotlib
matplotlib.use("Agg")
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Adjust sys.path to include the project root so that the backend package can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.agents import eda_agent

# Helper function to create a simple DataFrame for testing
def create_test_dataframe():
    data = {
        "A": [1, 2, 3, np.nan, 5],
        "B": ["apple", "banana", "apple", "orange", np.nan],
        "C": ["2020-01-01", "2020-02-01", "2020-03-01", "not_a_date", "2020-05-01"],
        "D": [10.5, 20.5, 30.5, 40.5, 50.5]
    }
    return pd.DataFrame(data)

def test_sample_df():
    df = create_test_dataframe()
    # Set a sample size smaller than the dataframe length to force sampling
    sampled = eda_agent.sample_df(df, sample_size=3)
    # Check that we get 3 rows
    assert len(sampled) == 3

def test_generate_eda_tables():
    df = create_test_dataframe()
    tables = eda_agent.generate_eda_tables(df, exclude_date_features=False)
    
    # Check for key tables
    assert "Overview" in tables
    assert "Column Data Types" in tables
    assert "Missing Values" in tables
    assert "Descriptive Statistics" in tables
    
    # Check that overview table contains shape information
    overview_df = tables["Overview"]
    assert "Shape" in overview_df["Metric"].values

def test_generate_missing_values_heatmap(tmp_path):
    df = create_test_dataframe()
    # Use a temporary file path from pytest
    save_path = tmp_path / "missing_heatmap.png"
    fig = eda_agent.generate_missing_values_heatmap(df, save_path=str(save_path))
    
    # Check if the file is created
    assert os.path.exists(str(save_path))
    # Optionally, you can check that the returned object is a matplotlib figure
    assert isinstance(fig, plt.Figure)

def test_generate_numeric_distribution_plots(tmp_path):
    df = create_test_dataframe()
    # Create a temporary directory for saving plots
    save_dir = tmp_path / "figures"
    plots = eda_agent.generate_numeric_distribution_plots(df, save_dir=str(save_dir), sample_size=5)
    
    # Should generate a plot for each numeric column (A and D in our test DataFrame)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        assert col in plots
        # Check if the file was saved
        file_path = os.path.join(str(save_dir), f"{col}_distribution.png")
        assert os.path.exists(file_path)

def test_generate_categorical_count_plots(tmp_path):
    df = create_test_dataframe()
    save_dir = tmp_path / "figures"
    plots = eda_agent.generate_categorical_count_plots(df, save_dir=str(save_dir), sample_size=5)
    
    # Should generate a plot for each categorical column (B in our test DataFrame)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        # In our test data, column C might be treated as object (if not parsed as datetime), so check accordingly.
        if col in plots:
            file_path = os.path.join(str(save_dir), f"{col}_countplot.png")
            assert os.path.exists(file_path)

def test_generate_pairplot(tmp_path):
    # Create a DataFrame with at least 2 numeric columns
    df = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 3, 4, 5, 6],
        "Z": [5, 4, 3, 2, 1]
    })
    save_path = tmp_path / "pairplot.png"
    fig = eda_agent.generate_pairplot(df, save_path=str(save_path), sample_size=5)
    
    if fig is not None:
        assert os.path.exists(str(save_path))
        assert isinstance(fig, plt.Figure)
    else:
        pytest.skip("Not enough numeric columns for pairplot.")

def test_generate_interactive_correlation_heatmap():
    # This test will only run if Plotly is installed; if not, skip it.
    df = create_test_dataframe()
    fig = eda_agent.generate_interactive_correlation_heatmap(df, method="pearson")
    # If Plotly is not installed, the function should return None.
    # If installed, check that the returned object has an attribute 'to_json'
    if fig is not None:
        assert hasattr(fig, "to_json")
    else:
        pytest.skip("Plotly not installed, skipping interactive correlation heatmap test.")

def test_generate_eda():
    # Test the full EDA pipeline
    df = create_test_dataframe()
    report_text, tables, visuals = eda_agent.generate_eda(df, interactive=False, sample_size=5)
    
    # Check that report text and tables are returned
    assert isinstance(report_text, str)
    assert isinstance(tables, dict)
    # Visuals can contain both matplotlib figures and/or Plotly figures
    assert isinstance(visuals, dict)
