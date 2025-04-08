import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sample_df(df: pd.DataFrame, sample_size: Optional[int] = 100000) -> pd.DataFrame:
    """
    Returns a sample of the DataFrame if its size exceeds sample_size.
    """
    if sample_size is not None and len(df) > sample_size:
        logging.info(f"Dataset has {len(df)} rows; sampling {sample_size} rows for EDA.")
        return df.sample(n=sample_size, random_state=42)
    return df

def generate_eda_tables(
    df: pd.DataFrame,
    exclude_date_features: bool = True,
    correlation_method: str = "pearson",
    sample_size: Optional[int] = 100000
) -> Dict[str, pd.DataFrame]:
    """
    Generates summary tables for EDA: overview, data types, missing values, and descriptive stats.
    """
    # Overview table
    overview_df = pd.DataFrame({"Metric": ["Shape"], "Value": [str(df.shape)]})

    # Data types table
    dtypes_df = (
        pd.DataFrame(df.dtypes, columns=["Data Type"])
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    dtypes_df["Data Type"] = dtypes_df["Data Type"].astype(str)
    if exclude_date_features:
        mask = dtypes_df["Column"].str.lower().str.startswith("date_")
        dtypes_df = dtypes_df[~mask]

    # Missing values table
    missing_series = df.isnull().sum()
    missing_df = (
        missing_series[missing_series > 0]
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Values"})
    )
    if missing_df.empty:
        missing_df = pd.DataFrame({"Column": ["No missing values found."], "Missing Values": [""]})

    # Descriptive statistics table
    df_sample = sample_df(df, sample_size)
    if exclude_date_features:
        cols = [c for c in df_sample.columns if not c.lower().startswith("date_")]
        desc_df = df_sample[cols].describe(include="all").T.reset_index().rename(columns={"index": "Feature"})
    else:
        desc_df = df_sample.describe(include="all").T.reset_index().rename(columns={"index": "Feature"})

    # Adding skewness and kurtosis for numeric columns
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        skewness = df_sample[numeric_cols].skew()
        kurtosis = df_sample[numeric_cols].kurtosis()
        desc_df['Skewness'] = desc_df['Feature'].apply(lambda x: round(skewness.get(x, np.nan), 4))
        desc_df['Kurtosis'] = desc_df['Feature'].apply(lambda x: round(kurtosis.get(x, np.nan), 4))

    tables = {
        "Overview": overview_df,
        "Column Data Types": dtypes_df,
        "Missing Values": missing_df,
        "Descriptive Statistics": desc_df
    }
    return tables

def generate_missing_values_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates a heatmap of missing values and saves it if a save path is provided.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    fig = plt.gcf()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logging.info(f"Missing values heatmap saved at {save_path}")
    return fig

def generate_numeric_distribution_plots(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    exclude_date_features: bool = True,
    sample_size: Optional[int] = 100000
) -> Dict[str, plt.Figure]:
    """
    Generates histograms with KDE for numeric columns.
    """
    plots = {}
    df_sample = sample_df(df, sample_size)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_date_features:
        numeric_cols = [c for c in numeric_cols if not c.lower().startswith("date_")]

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df_sample[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        fig = plt.gcf()
        plots[col] = fig
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{col}_distribution.png"), bbox_inches="tight")
        plt.close(fig)
    return plots

def generate_categorical_count_plots(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    exclude_date_features: bool = True,
    sample_size: Optional[int] = 100000
) -> Dict[str, plt.Figure]:
    """
    Generates count plots for categorical columns.
    """
    plots = {}
    df_sample = sample_df(df, sample_size)
    cat_cols = df_sample.select_dtypes(include=["object", "category"]).columns.tolist()
    if exclude_date_features:
        cat_cols = [c for c in cat_cols if not c.lower().startswith("date_")]

    for col in cat_cols:
        plt.figure()
        sns.countplot(data=df_sample, x=col, order=df_sample[col].value_counts().index)
        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        fig = plt.gcf()
        plots[col] = fig
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{col}_countplot.png"), bbox_inches="tight")
        plt.close(fig)
    return plots

def generate_pairplot(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    exclude_date_features: bool = True,
    max_numeric_cols: int = 8,
    sample_size: Optional[int] = 100000
) -> Optional[plt.Figure]:
    """
    Generates a pairplot for a subset of numeric columns.
    """
    df_sample = sample_df(df, sample_size)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_date_features:
        numeric_cols = [c for c in numeric_cols if not c.lower().startswith("date_")]
    if len(numeric_cols) < 2:
        logging.info("Not enough numeric columns for pairplot.")
        return None
    if len(numeric_cols) > max_numeric_cols:
        numeric_cols = numeric_cols[:max_numeric_cols]
        logging.info(f"Using first {max_numeric_cols} numeric columns for pairplot.")
    pairplot_fig = sns.pairplot(df_sample[numeric_cols].dropna())
    fig = pairplot_fig.fig
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig

def generate_interactive_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson"
) -> Optional[Any]:
    """
    Generates an interactive correlation heatmap using Plotly.
    """
    try:
        import plotly.express as px
    except ImportError:
        logging.warning("Plotly not installed, skipping interactive correlation heatmap.")
        return None
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        logging.info("Not enough numeric columns for correlation heatmap.")
        return None
    corr = numeric_df.corr(method=method)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        origin='lower',
        title=f"Correlation Heatmap ({method.title()})"
    )
    return fig

def generate_interactive_scatter_matrix(
    df: pd.DataFrame,
    sample_size: Optional[int] = 100000
) -> Optional[Any]:
    """
    Generates an interactive scatter matrix using Plotly.
    """
    try:
        import plotly.express as px
    except ImportError:
        logging.warning("Plotly not installed, skipping scatter matrix.")
        return None
    df_sample = sample_df(df, sample_size)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        logging.info("Not enough numeric columns for scatter matrix.")
        return None
    fig = px.scatter_matrix(
        df_sample,
        dimensions=numeric_cols,
        title="Interactive Scatter Matrix"
    )
    return fig

def save_eda_report_tables(
    df: pd.DataFrame,
    dataset_id: Optional[str] = None,
    report_path: Optional[str] = None,
    exclude_date_features: bool = True,
    correlation_method: str = "pearson",
    sample_size: Optional[int] = 100000
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Generates EDA tables and a textual summary report, then saves the report.
    """
    if not report_path and dataset_id:
        report_path = f"reports/eda_report_{dataset_id}.txt"
    elif not report_path:
        report_path = "reports/eda_report.txt"

    tables = generate_eda_tables(df, exclude_date_features, correlation_method, sample_size)
    report_lines = [
        "### Exploratory Data Analysis Summary\n",
        "Overview of dataset size, structure, and key characteristics.",
        "Interactive charts are available in the UI for a deeper dive.\n",
        "Key Points:",
        "- Identified numeric and categorical features",
        "- Summarized missing value patterns and distributions",
        "- Computed skewness and kurtosis for numeric features\n"
    ]
    report_text = "\n".join(report_lines)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"EDA report saved at {report_path}")
    return report_text, tables

def generate_visualizations(
    df: pd.DataFrame,
    save_dir: str = "reports/figures",
    exclude_date_features: bool = True,
    sample_size: Optional[int] = 100000,
    max_numeric_cols: int = 8
) -> Dict[str, Any]:
    """
    Generates a collection of static and interactive visualizations.
    """
    visuals = {}
    os.makedirs(save_dir, exist_ok=True)
    visuals["missing_values_heatmap"] = generate_missing_values_heatmap(
        df, save_path=os.path.join(save_dir, "missing_values_heatmap.png")
    )
    numeric_dists = generate_numeric_distribution_plots(
        df, save_dir=save_dir, exclude_date_features=exclude_date_features, sample_size=sample_size
    )
    for col, fig in numeric_dists.items():
        visuals[f"numeric_dist__{col}"] = fig
    cat_counts = generate_categorical_count_plots(
        df, save_dir=save_dir, exclude_date_features=exclude_date_features, sample_size=sample_size
    )
    for col, fig in cat_counts.items():
        visuals[f"categorical_count__{col}"] = fig
    pairplot_fig = generate_pairplot(
        df,
        save_path=os.path.join(save_dir, "pairplot.png"),
        exclude_date_features=exclude_date_features,
        max_numeric_cols=max_numeric_cols,
        sample_size=sample_size
    )
    if pairplot_fig is not None:
        visuals["pairplot"] = pairplot_fig
    scatter_fig = generate_interactive_scatter_matrix(df, sample_size)
    if scatter_fig is not None:
        visuals["interactive_scatter_matrix"] = scatter_fig
    return visuals

def generate_eda(
    df: pd.DataFrame,
    dataset_id: Optional[str] = None,
    exclude_date_features: bool = True,
    correlation_method: str = "pearson",
    interactive: bool = True,
    sample_size: Optional[int] = 100000,
    max_numeric_cols: int = 8
) -> Tuple[str, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Runs the full EDA pipeline: creates a report, summary tables, and visualizations.
    """
    logging.info("Starting EDA pipeline...")
    report_text, tables = save_eda_report_tables(
        df,
        dataset_id=dataset_id,
        exclude_date_features=exclude_date_features,
        correlation_method=correlation_method,
        sample_size=sample_size
    )
    interactive_figs = {}
    if interactive:
        corr_heatmap = generate_interactive_correlation_heatmap(df, correlation_method)
        if corr_heatmap is not None:
            interactive_figs["correlation_heatmap"] = corr_heatmap
    static_visuals = generate_visualizations(
        df,
        save_dir="reports/figures",
        exclude_date_features=exclude_date_features,
        sample_size=sample_size,
        max_numeric_cols=max_numeric_cols
    )
    all_figures = {**interactive_figs, **static_visuals}
    logging.info("EDA pipeline complete.")
    return report_text, tables, all_figures
