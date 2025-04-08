import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers from a numeric column using the IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
    removed = (~mask).sum()
    logger.info(f"Removed {removed} outliers from '{col}' using factor={factor}.")
    return df[mask]

def create_date_features(df: pd.DataFrame, col: str, drop_original: bool = False) -> pd.DataFrame:
    """Create new date-based features from a date column."""
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    if drop_original:
        df.drop(columns=[col], inplace=True)
        logger.info(f"Dropped original date column '{col}'.")
    else:
        logger.info(f"Date features created for '{col}' (original retained).")
    return df

def _impute_column(df: pd.DataFrame, col: str, strategy: str, knn_neighbors: int = 3) -> pd.DataFrame:
    """
    Impute missing values in a single column using the specified strategy.
    Supported strategies: mean, median, mode, knn, drop.
    """
    series = df[col]
    if series.isnull().sum() == 0:
        return df

    strategy_l = strategy.lower()
    if strategy_l == "mean":
        if pd.api.types.is_numeric_dtype(series):
            df[col] = series.fillna(series.mean())
        else:
            df[col] = series.fillna(series.mode().iloc[0])
        logger.info(f"Imputed '{col}' via mean/auto fallback.")
    elif strategy_l == "median":
        if pd.api.types.is_numeric_dtype(series):
            df[col] = series.fillna(series.median())
        else:
            df[col] = series.fillna(series.mode().iloc[0])
        logger.info(f"Imputed '{col}' via median/auto fallback.")
    elif strategy_l == "mode":
        df[col] = series.fillna(series.mode().iloc[0])
        logger.info(f"Imputed '{col}' via mode.")
    elif strategy_l == "knn":
        if pd.api.types.is_numeric_dtype(series):
            knn = KNNImputer(n_neighbors=knn_neighbors)
            df[[col]] = knn.fit_transform(df[[col]])
            logger.info(f"Imputed '{col}' via KNN (k={knn_neighbors}).")
        else:
            df[col] = series.fillna(series.mode().iloc[0])
            logger.info(f"Imputed '{col}' via fallback mode (KNN not applicable).")
    elif strategy_l == "drop":
        before = len(df)
        df.dropna(subset=[col], inplace=True)
        after = len(df)
        logger.info(f"Dropped {before - after} rows with missing values in '{col}'.")
    else:
        logger.warning(f"Unknown impute strategy '{strategy}' for column '{col}'. Skipping.")
    return df

def convert_column_dtype(df: pd.DataFrame, col: str, conversion: str) -> pd.DataFrame:
    """Convert the data type of a column based on the provided conversion."""
    if conversion == "numeric":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Converted '{col}' to numeric.")
    elif conversion == "category":
        df[col] = df[col].astype("category")
        logger.info(f"Converted '{col}' to category.")
    return df

def _scale_numeric(df: pd.DataFrame, scaler_name: str = "standard") -> pd.DataFrame:
    """Scale numeric columns using StandardScaler or MinMaxScaler."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        logger.info("No numeric columns for scaling.")
        return df

    scaler = StandardScaler() if scaler_name.lower() == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"Numeric columns scaled via {scaler_name} scaler.")
    return df

def _polynomial_features(df: pd.DataFrame, degree: int = 2, interaction_only: bool = False, include_bias: bool = False) -> pd.DataFrame:
    """Add polynomial features to numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        logger.info("No numeric columns found for polynomial features.")
        return df

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_data = poly.fit_transform(df[numeric_cols])
    poly_cols = poly.get_feature_names_out(numeric_cols)
    poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    logger.info(f"Polynomial features added (degree={degree}, interaction_only={interaction_only}).")
    return df

def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to strictly positive numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] > 0).all():
            df[f"log_{col}"] = np.log(df[col])
            logger.info(f"Log transform applied to {col}.")
        else:
            logger.warning(f"Column {col} has non-positive values; skipping log transform.")
    return df

def run_advanced_feature_engineering(
    df: pd.DataFrame,
    impute_plan: Optional[Dict[str, str]] = None,
    knn_neighbors: int = 3,
    cat_impute_value: str = "Unknown",
    outlier_plan: Optional[Dict[str, float]] = None,
    date_plan: Optional[Dict[str, bool]] = None,
    convert_plan: Optional[Dict[str, str]] = None,
    encoding_method: str = "one-hot",
    scaling_method: str = "standard",
    apply_poly: bool = False,
    poly_degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
    apply_log: bool = False,
    high_card_threshold: int = 10,          # NEW parameter
    high_card_option: str = "frequency"       # NEW parameter: 'one-hot', 'frequency', or 'drop'
) -> pd.DataFrame:
    """
    Run the full advanced feature engineering pipeline, including imputation,
    outlier removal, date feature creation, type conversion, encoding, scaling,
    polynomial feature expansion, and log transformation.
    """
    df = df.copy()
    try:
        # 1) Imputation plan
        if impute_plan:
            for col, strategy in impute_plan.items():
                if col in df.columns:
                    df = _impute_column(df, col, strategy, knn_neighbors)
        # 2) Fill leftover categorical nulls
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(cat_impute_value)
                logger.info(f"Filled missing in {col} with '{cat_impute_value}'.")
        # 3) Outlier removal
        if outlier_plan:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col, factor in outlier_plan.items():
                if col in numeric_cols and factor > 0:
                    df = remove_outliers_iqr(df, col, factor=factor)
        # 4) Date feature creation
        if date_plan:
            for date_col, drop_orig in date_plan.items():
                if date_col in df.columns:
                    df = create_date_features(df, date_col, drop_original=drop_orig)
        # 5) Data type conversion
        if convert_plan:
            for col, conv in convert_plan.items():
                if col in df.columns:
                    df = convert_column_dtype(df, col, conv)
        # 6) Categorical encoding with high-cardinality handling:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            let_low = [col for col in cat_cols if df[col].nunique() <= high_card_threshold]
            let_high = [col for col in cat_cols if df[col].nunique() > high_card_threshold]
            if let_low:
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[let_low].astype(str))
                encoded_cols = encoder.get_feature_names_out(let_low)
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                df.drop(columns=let_low, inplace=True)
                df = pd.concat([df, encoded_df], axis=1)
                logger.info("One-hot encoded low-cardinality categorical columns.")
            if let_high:
                if high_card_option == 'frequency':
                    for col in let_high:
                        freq = df[col].value_counts(normalize=True)
                        df[col + '_freq'] = df[col].map(freq)
                        df.drop(columns=[col], inplace=True)
                    logger.info("Frequency encoded high-cardinality categorical columns.")
                elif high_card_option == 'drop':
                    df.drop(columns=let_high, inplace=True)
                    logger.info("Dropped high-cardinality categorical columns.")
                elif high_card_option == 'one-hot':
                    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[let_high].astype(str))
                    encoded_cols = encoder.get_feature_names_out(let_high)
                    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                    df.drop(columns=let_high, inplace=True)
                    df = pd.concat([df, encoded_df], axis=1)
                    logger.info("One-hot encoded high-cardinality categorical columns.")
                else:
                    logger.warning(f"Unknown high_card_option '{high_card_option}'. Skipping high-card encoding.")
        # 7) Numeric scaling
        df = _scale_numeric(df, scaling_method)
        # 8) Polynomial features
        if apply_poly:
            df = _polynomial_features(df, degree=poly_degree, interaction_only=interaction_only, include_bias=include_bias)
        # 9) Log transformation
        if apply_log:
            df = _log_transform(df)
        logger.info("Advanced feature engineering completed.")
        return df
    except Exception as e:
        logger.exception("Error in advanced feature engineering pipeline")
        raise e

def apply_feature_transformations(df: pd.DataFrame, plan: Optional[dict] = None) -> pd.DataFrame:
    """
    A wrapper for run_advanced_feature_engineering for a simpler one-call approach.
    """
    if plan is None:
        plan = {}
    return run_advanced_feature_engineering(
        df,
        impute_plan=plan.get("impute_plan", None),
        knn_neighbors=plan.get("knn_neighbors", 3),
        cat_impute_value=plan.get("cat_impute_value", "Unknown"),
        outlier_plan=plan.get("outlier_plan", None),
        date_plan=plan.get("date_plan", None),
        convert_plan=plan.get("convert_plan", None),
        encoding_method=plan.get("encoding_method", "one-hot"),
        scaling_method=plan.get("scaling_method", "standard"),
        apply_poly=plan.get("apply_poly", False),
        poly_degree=plan.get("poly_degree", 2),
        interaction_only=plan.get("interaction_only", False),
        include_bias=plan.get("include_bias", False),
        apply_log=plan.get("apply_log", False),
        high_card_threshold=plan.get("high_card_threshold", 10),
        high_card_option=plan.get("high_card_option", "frequency")
    )
