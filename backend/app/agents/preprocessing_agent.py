import os
import logging
import pandas as pd
import numpy as np
import uuid
from typing import List, Optional, Tuple, Union

from ..utils.file_utils import load_dataset
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def identify_column_types(
    df: pd.DataFrame,
    explicit_datetime_cols: Optional[List[str]] = None,
    date_detection_threshold: float = 0.9
) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify numerical, categorical, and datetime columns in a DataFrame.
    """
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_features = []

    # Mark explicit datetime columns first
    if explicit_datetime_cols:
        for col in explicit_datetime_cols:
            if col in df.columns:
                datetime_features.append(col)

    # Attempt to detect datetime columns from remaining candidates
    remaining_candidates = set(df.columns) - set(numerical_features) - set(datetime_features)
    for col in remaining_candidates:
        if col in df.select_dtypes(include=['datetime']):
            datetime_features.append(col)
        elif col in categorical_features:
            series_sample = df[col].dropna()
            if not series_sample.empty:
                sample_size = min(1000, series_sample.shape[0])
                series_sample = series_sample.sample(sample_size, random_state=42)
                converted = pd.to_datetime(series_sample, errors='coerce')
                conversion_rate = converted.notnull().mean()
                if conversion_rate >= date_detection_threshold:
                    datetime_features.append(col)

    # Remove any detected datetime columns from categorical list
    categorical_features = [c for c in categorical_features if c not in datetime_features]

    logging.info("Identified columns:")
    logging.info(f"  • Numerical: {numerical_features}")
    logging.info(f"  • Categorical: {categorical_features}")
    logging.info(f"  • Datetime: {datetime_features}")

    return numerical_features, categorical_features, datetime_features

def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the IQR method.
    """
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        removed_count = (~mask).sum()
        logging.info(f"Removed {removed_count} outliers from '{col}' (factor={factor}).")
        return df[mask]
    except Exception as e:
        logging.error(f"Error removing outliers in column '{col}': {e}")
        raise

def process_date_columns(
    df: pd.DataFrame,
    datetime_features: List[str],
    drop_original_date_cols: bool = True
) -> pd.DataFrame:
    """
    Convert date columns to datetime and extract year, month, day, and dayofweek.
    Optionally drops the original date columns.
    """
    for col in datetime_features:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        if drop_original_date_cols:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Processed and dropped original datetime column '{col}'.")
        else:
            logging.info(f"Processed datetime column '{col}' (original retained).")
    return df

def run_preprocessing_pipeline(
    file: Union[str, object],
    save_processed: bool = True,
    knn_neighbors: int = 3,
    impute_numeric: str = 'knn',
    remove_outliers: bool = False,
    outlier_factor: float = 1.5,
    drop_original_date_cols: bool = True,
    date_detection_threshold: float = 0.9,
    explicit_datetime_cols: Optional[List[str]] = None,
    high_card_threshold: int = 10,           # NEW: maximum unique values for one-hot encoding
    high_card_option: str = 'frequency'        # NEW: options: 'one-hot', 'frequency', 'drop'
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], str]:
    """
    Advanced preprocessing pipeline:
      1. Load dataset.
      2. Identify column types.
      3. Impute numeric columns.
      4. Fill missing categorical values.
      5. Optionally remove outliers.
      6. Process date columns.
      7. Scale numeric columns.
      8. Encode categorical columns based on cardinality.
      9. Optionally save the processed dataset.
      
    Returns:
      (processed_df, processed_file_path, dataset_id, raw_dataset_id)
    """
    try:
        logging.info("Starting advanced preprocessing pipeline...")
        df = load_dataset(file)
        if df.empty:
            raise ValueError("Uploaded file contains no data or failed to load.")

        # Save the raw dataset
        raw_dataset_id = str(uuid.uuid4())
        raw_dir = "original_data"
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{raw_dataset_id}.csv")
        df.to_csv(raw_path, index=False)
        logging.info(f"Raw dataset saved at: {raw_path}")

        # Identify column types
        numerical_features, categorical_features, datetime_features = identify_column_types(
            df,
            explicit_datetime_cols=explicit_datetime_cols,
            date_detection_threshold=date_detection_threshold
        )

        # Impute numeric columns
        if numerical_features:
            if impute_numeric.lower() == 'knn':
                imputer = KNNImputer(n_neighbors=knn_neighbors)
            else:
                imputer = SimpleImputer(strategy=impute_numeric)
            df[numerical_features] = imputer.fit_transform(df[numerical_features])
            logging.info(f"Numeric columns imputed using '{impute_numeric}' strategy.")

        # Fill missing values for categorical columns
        if categorical_features:
            df[categorical_features] = df[categorical_features].fillna('Unknown')
            logging.info("Filled missing categorical values with 'Unknown'.")

        # Remove outliers column-wise
        if remove_outliers and numerical_features:
            for col in numerical_features:
                df = remove_outliers_iqr(df, col, factor=outlier_factor)

        # Process datetime columns
        if datetime_features:
            df = process_date_columns(df, datetime_features, drop_original_date_cols)

        # Scale numeric columns
        if numerical_features:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            logging.info("Scaled numeric columns using StandardScaler.")

        # Encode categorical columns with high cardinality handling
        if categorical_features:
            # Split into low and high cardinality columns
            let_low = [col for col in categorical_features if df[col].nunique() <= high_card_threshold]
            let_high = [col for col in categorical_features if df[col].nunique() > high_card_threshold]

            # One-hot encode low-cardinality columns
            if let_low:
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[let_low].astype(str))
                encoded_cols = encoder.get_feature_names_out(let_low)
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                df.drop(columns=let_low, inplace=True)
                df = pd.concat([df, encoded_df], axis=1)
                logging.info("One-hot encoded low-cardinality categorical columns.")

            # Process high-cardinality columns based on user option
            if let_high:
                if high_card_option == 'frequency':
                    for col in let_high:
                        freq = df[col].value_counts(normalize=True)
                        df[col + '_freq'] = df[col].map(freq)
                        df.drop(columns=[col], inplace=True)
                    logging.info("Frequency encoded high-cardinality categorical columns.")
                elif high_card_option == 'drop':
                    df.drop(columns=let_high, inplace=True)
                    logging.info("Dropped high-cardinality categorical columns.")
                elif high_card_option == 'one-hot':
                    # Warning: one-hot encoding high-card columns may create many features!
                    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[let_high].astype(str))
                    encoded_cols = encoder.get_feature_names_out(let_high)
                    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                    df.drop(columns=let_high, inplace=True)
                    df = pd.concat([df, encoded_df], axis=1)
                    logging.info("One-hot encoded high-cardinality categorical columns.")
                else:
                    logging.warning(f"Unknown high_card_option '{high_card_option}'. Skipping high-card encoding.")

        # Optionally save the processed dataset
        final_path = None
        dataset_id = None
        if save_processed:
            dataset_id = str(uuid.uuid4())
            processed_dir = "processed_data"
            os.makedirs(processed_dir, exist_ok=True)
            final_path = os.path.join(processed_dir, f"{dataset_id}.csv")
            df.to_csv(final_path, index=False)
            logging.info(f"Processed dataset saved at: {final_path}")

        logging.info("Preprocessing pipeline complete!")
        return df, final_path, dataset_id, raw_dataset_id

    except Exception as e:
        logging.error(f"Preprocessing pipeline failed: {e}")
        raise
