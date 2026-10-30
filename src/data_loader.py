"""
Data loader module for Medical Health Insurance Cost Prediction.
Supports both:
1. Primary Research Paper Dataset (986 records, Orji & Ukwandu 2024 - Medicalpremium.csv)
2. Extended Actuarial Benchmark Dataset (1,338 records - insurance.csv)
Provides full backward compatibility for all pipeline callers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Schema definitions
SCHEMA_986 = {
    "expected_columns": [
        "Age", "Diabetes", "BloodPressureProblems", "AnyTransplants",
        "AnyChronicDiseases", "Height", "Weight", "KnownAllergies",
        "HistoryOfCancerInFamily", "NumberOfMajorSurgeries", "PremiumPrice"
    ],
    "target_column": "PremiumPrice",
    "continuous_columns": ["Age", "Height", "Weight", "NumberOfMajorSurgeries"],
    "binary_columns": [
        "Diabetes", "BloodPressureProblems", "AnyTransplants",
        "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"
    ],
}

SCHEMA_1338 = {
    "expected_columns": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
    "target_column": "charges",
    "numerical_columns": ["age", "bmi", "children"],
    "categorical_columns": ["sex", "smoker", "region"],
}

# Backward compatibility exports for app.py and legacy callers
EXPECTED_COLUMNS = SCHEMA_1338["expected_columns"]
NUMERICAL_COLUMNS = SCHEMA_1338["numerical_columns"]
CATEGORICAL_COLUMNS = SCHEMA_1338["categorical_columns"]
TARGET_COLUMN = SCHEMA_1338["target_column"]


class DataLoaderError(Exception):
    """Custom exception raised for dataset loading and schema validation errors."""
    pass


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_default_data_path() -> Path:
    """Return default path to insurance.csv for backward compatibility."""
    return get_project_root() / "data" / "insurance.csv"


def get_default_dataset_paths() -> Dict[str, Path]:
    """Return verified default paths to both datasets."""
    root = get_project_root()
    return {
        "medical_premium_986": root / "data" / "raw" / "Medicalpremium.csv",
        "insurance_1338": root / "data" / "insurance.csv",
    }


def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Detect whether a DataFrame conforms to the 986-record clinical schema
    or the 1,338-record actuarial schema.
    """
    cols_clean = [str(c).strip().lower() for c in df.columns]
    if "premiumprice" in cols_clean:
        return "medical_premium_986"
    elif "charges" in cols_clean:
        return "insurance_1338"
    else:
        # Fallback check
        if any(c in cols_clean for c in ["anytransplants", "diabetes", "height"]):
            return "medical_premium_986"
        return "insurance_1338"


def load_dataset(
    dataset_name_or_path: Optional[Union[str, Path]] = None,
    dataset_type: Optional[str] = None,
    return_type: Optional[bool] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
    """
    Load and validate dataset.
    If called with no arguments (e.g. load_dataset()), returns DataFrame directly (backward compatible).
    If called with dataset name or return_type=True, returns (df, dataset_type).
    """
    default_paths = get_default_dataset_paths()

    if dataset_name_or_path is None:
        target_path = default_paths["insurance_1338"]
        if not target_path.exists():
            target_path = default_paths["medical_premium_986"]
    elif str(dataset_name_or_path) in default_paths:
        target_path = default_paths[str(dataset_name_or_path)]
    else:
        target_path = Path(dataset_name_or_path)

    if not target_path.exists():
        raise DataLoaderError(
            f"Dataset not found at '{target_path}'. Please ensure data files exist in data/ or data/raw/."
        )

    try:
        df = pd.read_csv(target_path)
    except Exception as exc:
        raise DataLoaderError(f"Failed to read CSV at '{target_path}': {exc}") from exc

    if df.empty:
        raise DataLoaderError(f"Dataset at '{target_path}' is completely empty.")

    detected_type = dataset_type or detect_dataset_type(df)

    # Validate against detected schema
    if detected_type == "medical_premium_986":
        expected = SCHEMA_986["expected_columns"]
        df_cols_map = {c.strip().lower(): c for c in df.columns}
        for exp in expected:
            if exp.lower() not in df_cols_map:
                raise DataLoaderError(f"Missing column '{exp}' in 986-record dataset.")
    else:
        expected = SCHEMA_1338["expected_columns"]
        df.columns = [str(c).strip().lower() for c in df.columns]
        for exp in expected:
            if exp not in df.columns:
                raise DataLoaderError(f"Missing column '{exp}' in 1338-record dataset.")

    # Return format logic
    if return_type is True:
        return df, detected_type
    elif return_type is False:
        return df

    # If called with no arguments, return DataFrame directly (for app.py backward compatibility)
    if dataset_name_or_path is None:
        return df
    return df, detected_type


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute structured summary diagnostics for the dataset.
    Fully compatible with legacy app.py summary expectations.
    """
    d_type = detect_dataset_type(df)
    target_col = SCHEMA_986["target_column"] if d_type == "medical_premium_986" else SCHEMA_1338["target_column"]

    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_values": missing_counts,
        "has_missing": any(v > 0 for v in missing_counts.values()),
        "duplicates_count": duplicates_count,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numerical_columns": SCHEMA_1338["numerical_columns"] if d_type == "insurance_1338" else SCHEMA_986["continuous_columns"],
        "categorical_columns": SCHEMA_1338["categorical_columns"] if d_type == "insurance_1338" else SCHEMA_986["binary_columns"],
        "target_column": target_col,
    }


def get_dataset_quality_report(df: pd.DataFrame, dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute quality diagnostics including statistics on target variable.
    """
    d_type = dataset_type or detect_dataset_type(df)
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())

    target_col = SCHEMA_986["target_column"] if d_type == "medical_premium_986" else SCHEMA_1338["target_column"]
    target_series = df[target_col]

    target_stats = {
        "mean": float(target_series.mean()),
        "std": float(target_series.std()),
        "min": float(target_series.min()),
        "median": float(target_series.median()),
        "max": float(target_series.max()),
        "has_negative": bool((target_series < 0).any()),
        "has_zero": bool((target_series == 0).any()),
    }

    return {
        "dataset_type": d_type,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_counts": missing,
        "has_missing": any(v > 0 for v in missing.values()),
        "duplicates_count": duplicates,
        "target_column": target_col,
        "target_stats": target_stats,
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def split_features_target(df: pd.DataFrame, dataset_type: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Safely split features and target, preventing target leakage.
    Accepts 1 or 2 arguments with automatic dataset type detection.
    """
    d_type = dataset_type or detect_dataset_type(df)
    target_col = SCHEMA_986["target_column"] if d_type == "medical_premium_986" else SCHEMA_1338["target_column"]
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
