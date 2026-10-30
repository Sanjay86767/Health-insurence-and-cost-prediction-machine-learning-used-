"""
Preprocessing module for Medical Health Insurance Cost Prediction.
Enforces strict prevention of data leakage by fitting transformers exclusively on training data.
Supports both the 986-record clinical dataset and the 1,338-record actuarial dataset.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import SCHEMA_986, SCHEMA_1338


def build_preprocessor_986(scale_continuous: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the 986-record clinical dataset (Orji & Ukwandu 2024).
    Continuous features (Age, Height, Weight, Major Surgeries) are standardized.
    Binary disease/transplant flags (0/1) are passed through cleanly.
    """
    cont_cols = SCHEMA_986["continuous_columns"]
    bin_cols = SCHEMA_986["binary_columns"]

    cont_pipe = Pipeline([("scaler", StandardScaler())]) if scale_continuous else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", cont_pipe, cont_cols),
            ("bin", "passthrough", bin_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_preprocessor_1338(scale_numerical: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the 1,338-record actuarial dataset.
    Numerical columns (age, bmi, children) are standardized.
    Categorical columns (sex, smoker, region) are OneHotEncoded with drop='first'.
    """
    num_cols = SCHEMA_1338["numerical_columns"]
    cat_cols = SCHEMA_1338["categorical_columns"]

    num_pipe = Pipeline([("scaler", StandardScaler())]) if scale_numerical else "passthrough"
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_preprocessor(dataset_type: str = "medical_premium_986", scale_features: bool = True) -> ColumnTransformer:
    """
    Factory function to construct the appropriate preprocessor based on dataset type.
    """
    if dataset_type == "medical_premium_986":
        return build_preprocessor_986(scale_continuous=scale_features)
    elif dataset_type == "insurance_1338":
        return build_preprocessor_1338(scale_numerical=scale_features)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def get_feature_names(preprocessor: ColumnTransformer, dataset_type: str = "medical_premium_986") -> List[str]:
    """
    Extract readable feature names after ColumnTransformer transformation.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        if dataset_type == "medical_premium_986":
            return SCHEMA_986["continuous_columns"] + SCHEMA_986["binary_columns"]
        else:
            names = list(SCHEMA_1338["numerical_columns"])
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for cat_col, cats in zip(SCHEMA_1338["categorical_columns"], cat_encoder.categories_):
                for cat in cats[1:]:
                    names.append(f"{cat_col}_{cat}")
            return names


def prepare_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_type: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    scale_numerical: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Split data into training and test sets and apply preprocessing.
    LEAKAGE PREVENTION: The ColumnTransformer is fitted strictly on X_train.
    """
    if scale_numerical is not None:
        scale_features = scale_numerical

    if dataset_type is None:
        cols_lower = [str(c).strip().lower() for c in X.columns]
        if any(c in cols_lower for c in ["height", "diabetes", "bloodpressureproblems"]):
            dataset_type = "medical_premium_986"
        else:
            dataset_type = "insurance_1338"

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(dataset_type=dataset_type, scale_features=scale_features)

    # Fit ONLY on training data
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    # Transform test data using statistics learned exclusively from training
    X_test_proc = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names(preprocessor, dataset_type=dataset_type)

    return (
        np.asarray(X_train_proc),
        np.asarray(X_test_proc),
        np.asarray(y_train_raw),
        np.asarray(y_test_raw),
        preprocessor,
        feature_names,
    )
