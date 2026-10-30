"""
Unit tests for data preprocessing and leakage prevention across both datasets.
"""

import numpy as np
import pytest

from src.data_loader import load_dataset, split_features_target
from src.preprocessing import build_preprocessor, prepare_train_test_data


def test_prepare_train_test_data_986():
    """Verify split and scaling for 986 clinical dataset."""
    df, d_type = load_dataset("medical_premium_986")
    X, y = split_features_target(df, d_type)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
        X, y, dataset_type="medical_premium_986", test_size=0.2, random_state=42
    )

    assert X_train.shape[0] == 788
    assert X_test.shape[0] == 198
    assert X_train.shape[1] == 10
    assert len(feature_names) == 10
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_prepare_train_test_data_1338():
    """Verify split and OneHotEncoding for 1,338 actuarial dataset."""
    df, d_type = load_dataset("insurance_1338")
    X, y = split_features_target(df, d_type)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
        X, y, dataset_type="insurance_1338", test_size=0.2, random_state=42
    )

    assert X_train.shape[0] == 1070
    assert X_test.shape[0] == 268
    assert X_train.shape[1] == len(feature_names)
    assert "smoker_yes" in feature_names or "smoker_no" in feature_names or any("smoker" in f for f in feature_names)
