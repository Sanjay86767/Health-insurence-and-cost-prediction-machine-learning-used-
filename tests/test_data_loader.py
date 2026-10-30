"""
Unit tests for multi-schema dataset loading and data quality reports.
"""

import pandas as pd
import pytest

from src.data_loader import (
    DataLoaderError,
    SCHEMA_986,
    SCHEMA_1338,
    detect_dataset_type,
    get_dataset_quality_report,
    load_dataset,
    split_features_target,
)


def test_load_dataset_986():
    """Verify loading primary research paper dataset (986 rows)."""
    df, d_type = load_dataset("medical_premium_986")
    assert d_type == "medical_premium_986"
    assert df.shape[0] == 986
    assert df.shape[1] == 11
    assert "PremiumPrice" in df.columns


def test_load_dataset_1338():
    """Verify loading extended actuarial dataset (1,338 rows)."""
    df, d_type = load_dataset("insurance_1338")
    assert d_type == "insurance_1338"
    assert df.shape[0] == 1338
    assert "charges" in df.columns


def test_missing_file_raises_error():
    """Ensure missing paths raise informative DataLoaderError."""
    with pytest.raises(DataLoaderError):
        load_dataset("non_existent_path.csv")


def test_quality_report():
    """Verify quality report calculation for 986 dataset."""
    df, d_type = load_dataset("medical_premium_986")
    report = get_dataset_quality_report(df, d_type)
    assert report["total_rows"] == 986
    assert report["has_missing"] is False
    assert report["target_stats"]["mean"] > 0


def test_split_features_target_no_leakage():
    """Ensure target variable is isolated from feature matrix."""
    df, d_type = load_dataset("medical_premium_986")
    X, y = split_features_target(df, d_type)
    assert "PremiumPrice" not in X.columns
    assert y.name == "PremiumPrice"
    assert len(X) == len(y)
