"""
Unit tests for evaluation metrics (R2, MAE, RMSE, safe MAPE).
"""

import numpy as np
import pytest

from src.evaluate_models import calculate_safe_mape, evaluate_single_model
from sklearn.linear_model import LinearRegression


def test_calculate_safe_mape_zeros():
    """Verify safe MAPE avoids division-by-zero crashes."""
    y_true = np.array([0.0, 100.0, 50.0])
    y_pred = np.array([5.0, 105.0, 48.0])
    mape = calculate_safe_mape(y_true, y_pred)
    assert np.isfinite(mape)
    assert mape >= 0.0


def test_evaluate_single_model():
    """Verify evaluation metric calculations against known dummy array."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = LinearRegression().fit(X, y)

    metrics = evaluate_single_model(model, X, y)
    assert np.isclose(metrics["R2"], 1.0)
    assert np.isclose(metrics["MAE"], 0.0)
    assert np.isclose(metrics["RMSE"], 0.0)
