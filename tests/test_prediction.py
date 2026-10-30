"""
Unit tests for input validation and inference on both datasets.
"""

import pytest

from src.data_loader import load_dataset, split_features_target
from src.predict import (
    PredictionInputError,
    predict_cost,
    validate_inputs_986,
    validate_inputs_1338,
)
from src.preprocessing import prepare_train_test_data
from src.train_models import get_core_paper_models


def test_validate_inputs_986_valid():
    """Verify sanitization of valid physiological inputs."""
    raw = {
        "Age": 45,
        "Diabetes": 1,
        "BloodPressureProblems": 0,
        "AnyTransplants": 0,
        "AnyChronicDiseases": 1,
        "Height": 172,
        "Weight": 78,
        "KnownAllergies": 0,
        "HistoryOfCancerInFamily": 0,
        "NumberOfMajorSurgeries": 1,
    }
    cleaned = validate_inputs_986(raw)
    assert cleaned["Age"] == 45
    assert cleaned["Diabetes"] == 1
    assert cleaned["Weight"] == 78.0


def test_validate_inputs_986_invalid():
    """Verify rejection of invalid bounds or non-binary flags."""
    with pytest.raises(PredictionInputError):
        validate_inputs_986({"Age": 12})  # Age < 18

    with pytest.raises(PredictionInputError):
        validate_inputs_986({"Age": 30, "Diabetes": 5})  # Non-binary


def test_end_to_end_predict_cost_986():
    """Test full inference pipeline on 986 dataset."""
    df, d_type = load_dataset("medical_premium_986")
    X, y = split_features_target(df, d_type)
    X_train, X_test, y_train, y_test, preprocessor, _ = prepare_train_test_data(X, y, d_type)

    models = get_core_paper_models()
    model = models["Random Forest"]
    model.fit(X_train, y_train)

    applicant = {
        "Age": 50,
        "Diabetes": 0,
        "BloodPressureProblems": 1,
        "AnyTransplants": 0,
        "AnyChronicDiseases": 0,
        "Height": 168,
        "Weight": 72,
        "KnownAllergies": 0,
        "HistoryOfCancerInFamily": 0,
        "NumberOfMajorSurgeries": 0,
    }

    pred, X_single, df_single = predict_cost(model, preprocessor, applicant, dataset_type=d_type)
    assert isinstance(pred, float)
    assert pred > 0.0
    assert X_single.shape == (1, 10)
