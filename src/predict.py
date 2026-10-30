"""
Inference and input validation module for Medical Health Insurance Cost Prediction.
Supports both the 986-record clinical dataset and the 1,338-record actuarial dataset.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

VALID_REGIONS = ["southwest", "southeast", "northwest", "northeast"]
VALID_SEX = ["male", "female"]
VALID_SMOKER = ["yes", "no"]


class PredictionInputError(Exception):
    """Exception raised when input features violate biological or schema constraints."""
    pass


def validate_inputs_986(inputs: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """Validate 10 physiological and clinical inputs for the 986 dataset."""
    age = int(inputs.get("Age", 0))
    if not (18 <= age <= 100):
        raise PredictionInputError(f"Age must be between 18 and 100. Received: {age}")

    height = float(inputs.get("Height", 0))
    if not (120.0 <= height <= 220.0):
        raise PredictionInputError(f"Height must be between 120 and 220 cm. Received: {height}")

    weight = float(inputs.get("Weight", 0))
    if not (30.0 <= weight <= 200.0):
        raise PredictionInputError(f"Weight must be between 30 and 200 kg. Received: {weight}")

    surgeries = int(inputs.get("NumberOfMajorSurgeries", 0))
    if not (0 <= surgeries <= 6):
        raise PredictionInputError(f"Major surgeries count must be between 0 and 6. Received: {surgeries}")

    binary_fields = [
        "Diabetes", "BloodPressureProblems", "AnyTransplants",
        "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"
    ]
    validated = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "NumberOfMajorSurgeries": surgeries,
    }

    for b in binary_fields:
        val = int(inputs.get(b, 0))
        if val not in [0, 1]:
            raise PredictionInputError(f"Field '{b}' must be binary 0 or 1. Received: {val}")
        validated[b] = val

    return validated


def validate_inputs_1338(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate inputs for the 1,338 actuarial dataset."""
    age = int(inputs.get("age", 0))
    if not (18 <= age <= 100):
        raise PredictionInputError(f"Age must be between 18 and 100. Received: {age}")

    bmi = float(inputs.get("bmi", 0.0))
    if not (12.0 <= bmi <= 65.0):
        raise PredictionInputError(f"BMI must be between 12.0 and 65.0. Received: {bmi}")

    children = int(inputs.get("children", 0))
    if not (0 <= children <= 10):
        raise PredictionInputError(f"Children must be between 0 and 10. Received: {children}")

    sex = str(inputs.get("sex", "")).strip().lower()
    if sex not in VALID_SEX:
        raise PredictionInputError(f"Sex must be 'male' or 'female'. Received: '{sex}'")

    smoker = str(inputs.get("smoker", "")).strip().lower()
    if smoker not in VALID_SMOKER:
        raise PredictionInputError(f"Smoker must be 'yes' or 'no'. Received: '{smoker}'")

    region = str(inputs.get("region", "")).strip().lower()
    if region not in VALID_REGIONS:
        raise PredictionInputError(f"Region must be one of {VALID_REGIONS}. Received: '{region}'")

    return {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }


def validate_applicant_inputs(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> Dict[str, Union[int, float, str]]:
    """Backward compatibility wrapper for 1,338 dataset inputs."""
    return validate_inputs_1338({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    })


def predict_cost(
    model: Any,
    preprocessor: ColumnTransformer,
    applicant_data: Dict[str, Any],
    dataset_type: str = "medical_premium_986",
) -> Tuple[float, np.ndarray, pd.DataFrame]:
    """
    Predict medical insurance cost or premium.
    Returns (predicted_cost, preprocessed_features, single_row_dataframe).
    """
    if dataset_type == "medical_premium_986":
        validated = validate_inputs_986(applicant_data)
        ordered_keys = [
            "Age", "Diabetes", "BloodPressureProblems", "AnyTransplants",
            "AnyChronicDiseases", "Height", "Weight", "KnownAllergies",
            "HistoryOfCancerInFamily", "NumberOfMajorSurgeries"
        ]
        df_single = pd.DataFrame([{k: validated[k] for k in ordered_keys}])
    else:
        validated = validate_inputs_1338(applicant_data)
        df_single = pd.DataFrame([validated])

    X_single_proc = preprocessor.transform(df_single)
    raw_pred = float(model.predict(X_single_proc)[0])
    prediction = max(raw_pred, 0.0)

    return prediction, np.asarray(X_single_proc), df_single


def predict_charges(
    model: Any,
    preprocessor: ColumnTransformer,
    applicant_data: Dict[str, Any],
) -> Tuple[float, np.ndarray, pd.DataFrame]:
    """Backward compatibility alias for 1,338 dataset prediction."""
    return predict_cost(
        model=model,
        preprocessor=preprocessor,
        applicant_data=applicant_data,
        dataset_type="insurance_1338",
    )
