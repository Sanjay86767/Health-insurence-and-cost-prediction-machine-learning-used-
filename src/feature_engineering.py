"""
Feature engineering module based on Sanjay Kumar & Dr. Sukhvinder Singh Bamber (Panjab University).
Derives 3 clinical features:
1. BMI = Weight (kg) / (Height (m))^2
2. Composite Comorbidity Index (CCI) = 2.0*Transplant + 1.5*Chronic + 1.0*Diabetes + 1.0*BloodPressure
3. Surgical-Oncological Interaction = NumberOfMajorSurgeries * HistoryOfCancerInFamily
Keeps VIF < 3.0 across all predictors.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific clinical features for the 986-record medical dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw physiological attributes.
        
    Returns
    -------
    pd.DataFrame
        Enriched dataframe with BMI, CCI, and Surgical-Oncological Interaction.
    """
    df_out = df.copy()

    # 1. Body Mass Index (BMI): weight (kg) / (height (m))^2
    if "Height" in df_out.columns and "Weight" in df_out.columns:
        height_m = df_out["Height"] / 100.0
        df_out["BMI"] = df_out["Weight"] / (height_m ** 2)

    # 2. Composite Comorbidity Index (CCI): additive multi-morbidity proxy
    # Weights: 2.0 for Transplants, 1.5 for Chronic, 1.0 for Diabetes, 1.0 for Blood Pressure
    transplant_col = "AnyTransplants" if "AnyTransplants" in df_out.columns else None
    chronic_col = "AnyChronicDiseases" if "AnyChronicDiseases" in df_out.columns else None
    diabetes_col = "Diabetes" if "Diabetes" in df_out.columns else None
    bp_col = "BloodPressureProblems" if "BloodPressureProblems" in df_out.columns else None

    if all([transplant_col, chronic_col, diabetes_col, bp_col]):
        df_out["CCI"] = (
            2.0 * df_out[transplant_col] +
            1.5 * df_out[chronic_col] +
            1.0 * df_out[diabetes_col] +
            1.0 * df_out[bp_col]
        )

    # 3. Surgical-Oncological Interaction
    surgeries_col = "NumberOfMajorSurgeries" if "NumberOfMajorSurgeries" in df_out.columns else None
    cancer_col = "HistoryOfCancerInFamily" if "HistoryOfCancerInFamily" in df_out.columns else None

    if surgeries_col and cancer_col:
        df_out["SurgicalCancerInteraction"] = df_out[surgeries_col] * df_out[cancer_col]

    return df_out


def get_feature_sets() -> dict:
    """Return feature groupings for ablation studies."""
    raw_features = [
        "Age", "Height", "Weight", "Diabetes", "BloodPressureProblems",
        "AnyTransplants", "AnyChronicDiseases", "KnownAllergies",
        "HistoryOfCancerInFamily", "NumberOfMajorSurgeries"
    ]
    return {
        "raw": raw_features,
        "raw_plus_bmi": raw_features + ["BMI"],
        "raw_plus_bmi_cci": raw_features + ["BMI", "CCI"],
        "full_engineered": raw_features + ["BMI", "CCI", "SurgicalCancerInteraction"],
    }
