"""
Counterfactual Recourse & Actionable Recommendations Engine
Based on Section III-E and Table VII of Sanjay Kumar & Dr. Sukhvinder Singh Bamber (Panjab University).
Finds the minimal, feasible lifestyle and clinical adjustments (e.g. BMI reduction, BP management)
that lower predicted health insurance premiums for new policyholders while holding immutable factors fixed.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


def generate_counterfactual_recourse(
    model: Any,
    preprocessor: ColumnTransformer,
    applicant_data: Dict[str, Any],
    dataset_type: str = "medical_premium_986",
    target_bmi_reduction: float = 4.0,
) -> Dict[str, Any]:
    """
    Generate tailored, actionable counterfactual recourse strategies for a new applicant.

    Parameters
    ----------
    model : Any
        Fitted regression model.
    preprocessor : ColumnTransformer
        Fitted preprocessor pipeline.
    applicant_data : Dict[str, Any]
        Raw applicant inputs.
    dataset_type : str
        'medical_premium_986' or 'insurance_1338'.
    target_bmi_reduction : float
        Simulated achievable BMI drop (default 4.0 units).

    Returns
    -------
    Dict[str, Any]
        Baseline estimate, recommended adjustments, scenario estimate, and financial savings.
    """
    from src.predict import predict_cost

    # 1. Calculate Baseline Prediction
    base_cost, _, _ = predict_cost(model, preprocessor, applicant_data, dataset_type)

    strategies = []

    if dataset_type == "medical_premium_986":
        curr_weight = float(applicant_data.get("Weight", 75.0))
        curr_height = float(applicant_data.get("Height", 170.0))
        height_m = curr_height / 100.0
        curr_bmi = curr_weight / (height_m ** 2)

        # Strategy A: Weight / BMI Reduction (Modest & Feasible)
        if curr_bmi > 24.5:
            target_bmi = max(24.0, curr_bmi - target_bmi_reduction)
            target_weight = round(target_bmi * (height_m ** 2), 1)
            weight_drop = round(curr_weight - target_weight, 1)

            mod_profile_a = dict(applicant_data)
            mod_profile_a["Weight"] = target_weight

            cost_a, _, _ = predict_cost(model, preprocessor, mod_profile_a, dataset_type)
            saving_a = base_cost - cost_a
            saving_pct_a = (saving_a / base_cost * 100) if base_cost > 0 else 0

            strategies.append({
                "strategy_name": "Weight Management & BMI Optimization",
                "action": f"Reduce weight by {weight_drop} kg (from {curr_weight} kg to {target_weight} kg; BMI {curr_bmi:.1f} → {target_bmi:.1f})",
                "scenario_cost": cost_a,
                "annual_saving": max(0.0, saving_a),
                "saving_percent": max(0.0, saving_pct_a),
                "modified_inputs": {"Weight": target_weight},
            })

        # Strategy B: Clinical Condition Control (BP / Chronic Disease Management)
        if applicant_data.get("BloodPressureProblems") == 1 or applicant_data.get("AnyChronicDiseases") == 1:
            mod_profile_b = dict(applicant_data)
            actions = []
            if mod_profile_b.get("BloodPressureProblems") == 1:
                mod_profile_b["BloodPressureProblems"] = 0
                actions.append("Manage & normalize blood pressure through treatment")
            if mod_profile_b.get("AnyChronicDiseases") == 1:
                mod_profile_b["AnyChronicDiseases"] = 0
                actions.append("Control chronic conditions through preventive care")

            cost_b, _, _ = predict_cost(model, preprocessor, mod_profile_b, dataset_type)
            saving_b = base_cost - cost_b
            saving_pct_b = (saving_b / base_cost * 100) if base_cost > 0 else 0

            strategies.append({
                "strategy_name": "Clinical Risk Management",
                "action": " + ".join(actions),
                "scenario_cost": cost_b,
                "annual_saving": max(0.0, saving_b),
                "saving_percent": max(0.0, saving_pct_b),
                "modified_inputs": {"BloodPressureProblems": 0, "AnyChronicDiseases": 0},
            })

        # Strategy C: Combined Lifestyle & Medical Protocol
        if len(strategies) >= 2:
            mod_profile_c = dict(applicant_data)
            mod_profile_c["Weight"] = strategies[0]["modified_inputs"]["Weight"]
            for k, v in strategies[1]["modified_inputs"].items():
                mod_profile_c[k] = v

            cost_c, _, _ = predict_cost(model, preprocessor, mod_profile_c, dataset_type)
            saving_c = base_cost - cost_c
            saving_pct_c = (saving_c / base_cost * 100) if base_cost > 0 else 0

            strategies.append({
                "strategy_name": "Comprehensive Health Protocol (Lifestyle + Medical)",
                "action": f"{strategies[0]['action']} AND {strategies[1]['action']}",
                "scenario_cost": cost_c,
                "annual_saving": max(0.0, saving_c),
                "saving_percent": max(0.0, saving_pct_c),
                "modified_inputs": mod_profile_c,
            })

    else:
        # 1,338 dataset recourse (Smoking cessation & BMI reduction)
        curr_bmi = float(applicant_data.get("bmi", 28.5))
        is_smoker = applicant_data.get("smoker") == "yes"

        if is_smoker:
            mod_profile_s = dict(applicant_data)
            mod_profile_s["smoker"] = "no"
            cost_s, _, _ = predict_cost(model, preprocessor, mod_profile_s, dataset_type)
            saving_s = base_cost - cost_s
            saving_pct_s = (saving_s / base_cost * 100) if base_cost > 0 else 0

            strategies.append({
                "strategy_name": "Smoking Cessation",
                "action": "Quit smoking completely (smoker: yes → no)",
                "scenario_cost": cost_s,
                "annual_saving": max(0.0, saving_s),
                "saving_percent": max(0.0, saving_pct_s),
                "modified_inputs": {"smoker": "no"},
            })

        if curr_bmi > 25.0:
            target_bmi = max(23.5, curr_bmi - 4.5)
            mod_profile_b = dict(applicant_data)
            mod_profile_b["bmi"] = round(target_bmi, 1)
            cost_b, _, _ = predict_cost(model, preprocessor, mod_profile_b, dataset_type)
            saving_b = base_cost - cost_b
            saving_pct_b = (saving_b / base_cost * 100) if base_cost > 0 else 0

            strategies.append({
                "strategy_name": "BMI Normalization",
                "action": f"Reduce BMI from {curr_bmi:.1f} to {target_bmi:.1f}",
                "scenario_cost": cost_b,
                "annual_saving": max(0.0, saving_b),
                "saving_percent": max(0.0, saving_pct_b),
                "modified_inputs": {"bmi": target_bmi},
            })

    return {
        "baseline_cost": base_cost,
        "applicant_profile": applicant_data,
        "recourse_strategies": strategies,
        "disclaimer": (
            "Counterfactual recourse identifies algorithmic sensitivities in the empirical pricing model. "
            "It does not represent clinical medical advice or an underwriter guarantee."
        ),
    }
