"""
Explainable AI (XAI) module using SHAP for Medical Health Insurance Cost Prediction.
Provides global feature attribution and individual applicant local prediction explanations.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ExplainabilityError(Exception):
    """Exception raised when SHAP computation encounters an unsupported model or failure."""
    pass


def get_shap_explainer(
    model: Any,
    X_train_proc: np.ndarray,
    feature_names: List[str],
) -> shap.Explainer:
    """
    Construct a suitable SHAP Explainer based on model family.
    Uses TreeExplainer for decision trees/boosters and general Explainer for others.

    Parameters
    ----------
    model : Any
        Trained model instance.
    X_train_proc : np.ndarray
        Background dataset for reference baseline.
    feature_names : List[str]
        Clean names of preprocessed features.

    Returns
    -------
    shap.Explainer
        Configured explainer.
    """
    model_type = type(model).__name__.lower()
    
    # Subsample background data to at most 100 samples for swift calculation
    background = shap.sample(X_train_proc, min(100, len(X_train_proc)), random_state=42)

    try:
        if any(tree_keyword in model_type for tree_keyword in ["forest", "gradient", "xgb", "lgbm", "catboost"]):
            explainer = shap.TreeExplainer(model, data=background)
        elif any(linear_keyword in model_type for linear_keyword in ["linear", "ridge", "lasso"]):
            explainer = shap.LinearExplainer(model, data=background)
        else:
            explainer = shap.Explainer(model.predict, background)
        return explainer
    except Exception as exc:
        # Generic fallback
        try:
            return shap.Explainer(model.predict, background)
        except Exception as inner_exc:
            raise ExplainabilityError(f"Failed to initialize SHAP explainer: {exc} | Fallback: {inner_exc}") from exc


def compute_shap_values(
    explainer: shap.Explainer,
    X_proc: np.ndarray,
) -> shap.Explanation:
    """
    Compute SHAP Explanation for a feature matrix.

    Parameters
    ----------
    explainer : shap.Explainer
        Fitted explainer.
    X_proc : np.ndarray
        Preprocessed feature matrix.

    Returns
    -------
    shap.Explanation
        Explanation object containing values, base_values, and data.
    """
    try:
        explanation = explainer(X_proc)
        return explanation
    except Exception as exc:
        raise ExplainabilityError(f"Error computing SHAP values: {exc}") from exc


def plot_global_shap_summary(
    explanation: shap.Explanation,
    feature_names: List[str],
    max_display: int = 10,
) -> plt.Figure:
    """
    Generate a global SHAP beeswarm / summary plot showing direction and impact.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP explanation on test data.
    feature_names : List[str]
        Transformed feature names.
    max_display : int, default=10
        Maximum features to display.

    Returns
    -------
    plt.Figure
        Figure containing the summary plot.
    """
    fig = plt.figure(figsize=(9, 6))
    
    # Ensure feature names are assigned to the explanation object for proper labelling
    explanation.feature_names = feature_names
    shap.summary_plot(
        explanation.values,
        features=explanation.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Global Feature Importance & Impact Direction", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()
    return fig


def plot_global_bar_importance(
    explanation: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature for a clean tabular summary.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP explanation object.
    feature_names : List[str]
        Feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and Mean |SHAP value| ($).
    """
    mean_abs_shap = np.mean(np.abs(explanation.values), axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value ($)": mean_abs_shap,
    }).sort_values(by="Mean Absolute SHAP Value ($)", ascending=False).reset_index(drop=True)
    return df_importance


def plot_local_waterfall(
    explanation: shap.Explanation,
    sample_index: int = 0,
    feature_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Generate a local SHAP waterfall plot explaining an individual's prediction.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP explanation object.
    sample_index : int, default=0
        Index of the sample to explain.
    feature_names : Optional[List[str]]
        Transformed feature names.

    Returns
    -------
    plt.Figure
        Waterfall plot figure.
    """
    if feature_names:
        explanation.feature_names = feature_names

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation[sample_index], show=False)
    plt.title("Individual Prediction Attribution (SHAP Waterfall)", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()
    return fig


def generate_natural_language_explanation(
    shap_sample: np.ndarray,
    feature_names: List[str],
    base_value: float,
    predicted_charge: float,
    top_n: int = 4,
) -> Dict[str, Any]:
    """
    Translate quantitative SHAP values into an intuitive, plain-English summary.

    Parameters
    ----------
    shap_sample : np.ndarray
        1D array of SHAP values for the specific applicant.
    feature_names : List[str]
        Names of transformed features.
    base_value : float
        Model baseline average charge.
    predicted_charge : float
        Final predicted insurance charge.
    top_n : int, default=4
        Number of leading contributors to highlight.

    Returns
    -------
    Dict[str, Any]
        Structured explanation with text bullets and attributions.
    """
    contributions = []
    for name, val in zip(feature_names, shap_sample):
        contributions.append({
            "feature": name,
            "shap_value": float(val),
            "abs_val": abs(float(val)),
            "direction": "increased" if val > 0 else "decreased",
        })

    # Sort by absolute impact
    contributions = sorted(contributions, key=lambda x: x["abs_val"], reverse=True)

    summary_bullets = []
    for item in contributions[:top_n]:
        feature_clean = item["feature"].replace("_", " ").title()
        sign = "+" if item["shap_value"] > 0 else "-"
        summary_bullets.append(
            f"**{feature_clean}**: {item['direction']} the estimated charge by **{sign}${item['abs_val']:,.2f}**."
        )

    return {
        "base_value": base_value,
        "predicted_charge": predicted_charge,
        "net_difference": predicted_charge - base_value,
        "top_contributions": contributions[:top_n],
        "summary_bullets": summary_bullets,
        "disclaimer": (
            "Ethical & Academic Notice: Feature contributions reflect statistical associations "
            "learned by the ML model on historical training data. They DO NOT establish medical causation, "
            "clinical diagnoses, or finalized quotations by an insurance underwriting carrier."
        ),
    }
