"""
Explainable AI (XAI) module for Medical Health Insurance Cost Prediction.
Implements:
1. SHAP (Shapley Additive Explanations) - Global Beeswarm & Local Waterfall
2. ICE (Individual Conditional Expectation) Plots - Feature Tipping Points & Non-linear Curves
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay


class ExplainabilityError(Exception):
    """Exception raised for explainability failures."""
    pass


def get_shap_explainer(model: Any, X_train_proc: np.ndarray) -> shap.Explainer:
    """Initialize suitable SHAP explainer."""
    model_name = type(model).__name__.lower()
    background = shap.sample(X_train_proc, min(80, len(X_train_proc)), random_state=42)

    try:
        if any(tree_k in model_name for tree_k in ["forest", "gradient", "xgb", "lgbm", "catboost"]):
            return shap.TreeExplainer(model, data=background)
        else:
            return shap.Explainer(model.predict, background)
    except Exception:
        return shap.Explainer(model.predict, background)


def compute_shap_explanation(explainer: shap.Explainer, X_proc: np.ndarray, feature_names: List[str]) -> shap.Explanation:
    """Compute SHAP explanation object with feature labels attached."""
    try:
        explanation = explainer(X_proc, check_additivity=False)
    except TypeError:
        explanation = explainer(X_proc)
    explanation.feature_names = feature_names
    return explanation


def plot_shap_summary(
    explanation: shap.Explanation,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    max_display: int = 10,
) -> plt.Figure:
    """Generate global SHAP beeswarm / summary plot."""
    fig = plt.figure(figsize=(9, 6))
    explanation.feature_names = feature_names
    shap.summary_plot(
        explanation.values,
        features=explanation.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Global Feature Attribution & Directionality", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_local_waterfall(
    explanation: shap.Explanation,
    sample_index: int = 0,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate local SHAP waterfall plot for individual applicant."""
    if feature_names:
        explanation.feature_names = feature_names

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation[sample_index], show=False)
    plt.title("Individual Prediction Attribution (SHAP Waterfall)", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ice_curves(
    model: Any,
    X_proc: np.ndarray,
    feature_names: List[str],
    features_to_plot: List[str],
    subsample_n: int = 40,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate Individual Conditional Expectation (ICE) curves alongside Partial Dependence.
    kind='both' displays individual patient lines (ICE) and the bold black mean line (PDP).
    """
    df_features = pd.DataFrame(X_proc, columns=feature_names)
    indices = [feature_names.index(f) for f in features_to_plot if f in feature_names]

    if not indices:
        raise ExplainabilityError(f"None of features {features_to_plot} found in {feature_names}")

    fig, ax = plt.subplots(figsize=(6 * len(indices), 5))
    if len(indices) == 1:
        ax = [ax]

    # Sample rows for clean rendering
    sample_df = df_features.sample(min(subsample_n, len(df_features)), random_state=42)

    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=sample_df,
        features=indices,
        feature_names=feature_names,
        kind="both",  # ICE individual curves + PDP average curve
        subsample=subsample_n,
        ax=ax,
        grid_resolution=30,
        random_state=42,
    )

    fig.suptitle("Individual Conditional Expectation (ICE) & Partial Dependence Curves", fontsize=13, weight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def generate_natural_language_explanation(
    shap_sample: np.ndarray,
    feature_names: List[str],
    base_value: float,
    predicted_value: float,
    top_n: int = 4,
) -> Dict[str, Any]:
    """Translate SHAP numerical values into an intuitive plain-English summary."""
    items = []
    for f_name, s_val in zip(feature_names, shap_sample):
        items.append({
            "feature": f_name,
            "shap_value": float(s_val),
            "abs_val": abs(float(s_val)),
            "direction": "increased" if s_val > 0 else "decreased",
        })

    items = sorted(items, key=lambda x: x["abs_val"], reverse=True)
    bullets = []
    for item in items[:top_n]:
        clean_name = item["feature"].replace("_", " ").title()
        sign = "+" if item["shap_value"] > 0 else "-"
        bullets.append(
            f"**{clean_name}**: {item['direction']} the predicted premium by **{sign}{item['abs_val']:,.2f}**."
        )

    return {
        "base_value": base_value,
        "predicted_value": predicted_value,
        "top_contributions": items[:top_n],
        "summary_bullets": bullets,
        "disclaimer": (
            "Academic Notice: SHAP attributions represent empirical model weightings on historical data; "
            "they do not imply medical causality or guaranteed underwriter pricing."
        ),
    }
