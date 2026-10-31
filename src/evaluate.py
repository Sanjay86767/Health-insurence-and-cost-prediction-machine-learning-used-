"""
Evaluation and visualization module for Medical Health Insurance Cost Prediction.
Computes MAE, RMSE, R2, safe MAPE, and generates publication-grade diagnostic plots.
"""

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set clean, modern plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"


def calculate_safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) with division-by-zero protection.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, default=1e-6
        Small constant to prevent zero-division.

    Returns
    -------
    float
        MAPE expressed as a percentage (0 to 100).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate test evaluation metrics for a single fitted regressor.

    Parameters
    ----------
    model : Any
        Fitted regression model.
    X_test : np.ndarray
        Preprocessed test features.
    y_test : np.ndarray
        Actual test charges.

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, R2, and MAPE.
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = calculate_safe_mape(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def compare_models(
    trained_results: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Generate a comprehensive model comparison DataFrame combining test metrics
    and training cross-validation scores.

    Parameters
    ----------
    trained_results : Dict[str, Dict[str, Any]]
        Output dictionary from `train_all_models`.
    X_test : np.ndarray
        Preprocessed test features.
    y_test : np.ndarray
        True test charges.

    Returns
    -------
    pd.DataFrame
        Sorted comparison table (best R2 on test set first).
    """
    records = []

    for name, res in trained_results.items():
        model = res["model"]
        test_metrics = evaluate_single_model(model, X_test, y_test)

        records.append({
            "Model": name,
            "Test R2": test_metrics["R2"],
            "Test MAE (₹)": test_metrics["MAE"],
            "Test RMSE (₹)": test_metrics["RMSE"],
            "Test MAPE (%)": test_metrics["MAPE"],
            "CV R2 (Train Mean)": res["cv_r2_mean"],
            "CV R2 (Train Std)": res["cv_r2_std"],
            "CV MAE (Train Mean)": res["cv_mae_mean"],
        })

    df_comp = pd.DataFrame(records)
    # Sort descending by Test R2 (higher is better)
    df_comp = df_comp.sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    return df_comp


def plot_metrics_comparison(df_comp: pd.DataFrame) -> plt.Figure:
    """
    Generate side-by-side bar plots comparing MAE, RMSE, and R2 across all models.

    Parameters
    ----------
    df_comp : pd.DataFrame
        Comparison table from `compare_models`.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Errors (MAE and RMSE) - Lower is better
    mae_col = "Test MAE (₹)" if "Test MAE (₹)" in df_comp.columns else "Test MAE ($)"
    rmse_col = "Test RMSE (₹)" if "Test RMSE (₹)" in df_comp.columns else "Test RMSE ($)"
    df_melted = df_comp.melt(
        id_vars=["Model"],
        value_vars=[mae_col, rmse_col],
        var_name="Metric",
        value_name="Error (₹)",
    )
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Error (₹)",
        hue="Metric",
        ax=axes[0],
        palette=["#3498db", "#e74c3c"],
    )
    axes[0].set_title("Test Error Comparison (Lower is Better)", fontsize=13, weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Error in Indian Rupees (₹)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=True)

    # Plot 2: R2 Score - Higher is better
    sns.barplot(
        data=df_comp,
        x="Model",
        y="Test R2",
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_title("Test R² Score Comparison (Higher is Better)", fontsize=13, weight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R² Coefficient of Determination", fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", rotation=35)

    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Best Model",
) -> plt.Figure:
    """
    Generate an Actual vs. Predicted scatter plot with perfect-prediction identity line.

    Parameters
    ----------
    model : Any
        Fitted model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Actual charges.
    model_name : str
        Model title display.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_test, y_pred, alpha=0.6, color="#2980b9", edgecolors="w", s=50)

    # Perfect prediction identity line y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")

    ax.set_title(f"Predicted vs. Actual Charges: {model_name}", fontsize=13, weight="bold")
    ax.set_xlabel("Actual Insurance Charges (₹)", fontsize=11)
    ax.set_ylabel("Predicted Insurance Charges (₹)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_residuals(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Best Model",
) -> plt.Figure:
    """
    Generate residual distribution and residual vs. predicted diagnostic plots.

    Parameters
    ----------
    model : Any
        Fitted model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Actual charges.
    model_name : str
        Model name.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter of residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#8e44ad", edgecolors="w", s=45)
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set_title(f"Residuals vs. Predicted ({model_name})", fontsize=12, weight="bold")
    axes[0].set_xlabel("Predicted Charges (₹)", fontsize=11)
    axes[0].set_ylabel("Residuals (Actual - Predicted) (₹)", fontsize=11)

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="#27ae60", bins=25)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_title("Residual Error Distribution", fontsize=12, weight="bold")
    axes[1].set_xlabel("Residual Error (₹)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)

    plt.tight_layout()
    return fig
