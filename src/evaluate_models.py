"""
Evaluation and diagnostic plotting module for Medical Health Insurance Cost Prediction.
Computes R², MAE, RMSE, safe MAPE, logs metrics to results/metrics.csv,
and exports high-resolution diagnostic plots to results/figures/.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetics
sns.set_theme(style="whitegrid", palette="muted")


def calculate_safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) protecting against zero-division.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_single_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calculate R2, MAE, RMSE, and MAPE for a single model."""
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


def compare_and_save_metrics(
    trained_results: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_csv: bool = True,
    figures_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute test metrics across all models, rank them, and persist results/metrics.csv.
    """
    records = []

    for name, res in trained_results.items():
        model = res["model"]
        test_metrics = evaluate_single_model(model, X_test, y_test)

        records.append({
            "Model": name,
            "Test R2": round(test_metrics["R2"], 4),
            "Test MAE": round(test_metrics["MAE"], 2),
            "Test RMSE": round(test_metrics["RMSE"], 2),
            "Test MAPE (%)": round(test_metrics["MAPE"], 2),
            "CV R2 (Mean)": round(res.get("cv_r2_mean", 0.0), 4),
            "CV R2 (Std)": round(res.get("cv_r2_std", 0.0), 4),
            "CV MAE (Mean)": round(res.get("cv_mae_mean", 0.0), 2),
            "Train Time (s)": round(res.get("train_duration_sec", 0.0), 4),
        })

    df_comp = pd.DataFrame(records).sort_values(by="Test R2", ascending=False).reset_index(drop=True)

    if save_csv:
        csv_path = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(csv_path, index=False)

    return df_comp


def plot_metrics_comparison(df_comp: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Generate bar chart comparing MAE, RMSE, and R2 across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_melted = df_comp.melt(
        id_vars=["Model"],
        value_vars=["Test MAE", "Test RMSE"],
        var_name="Metric",
        value_name="Error",
    )
    sns.barplot(data=df_melted, x="Model", y="Error", hue="Metric", ax=axes[0], palette=["#3498db", "#e74c3c"])
    axes[0].set_title("Test Error (MAE vs. RMSE) — Lower is Better", fontsize=12, weight="bold")
    axes[0].set_ylabel("Error Units", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=df_comp, x="Model", y="Test R2", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Test R² Score — Higher is Better", fontsize=12, weight="bold")
    axes[1].set_ylabel("R² Score", fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", rotation=30)

    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_predicted_vs_actual(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """Generate scatter plot of actual vs predicted."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_test, y_pred, alpha=0.6, color="#2980b9", edgecolors="w", s=50)
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Identity Line (y = x)")

    ax.set_title(f"Predicted vs. Actual: {model_name}", fontsize=13, weight="bold")
    ax.set_xlabel("Actual Values", fontsize=11)
    ax.set_ylabel("Predicted Values", fontsize=11)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_residuals(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """Generate residual error scatter and distribution plots."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#8e44ad", edgecolors="w", s=45)
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set_title(f"Residuals vs. Fitted ({model_name})", fontsize=12, weight="bold")
    axes[0].set_xlabel("Fitted Values", fontsize=11)
    axes[0].set_ylabel("Residuals", fontsize=11)

    sns.histplot(residuals, kde=True, ax=axes[1], color="#27ae60", bins=25)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_title("Residual Error Distribution", fontsize=12, weight="bold")
    axes[1].set_xlabel("Residual", fontsize=11)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
