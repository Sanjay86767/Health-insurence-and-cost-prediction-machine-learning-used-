"""
Command-line pipeline runner for Medical Health Insurance Cost Prediction.
Executes complete research workflow:
1. Loads verified dataset (986 records)
2. Fits leak-free preprocessing pipeline
3. Trains Random Forest, Gradient Boosting, and XGBoost
4. Calculates R2, MAE, RMSE, MAPE
5. Generates SHAP beeswarm and ICE curves
6. Saves model artifacts and metrics to disk
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import load_dataset, split_features_target
from src.evaluate_models import (
    compare_and_save_metrics,
    plot_metrics_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
)
from src.explainability import (
    compute_shap_explanation,
    get_shap_explainer,
    plot_ice_curves,
    plot_shap_summary,
)
from src.preprocessing import prepare_train_test_data
from src.train_models import save_artifact, train_models_pipeline


def run_full_pipeline(dataset_type: str = "medical_premium_986"):
    print("\n=======================================================")
    print(f"Running Full Pipeline for: {dataset_type}")
    print("=======================================================")

    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df, d_type = load_dataset(dataset_type)
    print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Split Features & Target
    X, y = split_features_target(df, d_type)

    # 3. Preprocess with strict leakage protection
    X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
        X, y, dataset_type=d_type, test_size=0.2, random_state=42
    )
    print(f"✓ Preprocessing completed. Training samples: {len(X_train_proc)}, Test samples: {len(X_test_proc)}")
    print(f"✓ Feature names ({len(feature_names)}): {feature_names}")

    # 4. Train Models
    print("✓ Training Random Forest, Gradient Boosting, and XGBoost with 5-Fold Cross Validation...")
    trained_results = train_models_pipeline(
        X_train=X_train_proc,
        y_train=y_train,
        use_core_only=True,
        cv_folds=5,
        random_state=42,
        log_experiment=True,
        dataset_name=d_type,
    )

    # 5. Evaluate and Save Metrics
    comp_df = compare_and_save_metrics(trained_results, X_test_proc, y_test, save_csv=True)
    print("\n🏆 Measured Test Metrics Leaderboard:")
    print(comp_df.to_string(index=False))

    best_model_name = comp_df.iloc[0]["Model"]
    best_model = trained_results[best_model_name]["model"]
    print(f"\n✓ Champion Model Selected: {best_model_name} (Highest Test R²: {comp_df.iloc[0]['Test R2']})")

    # 6. Save Artifacts
    save_artifact(best_model, f"best_model_{d_type}.joblib")
    save_artifact(preprocessor, f"preprocessor_{d_type}.joblib")
    print(f"✓ Artifacts saved to models/ directory.")

    # 7. Generate Evaluation Figures
    plot_metrics_comparison(comp_df, save_path=figures_dir / f"metrics_comparison_{d_type}.png")
    plot_predicted_vs_actual(best_model, X_test_proc, y_test, best_model_name, save_path=figures_dir / f"pred_vs_actual_{d_type}.png")
    plot_residuals(best_model, X_test_proc, y_test, best_model_name, save_path=figures_dir / f"residuals_{d_type}.png")
    print("✓ Evaluation diagnostic plots exported to results/figures/.")

    # 8. Explainability: SHAP & ICE
    print("✓ Computing Explainable AI outputs (SHAP & ICE)...")
    try:
        explainer = get_shap_explainer(best_model, X_train_proc)
        shap_exp = compute_shap_explanation(explainer, X_test_proc[:100], feature_names)
        plot_shap_summary(shap_exp, feature_names, save_path=figures_dir / f"shap_summary_{d_type}.png")
        print("✓ SHAP beeswarm summary plot saved.")

        # ICE plots
        features_to_plot = ["Age", "Weight"] if d_type == "medical_premium_986" else ["age", "bmi"]
        plot_ice_curves(best_model, X_test_proc, feature_names, features_to_plot, save_path=figures_dir / f"ice_curves_{d_type}.png")
        print("✓ ICE individual conditional expectation plots saved.")
    except Exception as exc:
        print(f"⚠️ Explainability warning: {exc}")

    print("\n✅ Full Pipeline Execution Complete!")


if __name__ == "__main__":
    run_full_pipeline("medical_premium_986")
