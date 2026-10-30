"""
Model training module for Medical Health Insurance Cost Prediction.
Implements 8 regression models with graceful dependency handling,
K-fold cross-validation on training data, and joblib artifact persistence.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score

# Optional Gradient Boosting libraries with clean status tracking
MODELS_AVAILABILITY = {
    "Linear Regression": True,
    "Ridge Regression": True,
    "Lasso Regression": True,
    "Random Forest": True,
    "Gradient Boosting": True,
    "XGBoost": False,
    "LightGBM": False,
    "CatBoost": False,
}

try:
    from xgboost import XGBRegressor
    MODELS_AVAILABILITY["XGBoost"] = True
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    MODELS_AVAILABILITY["LightGBM"] = True
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
    MODELS_AVAILABILITY["CatBoost"] = True
except ImportError:
    CatBoostRegressor = None


def get_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """
    Initialize instances of the 8 regression models with consistent random seeds.
    Only includes boosting libraries if they are installed in the environment.

    Parameters
    ----------
    random_state : int, default=42
        Seed for reproducibility across stochastic models.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to regressor instances.
    """
    registry = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=1.0, random_state=random_state, max_iter=2000),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=random_state),
    }

    if MODELS_AVAILABILITY["XGBoost"] and XGBRegressor is not None:
        registry["XGBoost"] = XGBRegressor(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=3,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )

    if MODELS_AVAILABILITY["LightGBM"] and LGBMRegressor is not None:
        registry["LightGBM"] = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=3,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )

    if MODELS_AVAILABILITY["CatBoost"] and CatBoostRegressor is not None:
        registry["CatBoost"] = CatBoostRegressor(
            iterations=150,
            learning_rate=0.07,
            depth=4,
            random_seed=random_state,
            verbose=0,
        )

    return registry


def train_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a single regression model on training data and evaluate via cross-validation.

    Parameters
    ----------
    model : Any
        Regressor instance.
    X_train : np.ndarray
        Preprocessed training feature matrix.
    y_train : np.ndarray
        Target vector.
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        CV shuffle seed.

    Returns
    -------
    Dict[str, Any]
        Fitted model, mean CV R2, std CV R2, and mean CV MAE.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 5-Fold Cross-Validation on training data to prevent test leakage
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

    # Fit final model on complete training split
    model.fit(X_train, y_train)

    return {
        "model": model,
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "cv_r2_std": float(np.std(cv_r2_scores)),
        "cv_mae_mean": float(np.mean(cv_mae_scores)),
        "cv_mae_std": float(np.std(cv_mae_scores)),
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all registered and available regression models.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Target values.
    cv_folds : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of model names to training results.
    """
    registry = get_model_registry(random_state=random_state)
    trained_results = {}

    for name, model in registry.items():
        trained_results[name] = train_single_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            random_state=random_state,
        )

    return trained_results


def save_artifact(obj: Any, file_name: str, directory: Optional[str] = None) -> Path:
    """
    Persist trained model, preprocessor, or evaluation metrics using joblib.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    file_name : str
        Target file name (e.g. 'best_model.joblib').
    directory : Optional[str]
        Folder to save in. Defaults to project 'models/' directory.

    Returns
    -------
    Path
        Saved file path.
    """
    save_dir = Path(directory) if directory else Path(__file__).resolve().parent.parent / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    target_path = save_dir / file_name
    joblib.dump(obj, target_path)
    return target_path


def load_artifact(file_name: str, directory: Optional[str] = None) -> Any:
    """
    Load a serialized object from disk.

    Parameters
    ----------
    file_name : str
        Name of file to load.
    directory : Optional[str]
        Directory path.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    target_dir = Path(directory) if directory else Path(__file__).resolve().parent.parent / "models"
    target_path = target_dir / file_name
    if not target_path.exists():
        raise FileNotFoundError(f"Artifact not found at: {target_path}")
    return joblib.load(target_path)
