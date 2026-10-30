# Existing Project Audit Report

**Audit Date:** October 30, 2026  
**Auditor:** Senior ML Engineer & Project Mentor  
**Project:** Medical Health Insurance Cost Prediction Using Explainable Machine Learning  

---

## 1. Executive Summary

This audit assesses the initial workspace state against the requirements of the **Orji & Ukwandu (2024)** research paper (*"Machine Learning for an Explainable Cost Prediction of Medical Insurance"*, Elsevier MLWA) and the college project specifications.

---

## 2. Component Status Audit Table

| Component | Existing Status | Problem Identified | Required Improvement | Priority |
|---|---|---|---|---|
| **Dataset Ingestion (`data/`)** | **Partially working** | Workspace initially only contained the 1,338-record `insurance.csv` (Brett Lantz dataset). The primary research paper explicitly analyzes the 986-record `Medicalpremium.csv` with 10 clinical features. | Separate raw and processed data; support both the 986-record Primary Reproduction dataset and the 1,338-record Extended Actuarial dataset with automated schema checks. | **High** |
| **Data Loader (`src/data_loader.py`)** | **Working** | Implemented only for 7 columns of the 1,338 dataset. Cannot load or validate the 11-column 986-record dataset. | Extend loader to dynamically validate and ingest both schemas with clear type checks, null checks, and range verifications. | **High** |
| **Preprocessing Pipeline (`src/preprocessing.py`)** | **Partially working** | Fitted strictly on training data (leak-free), but handles only 1,338-record categorical features. No pipeline for the 986-record clinical features. | Implement dedicated `ColumnTransformer` and scalers for both the 986 clinical dataset and the 1,338 actuarial dataset. | **High** |
| **Model Development (`src/train_models.py`)** | **Partially working** | Implemented general training for 8 regressors, but lacks runtime logging, training duration tracking, and experimental logging into `results/experiment_log.csv`. | Implement paper-specific core models: Random Forest, GBM, and XGBoost; record seeds, training time, and parameters. | **High** |
| **Evaluation Suite (`src/evaluate_models.py`)** | **Working** | Computes $R^2$, MAE, RMSE, MAPE, but outputs figures to in-memory buffers rather than persisting to `results/figures/` and `results/metrics.csv`. | Auto-save metrics to `results/metrics.csv` and export publication-ready SVG/PNG figures to `results/figures/`. | **High** |
| **Explainable AI (`src/explainability.py`)** | **Partially working** | SHAP Global summary and Local waterfall are implemented. **ICE (Individual Conditional Expectation) plots** required by the 2024 paper are missing. | Implement ICE curve generators using `PartialDependenceDisplay` to inspect non-linear tipping points (e.g., Age, Weight, Transplants). | **High** |
| **Jupyter Notebooks (`notebooks/`)** | **Not implemented** | Directory is empty; students need reproducible, cell-by-cell exploratory notebooks for submission. | Generate clean, executable notebooks: `01_data_exploration.ipynb`, `02_preprocessing.ipynb`, `03_model_training.ipynb`, and `04_shap_ice_analysis.ipynb`. | **Medium** |
| **What-If Analysis (Counterfactual Simulator)** | **Not implemented** | Web application only provides static point prediction; lacks interactive counterfactual delta comparison. | Add a dedicated What-If comparison tool in Streamlit to simulate how shifting specific inputs modifies predicted costs. | **High** |
| **Streamlit Application (`app.py`)** | **Working** | UI is responsive, but configured only for the 1,338 dataset. Lacks ICE plots, What-If simulation, and research paper reproduction mode. | Upgrade `app.py` with 7 dedicated pages/tabs: Home, Predictor, Model Performance, Explainable AI (SHAP & ICE), What-If Analysis, Research Methodology, and Model Card/About. | **High** |
| **Unit Test Suite (`tests/`)** | **Working** | 9 unit tests passing on general pipeline. Needs dedicated coverage for 986 dataset, ICE routines, and bounds checking. | Expand tests to 15+ assertions covering both datasets, metric calculations, and inference bounds. | **Medium** |

---

## 3. Decision & Implementation Strategy

1. **Dual-Track Research Architecture:**
   - **Track 1 (Core Paper Reproduction):** Implements the exact 986-record Orji & Ukwandu (2024) pipeline: `Medicalpremium.csv`, RF, GBM, XGBoost, SHAP, and ICE.
   - **Track 2 (Extended Actuarial Comparison):** Preserves the 1,338-record `insurance.csv` 8-model benchmark (Ridge, Lasso, CatBoost, LightGBM, etc.) as an extension.
2. **Persistence Guarantee:**
   - Real metrics will be logged to `results/metrics.csv` and `results/experiment_log.csv`.
   - Figures will be saved to `results/figures/`.
   - Trained artifacts (`best_model.joblib`, `preprocessor.joblib`) will be serialized into `models/`.
