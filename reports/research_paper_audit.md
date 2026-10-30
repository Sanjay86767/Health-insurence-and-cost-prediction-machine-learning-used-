# Research Paper Audit Report
**Project:** Medical Health Insurance Cost Prediction Using Explainable Machine Learning  
**Author:** B.Tech Computer Science & Engineering Research Project  
**Date:** October 30, 2026  

---

## 1. Primary Paper Citation & Metadata

- **Title:** *Machine Learning for an Explainable Cost Prediction of Medical Insurance*
- **Authors:** Ugochukwu Orji and Elochukwu Ukwandu
- **Journal:** *Machine Learning with Applications* (Elsevier), Volume 15, Article 100516, 2024.
- **DOI:** [10.1016/j.mlwa.2023.100516](https://doi.org/10.1016/j.mlwa.2023.100516)
- **Preprint:** [arXiv:2311.14139](https://arxiv.org/abs/2311.14139)
- **Publication Date:** 2024 (Accepted / Online 2023)

---

## 2. Research Problem & Objectives

### 2.1 The Problem
Medical insurance premium determination is traditionally performed using actuarial lookup tables and Generalized Linear Models (GLMs). These legacy methods often fail to capture complex non-linear combinations of physiological risk indicators (such as multi-morbidities: diabetes + hypertension + transplants). Furthermore, modern high-performing black-box machine learning models are resisted by insurance regulators and policyholders due to a lack of interpretability.

### 2.2 Core Objectives of the Study
1. Develop and compare three non-linear ensemble decision-tree regression models:
   - **Random Forest (RF)**
   - **Gradient Boosting Machine (GBM)**
   - **Extreme Gradient Boosting (XGBoost)**
2. Evaluate performance using four quantitative regression metrics:
   - Coefficient of Determination ($R^2$)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
3. Implement post-hoc Explainable AI (XAI) frameworks:
   - **SHAP (SHapley Additive exPlanations)** for global feature attribution and individual prediction breakdowns.
   - **Individual Conditional Expectation (ICE) plots** to assess localized marginal effects and non-linear feature interactions.

---

## 3. Dataset Audit: Verified Findings & Discrepancy Analysis

### 3.1 The 986-Record Dataset in Orji & Ukwandu (2024)
- **Source:** Kaggle (*Medical Insurance Premium Prediction* by tejashvi14).
- **Total Records:** **986 rows** (no missing values, all numeric/integer encoded).
- **Target Variable ($y$):** `PremiumPrice` (Annual insurance premium in Indian Rupees / Currency Units, range: 15,000 to 40,000).
- **Input Features ($X$ - 10 Attributes):**
  1. `Age` (Continuous/Discrete: 18–66 years)
  2. `Diabetes` (Binary: 0 = No, 1 = Yes)
  3. `BloodPressureProblems` (Binary: 0 = No, 1 = Yes)
  4. `AnyTransplants` (Binary: 0 = No, 1 = Yes)
  5. `AnyChronicDiseases` (Binary: 0 = No, 1 = Yes)
  6. `Height` (Continuous: in cm, range: 145–188)
  7. `Weight` (Continuous: in kg, range: 51–132)
  8. `KnownAllergies` (Binary: 0 = No, 1 = Yes)
  9. `HistoryOfCancerInFamily` (Binary: 0 = No, 1 = Yes)
  10. `NumberOfMajorSurgeries` (Discrete: 0, 1, 2, 3)

### 3.2 Comparison with the 1,338-Record Kaggle Dataset (Brett Lantz / Miri Choi)
Many online tutorials and previous studies cite the 1,338-record dataset:
- Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.
- **Audit Verification:** The Orji & Ukwandu (2024) paper specifically uses the **986-record** dataset (`Medicalpremium.csv`), NOT the 1,338-record dataset.
- **Architectural Resolution:** To preserve strict reproduction fidelity while honoring broader actuarial literature, our project will:
  1. Set the **986-record dataset** as the **Primary Research Reproduction Track** (Orji & Ukwandu 2024: RF, GBM, XGBoost, SHAP, ICE).
  2. Maintain the **1,338-record dataset** as an **Extended Actuarial Benchmark Track** (8 models: Linear, Ridge, Lasso, RF, GBM, XGBoost, LightGBM, CatBoost) to allow multi-dataset generalization.

---

## 4. Models, Hyperparameters & Training Protocols

| Model | Paper Architecture & Rationale | Default / Tuned Parameter Strategy |
|---|---|---|
| **Random Forest Regressor** | Bagging ensemble of de-correlated decision trees; minimizes variance. | `n_estimators=100-200`, `max_depth=None/10`, `random_state=42`. |
| **Gradient Boosting Machine** | Sequential boosting optimizing residual loss stage-by-stage. | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `loss='squared_error'`. |
| **XGBoost Regressor** | Optimized distributed gradient boosting with 2nd-order Taylor expansions. | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.8`. |

### Validation Protocol
- **Split Ratio:** 80% Training ($N \approx 788$), 20% Testing ($N \approx 198$).
- **Cross-Validation:** 5-Fold Stratified / K-Fold Cross-Validation on training data to prevent data leakage.
- **Leakage Safeguard:** Scaling (MinMaxScaler or StandardScaler for Height/Weight/Age) fitted strictly on $X_{\text{train}}$.

---

## 5. Explainability Framework (XAI) in the Paper

1. **SHAP (Shapley Additive Explanations):**
   - Game-theoretic attribution: $\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} (f(S \cup \{i\}) - f(S))$
   - Global Beeswarm plot: Identifies top risk contributors across the population (`Age`, `AnyTransplants`, `Weight`, `AnyChronicDiseases`).
   - Local Waterfall / Force plot: Individual customer premium breakdown.
2. **ICE (Individual Conditional Expectation) Plots:**
   - Plots the functional relationship between the predicted premium and an individual input feature (e.g., `Age` or `Weight`) while keeping all other features constant for individual instances.
   - Revealing non-linear tipping points and localized interactions that global partial dependence plots (PDP) average out.

---

## 6. Implementation Decisions & Scientific Disclaimers

1. **Exact Reproduction vs. Extension:**
   - **Exact Reproduction:** Orji & Ukwandu (2024) pipeline on `Medicalpremium.csv` (986 rows) with RF, GBM, XGBoost, SHAP, and ICE.
   - **Extension:** Multi-dataset toggle, What-If counterfactual simulator, and 8-model actuarial comparison.
2. **No Data Fabrication:**
   - All metrics (MAE, RMSE, $R^2$, MAPE) are calculated from actual test set predictions, never hardcoded.
3. **Ethical Framing:**
   - Predictive outputs represent statistical model outputs; they do not imply direct medical causation or guaranteed insurance contract rates.
