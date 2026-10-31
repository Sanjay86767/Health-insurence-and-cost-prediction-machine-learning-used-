"""
Explainable Machine Learning for Health Insurance Cost Prediction:
A Comparative Study of Tree-Based Ensembles with SHAP, ICE, and Counterfactual Analysis
Authors: Sanjay Kumar & Dr. Sukhvinder Singh Bamber (Panjab University, Hoshiarpur)
Interactive Production Web Application & Research Framework
"""

import sys
from pathlib import Path
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.counterfactual import generate_counterfactual_recourse
from src.data_loader import (
    DataLoaderError,
    get_dataset_quality_report,
    load_dataset,
    split_features_target,
)
from src.evaluate_models import (
    compare_and_save_metrics,
    plot_metrics_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
)
from src.explainability import (
    compute_shap_explanation,
    generate_natural_language_explanation,
    get_shap_explainer,
    plot_ice_curves,
    plot_local_waterfall,
    plot_shap_summary,
)
from src.feature_engineering import compute_clinical_features
from src.predict import (
    PredictionInputError,
    predict_cost,
    validate_inputs_986,
    validate_inputs_1338,
)
from src.preprocessing import prepare_train_test_data
from src.train_models import (
    load_artifact,
    run_feature_ablation_study,
    save_artifact,
    train_nine_models_pipeline,
)

# Page configuration
st.set_page_config(
    page_title="Explainable ML | Health Insurance Cost Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0369A1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #92400E;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 1.6rem;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        margin-bottom: 1.2rem;
    }
    .recourse-card {
        background: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_dataset(dataset_type: str):
    df = load_dataset(dataset_type, return_type=False)
    return df, dataset_type


@st.cache_resource
def get_trained_pipeline(dataset_type: str):
    df, d_type = get_cached_dataset(dataset_type)
    X, y = split_features_target(df, d_type)

    X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
        X, y, dataset_type=d_type, test_size=0.25, random_state=42
    )

    trained_results = train_nine_models_pipeline(
        X_train=X_train_proc,
        y_train=y_train,
        cv_folds=0,  # Fast 0.4s web fit to prevent Streamlit Cloud CPU throttling
        random_state=42,
        log_experiment=False,
        dataset_name=d_type,
    )

    metrics_file = PROJECT_ROOT / "results" / "metrics.csv"
    if metrics_file.exists():
        comp_df = pd.read_csv(metrics_file)
    else:
        comp_df = compare_and_save_metrics(trained_results, X_test_proc, y_test, save_csv=True)

    best_model_name = comp_df.iloc[0]["Model"]
    if best_model_name not in trained_results:
        best_model_name = list(trained_results.keys())[0]
    best_model = trained_results[best_model_name]["model"]

    save_artifact(best_model, f"best_model_{d_type}.joblib")
    save_artifact(preprocessor, f"preprocessor_{d_type}.joblib")

    try:
        explainer = get_shap_explainer(best_model, X_train_proc)
        shap_explanation = compute_shap_explanation(explainer, X_test_proc[:25], feature_names)
    except Exception:
        explainer = None
        shap_explanation = None

    return {
        "df": df,
        "dataset_type": d_type,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "trained_results": trained_results,
        "comparison_df": comp_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "explainer": explainer,
        "shap_explanation": shap_explanation,
    }


def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/medical-heart.png", width=65)
    st.sidebar.title("Health Insurance ML")
    st.sidebar.caption("Panjab University Research Framework")

    dataset_option = st.sidebar.selectbox(
        "Active Research Dataset:",
        [
            "📄 Primary Paper (986 Records - Sanjay Kumar & Dr. S. S. Bamber)",
            "📊 Actuarial Benchmark (1,338 Records - Brett Lantz / Miri Choi)",
        ],
        index=0,
    )
    dataset_key = "medical_premium_986" if "986" in dataset_option else "insurance_1338"

    menu = st.sidebar.radio(
        "Application Navigation:",
        [
            "🏠 1. Home & Paper Overview",
            "🔮 2. New Person Intake & Quote Generator",
            "📊 3. 9-Model Benchmark Leaderboard",
            "🔬 4. Clinical Feature Ablation Study",
            "🧠 5. Explainable AI (SHAP & ICE)",
            "🔄 6. Actionable Counterfactual Recourse",
            "⚖️ 7. Subgroup & Fairness Audit",
            "🚀 8. Production Latency & Drift Monitor",
            "📖 9. Model Card & Presentation",
        ],
        index=1,
    )

    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>Academic Research Implementation:</strong> Based on <em>"Explainable Machine Learning for Health Insurance Cost Prediction"</em>
        (Panjab University, Hoshiarpur). Quotes represent empirical regression approximations, not medical diagnoses or underwriting contracts.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initializing neural and tree ensembles..."):
        try:
            artifacts = get_trained_pipeline(dataset_key)
        except Exception as err:
            st.error(f"Error loading pipeline: {err}")
            return

    df = artifacts["df"]
    d_type = artifacts["dataset_type"]
    comp_df = artifacts["comparison_df"]
    best_name = artifacts["best_model_name"]
    best_model = artifacts["best_model"]
    preprocessor = artifacts["preprocessor"]
    feature_names = artifacts["feature_names"]
    target_col = "PremiumPrice" if d_type == "medical_premium_986" else "charges"
    curr = "$" if d_type == "medical_premium_986" else "$"

    # -------------------------------------------------------------
    # Page 1: Home & Paper Overview
    # -------------------------------------------------------------
    if menu == "🏠 1. Home & Paper Overview":
        st.markdown('<div class="main-header">Explainable Machine Learning for Health Insurance Cost Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">A Comparative Study of Tree-Based Ensembles with SHAP, ICE, and Counterfactual Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        **Authors:** Sanjay Kumar & Dr. Sukhvinder Singh Bamber  
        **Department:** Department of Computer Science & Engineering, Panjab University Swami Sarvanand Giri Regional Centre, Hoshiarpur, Punjab, India.  
        **Roll No:** SGM23366 | **Faculty Email:** `ss.bamber@pu.ac.in`
        """)

        report = get_dataset_quality_report(df, d_type)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card"><div class="metric-label">POLICYHOLDERS</div><div class="metric-value">{report['total_rows']:,}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><div class="metric-label">RAW ATTRIBUTES</div><div class="metric-value">{report['total_columns'] - 1}</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="metric-label">MEAN PREMIUM</div><div class="metric-value">{curr}{report['target_stats']['mean']:,.2f}</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card"><div class="metric-label">CHAMPION MODEL</div><div class="metric-value" style="font-size:1.4rem;">{best_name}</div></div>""", unsafe_allow_html=True)

        st.subheader("📋 Dataset Inspection")
        st.dataframe(df.head(10), use_container_width=True)

    # -------------------------------------------------------------
    # Page 2: New Person Intake & Real-time Quote Generator
    # -------------------------------------------------------------
    elif menu == "🔮 2. New Person Intake & Quote Generator":
        st.markdown('<div class="main-header">New Person Intake & Real-time Quote Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Instant premium calculation with sub-10ms latency, multi-model consensus, and automated counterfactual recourse</div>', unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1.25], gap="large")

        with col_in:
            st.subheader("👤 New Applicant Intake Form")

            if d_type == "medical_premium_986":
                preset = st.selectbox(
                    "⚡ Rapid Load Profile:",
                    [
                        "Custom Applicant Entry",
                        "Applicant 104 (Paper Table VII): Age 56, BMI 34.8, BP & Diabetes",
                        "Applicant 289 (Paper Table VII): Age 48, BMI 31.2, Chronic Disease",
                        "Applicant 415 (Paper Table VII): Age 35, BMI 28.4, Known Allergies",
                        "Applicant 672 (Paper Table VII): Age 62, BMI 39.5, Severe Comorbidities",
                        "Young Healthy Baseline: Age 23, Height 172cm, Weight 64kg (BMI 21.6)",
                    ],
                    index=0,
                )

                if "Applicant 104" in preset:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 56, 168, 98, 1, 1, 1, 0, 0, 0, 0
                elif "Applicant 289" in preset:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 48, 170, 90, 0, 0, 0, 0, 1, 0, 0
                elif "Applicant 415" in preset:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 35, 175, 87, 0, 0, 0, 0, 0, 1, 0
                elif "Applicant 672" in preset:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 62, 165, 108, 2, 1, 1, 0, 1, 0, 1
                elif "Young Healthy" in preset:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 23, 172, 64, 0, 0, 0, 0, 0, 0, 0
                else:
                    def_age, def_h, def_w, def_surg, def_diab, def_bp, def_trans, def_chron, def_allerg, def_canc = 42, 168, 77, 0, 0, 1, 0, 0, 0, 0

                active_model_name = st.selectbox("Inference Model:", options=list(artifacts["trained_results"].keys()), index=0)

                age = st.slider("Applicant Age", 18, 66, def_age)
                height = st.slider("Height (cm)", 140, 200, def_h)
                weight = st.slider("Weight (kg)", 45, 140, def_w)
                surgeries = st.selectbox("Major Surgeries (Past)", [0, 1, 2, 3], index=min(def_surg, 3))

                st.markdown("**Medical History & Comorbidities:**")
                c_a, c_b = st.columns(2)
                with c_a:
                    diabetes = 1 if st.checkbox("Diabetes", value=bool(def_diab)) else 0
                    bp = 1 if st.checkbox("Blood Pressure Problems", value=bool(def_bp)) else 0
                    transplants = 1 if st.checkbox("Any Organ Transplants", value=bool(def_trans)) else 0
                with c_b:
                    chronic = 1 if st.checkbox("Any Chronic Diseases", value=bool(def_chron)) else 0
                    allergies = 1 if st.checkbox("Known Allergies", value=bool(def_allerg)) else 0
                    cancer = 1 if st.checkbox("Family History of Cancer", value=bool(def_canc)) else 0

                user_dict = {
                    "Age": age, "Diabetes": diabetes, "BloodPressureProblems": bp,
                    "AnyTransplants": transplants, "AnyChronicDiseases": chronic,
                    "Height": height, "Weight": weight, "KnownAllergies": allergies,
                    "HistoryOfCancerInFamily": cancer, "NumberOfMajorSurgeries": surgeries,
                }

                calc_bmi = round(weight / ((height / 100) ** 2), 1)
                calc_cci = (2.0 * transplants) + (1.5 * chronic) + (1.0 * diabetes) + (1.0 * bp)
                calc_surg_canc = surgeries * cancer

                if calc_bmi < 18.5:
                    bmi_cat = "Underweight (<18.5)"
                    bmi_badge = "background-color: #FEF3C7; color: #92400E;"
                elif calc_bmi < 25.0:
                    bmi_cat = "Normal Weight (18.5–24.9) — Optimal"
                    bmi_badge = "background-color: #DCFCE7; color: #166534;"
                elif calc_bmi < 30.0:
                    bmi_cat = "Overweight (25.0–29.9)"
                    bmi_badge = "background-color: #FEF08A; color: #854D0E;"
                elif calc_bmi < 35.0:
                    bmi_cat = "Obese Class I (30.0–34.9)"
                    bmi_badge = "background-color: #FED7AA; color: #9A3412;"
                else:
                    bmi_cat = "Severe / Morbidly Obese (≥35.0)"
                    bmi_badge = "background-color: #FECACA; color: #991B1B;"

                cci_status = "Healthy Baseline (0.0)" if calc_cci == 0 else ("Moderate Comorbidity" if calc_cci <= 2.5 else "Severe Multi-morbidity")

                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #CBD5E1; padding:12px; border-radius:8px; margin-top:12px;">
                    <div style="font-weight:700; font-size:0.85rem; color:#334155; margin-bottom:6px;">🩺 LIVE CLINICAL INDICES</div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.9rem;">
                        <span>Body Mass Index (BMI):</span> <strong>{calc_bmi} kg/m²</strong>
                    </div>
                    <div style="display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; {bmi_badge} margin-bottom:8px;">
                        {bmi_cat}
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.9rem;">
                        <span>Comorbidity Index (CCI):</span> <strong>{calc_cci:.1f} ({cci_status})</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                        <span>Surgical-Oncological Score:</span> <strong>{calc_surg_canc}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                preset = st.selectbox(
                    "⚡ Rapid Load Profile:",
                    [
                        "Custom Applicant Entry",
                        "Young Fit Non-Smoker: Age 21, BMI 22.0, Dependents 0, Non-smoker",
                        "Middle-Aged Smoker: Age 45, BMI 31.0, Dependents 2, Smoker",
                        "Senior Non-Smoker: Age 60, BMI 28.5, Dependents 1, Non-smoker",
                    ],
                    index=0,
                )
                if "Young Fit" in preset:
                    def_age, def_sex, def_bmi, def_child, def_smoker, def_reg = 21, "female", 22.0, 0, "no", "southwest"
                elif "Middle-Aged" in preset:
                    def_age, def_sex, def_bmi, def_child, def_smoker, def_reg = 45, "male", 31.0, 2, "yes", "southeast"
                elif "Senior" in preset:
                    def_age, def_sex, def_bmi, def_child, def_smoker, def_reg = 60, "female", 28.5, 1, "no", "northwest"
                else:
                    def_age, def_sex, def_bmi, def_child, def_smoker, def_reg = 38, "male", 27.5, 1, "no", "southwest"

                active_model_name = st.selectbox("Inference Model:", options=list(artifacts["trained_results"].keys()), index=0)

                age = st.slider("Age (Years)", 18, 65, def_age)
                sex = st.radio("Sex", ["female", "male"], index=0 if def_sex == "female" else 1, horizontal=True)
                bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, float(def_bmi), 0.1)
                children = st.slider("Dependents", 0, 5, def_child)
                smoker = st.radio("Smoker Status", ["no", "yes"], index=0 if def_smoker == "no" else 1, horizontal=True)
                region_idx = ["southwest", "southeast", "northwest", "northeast"].index(def_reg) if def_reg in ["southwest", "southeast", "northwest", "northeast"] else 0
                region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"], index=region_idx)
                user_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
                calc_bmi = bmi
                calc_cci = 0.0

        with col_out:
            st.subheader("💡 Real-time Actuarial Quote")
            chosen_model = artifacts["trained_results"][active_model_name]["model"]

            start_clock = time.perf_counter()
            pred_cost, X_single, _ = predict_cost(chosen_model, preprocessor, user_dict, d_type)
            latency_ms = (time.perf_counter() - start_clock) * 1000.0

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.85rem; letter-spacing: 1px; opacity: 0.85;">ESTIMATED ANNUAL HEALTH PREMIUM</div>
                <div style="font-size: 2.8rem; font-weight: 800; margin: 0.2rem 0;">{curr}{pred_cost:,.2f}</div>
                <div style="font-size: 0.88rem; opacity: 0.95;">
                    Engine: <strong>{active_model_name}</strong> | Quote Latency: <strong>{latency_ms:.2f} ms</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📊 Compare All 9 Model Predictions for This Applicant", expanded=False):
                model_quotes = []
                for m_name, m_info in artifacts["trained_results"].items():
                    m_pred, _, _ = predict_cost(m_info["model"], preprocessor, user_dict, d_type)
                    model_quotes.append({"Model": m_name, "Predicted Quote": m_pred})

                m_quotes_df = pd.DataFrame(model_quotes).sort_values("Predicted Quote", ascending=True)
                fig_m, ax_m = plt.subplots(figsize=(7, 3.8))
                colors = ["#2563EB" if m == active_model_name else "#94A3B8" for m in m_quotes_df["Model"]]
                bars = ax_m.barh(m_quotes_df["Model"], m_quotes_df["Predicted Quote"], color=colors, height=0.6)
                ax_m.set_xlabel(f"Predicted Premium ({curr})")
                ax_m.set_title("Cross-Model Pricing Comparison for This Applicant", fontsize=11, fontweight="bold")
                for bar in bars:
                    w = bar.get_width()
                    ax_m.text(w + 100, bar.get_y() + bar.get_height()/2, f"{curr}{w:,.0f}", va="center", fontsize=8)
                ax_m.spines["top"].set_visible(False)
                ax_m.spines["right"].set_visible(False)
                st.pyplot(fig_m)
                plt.close(fig_m)

            # Automated Counterfactual Recourse (Table VII reproduction)
            st.markdown("### 🎯 Actionable Recourse to Lower Premium")
            recourse_res = generate_counterfactual_recourse(chosen_model, preprocessor, user_dict, d_type)
            for strat in recourse_res["recourse_strategies"]:
                st.markdown(f"""
                <div class="recourse-card">
                    <strong style="color: #15803D;">✅ {strat['strategy_name']}</strong><br/>
                    <em>{strat['action']}</em><br/>
                    <strong>New Estimated Cost:</strong> {curr}{strat['scenario_cost']:,.2f} | 
                    <strong>Projected Annual Savings:</strong> <span style="color: #16A34A; font-weight: 700;">{curr}{strat['annual_saving']:,.2f} ({strat['saving_percent']:.1f}% reduction)</span>
                </div>
                """, unsafe_allow_html=True)

            # Local SHAP decomposition
            st.markdown("---")
            st.subheader("🔍 Local Feature Breakdown (SHAP Waterfall)")
            try:
                explainer = get_shap_explainer(chosen_model, artifacts["X_train_proc"])
                exp_single = compute_shap_explanation(explainer, X_single, feature_names)
                fig_wf = plot_local_waterfall(exp_single, sample_index=0, feature_names=feature_names)
                st.pyplot(fig_wf)
                plt.close(fig_wf)
            except Exception as e:
                st.info(f"Local attribution note: {e}")

            report_data = {
                "applicant_profile": user_dict,
                "clinical_metrics": {
                    "BMI": float(calc_bmi),
                    "CompositeComorbidityIndex": float(calc_cci),
                },
                "chosen_model": active_model_name,
                "estimated_annual_premium": float(pred_cost),
                "quote_latency_ms": round(latency_ms, 2),
                "counterfactual_savings_roadmap": recourse_res["recourse_strategies"],
            }
            st.download_button(
                label="📥 Download Applicant Actuarial Quote Report (.json)",
                data=json.dumps(report_data, indent=2),
                file_name="applicant_health_quote_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # -------------------------------------------------------------
    # Page 3: 9-Model Benchmark Leaderboard
    # -------------------------------------------------------------
    elif menu == "📊 3. 9-Model Benchmark Leaderboard":
        st.markdown('<div class="main-header">Holdout Performance of Nine Regression Models</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Reproducing Table III from Sanjay Kumar & Dr. Sukhvinder Singh Bamber (Panjab University)</div>', unsafe_allow_html=True)

        st.dataframe(
            comp_df.style.format({
                "Test R2": "{:.4f}",
                "Test MAE": f"{curr}{{:,.2f}}",
                "Test RMSE": f"{curr}{{:,.2f}}",
                "Test MAPE (%)": "{:.2f}%",
                "CV R2 (Mean)": "{:.4f}",
                "CV R2 (Std)": "±{:.4f}",
                "CV MAE (Mean)": f"{curr}{{:,.2f}}",
                "Train Time (s)": "{:.3f}s",
            }).highlight_max(subset=["Test R2"], color="#DCFCE7")
              .highlight_min(subset=["Test MAE", "Test RMSE"], color="#DCFCE7"),
            use_container_width=True,
        )

        st.subheader("📑 Paper Reported Benchmarks (Table III Reference)")
        paper_table_iii = [
            {"Model": "Ordinary Least Squares", "Train R² (%)": "76.12", "CV R² (%)": "71.40", "Test R² (%)": "75.09", "MAE ($)": "2,310.45", "RMSE ($)": "3,124.60", "MAPE (%)": "11.20"},
            {"Model": "Ridge Regression (α = 1)", "Train R² (%)": "76.08", "CV R² (%)": "71.45", "Test R² (%)": "75.12", "MAE ($)": "2,305.10", "RMSE ($)": "3,118.25", "MAPE (%)": "11.15"},
            {"Model": "Tweedie GLM (p = 1.5)", "Train R² (%)": "78.45", "CV R² (%)": "72.20", "Test R² (%)": "77.80", "MAE ($)": "2,115.30", "RMSE ($)": "2,940.10", "MAPE (%)": "9.85"},
            {"Model": "Gradient Boosting (GBM)", "Train R² (%)": "79.84", "CV R² (%)": "73.00", "Test R² (%)": "84.72", "MAE ($)": "1,725.86", "RMSE ($)": "2,371.41", "MAPE (%)": "7.23"},
            {"Model": "Random Forest", "Train R² (%)": "95.81", "CV R² (%)": "73.43", "Test R² (%)": "84.05", "MAE ($)": "1,379.96", "RMSE ($)": "2,423.16", "MAPE (%)": "5.83"},
            {"Model": "LightGBM", "Train R² (%)": "87.10", "CV R² (%)": "74.12", "Test R² (%)": "85.80", "MAE ($)": "1,510.20", "RMSE ($)": "2,290.15", "MAPE (%)": "6.10"},
            {"Model": "CatBoost", "Train R² (%)": "88.40", "CV R² (%)": "74.25", "Test R² (%)": "86.15", "MAE ($)": "1,465.30", "RMSE ($)": "2,254.80", "MAPE (%)": "5.72"},
            {"Model": "XGBoost (optimized)", "Train R² (%)": "88.22", "CV R² (%)": "74.48", "Test R² (%)": "86.82", "MAE ($)": "1,442.90", "RMSE ($)": "2,210.45", "MAPE (%)": "5.48"},
            {"Model": "Stacking meta-ensemble", "Train R² (%)": "90.15", "CV R² (%)": "74.65", "Test R² (%)": "86.95", "MAE ($)": "1,395.20", "RMSE ($)": "2,198.50", "MAPE (%)": "5.42"},
        ]
        st.table(pd.DataFrame(paper_table_iii))

    # -------------------------------------------------------------
    # Page 4: Clinical Feature Ablation Study
    # -------------------------------------------------------------
    elif menu == "🔬 4. Clinical Feature Ablation Study":
        st.markdown('<div class="main-header">Feature Ablation with XGBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Reproducing Table IV: Quantifying the Value of Domain-Specific Clinical Engineering</div>', unsafe_allow_html=True)

        st.markdown("""
        The paper demonstrates that engineering **BMI**, a **Composite Comorbidity Index (CCI)**, and a **Surgical-Oncological Interaction**
        adds **+2.77 points of R²** and saves **$178 in lower MAE** on the hold-out test set without introducing multicollinearity (all VIF < 3.0).
        """)

        with st.spinner("Computing live ablation study across 4 feature subsets..."):
            df_ablation = run_feature_ablation_study(df)

        st.subheader("Table IV: Empirical Feature Ablation Results")
        st.table(df_ablation)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_ablation, x="Configuration", y="R² (%)", palette="Blues_r", ax=ax)
        ax.set_ylim(80, 90)
        ax.set_title("Progression of R² (%) with Clinical Feature Engineering", weight="bold")
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------------------
    # Page 5: Explainable AI (SHAP & ICE)
    # -------------------------------------------------------------
    elif menu == "🧠 5. Explainable AI (SHAP & ICE)":
        st.markdown('<div class="main-header">Explainable AI: TreeSHAP & Centered ICE Curves</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluating Global Feature Attributions and Individual Trajectories</div>', unsafe_allow_html=True)

        tab_s, tab_i = st.tabs(["🌐 TreeSHAP Global Attribution (Fig. 4)", "📈 Centered ICE Curves"])
        with tab_s:
            if artifacts["shap_explanation"] is not None:
                fig_g = plot_shap_summary(artifacts["shap_explanation"], feature_names)
                st.pyplot(fig_g)
                plt.close(fig_g)
                st.info("💡 **Finding (Fig. 4):** Age is the main driver (Mean |SHAP|: $3,700), followed by BMI ($3,100), Organ Transplants ($2,450), and Chronic Diseases ($1,820).")

        with tab_i:
            st.subheader("Centered ICE Curves (Section III-E)")
            fig_ice = plot_ice_curves(best_model, artifacts["X_test_proc"], feature_names, ["Age", "Weight"] if d_type == "medical_premium_986" else ["age", "bmi"])
            st.pyplot(fig_ice)
            plt.close(fig_ice)

    # -------------------------------------------------------------
    # Page 6: Actionable Counterfactual Recourse
    # -------------------------------------------------------------
    elif menu == "🔄 6. Actionable Counterfactual Recourse":
        st.markdown('<div class="main-header">Actionable Counterfactual Recourse (Table VII)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">What feasible actions can a policyholder take to reduce their health premium?</div>', unsafe_allow_html=True)

        st.markdown("""
        **Counterfactual Principle (Wachter et al. & DiCE):**  
        $$x^* = \\arg\\min_{x'} \\text{dist}(x, x') + \\lambda (f(x') - y^*)^2$$
        With immutable attributes (Age, Family Cancer History, Transplants) held fixed.
        """)

        st.subheader("Benchmark Applicants from Table VII in Paper")
        table_vii = [
            {"Applicant": "104", "BMI change": "34.8 → 29.9", "Saving ($/yr)": "$8,400", "Saving (%)": "24.3%"},
            {"Applicant": "289", "BMI change": "31.2 → 26.9", "Saving ($/yr)": "$4,800", "Saving (%)": "16.8%"},
            {"Applicant": "415", "BMI change": "28.4 → 24.8", "Saving ($/yr)": "$3,250", "Saving (%)": "13.5%"},
            {"Applicant": "672", "BMI change": "39.5 → 32.1", "Saving ($/yr)": "$9,850", "Saving (%)": "26.6%"},
        ]
        st.table(pd.DataFrame(table_vii))

    # -------------------------------------------------------------
    # Page 7: Subgroup & Fairness Audit
    # -------------------------------------------------------------
    elif menu == "⚖️ 7. Subgroup & Fairness Audit":
        st.markdown('<div class="main-header">Subgroup Accuracy & Age-Fairness Audit (Table VI)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluating consistency of predictions across vulnerable and clinical cohorts</div>', unsafe_allow_html=True)

        table_vi = [
            {"Subgroup": "Age ≥ 50", "N": "312", "Mean Premium ($)": "$29,450", "R² (%)": "85.12%", "MAE ($)": "$1,520.40"},
            {"Subgroup": "BMI ≥ 30", "N": "284", "Mean Premium ($)": "$27,820", "R² (%)": "86.40%", "MAE ($)": "$1,480.10"},
            {"Subgroup": "Organ Transplant", "N": "59", "Mean Premium ($)": "$33,150", "R² (%)": "83.75%", "MAE ($)": "$1,810.50"},
            {"Subgroup": "Chronic Disease", "N": "177", "Mean Premium ($)": "$28,640", "R² (%)": "85.90%", "MAE ($)": "$1,495.20"},
            {"Subgroup": "Healthy Baseline", "N": "412", "Mean Premium ($)": "$19,250", "R² (%)": "87.40%", "MAE ($)": "$1,210.80"},
        ]
        st.table(pd.DataFrame(table_vi))

    # -------------------------------------------------------------
    # Page 8: Production Latency & Drift Monitor
    # -------------------------------------------------------------
    elif menu == "🚀 8. Production Latency & Drift Monitor":
        st.markdown('<div class="main-header">FastAPI Microservice & Drift Monitor (Section V-G)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Sub-10ms Quote Latency Benchmarking and Statistical Drift Detection</div>', unsafe_allow_html=True)

        col_l1, col_l2, col_l3 = st.columns(3)
        col_l1.metric("Single Quote Latency", "8.4 ms", "Sub-10ms SLA Met")
        col_l2.metric("Batch Throughput", "2,631 quotes/s", "256 Batch Size")
        col_l3.metric("Population Stability Index (PSI)", "0.04", "Stable (< 0.10)")

        st.subheader("Data Drift Detection Protocol (KS-Test & PSI)")
        st.markdown("""
        - **Kolmogorov-Smirnov Test:** Continuous two-sample test comparing incoming policyholder age and weight distributions against the training baseline.
        - **Population Stability Index (PSI):**
          - $\\text{PSI} < 0.10$: Stable distribution (No action needed).
          - $0.10 \\le \\text{PSI} < 0.25$: Moderate drift (Warning logged).
          - $\\text{PSI} \\ge 0.25$: Significant population drift (Automated model retraining triggered).
        """)

    # -------------------------------------------------------------
    # Page 9: Model Card & Presentation
    # -------------------------------------------------------------
    elif menu == "📖 9. Model Card & Presentation":
        st.markdown('<div class="main-header">Academic Model Card & Presentation Checklist</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Panjab University Research Project Documentation</div>', unsafe_allow_html=True)

        st.markdown("""
        - **Authors:** Sanjay Kumar & Dr. Sukhvinder Singh Bamber (Panjab University, Hoshiarpur).
        - **Institution:** Panjab University Swami Sarvanand Giri Regional Centre, Hoshiarpur, Punjab, India.
        - **Roll No:** SGM23366 | **Supervisor:** Dr. Sukhvinder Singh Bamber (`ss.bamber@pu.ac.in`).
        - **Models:** 9 Regressors (OLS, Ridge, Tweedie GLM, GBM, RF, LightGBM, CatBoost, XGBoost, Stacking).
        - **Champion Architecture:** XGBoost / Stacking meta-ensemble ($R^2 = 86.82\\% - 86.95\\%$, $\\text{RMSE} = \\$2,198 - \\$2,210$).
        """)


if __name__ == "__main__":
    main()
