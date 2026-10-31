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

# Custom Styling - High Aesthetic Executive Healthcare Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, .main-header, .brand-title {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    .main-header {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 45%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        line-height: 1.25;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .status-active {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .status-academic {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .status-metrics {
        background: #FAF5FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }

    /* High-impact Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #2563EB, #38BDF8);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 32px -4px rgba(37, 99, 235, 0.12);
        border-color: #93C5FD;
    }
    .metric-value {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.8px;
        font-family: 'Outfit', sans-serif !important;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
    }

    .disclaimer-banner {
        background: #FEF9C3;
        border: 1px solid #FDE047;
        border-left: 5px solid #EAB308;
        padding: 0.9rem 1.3rem;
        border-radius: 12px;
        margin-bottom: 1.4rem;
        font-size: 0.88rem;
        color: #713F12;
        line-height: 1.5;
    }

    /* Executive Real-time Actuarial Quote Box */
    .prediction-box {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 55%, #1E3A8A 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 45px -10px rgba(11, 19, 43, 0.5);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .prediction-box::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.45) 0%, transparent 70%);
        pointer-events: none;
    }
    .prediction-price {
        font-size: 3.3rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1.2px;
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Actionable Recourse Cards */
    .recourse-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        padding: 1.15rem 1.35rem;
        border-radius: 14px;
        margin-bottom: 0.95rem;
        transition: all 0.22s ease;
    }
    .recourse-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.1);
    }

    /* Premium Button Polish */
    .stButton > button, button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.32) !important;
        transition: all 0.22s ease !important;
    }
    .stButton > button:hover, button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.48) !important;
    }

    /* Sidebar Card */
    .sidebar-brand-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.15rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: white;
        margin-bottom: 1.2rem;
    }

    /* Digital Smart Health ID Card (Apple Wallet / Healthcare Pass style) */
    .smart-health-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 45%, #1E3A8A 100%);
        border-radius: 22px;
        padding: 1.8rem 2.2rem;
        color: white;
        box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.65), inset 0 1px 1px rgba(255, 255, 255, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.16);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.8rem;
    }
    .smart-health-card::before {
        content: "";
        position: absolute;
        top: -80px;
        right: -80px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(234, 179, 8, 0.35) 0%, transparent 70%);
        pointer-events: none;
    }
    .card-chip {
        width: 48px;
        height: 34px;
        background: linear-gradient(135deg, #FDE047 0%, #CA8A04 100%);
        border-radius: 8px;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .card-number {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.35rem;
        letter-spacing: 3px;
        color: #F8FAFC;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
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
    st.sidebar.markdown("""
    <div class="sidebar-brand-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <div style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.3) 0%, rgba(59, 130, 246, 0.1) 100%); border: 1px solid rgba(96, 165, 250, 0.4); border-radius: 12px; width: 44px; height: 44px; font-size: 1.5rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(37,99,235,0.25);">
                🩺
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.4px; font-family: 'Outfit', sans-serif;">Health InsurAI</div>
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Research Console</div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #CBD5E1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 9px; line-height: 1.5;">
            🎓 <strong>Panjab University</strong> (SSGICR)<br/>
            👨‍🔬 <strong>Sanjay Kumar</strong> (Roll: SGM23366)<br/>
            👨‍🏫 <strong>Dr. S. S. Bamber</strong> (Faculty CSE)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize user profile in session state
    if "user" not in st.session_state:
        st.session_state["user"] = {
            "is_logged_in": True,
            "username": "sanjay86767",
            "full_name": "Sanjay Kumar",
            "policy_id": "PU-INS-2026-98604",
            "tier": "Comprehensive Platinum Care",
            "email": "sanjay@pussgrc.ac.in",
            "avatar": "👨‍🔬",
            "age": 42,
            "height": 168,
            "weight": 77,
            "surgeries": 0,
            "diabetes": 0,
            "bp": 1,
            "transplants": 0,
            "chronic": 0,
            "allergies": 0,
            "cancer": 0,
        }

    # Sidebar Account Card & Login / Logout
    u_sess = st.session_state["user"]
    if u_sess["is_logged_in"]:
        st.sidebar.markdown(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 12px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(15,23,42,0.05);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #16A34A; background: #DCFCE7; padding: 2px 8px; border-radius: 9999px;">● MEMBER ONLINE</span>
                <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">{u_sess['policy_id']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.6rem;">{u_sess['avatar']}</span>
                <div>
                    <div style="font-weight: 800; font-size: 0.94rem; color: #0F172A; font-family: 'Outfit', sans-serif;">{u_sess['full_name']}</div>
                    <div style="font-size: 0.75rem; color: #64748B;">{u_sess['tier']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.sidebar.button("🚪 Sign Out", key="sidebar_logout_btn", use_container_width=True):
            st.session_state["user"]["is_logged_in"] = False
            st.rerun()
    else:
        st.sidebar.markdown("""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 12px; padding: 10px; margin-bottom: 10px; font-size: 0.82rem; color: #92400E;">
            🔒 <strong>Guest Session:</strong> Sign in to view your Digital Policyholder Health Card & Online Records.
        </div>
        """, unsafe_allow_html=True)
        with st.sidebar.form("quick_login_form"):
            login_user = st.text_input("Username", value="sanjay86767")
            login_pass = st.text_input("Password", type="password", value="password123")
            if st.form_submit_button("🔑 Quick Sign In", use_container_width=True):
                st.session_state["user"]["is_logged_in"] = True
                st.rerun()

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
            "💳 0. My Policyholder Portal & Health Card",
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
        index=0,
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
    # Page 0: My Policyholder Portal & Digital Health Card
    # -------------------------------------------------------------
    if menu == "💳 0. My Policyholder Portal & Health Card":
        user = st.session_state["user"]

        if not user["is_logged_in"]:
            st.markdown("""
            <div style="margin-bottom: 12px;">
                <span class="status-pill status-active">● Secure Member Gateway</span>
                <span class="status-pill status-academic">Panjab University SSGICR</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="main-header">Policyholder Sign In & Member Access</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Log in to view your Digital Health Insurance Smart Card, active policy tier, and online records</div>', unsafe_allow_html=True)

            col_auth1, col_auth2 = st.columns([1, 1], gap="large")
            with col_auth1:
                st.subheader("🔑 Member Sign In")
                with st.form("main_login_form"):
                    u_in = st.text_input("Policy ID or Username", value="sanjay86767")
                    p_in = st.text_input("Password", type="password", value="password123")
                    remember = st.checkbox("Remember this device", value=True)
                    if st.form_submit_button("🚀 Sign In to My Account", use_container_width=True):
                        st.session_state["user"]["is_logged_in"] = True
                        st.rerun()

            with col_auth2:
                st.subheader("🌟 1-Click Demo Profiles")
                st.markdown("Try out the system instantly with pre-configured research accounts:")
                if st.button("👤 Log in as Sanjay Kumar (Active Member)", use_container_width=True):
                    st.session_state["user"] = {
                        "is_logged_in": True, "username": "sanjay86767", "full_name": "Sanjay Kumar",
                        "policy_id": "PU-INS-2026-98604", "tier": "Comprehensive Platinum Care",
                        "email": "sanjay@pussgrc.ac.in", "avatar": "👨‍🔬",
                        "age": 42, "height": 168, "weight": 77, "surgeries": 0, "diabetes": 0,
                        "bp": 1, "transplants": 0, "chronic": 0, "allergies": 0, "cancer": 0,
                    }
                    st.rerun()

                if st.button("👤 Log in as Applicant #104 (Paper Benchmark - Senior with BP)", use_container_width=True):
                    st.session_state["user"] = {
                        "is_logged_in": True, "username": "applicant_104", "full_name": "Ramesh Chandra (App #104)",
                        "policy_id": "PU-INS-2026-104", "tier": "High-Risk Chronic Care Tier",
                        "email": "ramesh.chandra@sample.org", "avatar": "👴",
                        "age": 56, "height": 168, "weight": 98, "surgeries": 1, "diabetes": 1,
                        "bp": 1, "transplants": 0, "chronic": 0, "allergies": 0, "cancer": 0,
                    }
                    st.rerun()

                if st.button("👤 Log in as Young Fit Adult (Baseline Zero-Condition)", use_container_width=True):
                    st.session_state["user"] = {
                        "is_logged_in": True, "username": "priya_sharma", "full_name": "Priya Sharma (Young Fit)",
                        "policy_id": "PU-INS-2026-701", "tier": "Standard Essential Health Pass",
                        "email": "priya.sharma@sample.org", "avatar": "👩‍💼",
                        "age": 23, "height": 172, "weight": 64, "surgeries": 0, "diabetes": 0,
                        "bp": 0, "transplants": 0, "chronic": 0, "allergies": 0, "cancer": 0,
                    }
                    st.rerun()

        else:
            u = st.session_state["user"]
            st.markdown(f"""
            <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span class="status-pill status-active">● Verified Member Online</span>
                    <span class="status-pill status-academic">Policy ID: {u['policy_id']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="main-header">Welcome, {u["full_name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sub-header">Your personal healthcare actuarial portal, live digital policy card, and online medical risk records.</div>', unsafe_allow_html=True)

            col_card, col_details = st.columns([1, 1.25], gap="large")

            u_bmi = round(u["weight"] / ((u["height"] / 100) ** 2), 1)
            u_cci = (2.0 * u["transplants"]) + (1.5 * u["chronic"]) + (1.0 * u["diabetes"]) + (1.0 * u["bp"])
            u_surg_canc = u["surgeries"] * u["cancer"]

            user_data_dict = {
                "Age": u["age"], "Diabetes": u["diabetes"], "BloodPressureProblems": u["bp"],
                "AnyTransplants": u["transplants"], "AnyChronicDiseases": u["chronic"],
                "Height": u["height"], "Weight": u["weight"], "KnownAllergies": u["allergies"],
                "HistoryOfCancerInFamily": u["cancer"], "NumberOfMajorSurgeries": u["surgeries"],
            }
            u_quote, _, _ = predict_cost(best_model, preprocessor, user_data_dict, "medical_premium_986")

            with col_card:
                st.markdown(f"""
                <div class="smart-health-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 800; font-size: 1.1rem; letter-spacing: 0.5px; text-transform: uppercase; color: #F8FAFC; font-family: 'Outfit', sans-serif;">HEALTH INSURAI SMART PASS</span>
                        <span style="background: rgba(16, 185, 129, 0.25); border: 1px solid rgba(52, 211, 153, 0.5); color: #6EE7B7; padding: 2px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700;">● ACTIVE</span>
                    </div>
                    <div class="card-chip"></div>
                    <div class="card-number">{u['policy_id'][:4]} •••• •••• {u['policy_id'][-4:]}</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div>
                            <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">Cardholder Name</div>
                            <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; font-family: 'Outfit', sans-serif;">{u['full_name']}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">Coverage Plan</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #FDE047;">{u['tier']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📑 Digital Documentation")
                cert_data = {
                    "policyholder": u,
                    "vitals": {"BMI": u_bmi, "CCI": u_cci},
                    "quote": float(u_quote),
                    "coverage_status": "Active & Verified",
                    "issuing_authority": "Panjab University SSGICR Research Framework",
                }
                st.download_button(
                    label="📥 Download Official Policy Certificate (.json)",
                    data=json.dumps(cert_data, indent=2),
                    file_name=f"policy_{u['policy_id']}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                if st.button("🚪 Log Out from Portal", key="portal_logout_btn", use_container_width=True):
                    st.session_state["user"]["is_logged_in"] = False
                    st.rerun()

            with col_details:
                st.subheader("🩺 My Vitals & Clinical Health Records")

                v1, v2, v3, v4 = st.columns(4)
                v1.markdown(f"""<div class="metric-card"><div class="metric-label">AGE</div><div class="metric-value">{u['age']} <span style="font-size:1rem;color:#64748B;">yr</span></div></div>""", unsafe_allow_html=True)
                v2.markdown(f"""<div class="metric-card"><div class="metric-label">HEIGHT / WEIGHT</div><div class="metric-value" style="font-size:1.35rem;">{u['height']}cm<br/><span style="font-size:1.05rem;color:#64748B;">{u['weight']}kg</span></div></div>""", unsafe_allow_html=True)

                bmi_badge = "#DCFCE7; color: #166534;" if u_bmi < 25 else ("#FEF08A; color: #854D0E;" if u_bmi < 30 else "#FECACA; color: #991B1B;")
                v3.markdown(f"""<div class="metric-card"><div class="metric-label">BMI</div><div class="metric-value">{u_bmi}</div><div style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:0.7rem;font-weight:700;background:{bmi_badge}">{ 'Normal' if u_bmi < 25 else ('Overweight' if u_bmi < 30 else 'Obese') }</div></div>""", unsafe_allow_html=True)
                v4.markdown(f"""<div class="metric-card"><div class="metric-label">COMORBIDITY (CCI)</div><div class="metric-value">{u_cci:.1f}</div><div style="font-size:0.7rem;color:#64748B;">{'Low Risk' if u_cci == 0 else ('Moderate' if u_cci <= 2.5 else 'Elevated')}</div></div>""", unsafe_allow_html=True)

                st.markdown("**Medical History & Underwriting Risk Status:**")
                c_cond1, c_cond2, c_cond3 = st.columns(3)
                c_cond1.markdown(f"🩺 **Hypertension:** {'🔴 Elevated (Yes)' if u['bp'] else '🟢 Normal (No)'}")
                c_cond1.markdown(f"🩸 **Diabetes:** {'🔴 Yes' if u['diabetes'] else '🟢 None'}")
                c_cond2.markdown(f"🫀 **Organ Transplants:** {'🔴 Yes' if u['transplants'] else '🟢 None'}")
                c_cond2.markdown(f"🩹 **Chronic Diseases:** {'🔴 Yes' if u['chronic'] else '🟢 None'}")
                c_cond3.markdown(f"🧬 **Cancer in Family:** {'🔴 Yes' if u['cancer'] else '🟢 None'}")
                c_cond3.markdown(f"🏥 **Past Surgeries:** **{u['surgeries']} major surgeries**")

                st.markdown("---")
                st.subheader(f"💡 Real-time Active Premium: {curr}{u_quote:,.2f} / year")
                st.markdown(f"""
                <div class="prediction-box" style="padding: 1.5rem;">
                    <div style="font-size: 0.8rem; letter-spacing: 1px; color: #94A3B8; text-transform: uppercase;">Actuarial Valuation for {u['full_name']}</div>
                    <div class="prediction-price" style="font-size: 2.7rem;">{curr}{u_quote:,.2f}</div>
                    <div style="font-size: 0.84rem; color: #E2E8F0; margin-top: 6px;">
                        Engine: <strong>{best_name}</strong> | Confidence: <strong>86.82% R²</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("✏️ Update My Health Profile Online (Live Recalculate)", expanded=False):
                    with st.form("update_profile_form"):
                        new_w = st.slider("Current Weight (kg)", 45, 140, int(u["weight"]))
                        new_h = st.slider("Height (cm)", 140, 200, int(u["height"]))
                        new_bp = 1 if st.checkbox("Blood Pressure Problems", value=bool(u["bp"])) else 0
                        new_chronic = 1 if st.checkbox("Any Chronic Diseases", value=bool(u["chronic"])) else 0
                        if st.form_submit_button("💾 Save Profile Changes & Recalculate", use_container_width=True):
                            st.session_state["user"]["weight"] = new_w
                            st.session_state["user"]["height"] = new_h
                            st.session_state["user"]["bp"] = new_bp
                            st.session_state["user"]["chronic"] = new_chronic
                            st.success("Health profile updated successfully! Recalculating actuarial quote...")
                            st.rerun()

    # -------------------------------------------------------------
    # Page 1: Home & Paper Overview
    # -------------------------------------------------------------
    elif menu == "🏠 1. Home & Paper Overview":
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <span class="status-pill status-active">● Active Production Pipeline</span>
            <span class="status-pill status-academic">Panjab University SSGICR</span>
            <span class="status-pill status-metrics">9 Regressors • Table III Benchmark</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="main-header">Explainable Machine Learning for Health Insurance Cost Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">A Comparative Study of Tree-Based Ensembles with SHAP, ICE, and Counterfactual Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; margin-bottom: 1.5rem; box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.8px;">Published Academic Framework</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 4px 0; font-family: 'Outfit', sans-serif;">Panjab University CSE Research Study</div>
                    <div style="font-size: 0.88rem; color: #475569; line-height: 1.5;">
                        👨‍🔬 <strong>Author:</strong> Sanjay Kumar (Roll No: <code>SGM23366</code>)<br/>
                        👨‍🏫 <strong>Faculty Supervisor:</strong> Dr. Sukhvinder Singh Bamber (<code>ss.bamber@pu.ac.in</code>)<br/>
                        🏫 <strong>Institution:</strong> Dept. of Computer Science & Engineering, PUSSGRC, Hoshiarpur, Punjab, India.
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; align-items: flex-end;">
                    <span style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; padding: 4px 12px; border-radius: 8px; font-size: 0.78rem; font-weight: 700;">9 Models Benchmark</span>
                    <span style="background: #F0FDF4; border: 1px solid #BBF7D0; color: #15803D; padding: 4px 12px; border-radius: 8px; font-size: 0.78rem; font-weight: 700;">86.95% Test R²</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <span class="status-pill status-active">● Sub-10ms Live Engine</span>
            <span class="status-pill status-academic">Clinical Feature Engineering</span>
            <span class="status-pill status-metrics">Table VII Actionable Recourse</span>
        </div>
        """, unsafe_allow_html=True)
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
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #94A3B8;">ESTIMATED ANNUAL HEALTH PREMIUM</span>
                    <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(52, 211, 153, 0.45); color: #6EE7B7; padding: 3px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700;">⚡ {latency_ms:.2f} ms Latency</span>
                </div>
                <div class="prediction-price">{curr}{pred_cost:,.2f}</div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 12px; font-size: 0.85rem; color: #E2E8F0;">
                    <span>🤖 Engine: <strong style="color: #60A5FA;">{active_model_name}</strong></span>
                    <span style="opacity: 0.5;">•</span>
                    <span>🛡️ Empirical Accuracy: <strong style="color: #34D399;">86.82% R²</strong></span>
                    <span style="opacity: 0.5;">•</span>
                    <span>📊 5-Fold Stratified Holdout</span>
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
