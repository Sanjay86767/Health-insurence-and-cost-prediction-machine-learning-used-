import pytest
from src.pdf_generator import generate_policyholder_pdf_report

def test_generate_pdf_certificate():
    applicant_data = {
        "Age": 28,
        "Height": 172,
        "Weight": 70,
        "Diabetes": 0,
        "BloodPressureProblems": 0,
        "AnyTransplants": 0,
        "AnyChronicDiseases": 0,
        "KnownAllergies": 0,
        "HistoryOfCancerInFamily": 0,
        "NumberOfMajorSurgeries": 0
    }
    pdf_bytes = generate_policyholder_pdf_report(
        applicant_data=applicant_data,
        quote_inr=17850.0,
        model_name="XGBoost Regressor (v1.0-Paper)",
        bmi=23.66,
        cci=0.0,
        counterfactual_strategies=[
            ("Maintain Baseline", "BMI 23.66 Normal Range", "None", 17850.0, 0.0)
        ],
        cardholder_name="Sanjay Kumar",
        policy_id="PU-SGM23366-TEST"
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
