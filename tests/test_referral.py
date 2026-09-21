"""
Unit tests for Bilingual Referral Slip & ICMR Decision Support:
- Referral slip payload generation
- HTML print rendering
- WHO-SEARO / ICMR South Asian BMI criteria
"""

from src.interop.referral import generate_referral_slip_data, render_referral_slip_html
from src.interop.icmr_guidelines import get_south_asian_bmi_category, get_icmr_clinical_recommendations


def test_south_asian_bmi_criteria():
    # South Asian normal
    norm = get_south_asian_bmi_category(22.0)
    assert norm["asian_category"] == "Normal (Asian Indian)"
    assert norm["is_asian_overweight"] is False

    # South Asian overweight (>=23)
    ow = get_south_asian_bmi_category(23.8)
    assert "Overweight" in ow["asian_category"]
    assert ow["is_asian_overweight"] is True

    # South Asian obese (>=25)
    ob = get_south_asian_bmi_category(26.5)
    assert "Obese" in ob["asian_category"]
    assert ob["is_asian_overweight"] is True


def test_icmr_recommendations_with_bmi():
    recs = get_icmr_clinical_recommendations(
        probability=0.72,
        ap_hi=150,
        ap_lo=95,
        cholesterol=2,
        gluc=1,
        smoke=0,
        bmi=24.2,  # Overweight under South Asian cutoff
    )
    assert "anthropometric_risk" in recs
    assert recs["anthropometric_risk"]["is_asian_overweight"] is True
    assert any("Asian Indian" in act for act in recs["clinical_actions"])


def test_generate_referral_slip_data_and_html():
    patient = {
        "name": "Dr. Sunita Kapoor",
        "age_years": 54,
        "gender": 1,
        "ap_hi": 142,
        "ap_lo": 90,
        "bmi": 24.5,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "abha_id": "91-0552-2867-3285",
    }
    risk_res = {
        "model": "CatBoost Champion",
        "probability": 0.584,
        "prediction": 1,
        "risk_tier": "High Estimated Risk (>50%)",
        "conformal_prediction": {
            "confidence_level": 0.90,
            "risk_band": [0.51, 0.66],
            "clinical_interpretation": "Definitive high risk",
        },
        "icmr_recommendations": {
            "urgency": "MODERATE_PRIORITY",
            "triage_level": "Primary Health Centre (PHC) Medical Officer",
            "follow_up_window": "Within 14 days",
            "clinical_actions": ["Initiate dietary salt reduction."],
            "lifestyle_prescriptions": ["150 mins aerobic exercise/week."],
        },
        "top_factors": [
            {"feature": "ap_hi", "direction": "increases", "shap_value": 0.24},
        ],
    }

    ref_data = generate_referral_slip_data(patient, risk_res, screening_id="SCR-TEST-001")
    assert ref_data["referral_id"] == "SCR-TEST-001"
    assert ref_data["patient"]["name"] == "Dr. Sunita Kapoor"
    assert ref_data["ai_risk_assessment"]["probability_percent"] == 58.4
    assert len(ref_data["emergency_warnings"]) >= 2

    html = render_referral_slip_html(ref_data)
    assert "<!DOCTYPE html>" in html
    assert "SCR-TEST-001" in html
    assert "Dr. Sunita Kapoor" in html
    assert "58.4%" in html
    assert "@media print" in html
