"""
Unit tests for National Health Stack Interoperability in CardioQ:
- ABDM 14-digit ABHA validation & test generation (Luhn mod-10)
- HL7 FHIR R4 RiskAssessment & Observation JSON Bundle generation
- ICMR NP-NCD clinical risk tiering, triage recommendations, and action plans
"""

from __future__ import annotations

import json
from src.interop.abha import validate_abha_id, generate_demo_abha
from src.interop.fhir import generate_fhir_risk_assessment_bundle
from src.interop.icmr_guidelines import get_icmr_clinical_recommendations


def test_abha_validator_valid_and_invalid():
    # Generate 5 valid test ABHAs and verify all pass
    for _ in range(5):
        demo_id = generate_demo_abha()
        is_valid, msg, clean = validate_abha_id(demo_id)
        assert is_valid is True, f"Generated ABHA {demo_id} should be valid, but failed with: {msg}"
        assert clean is not None and len(clean) == 17  # XX-XXXX-XXXX-XXXX

    # Verify invalid formats
    assert validate_abha_id("")[0] is False
    assert validate_abha_id("12345")[0] is False
    assert validate_abha_id("ab-cdef-ghij-klmn")[0] is False
    assert validate_abha_id("12-3456-7890-123X")[0] is False

    # Corrupt check digit of a valid demo
    valid_demo = generate_demo_abha()
    clean_digits = valid_demo.replace("-", "")
    last_digit = int(clean_digits[-1])
    corrupted_last = (last_digit + 1) % 10
    corrupted_clean = clean_digits[:-1] + str(corrupted_last)
    corrupted_formatted = f"{corrupted_clean[:2]}-{corrupted_clean[2:6]}-{corrupted_clean[6:10]}-{corrupted_clean[10:]}"
    is_valid, _, _ = validate_abha_id(corrupted_formatted)
    assert is_valid is False, "Corrupted Luhn check digit must fail validation"


def test_fhir_bundle_generation():
    patient_data = {
        "name": "Pooja Sharma",
        "age_years": 62,
        "gender": 0,
        "height": 160,
        "weight": 72,
        "ap_hi": 165,
        "ap_lo": 95,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "bmi": 28.1,
    }
    risk_result = {
        "probability": 0.88,
        "prediction": 1,
        "risk_tier": "High Estimated Risk (>50%)",
        "model": "vqc",
    }
    bundle = generate_fhir_risk_assessment_bundle(
        patient_data=patient_data,
        risk_result=risk_result,
        screening_id="SCR-FHIR-001",
        abha_id="12-3456-7890-1234",
    )

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) >= 3

    # Inspect entries
    resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
    assert "Patient" in resource_types
    assert "RiskAssessment" in resource_types
    assert "Observation" in resource_types

    # Find RiskAssessment entry
    risk_entry = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "RiskAssessment")
    assert risk_entry["prediction"][0]["probabilityDecimal"] == 0.88
    assert "High" in risk_entry["prediction"][0]["qualitativeRisk"]["text"]
    assert "loinc" in risk_entry["code"]["coding"][0]["system"].lower()
    assert "snomed" in risk_entry["prediction"][0]["outcome"]["coding"][0]["system"].lower()


    # Find BP observation
    bp_obs = next(
        e["resource"]
        for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Observation"
        and e["resource"]["code"]["coding"][0]["code"] == "85354-9"
    )
    assert bp_obs is not None
    assert len(bp_obs["component"]) == 2

    # Verify JSON serializability
    bundle_json = json.dumps(bundle)
    assert len(bundle_json) > 500


def test_icmr_guidelines_critical_triage():
    recs = get_icmr_clinical_recommendations(
        probability=0.91,
        ap_hi=185,
        ap_lo=110,
        cholesterol=3,
        gluc=3,
        smoke=1,
    )

    assert recs["urgency"] == "URGENT_REFERRAL"
    assert "Cardiology Referral" in recs["triage_level"] or "Hospital" in recs["triage_level"]
    assert len(recs["clinical_actions"]) > 0
    assert any("Hypertension" in a for a in recs["clinical_actions"])
    assert any("statin" in a.lower() for a in recs["clinical_actions"])
    assert any("Tobacco" in a for a in recs["lifestyle_prescriptions"])
    assert recs["follow_up_window"] == "Within 48 hours"


def test_icmr_guidelines_low_risk_triage():
    recs = get_icmr_clinical_recommendations(
        probability=0.15,
        ap_hi=118,
        ap_lo=76,
        cholesterol=1,
        gluc=1,
        smoke=0,
    )

    assert recs["urgency"] == "ANNUAL_CHECKUP"
    assert "ASHA" in recs["triage_level"] or "Self-Care" in recs["triage_level"]
    assert any("Normotensive" in a for a in recs["clinical_actions"])
