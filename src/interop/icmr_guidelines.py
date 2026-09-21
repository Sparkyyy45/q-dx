"""
Indian Council of Medical Research (ICMR) and National Programme for Prevention
and Control of Non-Communicable Diseases (NP-NCD) Clinical Decision Rules.
Includes WHO-SEARO South Asian specific anthropometric risk thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_south_asian_bmi_category(bmi: float) -> Dict[str, Any]:
    """
    WHO South-East Asia Region (SEARO) and ICMR revised BMI classifications for Asian Indians:
    - Underweight: < 18.5 kg/m^2
    - Normal: 18.5 - 22.9 kg/m^2
    - Overweight: 23.0 - 24.9 kg/m^2 (triggers visceral adiposity screening)
    - Obese: >= 25.0 kg/m^2 (triggers intensive metabolic intervention)
    Reference: Misra et al., Consensus Statement for Diagnosis of Obesity,
    Abdominal Obesity and the Metabolic Syndrome for Asian Indians (JAPI 2009).
    """
    b = float(bmi)
    if b < 18.5:
        cat = "Underweight"
        risk_level = "LOW_WEIGHT_RISK"
        advice = "Nutritional assessment; ensure balanced protein and caloric intake."
    elif b < 23.0:
        cat = "Normal (Asian Indian)"
        risk_level = "OPTIMAL"
        advice = "Maintain current caloric equilibrium and routine physical activity."
    elif b < 25.0:
        cat = "Overweight (Asian Indian Threshold >=23 kg/m²)"
        risk_level = "ELEVATED_VISCERAL_ADIPOSITY"
        advice = "South Asian visceral adiposity risk. Initiate 500 kcal/day dietary deficit and waist circumference monitoring."
    else:
        cat = "Obese (Asian Indian Threshold >=25 kg/m²)"
        risk_level = "HIGH_CARDIOMETABOLIC_RISK"
        advice = "Class I/II Asian Indian Obesity. Structured weight loss target (5-10% body weight reduction) + endocrinology review."

    return {
        "bmi": round(b, 2),
        "asian_category": cat,
        "risk_level": risk_level,
        "clinical_advice": advice,
        "is_asian_overweight": b >= 23.0,
        "standard_global_cutoff_comparison": "Global cutoff >=25/30 underestimates cardiometabolic risk in South Asians by ~10-15%.",
    }


def get_icmr_clinical_recommendations(
    probability: float,
    ap_hi: float,
    ap_lo: float,
    cholesterol: int,
    gluc: int,
    smoke: int,
    bmi: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate evidence-based clinical next-steps per ICMR-NCD guidelines:
    - Primary Health Centre (PHC) triage level
    - Pharmacotherapy threshold alerts (Hypertension / Dyslipidemia / Diabetes)
    - Recommended follow-up interval
    - Lifestyle modification prescription
    - WHO-SEARO Asian Indian anthropometric risk stratification
    """
    triage_level: str
    urgency: str
    actions: List[str] = []
    lifestyle: List[str] = []

    # 1. Hemodynamic Stratification
    if ap_hi >= 160 or ap_lo >= 100:
        urgency = "URGENT_REFERRAL"
        triage_level = "Community Health Centre (CHC) / District Hospital Cardiology Referral"
        actions.append("Severe Stage 2 Hypertension detected (>=160/100 mmHg). Immediate physician review and dual antihypertensive therapy evaluation required.")
    elif ap_hi >= 140 or ap_lo >= 90:
        urgency = "MODERATE_PRIORITY"
        triage_level = "Primary Health Centre (PHC) Medical Officer Review"
        actions.append("Stage 2 Hypertension detected. Initiate salt-reduction protocol (<5g/day) and evaluate for first-line antihypertensive therapy (CCB or ARB).")
    elif ap_hi >= 130 or ap_lo >= 80:
        urgency = "ROUTINE_MONITORING"
        triage_level = "Ayushman Arogya Mandir (Health & Wellness Centre)"
        actions.append("Stage 1 Hypertension. Re-measure ambulatory blood pressure in 2 weeks.")
    else:
        urgency = "ANNUAL_CHECKUP"
        triage_level = "Self-Care & Community Health Worker (ASHA) Monitoring"
        actions.append("Normotensive. Maintain routine annual NCD screening.")

    # 2. Metabolic & Glycemic Stratification
    if gluc == 3:
        actions.append("Suspected uncontrolled diabetes (fasting glucose >=126 mg/dL). Confirm with HbA1c and refer to Medical Officer.")
    elif gluc == 2:
        actions.append("Impaired fasting glucose. Advise glycemic dietary restriction and repeat fasting glucose in 3 months.")

    if cholesterol == 3:
        actions.append("Severe hypercholesterolemia (>=240 mg/dL). Evaluate for high-intensity statin therapy (Atorvastatin 20-40mg).")
    elif cholesterol == 2:
        actions.append("Borderline hypercholesterolemia (200-239 mg/dL). Advise dietary lipid reduction and monitor in 6 months.")

    # 3. Anthropometric Stratification (South Asian Specific)
    anthropometric_info = None
    if bmi is not None and bmi > 0:
        anthropometric_info = get_south_asian_bmi_category(bmi)
        if anthropometric_info["is_asian_overweight"]:
            actions.append(f"Asian Indian Anthropometric Trigger (BMI {bmi:.1f} >= 23 kg/m²): Screen for insulin resistance and metabolic syndrome.")
            lifestyle.append(anthropometric_info["clinical_advice"])

    # 4. Smoking Cessation
    if smoke == 1:
        lifestyle.append("Tobacco cessation intervention: Enroll in National Tobacco Control Programme (NTCP) counseling.")

    lifestyle.append("Dietary Approach: Restrict dietary salt to <5g/day; increase green leafy vegetable intake.")
    lifestyle.append("Physical Activity: Minimum 150 minutes of moderate-intensity aerobic exercise (e.g., brisk walking) per week.")

    rec: Dict[str, Any] = {
        "guideline_source": "ICMR Guidelines for Management of Type 2 Diabetes & Hypertension (NP-NCD)",
        "urgency": urgency,
        "triage_level": triage_level,
        "clinical_actions": actions,
        "lifestyle_prescriptions": lifestyle,
        "follow_up_window": "Within 48 hours" if urgency == "URGENT_REFERRAL" else ("Within 14 days" if urgency == "MODERATE_PRIORITY" else "Within 3-6 months"),
    }
    if anthropometric_info:
        rec["anthropometric_risk"] = anthropometric_info

    return rec
