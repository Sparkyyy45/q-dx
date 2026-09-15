"""
Indian Council of Medical Research (ICMR) and National Programme for Prevention
and Control of Non-Communicable Diseases (NP-NCD) Clinical Decision Rules.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_icmr_clinical_recommendations(
    probability: float,
    ap_hi: float,
    ap_lo: float,
    cholesterol: int,
    gluc: int,
    smoke: int,
) -> Dict[str, Any]:
    """
    Generate evidence-based clinical next-steps per ICMR-NCD guidelines:
    - Primary Health Centre (PHC) triage level
    - Pharmacotherapy threshold alerts (Hypertension / Dyslipidemia / Diabetes)
    - Recommended follow-up interval
    - Lifestyle modification prescription
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

    # 3. Smoking Cessation
    if smoke == 1:
        lifestyle.append("Tobacco cessation intervention: Enroll in National Tobacco Control Programme (NTCP) counseling.")

    lifestyle.append("Dietary Approach: Restrict dietary salt to <5g/day; increase green leafy vegetable intake.")
    lifestyle.append("Physical Activity: Minimum 150 minutes of moderate-intensity aerobic exercise (e.g., brisk walking) per week.")

    return {
        "guideline_source": "ICMR Guidelines for Management of Type 2 Diabetes & Hypertension (NP-NCD)",
        "urgency": urgency,
        "triage_level": triage_level,
        "clinical_actions": actions,
        "lifestyle_prescriptions": lifestyle,
        "follow_up_window": "Within 48 hours" if urgency == "URGENT_REFERRAL" else ("Within 14 days" if urgency == "MODERATE_PRIORITY" else "Within 3-6 months"),
    }
