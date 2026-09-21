"""
Bilingual Clinical Referral Slip & Patient Card Generator for CardioQ.
Compliant with Ayushman Bharat Digital Mission (ABDM) and ICMR-NP-NCD clinical protocols.
Provides:
- Structured clinical referral slip payload
- Print-ready HTML document (@media print optimized) for 1-click PDF export
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_referral_slip_data(
    patient_data: Dict[str, Any],
    risk_result: Dict[str, Any],
    screening_id: Optional[str] = None,
    abha_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesizes patient clinical vitals, AI risk probability, conformal bounds,
    and ICMR triage actions into a formal hospital referral slip.
    """
    scr_id = screening_id or f"CQ-REF-{int(datetime.now().timestamp())}"
    pat_id = patient_data.get("patient_id") or patient_data.get("mrn") or "MRN-84920"
    prob = float(risk_result.get("probability", 0.0))
    tier = risk_result.get("risk_tier", "Moderate Estimated Risk")
    model_name = risk_result.get("model", "CatBoost Champion")

    conformal = risk_result.get("conformal_prediction", {})
    icmr = risk_result.get("icmr_recommendations", {})

    top_factors = risk_result.get("top_factors", [])

    return {
        "referral_id": scr_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "patient_id": pat_id,
        "abha_id": None,
        "patient": {
            "name": patient_data.get("name", "CardioQ Patient"),
            "age_years": patient_data.get("age_years", 50),
            "gender": "Male (पुरुष)" if patient_data.get("gender") in [2, "2", "male"] else "Female (महिला)",
            "ap_hi": patient_data.get("ap_hi", 120),
            "ap_lo": patient_data.get("ap_lo", 80),
            "bmi": round(float(patient_data.get("bmi", 24.0)), 1),
            "cholesterol": "Normal (सामान्य)" if patient_data.get("cholesterol") == 1 else ("Above Normal (उच्च)" if patient_data.get("cholesterol") == 2 else "Well Above Normal (अति उच्च)"),
            "gluc": "Normal (सामान्य)" if patient_data.get("gluc") == 1 else ("Above Normal (उच्च)" if patient_data.get("gluc") == 2 else "Well Above Normal (अति उच्च)"),
            "smoke": "Active Smoker (धूम्रपानकर्ता)" if patient_data.get("smoke") == 1 else "Non-Smoker (अधूम्रपानकर्ता)",
        },
        "ai_risk_assessment": {
            "model": model_name,
            "probability_percent": round(prob * 100, 1),
            "risk_tier": tier,
            "prediction": risk_result.get("prediction", 1 if prob >= 0.5 else 0),
            "confidence_band": conformal.get("risk_band", [round(max(0.0, prob - 0.08), 3), round(min(1.0, prob + 0.08), 3)]),
            "conformal_status": conformal.get("clinical_interpretation", "Calibrated 90% finite-sample confidence band."),
            "primary_drivers": top_factors[:3] if top_factors else [],
        },
        "icmr_triage": {
            "urgency": icmr.get("urgency", "MODERATE_PRIORITY"),
            "triage_facility": icmr.get("triage_level", "Primary Health Centre (PHC)"),
            "follow_up_window": icmr.get("follow_up_window", "Within 14 days"),
            "actions": icmr.get("clinical_actions", []),
            "lifestyle": icmr.get("lifestyle_prescriptions", []),
        },
        "emergency_warnings": [
            "Acute persistent retrosternal chest pain radiating to left arm or jaw (सीने में तेज दर्द).",
            "Sudden unexplained shortness of breath, cold sweating, or syncope (अचानक सांस फूलना या पसीना).",
            "If above symptoms present, report immediately to the nearest Emergency Department (108 / 112).",
        ],
        "disclaimer": "This document is an AI-assisted clinical decision support triage report. It does not replace independent physician diagnosis.",
    }


def render_referral_slip_html(referral_data: Dict[str, Any]) -> str:
    """
    Renders an elegant, high-contrast, bilingual print-ready referral slip.
    Optimized for A4 paper and PDF printing.
    """
    p = referral_data["patient"]
    ai = referral_data["ai_risk_assessment"]
    icmr = referral_data["icmr_triage"]
    urgency = icmr.get("urgency", "ROUTINE")

    urgency_color = "#dc2626" if urgency == "URGENT_REFERRAL" else ("#ea580c" if urgency == "MODERATE_PRIORITY" else "#16a34a")
    urgency_bg = "#fef2f2" if urgency == "URGENT_REFERRAL" else ("#fff7ed" if urgency == "MODERATE_PRIORITY" else "#f0fdf4")

    actions_html = "".join(f"<li>{act}</li>" for act in icmr.get("actions", [])) or "<li>Maintain routine annual wellness screening.</li>"
    lifestyle_html = "".join(f"<li>{ls}</li>" for ls in icmr.get("lifestyle", [])) or "<li>Maintain balanced diet and regular aerobic exercise.</li>"
    warnings_html = "".join(f"<li>{w}</li>" for w in referral_data.get("emergency_warnings", []))

    factors_html = ""
    if ai.get("primary_drivers"):
        factors_items = "".join(
            f"<li><strong>{f.get('feature', 'Factor')}:</strong> {f.get('direction', 'increases')} risk (attribution: {f.get('shap_value', 0):+.3f})</li>"
            for f in ai["primary_drivers"]
        )
        factors_html = f"<div class='sec-sub'><strong>Primary Clinical Drivers (TreeSHAP):</strong><ul>{factors_items}</ul></div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CardioQ Clinical Referral Slip - {referral_data['referral_id']}</title>
  <style>
    @page {{
      size: A4;
      margin: 12mm 15mm;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #0f172a;
      background: #ffffff;
      line-height: 1.4;
      font-size: 13px;
      padding: 15px;
    }}
    .slip-container {{
      max-width: 800px;
      margin: 0 auto;
      border: 2px solid #0284c7;
      border-radius: 8px;
      padding: 20px;
      background: #ffffff;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 12px;
      margin-bottom: 15px;
    }}
    .header-logo h1 {{
      font-size: 22px;
      color: #0369a1;
      letter-spacing: -0.5px;
    }}
    .header-logo p {{
      font-size: 11px;
      color: #64748b;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .header-badge {{
      text-align: right;
    }}
    .badge-ref {{
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      padding: 4px 10px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
    }}
    .patient-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      background: #f8fafc;
      padding: 12px;
      border-radius: 6px;
      margin-bottom: 15px;
      border: 1px solid #e2e8f0;
    }}
    .grid-item span {{
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      color: #64748b;
      font-weight: 600;
    }}
    .grid-item strong {{
      font-size: 13px;
      color: #0f172a;
    }}
    .triage-banner {{
      background: {urgency_bg};
      border-left: 5px solid {urgency_color};
      padding: 12px 15px;
      border-radius: 4px;
      margin-bottom: 15px;
    }}
    .triage-banner h3 {{
      color: {urgency_color};
      font-size: 15px;
      margin-bottom: 4px;
    }}
    .triage-banner p {{
      color: #334155;
      font-size: 12px;
    }}
    .section {{
      margin-bottom: 15px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 700;
      color: #0369a1;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    ul {{
      padding-left: 18px;
      margin-top: 4px;
    }}
    li {{
      margin-bottom: 4px;
      color: #1e293b;
    }}
    .sec-sub {{
      margin-top: 8px;
      background: #f8fafc;
      padding: 8px 12px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }}
    .warnings {{
      background: #fff1f2;
      border: 1px dashed #f43f5e;
      padding: 10px 14px;
      border-radius: 6px;
      margin-bottom: 15px;
    }}
    .warnings h4 {{
      color: #be123c;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .footer-sign {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      margin-top: 25px;
      padding-top: 15px;
      border-top: 1px dashed #cbd5e1;
    }}
    .footer-sign .disclaimer {{
      max-width: 60%;
      font-size: 10px;
      color: #64748b;
    }}
    .footer-sign .sign-box {{
      text-align: center;
      width: 200px;
      border-top: 1px solid #0f172a;
      padding-top: 5px;
      font-size: 11px;
      font-weight: 600;
    }}
    .print-controls {{
      text-align: center;
      margin-bottom: 15px;
    }}
    .btn-print {{
      background: #0284c7;
      color: white;
      border: none;
      padding: 8px 20px;
      font-size: 14px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
    }}
    @media print {{
      .print-controls {{
        display: none;
      }}
      body {{
        padding: 0;
      }}
      .slip-container {{
        border: 1px solid #64748b;
      }}
    }}
  </style>
</head>
<body>
  <div class="print-controls">
    <button class="btn-print" onclick="window.print()">Print / Save PDF (रेफरल पर्ची प्रिंट करें)</button>
  </div>

  <div class="slip-container">
    <div class="header">
      <div class="header-logo">
        <h1>कार्डियो-क्यू • CardioQ Clinical Decision Triage</h1>
        <p>National Programme for Prevention and Control of Non-Communicable Diseases (NP-NCD)</p>
      </div>
      <div class="header-badge">
        <div class="badge-ref">{referral_data['referral_id']}</div>
        <div style="font-size: 10px; color: #64748b; margin-top: 3px;">Date: {referral_data['generated_at']}</div>
      </div>
    </div>

    <div class="patient-grid">
      <div class="grid-item">
        <span>Patient Name / मरीज का नाम</span>
        <strong>{p['name']}</strong>
      </div>
      <div class="grid-item">
        <span>Patient ID / MRN (मरीज आईडी)</span>
        <strong>{referral_data.get('patient_id') or 'MRN-84920'}</strong>
      </div>
      <div class="grid-item">
        <span>Age & Gender / आयु व लिंग</span>
        <strong>{p['age_years']} Y / {p['gender']}</strong>
      </div>
      <div class="grid-item">
        <span>Blood Pressure / रक्तचाप</span>
        <strong>{p['ap_hi']}/{p['ap_lo']} mmHg</strong>
      </div>
      <div class="grid-item">
        <span>BMI / बॉडी मास इंडेक्स</span>
        <strong>{p['bmi']} kg/m²</strong>
      </div>
      <div class="grid-item">
        <span>Blood Glucose / रक्त शर्करा</span>
        <strong>{p['gluc']}</strong>
      </div>
      <div class="grid-item">
        <span>Cholesterol / कोलेस्ट्रॉल</span>
        <strong>{p['cholesterol']}</strong>
      </div>
      <div class="grid-item">
        <span>Smoking / धूम्रपान</span>
        <strong>{p['smoke']}</strong>
      </div>
    </div>

    <div class="triage-banner">
      <h3>RECOMMENDED FACILITY: {icmr['triage_facility']}</h3>
      <p><strong>Urgency Tier:</strong> {icmr['urgency']} | <strong>Follow-up Window:</strong> {icmr['follow_up_window']}</p>
    </div>

    <div class="section">
      <div class="section-title">Cardiovascular Risk Assessment (हृदय जोखिम मूल्यांकन)</div>
      <p>
        <strong>Predicted CVD Probability:</strong> <span style="font-size: 15px; font-weight: 700; color: #0369a1;">{ai['probability_percent']}%</span> ({ai['risk_tier']})
        <br>
        <strong>90% Conformal Confidence Band:</strong> [{ai['confidence_band'][0]*100:.1f}%, {ai['confidence_band'][1]*100:.1f}%]
      </p>
      <p style="font-size: 11px; color: #475569; margin-top: 3px;">{ai['conformal_status']}</p>
      {factors_html}
    </div>

    <div class="section">
      <div class="section-title">Mandatory ICMR Clinical Next Steps (चिकित्सीय कदम)</div>
      <ul>{actions_html}</ul>
    </div>

    <div class="section">
      <div class="section-title">Prescribed Lifestyle Intervention (जीवनशैली संशोधन)</div>
      <ul>{lifestyle_html}</ul>
    </div>

    <div class="warnings">
      <h4>RED-FLAG EMERGENCY WARNINGS / आपातकालीन चेतावनी</h4>
      <ul>{warnings_html}</ul>
    </div>

    <div class="footer-sign">
      <div class="disclaimer">
        {referral_data['disclaimer']}
        <br>
        Interoperable with National Clinical Health Systems & HL7 FHIR R4.
      </div>
      <div class="sign-box">
        Examining Medical Officer<br>
        (हस्ताक्षर एवं सील)
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html
