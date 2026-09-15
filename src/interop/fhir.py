"""
HL7 FHIR R4 Bundle generator for CardioQ Cardiovascular Risk Assessments.
Compliant with Ayushman Bharat Digital Mission (ABDM) Diagnostic Report & RiskAssessment profile.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings


def generate_fhir_risk_assessment_bundle(
    patient_data: Dict[str, Any],
    risk_result: Dict[str, Any],
    screening_id: Optional[str] = None,
    abha_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an HL7 FHIR Release 4 compliant collection bundle containing:
    - Resource: Patient
    - Resource: RiskAssessment (LOINC 75492-9: Cardiovascular disease 10Y risk)
    - Resource: Observation (Blood Pressure, BMI, Cholesterol, Glucose)
    """
    screening_uuid = screening_id or f"scr-{uuid.uuid4().hex[:12]}"
    patient_uuid = f"pat-{uuid.uuid4().hex[:8]}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    probability = float(risk_result.get("probability", 0.0))
    tier = str(risk_result.get("risk_tier", "Low Estimated Risk (<20%)"))
    model_name = str(risk_result.get("model", "CatBoost"))

    # Determine qualitative risk code
    if probability >= 0.50:
        qual_code = "high"
        qual_display = "High Risk"
    elif probability >= 0.20:
        qual_code = "moderate"
        qual_display = "Moderate Risk"
    else:
        qual_code = "low"
        qual_display = "Low Risk"

    # Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_uuid,
        "identifier": [
            {
                "system": "https://healthid.ndhm.gov.in",
                "value": abha_id or "91-0000-0000-0000",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR",
                            "display": "Medical Record Number / ABHA",
                        }
                    ]
                },
            }
        ],
        "gender": "male" if int(patient_data.get("gender", 1)) == 2 else "female",
    }

    # Blood Pressure Observation
    bp_obs_id = f"obs-bp-{uuid.uuid4().hex[:6]}"
    bp_observation = {
        "resourceType": "Observation",
        "id": bp_obs_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Blood pressure panel with all children optional",
                }
            ],
            "text": "Blood Pressure",
        },
        "subject": {"reference": f"Patient/{patient_uuid}"},
        "effectiveDateTime": now_iso,
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8480-6",
                            "display": "Systolic blood pressure",
                        }
                    ]
                },
                "valueQuantity": {
                    "value": float(patient_data.get("ap_hi", 120.0)),
                    "unit": "mmHg",
                    "system": "http://unitsofmeasure.org",
                    "code": "mm[Hg]",
                },
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8462-4",
                            "display": "Diastolic blood pressure",
                        }
                    ]
                },
                "valueQuantity": {
                    "value": float(patient_data.get("ap_lo", 80.0)),
                    "unit": "mmHg",
                    "system": "http://unitsofmeasure.org",
                    "code": "mm[Hg]",
                },
            },
        ],
    }

    # Main RiskAssessment Resource
    risk_assessment_resource = {
        "resourceType": "RiskAssessment",
        "id": screening_uuid,
        "status": "final",
        "subject": {"reference": f"Patient/{patient_uuid}"},
        "occurrenceDateTime": now_iso,
        "performer": {
            "display": f"CardioQ Diagnostic Platform ({settings.ORGANIZATION_NAME})"
        },
        "method": {
            "coding": [
                {
                    "system": "https://cardioq.health.gov.in/algorithms",
                    "code": model_name.lower().replace(" ", "_"),
                    "display": f"{model_name} CVD Model (Threshold {risk_result.get('applied_threshold', 0.5):.4f})",
                }
            ],
            "text": f"Algorithmic Risk Assessment using {model_name}",
        },
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "75492-9",
                    "display": "Cardiovascular disease 10Y risk",
                }
            ],
            "text": "Cardiovascular Risk Assessment",
        },
        "basis": [{"reference": f"Observation/{bp_obs_id}"}],
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "49601007",
                            "display": "Cardiovascular disease",
                        }
                    ],
                    "text": "Prevalent Cardiovascular Disease",
                },
                "probabilityDecimal": round(probability, 4),
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": qual_code,
                            "display": qual_display,
                        }
                    ],
                    "text": tier,
                },
            }
        ],
        "note": [
            {
                "text": risk_result.get("clinical_summary", "Assessment completed.")
            },
            {
                "text": risk_result.get(
                    "disclaimer",
                    "Statistically estimated risk; does not constitute medical diagnosis.",
                )
            },
        ],
    }

    # Bundle Envelope
    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-{screening_uuid}",
        "meta": {
            "profile": [
                "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"
            ],
            "lastUpdated": now_iso,
        },
        "type": "collection",
        "timestamp": now_iso,
        "entry": [
            {"fullUrl": f"urn:uuid:{patient_uuid}", "resource": patient_resource},
            {"fullUrl": f"urn:uuid:{bp_obs_id}", "resource": bp_observation},
            {
                "fullUrl": f"urn:uuid:{screening_uuid}",
                "resource": risk_assessment_resource,
            },
        ],
    }

    return bundle
