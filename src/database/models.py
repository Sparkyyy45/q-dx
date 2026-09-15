"""
Data classes representing database entities for CardioQ platform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PatientRecord:
    id: str = ""
    name: str = "Anonymous Patient"
    age_years: float = 50.0
    gender: int = 1
    abha_id: Optional[str] = None
    phone: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        id: str = "",
        name: str = "Anonymous Patient",
        age_years: Optional[float] = None,
        gender: Optional[int] = None,
        abha_id: Optional[str] = None,
        phone: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        patient_id: Optional[str] = None,
        age: Optional[float] = None,
        sex: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        **kwargs: Any,
    ):
        self.id = id or patient_id or ""
        self.name = name
        self.age_years = float(age_years if age_years is not None else (age if age is not None else 50.0))
        self.gender = int(gender if gender is not None else (sex if sex is not None else 1))
        self.abha_id = abha_id
        self.phone = phone
        self.state = state
        self.district = district
        self.created_at = created_at
        self.updated_at = updated_at

    @property
    def patient_id(self) -> str:
        return self.id

    @property
    def age(self) -> float:
        return self.age_years

    @property
    def sex(self) -> int:
        return self.gender

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.id,
            "name": self.name,
            "age_years": self.age_years,
            "age": self.age_years,
            "gender": self.gender,
            "sex": self.gender,
            "abha_id": self.abha_id,
            "phone": self.phone,
            "state": self.state,
            "district": self.district,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ScreeningRecord:
    id: str = ""
    model_name: str = "CatBoost"
    age_years: float = 50.0
    gender: int = 1
    height: float = 165.0
    weight: float = 70.0
    ap_hi: float = 120.0
    ap_lo: float = 80.0
    cholesterol: int = 1
    gluc: int = 1
    smoke: int = 0
    alco: int = 0
    active: int = 1
    bmi: float = 24.5
    probability: float = 0.0
    prediction: int = 0
    risk_tier: str = "Low Estimated Risk (<20%)"
    applied_threshold: float = 0.50
    patient_id: Optional[str] = None
    abha_id: Optional[str] = None
    explanation_method: Optional[str] = None
    top_factors_json: Optional[str] = None
    clinical_summary: Optional[str] = None
    fhir_bundle_json: Optional[str] = None
    input_payload: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    icmr_guidance: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    def __init__(
        self,
        id: str = "",
        model_name: str = "CatBoost",
        age_years: Optional[float] = None,
        gender: Optional[int] = None,
        height: float = 165.0,
        weight: float = 70.0,
        ap_hi: float = 120.0,
        ap_lo: float = 80.0,
        cholesterol: int = 1,
        gluc: int = 1,
        smoke: int = 0,
        alco: int = 0,
        active: int = 1,
        bmi: float = 24.5,
        probability: Optional[float] = None,
        prediction: Optional[int] = None,
        risk_tier: str = "Low Estimated Risk (<20%)",
        applied_threshold: float = 0.50,
        patient_id: Optional[str] = None,
        abha_id: Optional[str] = None,
        explanation_method: Optional[str] = None,
        top_factors_json: Optional[str] = None,
        clinical_summary: Optional[str] = None,
        fhir_bundle_json: Optional[str] = None,
        screening_id: Optional[str] = None,
        risk_probability: Optional[float] = None,
        risk_prediction: Optional[int] = None,
        input_payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        icmr_guidance: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        **kwargs: Any,
    ):
        self.id = id or screening_id or ""
        self.model_name = model_name
        self.age_years = float(age_years if age_years is not None else 50.0)
        self.gender = int(gender if gender is not None else 1)
        self.height = float(height)
        self.weight = float(weight)
        self.ap_hi = float(ap_hi)
        self.ap_lo = float(ap_lo)
        self.cholesterol = int(cholesterol)
        self.gluc = int(gluc)
        self.smoke = int(smoke)
        self.alco = int(alco)
        self.active = int(active)
        self.bmi = float(bmi)
        self.probability = float(probability if probability is not None else (risk_probability if risk_probability is not None else 0.0))
        self.prediction = int(prediction if prediction is not None else (risk_prediction if risk_prediction is not None else 0))
        self.risk_tier = risk_tier
        self.applied_threshold = float(applied_threshold)
        self.patient_id = patient_id
        self.abha_id = abha_id
        self.explanation_method = explanation_method
        self.top_factors_json = top_factors_json
        self.clinical_summary = clinical_summary
        self.fhir_bundle_json = fhir_bundle_json
        self.input_payload = input_payload or {}
        self.metrics = metrics or {}
        self.icmr_guidance = icmr_guidance or {}
        self.created_at = created_at

    @property
    def screening_id(self) -> str:
        return self.id

    @property
    def risk_probability(self) -> float:
        return self.probability

    @property
    def risk_prediction(self) -> int:
        return self.prediction

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["screening_id"] = self.id
        d["risk_probability"] = self.probability
        d["risk_prediction"] = self.prediction
        return d


@dataclass
class AuditLogRecord:
    id: str
    action: str
    user_role: str = "clinician"
    details_json: Optional[str] = None
    client_ip: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
