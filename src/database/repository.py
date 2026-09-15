"""
Repository layer providing clean CRUD operations for clinical records, patient histories, and audit trails.
Supports both instance-based access (with custom database paths) and class-level operations.
"""

from __future__ import annotations

import json
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.database.connection import get_db_connection, init_db
from src.database.models import AuditLogRecord, PatientRecord, ScreeningRecord


class _hybridmethod:
    """Descriptor allowing a method to be invoked on either an instance or class directly."""

    def __init__(self, func: Callable):
        self.func = func

    def __get__(self, instance: Any, owner: Any):
        if instance is None:
            inst = owner()
            return lambda *args, **kwargs: self.func(inst, *args, **kwargs)
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)


class ClinicalRepository:
    """Data access object for persistent clinical records and audit logging."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            init_db(self.db_path)

    def _get_conn(self, db_path_override: Optional[Union[str, Path]] = None):
        target = db_path_override or self.db_path
        return get_db_connection(target)

    # -------------------------------------------------------------------------
    # Patient Operations
    # -------------------------------------------------------------------------

    @_hybridmethod
    def upsert_patient(self, patient: Union[PatientRecord, Dict[str, Any]]) -> PatientRecord:
        """Insert or update patient demographic and identification record."""
        if isinstance(patient, dict):
            p = PatientRecord(**patient)
        else:
            p = patient

        patient_id = p.id or f"pat_{uuid.uuid4().hex[:12]}"
        p.id = patient_id

        with self._get_conn() as conn:
            existing = None
            if p.id:
                cur = conn.execute("SELECT id FROM patients WHERE id = ?", (p.id,))
                existing = cur.fetchone()
            if not existing and p.abha_id:
                cur = conn.execute("SELECT id FROM patients WHERE abha_id = ?", (p.abha_id,))
                existing = cur.fetchone()

            if existing:
                p.id = existing["id"]
                conn.execute(
                    """
                    UPDATE patients
                    SET name = ?, age_years = ?, gender = ?, abha_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (p.name, p.age_years, p.gender, p.abha_id, p.id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO patients (id, abha_id, name, age_years, gender)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (p.id, p.abha_id, p.name, p.age_years, p.gender),
                )

        return p

    @_hybridmethod
    def get_or_create_patient(
        self,
        name: str,
        age_years: float,
        gender: int,
        abha_id: Optional[str] = None,
    ) -> PatientRecord:
        """Fetch existing patient by ABHA ID or create a new patient record."""
        with self._get_conn() as conn:
            if abha_id:
                cur = conn.execute("SELECT * FROM patients WHERE abha_id = ?", (abha_id,))
                row = cur.fetchone()
                if row:
                    return PatientRecord(
                        id=row["id"],
                        abha_id=row["abha_id"],
                        name=row["name"],
                        age_years=row["age_years"],
                        gender=row["gender"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )

            patient_id = f"pat_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO patients (id, abha_id, name, age_years, gender)
                VALUES (?, ?, ?, ?, ?)
                """,
                (patient_id, abha_id, name, age_years, gender),
            )
            return PatientRecord(
                id=patient_id,
                abha_id=abha_id,
                name=name,
                age_years=age_years,
                gender=gender,
            )

    @_hybridmethod
    def get_patient_by_id(self, patient_id: str) -> Optional[PatientRecord]:
        """Fetch patient record by internal identifier."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            row = cur.fetchone()
            if not row:
                return None
            return PatientRecord(
                id=row["id"],
                abha_id=row["abha_id"],
                name=row["name"],
                age_years=row["age_years"],
                gender=row["gender"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    @_hybridmethod
    def get_patient_by_abha(self, abha_id: str) -> Optional[PatientRecord]:
        """Fetch patient record by 14-digit ABHA ID."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM patients WHERE abha_id = ?", (abha_id,))
            row = cur.fetchone()
            if not row:
                return None
            return PatientRecord(
                id=row["id"],
                abha_id=row["abha_id"],
                name=row["name"],
                age_years=row["age_years"],
                gender=row["gender"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    # -------------------------------------------------------------------------
    # Screening Operations
    # -------------------------------------------------------------------------

    @_hybridmethod
    def save_screening(self, screening_data: Union[ScreeningRecord, Dict[str, Any]]) -> ScreeningRecord:
        """Persist a completed patient risk assessment."""
        if isinstance(screening_data, ScreeningRecord):
            rec = screening_data
            d = rec.to_dict()
        else:
            d = dict(screening_data)
            rec = ScreeningRecord(**d)

        screening_id = rec.id or f"scr_{uuid.uuid4().hex[:12]}"
        rec.id = screening_id

        # Format top factors & clinical payload
        top_factors = d.get("top_factors") or rec.metrics.get("top_factors")
        top_factors_json = json.dumps(top_factors) if top_factors else rec.top_factors_json
        clinical_summary = d.get("clinical_summary") or rec.clinical_summary or f"Estimated CVD probability: {rec.probability*100:.1f}%"
        fhir_bundle_json = d.get("fhir_bundle_json") or rec.fhir_bundle_json

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO screenings (
                    id, patient_id, abha_id, model_name, age_years, gender,
                    height, weight, ap_hi, ap_lo, cholesterol, gluc,
                    smoke, alco, active, bmi, probability, prediction,
                    risk_tier, applied_threshold, explanation_method,
                    top_factors_json, clinical_summary, fhir_bundle_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    screening_id,
                    rec.patient_id,
                    rec.abha_id,
                    rec.model_name,
                    rec.age_years,
                    rec.gender,
                    rec.height,
                    rec.weight,
                    rec.ap_hi,
                    rec.ap_lo,
                    rec.cholesterol,
                    rec.gluc,
                    rec.smoke,
                    rec.alco,
                    rec.active,
                    rec.bmi,
                    rec.probability,
                    rec.prediction,
                    rec.risk_tier,
                    rec.applied_threshold,
                    rec.explanation_method,
                    top_factors_json,
                    clinical_summary,
                    fhir_bundle_json,
                ),
            )

        return rec

    @_hybridmethod
    def list_recent_screenings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent screening events ordered by creation timestamp descending."""
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                SELECT s.*, p.name as patient_name
                FROM screenings s
                LEFT JOIN patients p ON s.patient_id = p.id
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["screening_id"] = item["id"]
                item["risk_probability"] = item["probability"]
                item["risk_prediction"] = item["prediction"]
                if item.get("top_factors_json"):
                    try:
                        item["top_factors"] = json.loads(item["top_factors_json"])
                    except Exception:
                        item["top_factors"] = []
                results.append(item)
            return results

    @_hybridmethod
    def get_screening_by_id(self, screening_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single screening event by ID."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["screening_id"] = res["id"]
            res["risk_probability"] = res["probability"]
            res["risk_prediction"] = res["prediction"]
            res["input_payload"] = {
                "age": res["age_years"],
                "trestbps": res["ap_hi"],
                "chol": res["cholesterol"] * 100 + 100 if res["cholesterol"] > 0 else 200,
            }
            if res.get("top_factors_json"):
                try:
                    res["top_factors"] = json.loads(res["top_factors_json"])
                except Exception:
                    res["top_factors"] = []
            return res

    @_hybridmethod
    def get_screenings_count(self) -> int:
        """Return total count of recorded screenings."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM screenings")
            return int(cur.fetchone()[0])

    # -------------------------------------------------------------------------
    # Audit Logging
    # -------------------------------------------------------------------------

    @_hybridmethod
    def log_audit(
        self,
        action: str,
        user_role: str = "clinician",
        details: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
    ) -> str:
        """Write an immutable entry to the clinical audit trail."""
        log_id = f"aud_{uuid.uuid4().hex[:12]}"
        details_str = json.dumps(details) if details else None
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (id, action, user_role, details_json, client_ip)
                VALUES (?, ?, ?, ?, ?)
                """,
                (log_id, action, user_role, details_str, client_ip),
            )
        return log_id

    @_hybridmethod
    def log_audit_event(
        self,
        action: str,
        actor_id: str = "clinician",
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write structured audit event."""
        full_details = dict(details or {})
        if resource_id:
            full_details["resource_id"] = resource_id
        return self.log_audit(action=action, user_role=actor_id, details=full_details)
