"""
Repository layer providing clean CRUD operations for clinical records, patient histories, and audit trails.
Supports both instance-based access (with custom database paths) and class-level operations.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import secrets
import uuid
from datetime import timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.database.connection import get_db_connection, init_db
from src.database.models import (
    AuditLogRecord,
    PatientRecord,
    ScreeningRecord,
    UserRecord,
    UserSessionRecord,
)


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

    # -------------------------------------------------------------------------
    # Authentication & User Management
    # -------------------------------------------------------------------------

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plain text password with a unique salt using PBKDF2-HMAC-SHA256."""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            100_000,
        ).hex()
        return f"{salt}${pwd_hash}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Constant-time verification of password against stored PBKDF2 hash."""
        try:
            salt_hex, hash_hex = stored_hash.split("$")
            computed = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                100_000,
            ).hex()
            return hmac.compare_digest(computed, hash_hex)
        except Exception:
            return False

    @_hybridmethod
    def create_user(
        self,
        name: str,
        email: str,
        password: str,
        role: str = "Clinician",
    ) -> Tuple[bool, Optional[UserRecord], str]:
        """Register a new user account with unique email and hashed password."""
        norm_email = email.strip().lower()
        if not norm_email or "@" not in norm_email:
            return False, None, "Invalid email address format."
        if not name or len(name.strip()) < 2:
            return False, None, "Full name must be at least 2 characters."
        if not password or len(password) < 6:
            return False, None, "Password must be at least 6 characters long."

        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        pwd_hash = self.hash_password(password)

        with self._get_conn() as conn:
            # Check existing email
            cur = conn.execute("SELECT id FROM users WHERE email = ?", (norm_email,))
            if cur.fetchone():
                return False, None, "An account with this email address already exists."

            conn.execute(
                """
                INSERT INTO users (id, name, email, password_hash, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, name.strip(), norm_email, pwd_hash, role),
            )

        user = self.get_user_by_id(user_id)
        self.log_audit(
            action="USER_REGISTERED",
            user_role=role,
            details={"user_id": user_id, "email": norm_email},
        )
        return True, user, "User registered successfully."

    @_hybridmethod
    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Retrieve user record by email address."""
        norm_email = email.strip().lower()
        with self._get_conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            cur = conn.execute(
                "SELECT id, name, email, password_hash, role, created_at, last_login FROM users WHERE email = ?",
                (norm_email,),
            )
            row = cur.fetchone()
            if row:
                return UserRecord(**row)
        return None

    @_hybridmethod
    def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Retrieve user record by user ID."""
        with self._get_conn() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            cur = conn.execute(
                "SELECT id, name, email, password_hash, role, created_at, last_login FROM users WHERE id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                return UserRecord(**row)
        return None

    @_hybridmethod
    def authenticate_user(
        self,
        email: str,
        password: str,
        client_ip: Optional[str] = None,
    ) -> Tuple[bool, Optional[UserRecord], str]:
        """Authenticate user credentials and update last_login timestamp."""
        user = self.get_user_by_email(email)
        if not user:
            return False, None, "Invalid email address or password."

        if not self.verify_password(password, user.password_hash):
            self.log_audit(
                action="USER_LOGIN_FAILED",
                user_role=user.role,
                details={"email": email.strip().lower()},
                client_ip=client_ip,
            )
            return False, None, "Invalid email address or password."

        # Update last_login
        now_iso = datetime.datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (now_iso, user.id),
            )
        user.last_login = now_iso

        self.log_audit(
            action="USER_LOGGED_IN",
            user_role=user.role,
            details={"user_id": user.id, "email": user.email},
            client_ip=client_ip,
        )
        return True, user, "Authentication successful."

    @_hybridmethod
    def create_session(self, user_id: str, duration_days: int = 7) -> str:
        """Generate and persist a secure session token for authenticated user."""
        token = secrets.token_urlsafe(32)
        expires_at = (
            datetime.datetime.now(timezone.utc) + timedelta(days=duration_days)
        ).isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions (token, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (token, user_id, expires_at),
            )
        return token

    @_hybridmethod
    def get_session_user(self, token: str) -> Optional[UserRecord]:
        """Validate session token and retrieve corresponding user if not expired."""
        if not token or not str(token).strip():
            return None

        clean_token = str(token).strip()
        now_iso = datetime.datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            cur = conn.execute(
                """
                SELECT user_id, expires_at FROM user_sessions
                WHERE token = ?
                """,
                (clean_token,),
            )
            row = cur.fetchone()
            if not row:
                return None

            user_id, expires_at = row[0], row[1]
            if expires_at and expires_at < now_iso:
                # Expired session; prune it
                conn.execute("DELETE FROM user_sessions WHERE token = ?", (clean_token,))
                return None

        return self.get_user_by_id(user_id)

    @_hybridmethod
    def delete_session(self, token: str) -> bool:
        """Invalidate and remove session token from persistence."""
        if not token:
            return False
        with self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM user_sessions WHERE token = ?",
                (str(token).strip(),),
            )
            return cur.rowcount > 0

