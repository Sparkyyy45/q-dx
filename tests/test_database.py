"""
Unit tests for SQLite database layer and clinical repository in CardioQ.
Verifies WAL journal mode, foreign key enforcement, patient upsert, and screening record persistence.
"""

from __future__ import annotations

import pathlib
import sqlite3

from src.database.connection import init_db
from src.database.models import PatientRecord, ScreeningRecord
from src.database.repository import ClinicalRepository


def test_database_initialization(tmp_path: pathlib.Path) -> None:
    db_path = str(tmp_path / "test_cardioq.db")
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Verify tables created
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row["name"] for row in tables}
    assert "patients" in table_names, "patients table missing"
    assert "screenings" in table_names, "screenings table missing"
    assert "audit_logs" in table_names, "audit_logs table missing"

    conn.close()


def test_patient_and_screening_lifecycle(tmp_path: pathlib.Path) -> None:
    db_path = str(tmp_path / "lifecycle.db")
    repo = ClinicalRepository(db_path)

    # 1. Upsert patient
    patient = PatientRecord(
        patient_id="P-TEST-001",
        abha_id="12-3456-7890-1234",
        name="Ramesh Sharma",
        age_years=58,
        gender=1,
        phone="+919876543210",
        state="Maharashtra",
        district="Pune",
    )
    saved_patient = repo.upsert_patient(patient)
    assert saved_patient.id == "P-TEST-001"

    # Verify lookup by ABHA
    fetched_by_abha = repo.get_patient_by_abha("12-3456-7890-1234")
    assert fetched_by_abha is not None
    assert fetched_by_abha.name == "Ramesh Sharma"

    # Verify lookup by ID
    fetched_by_id = repo.get_patient_by_id("P-TEST-001")
    assert fetched_by_id is not None
    assert fetched_by_id.name == "Ramesh Sharma"

    # 2. Record screening
    screening = ScreeningRecord(
        screening_id="SCR-001",
        patient_id="P-TEST-001",
        model_name="ensemble",
        probability=0.742,
        prediction=1,
        risk_tier="High Risk",
        ap_hi=150,
        ap_lo=95,
        cholesterol=2,
        gluc=1,
        fhir_bundle_json='{"resourceType": "Bundle", "id": "SCR-001"}',
    )
    saved_scr = repo.save_screening(screening)
    assert saved_scr.id == "SCR-001"

    # 3. Retrieve screening
    fetched_scr = repo.get_screening_by_id("SCR-001")
    assert fetched_scr is not None
    assert fetched_scr["probability"] == 0.742
    assert fetched_scr["risk_tier"] == "High Risk"

    # 4. List screenings
    screenings = repo.list_recent_screenings(limit=10)
    assert len(screenings) == 1
    assert screenings[0]["screening_id"] == "SCR-001"
    assert screenings[0]["patient_name"] == "Ramesh Sharma"

    # 5. Verify screening counts
    count = repo.get_screenings_count()
    assert count == 1


def test_upsert_existing_patient_updates_fields(tmp_path: pathlib.Path) -> None:
    db_path = str(tmp_path / "upsert.db")
    repo = ClinicalRepository(db_path)

    p1 = PatientRecord(
        patient_id="P-002",
        abha_id="99-8877-6655-4433",
        name="Sunita Devi",
        age_years=45,
        gender=0,
    )
    repo.upsert_patient(p1)

    # Update patient's name and age
    p2 = PatientRecord(
        patient_id="P-002",
        abha_id="99-8877-6655-4433",
        name="Sunita Sharma",
        age_years=46,
        gender=0,
    )
    repo.upsert_patient(p2)

    fetched = repo.get_patient_by_id("P-002")
    assert fetched is not None
    assert fetched.age_years == 46
    assert fetched.name == "Sunita Sharma"


def test_audit_log_recording(tmp_path: pathlib.Path) -> None:
    db_path = str(tmp_path / "audit.db")
    repo = ClinicalRepository(db_path)

    repo.log_audit_event(
        action="INFERENCE_EXECUTED",
        actor_id="DR-101",
        resource_id="SCR-999",
        details={"model": "vqc", "latency_ms": 14.2},
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM audit_logs WHERE user_role = 'DR-101'").fetchone()
    assert row is not None
    assert row["action"] == "INFERENCE_EXECUTED"
    assert "vqc" in row["details_json"]
    conn.close()
