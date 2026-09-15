"""
Unit tests validating zero-hardcoding invariants in CardioQ:
- Configuration dynamism: settings dynamically update from environment variables.
- Real model inference: models compute genuine continuous mathematical probabilities, not hardcoded mock numbers.
- Genuine clinical guideline evaluation: dynamic thresholds and rules produce context-aware clinical outputs.
- Genuine SQLite persistence: state actually persists across sessions.
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
from config.settings import CardioQSettings
from src.models import MODEL_REGISTRY, create_model
from src.interop.icmr_guidelines import get_icmr_clinical_recommendations
from src.database.connection import get_db_connection, init_db
from src.database.models import PatientRecord, ScreeningRecord
from src.database.repository import ClinicalRepository


def test_settings_dynamic_environment_overrides():
    # Set env vars and verify CardioQSettings respects them
    os.environ["CARDIOQ_PORT"] = "9999"
    os.environ["CARDIOQ_HOST"] = "0.0.0.0"
    os.environ["CARDIOQ_MAX_UPLOAD_MB"] = "75"
    os.environ["CARDIOQ_DEFAULT_MODEL"] = "catboost"

    custom_settings = CardioQSettings()
    try:
        assert custom_settings.PORT == 9999
        assert custom_settings.HOST == "0.0.0.0"
        assert custom_settings.MAX_UPLOAD_MB == 75
        assert custom_settings.DEFAULT_MODEL == "catboost"
    finally:
        del os.environ["CARDIOQ_PORT"]
        del os.environ["CARDIOQ_HOST"]
        del os.environ["CARDIOQ_MAX_UPLOAD_MB"]
        del os.environ["CARDIOQ_DEFAULT_MODEL"]


def test_models_produce_real_continuous_probabilities():
    # Verify that models do not return a single hardcoded constant
    # Generate 5 distinct patient feature profiles
    rng = np.random.RandomState(42)
    X = rng.randn(10, 13)

    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        if model_name not in MODEL_REGISTRY:
            continue
        model = create_model(model_name)
        # Train on toy data
        y = (X[:, 0] + X[:, 3] > 0).astype(int)
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert len(probs) == 10
        # Probabilities must be valid continuous numbers in [0.0, 1.0]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        # Probabilities should vary across distinct samples (not all identical constants)
        assert len(np.unique(np.round(probs, 4))) > 1, f"{model_name} should produce sample-dependent probabilities"


def test_icmr_recommendations_are_dynamic():
    # Test that varying inputs produce distinct triage outputs
    rec_low = get_icmr_clinical_recommendations(
        probability=0.12,
        ap_hi=110,
        ap_lo=70,
        cholesterol=1,
        gluc=1,
        smoke=0,
    )
    rec_high = get_icmr_clinical_recommendations(
        probability=0.94,
        ap_hi=190,
        ap_lo=115,
        cholesterol=3,
        gluc=3,
        smoke=1,
    )

    assert rec_low["urgency"] != rec_high["urgency"]
    assert rec_low["triage_level"] != rec_high["triage_level"]
    assert rec_low["follow_up_window"] != rec_high["follow_up_window"]
    assert len(rec_low["clinical_actions"]) < len(rec_high["clinical_actions"])



def test_real_sqlite_persistence_roundtrip(tmp_path: pathlib.Path):
    db_file = str(tmp_path / "persistence_test.db")
    repo = ClinicalRepository(db_file)

    p = PatientRecord(
        patient_id="P-PERSIST-1",
        abha_id="11-2233-4455-6677",
        name="Ananya Roy",
        age=52,
        sex=0,
    )
    repo.upsert_patient(p)

    s = ScreeningRecord(
        screening_id="SCR-P-1",
        patient_id="P-PERSIST-1",
        model_name="xgboost",
        risk_probability=0.684,
        risk_prediction=1,
        risk_tier="High Risk",
        input_payload={"age": 52, "trestbps": 140},
    )
    repo.save_screening(s)

    # Reopen brand new repository instance to confirm disk persistence
    new_repo = ClinicalRepository(db_file)
    loaded_p = new_repo.get_patient_by_id("P-PERSIST-1")
    loaded_s = new_repo.get_screening_by_id("SCR-P-1")

    assert loaded_p is not None
    assert loaded_p.name == "Ananya Roy"
    assert loaded_s is not None
    assert loaded_s["risk_probability"] == 0.684
    assert loaded_s["model_name"] == "xgboost"

