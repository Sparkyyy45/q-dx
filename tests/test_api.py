"""
Unit tests for the Clinical Software Platform REST API and dashboard handler.
Tests:
- /health and /api/models catalogs
- Strict HTTP 400 rejection on inverted blood pressure and out-of-bounds physiology
- Strict HTTP 503 rejection when model artifact is missing (no synthetic patient fallback)
- Prospective locked threshold enforcement
- Genuine model-level explainability (/api/explain)
"""

from __future__ import annotations

import io
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.api.server import ClinicalPlatformHandler
from src.config import PipelineConfig
from src.features import ClinicalFeatureEngineer
from src.models import create_model
from src.models.serialization import ProductionPipeline
from src.preprocess import build_preprocessor, build_catboost_preprocessor, fit_preprocessor


class MockSocket:
    """Mock socket implementing makefile and sendall for hermetic API testing."""

    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, b: bytes):
        self.wfile.write(b)


def make_request(method: str, path: str, body: Optional[dict] = None) -> Tuple[int, dict, str]:
    """Execute hermetic HTTP request against ClinicalPlatformHandler."""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        raw = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: localhost\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Content-Type: application/json\r\n\r\n"
        ).encode("utf-8") + body_bytes
    else:
        raw = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("utf-8")

    mock = MockSocket(raw)
    _ = ClinicalPlatformHandler(mock, ("127.0.0.1", 54321), None)

    response_raw = mock.wfile.getvalue().decode("utf-8")
    lines = response_raw.split("\r\n")
    status_line = lines[0]
    status_code = int(status_line.split(" ")[1])

    header_end = response_raw.find("\r\n\r\n")
    body_str = response_raw[header_end + 4 :] if header_end != -1 else ""

    try:
        data = json.loads(body_str)
    except Exception:
        data = {}

    return status_code, data, body_str


VALID_TEST_PATIENT: dict = {
    "age_years": 54.0,
    "gender": 2,
    "height": 170.0,
    "weight": 75.0,
    "ap_hi": 130.0,
    "ap_lo": 85.0,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}


def _ensure_test_server_models():
    """Setup lightweight genuine models for testing API endpoints."""
    if "catboost" in ClinicalPlatformHandler.server_models and "vqc" in ClinicalPlatformHandler.server_models:
        return

    cfg = PipelineConfig()
    fe = ClinicalFeatureEngineer()
    df = pd.DataFrame({
        "age": [18250, 20000, 19000, 21000, 18500, 22000],
        "gender": [1, 2, 1, 2, 1, 2],
        "height": [165, 175, 160, 180, 168, 172],
        "weight": [65, 80, 70, 85, 60, 90],
        "ap_hi": [120, 140, 130, 150, 110, 160],
        "ap_lo": [80, 90, 85, 95, 70, 100],
        "cholesterol": [1, 2, 1, 3, 1, 2],
        "gluc": [1, 1, 2, 1, 1, 2],
        "smoke": [0, 1, 0, 1, 0, 1],
        "alco": [0, 0, 1, 0, 0, 1],
        "active": [1, 1, 1, 0, 1, 0],
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    fe.fit(df)
    df_fe = fe.transform(df).drop(columns=["age"])
    num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Classical CatBoost Model Pipeline
    prep_cb = build_catboost_preprocessor(cfg, num_cols, cat_cols)
    prep_art_cb = fit_preprocessor(prep_cb, df_fe, y, num_cols, cat_cols, cfg)
    cb_model = create_model("catboost", config=cfg)
    cb_model.fit(df_fe, y)
    pipe_cb = ProductionPipeline(
        model=cb_model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art_cb,
        preprocessor=prep_cb,
        metadata={"model_key": "catboost", "locked_threshold": 0.485, "model_name": "CatBoost"},
        is_catboost=True,
    )
    ClinicalPlatformHandler.server_models["catboost"] = pipe_cb

    # 2. Quantum Model Pipeline
    prep_q = build_preprocessor(cfg, num_cols, cat_cols)
    prep_art_q = fit_preprocessor(prep_q, df_fe, y, num_cols, cat_cols, cfg)
    vqc_model = create_model("vqc", config=cfg)
    vqc_model.max_train_samples = 10
    vqc_model.n_epochs = 2
    vqc_model.fit(df_fe, y)
    pipe_vqc = ProductionPipeline(
        model=vqc_model,
        feature_engineer=fe,
        preprocessor_artifact=prep_art_q,
        metadata={"model_key": "vqc", "locked_threshold": 0.520, "model_name": "Variational Quantum Classifier"},
    )
    ClinicalPlatformHandler.server_models["vqc"] = pipe_vqc


def test_api_health_endpoint():
    status, data, _ = make_request("GET", "/health")
    assert status == 200
    assert data["status"] == "HEALTHY"
    assert data["registered_models_count"] == 11
    assert "2.0.0" in data["version"]


def test_api_models_catalog():
    status, data, _ = make_request("GET", "/api/models")
    assert status == 200
    assert isinstance(data, list)
    assert len(data) == 11
    model_ids = {m["id"] for m in data}
    assert "vqc" in model_ids
    assert "qsvm" in model_ids
    assert "hybrid_qnn" in model_ids
    assert "catboost" in model_ids


def test_api_predict_missing_model_returns_503():
    """Verify HTTP 503 Service Unavailable when model is not loaded (synthetic fallback removed)."""
    payload = {
        "model": "nonexistent_model",
        "patient": dict(VALID_TEST_PATIENT),
    }
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 503
    assert data["error"] == "MODEL_NOT_LOADED"
    assert "not found" in data["message"]


def test_api_predict_inverted_bp_returns_400():
    """Verify HTTP 400 Bad Request when hemodynamic inversion occurs (ap_hi <= ap_lo)."""
    p = dict(VALID_TEST_PATIENT)
    p["ap_hi"] = 80.0
    p["ap_lo"] = 120.0
    payload = {"model": "catboost", "patient": p}
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400
    assert data["error"] == "INVALID_INPUT"
    assert any("strictly greater than diastolic BP" in detail for detail in data["details"])


def test_api_predict_out_of_bounds_age_returns_400():
    """Verify HTTP 400 Bad Request when patient age violates physiological bounds."""
    p = dict(VALID_TEST_PATIENT)
    p["age_years"] = 150.0
    payload = {"model": "catboost", "patient": p}
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400
    assert data["error"] == "INVALID_INPUT"
    assert any("Patient age (150.0 years) must be between 18 and 120" in detail for detail in data["details"])


def test_api_predict_age_schema_and_boundary_conditions():
    """
    Phase 7 Audit Validation:
    Verify explicit age schema conversion rule and boundary behavior around 120/121:
    - age_years == 120: Valid (maximum allowable age in years)
    - age_years == 121: Rejected (exceeds 120-year maximum)
    - age == 120: Interpreted as 120 years (<= 120) -> Valid
    - age == 121: Interpreted as 121 days (> 120) -> 0.33 years -> Rejected (< 18-year minimum)
    - age == 18.0: Valid (minimum allowable age in years)
    - age_years == 17.9: Rejected (< 18-year minimum)
    - age == 20000: Interpreted as days -> 54.76 years -> Valid
    """
    _ensure_test_server_models()

    # 1. age_years == 120.0 -> Valid
    p1 = dict(VALID_TEST_PATIENT)
    p1["age_years"] = 120.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p1})
    assert status == 200

    # 2. age_years == 121.0 -> Rejected
    p2 = dict(VALID_TEST_PATIENT)
    p2["age_years"] = 121.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p2})
    assert status == 400
    assert any("between 18 and 120 years" in d for d in data.get("details", []))

    # 3. age == 120.0 (treated as 120 years) -> Valid
    p3 = dict(VALID_TEST_PATIENT)
    del p3["age_years"]
    p3["age"] = 120.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p3})
    assert status == 200

    # 4. age == 121.0 (treated as 121 days -> 0.33 years) -> Rejected (< 18)
    p4 = dict(VALID_TEST_PATIENT)
    del p4["age_years"]
    p4["age"] = 121.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p4})
    assert status == 400
    assert any("between 18 and 120 years" in d for d in data.get("details", []))

    # 5. age == 18.0 -> Valid
    p5 = dict(VALID_TEST_PATIENT)
    p5["age_years"] = 18.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p5})
    assert status == 200

    # 6. age_years == 17.9 -> Rejected
    p6 = dict(VALID_TEST_PATIENT)
    p6["age_years"] = 17.9
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p6})
    assert status == 400

    # 7. age == 20000.0 (days -> ~54.76 years) -> Valid
    p7 = dict(VALID_TEST_PATIENT)
    del p7["age_years"]
    p7["age"] = 20000.0
    status, data, _ = make_request("POST", "/api/predict", body={"model": "catboost", "patient": p7})
    assert status == 200


def test_api_predict_classical_model():
    """Verify inference on classical model uses prospective locked threshold."""
    _ensure_test_server_models()
    p = dict(VALID_TEST_PATIENT)
    p.update({
        "age_years": 56.0,
        "gender": 2,
        "ap_hi": 145.0,
        "ap_lo": 92.0,
        "bmi": 29.5,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
    })
    payload = {"model": "catboost", "patient": p}
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 200
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["applied_threshold"] == 0.485  # Model locked threshold
    assert data["risk_tier"] in [
        "Low Estimated Risk (<20%)",
        "Moderate Estimated Risk (20-50%)",
        "High Estimated Risk (>50%)",
    ]
    assert len(data["top_factors"]) > 0
    assert "DISCLAIMER" in data["disclaimer"]


def test_api_predict_quantum_model():
    """Verify inference on quantum model uses locked threshold."""
    _ensure_test_server_models()
    p = dict(VALID_TEST_PATIENT)
    p.update({
        "age_years": 62.0,
        "gender": 1,
        "ap_hi": 150.0,
        "ap_lo": 95.0,
        "bmi": 31.0,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 0,
    })
    payload = {"model": "vqc", "patient": p}
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 200
    assert data["model"] == "Variational Quantum Classifier"
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["applied_threshold"] == 0.520


def test_api_explain_endpoint():
    """Verify /api/explain returns signed feature attributions and method name."""
    _ensure_test_server_models()
    p = dict(VALID_TEST_PATIENT)
    p.update({
        "age_years": 58.0,
        "gender": 2,
        "ap_hi": 140.0,
        "ap_lo": 90.0,
        "bmi": 28.0,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
    })
    payload = {"model": "catboost", "patient": p}
    status, data, _ = make_request("POST", "/api/explain", body=payload)
    assert status == 200
    assert "explanation_method" in data
    assert "top_factors" in data
    assert len(data["top_factors"]) > 0
    first_factor = data["top_factors"][0]
    assert "feature" in first_factor
    assert "attribution" in first_factor
    assert "is_risk_increasing" in first_factor


def test_api_html_dashboard():
    status, _, body = make_request("GET", "/")
    assert status == 200
    assert "<!DOCTYPE html>" in body
    assert "CardioQ Platform" in body
    assert "SIH Problem Statement 3" in body


def test_api_predict_rejects_client_threshold_field():
    """
    ISSUE 2 Validation:
    Verify that any client attempt to submit a custom threshold is strictly rejected with HTTP 400.
    """
    _ensure_test_server_models()
    payload = {
        "model": "catboost",
        "threshold": 0.45,
        "patient": dict(VALID_TEST_PATIENT),
    }
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400
    assert data["error"] == "CLIENT_THRESHOLD_PROHIBITED"
    assert "cannot be altered" in data["message"] or "prohibited" in data["message"].lower()


def test_api_predict_rejects_negative_threshold():
    """ISSUE 2 Validation: Negative threshold must be rejected with HTTP 400."""
    _ensure_test_server_models()
    payload = {
        "model": "catboost",
        "threshold": -0.25,
        "patient": dict(VALID_TEST_PATIENT),
    }
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400


def test_api_predict_rejects_threshold_greater_than_one():
    """ISSUE 2 Validation: Threshold > 1.0 must be rejected with HTTP 400."""
    _ensure_test_server_models()
    payload = {
        "model": "catboost",
        "threshold": 1.75,
        "patient": dict(VALID_TEST_PATIENT),
    }
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400


def test_api_predict_patient_dict_rejects_threshold():
    """ISSUE 2 Validation: Threshold nested inside patient dict must also be rejected with HTTP 400."""
    _ensure_test_server_models()
    p = dict(VALID_TEST_PATIENT)
    p["threshold"] = 0.35
    payload = {
        "model": "catboost",
        "patient": p,
    }
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400


def test_api_explain_uses_same_model_artifact_and_locked_threshold():
    """
    ISSUE 15 Validation:
    Verify /api/explain and /api/predict load the exact same model artifact,
    produce identical probabilities, and enforce the same locked threshold.
    """
    _ensure_test_server_models()
    p = dict(VALID_TEST_PATIENT)
    p.update({
        "age_years": 62.0,
        "gender": 1,
        "ap_hi": 155.0,
        "ap_lo": 95.0,
        "bmi": 29.5,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 0,
    })
    payload = {"model": "catboost", "patient": p}
    status_pred, data_pred, _ = make_request("POST", "/api/predict", body=payload)
    status_exp, data_exp, _ = make_request("POST", "/api/explain", body=payload)

    assert status_pred == 200
    assert status_exp == 200
    assert data_pred["risk_probability"] == data_exp["risk_probability"]
    assert data_pred["applied_threshold"] == data_exp["applied_threshold"]
    assert data_pred["prediction"] == data_exp["prediction"]
    assert data_exp["threshold_source"] == "OOF_Youden"
    assert len(data_exp["top_factors"]) > 0


def test_api_predict_rejects_invalid_height():
    """P0-8 Validation: Height must be between 80 and 250 cm."""
    for bad_ht in [70.0, 260.0]:
        p = dict(VALID_TEST_PATIENT)
        p["height"] = bad_ht
        payload = {"model": "catboost", "patient": p}
        status, data, _ = make_request("POST", "/api/predict", body=payload)
        assert status == 400
        assert data["error"] == "INVALID_INPUT"
        assert any("Height" in err and "between 80 and 250" in err for err in data["details"])


def test_api_predict_rejects_invalid_weight():
    """P0-8 Validation: Weight must be between 20 and 350 kg."""
    for bad_wt in [15.0, 380.0]:
        p = dict(VALID_TEST_PATIENT)
        p["weight"] = bad_wt
        payload = {"model": "catboost", "patient": p}
        status, data, _ = make_request("POST", "/api/predict", body=payload)
        assert status == 400
        assert data["error"] == "INVALID_INPUT"
        assert any("Weight" in err and "between 20 and 350" in err for err in data["details"])


def test_api_predict_rejects_invalid_systolic_bp():
    """P0-8 Validation: Systolic BP must be between 50 and 300 mmHg."""
    for bad_hi in [40.0, 350.0]:
        p = dict(VALID_TEST_PATIENT)
        p["ap_hi"] = bad_hi
        payload = {"model": "catboost", "patient": p}
        status, data, _ = make_request("POST", "/api/predict", body=payload)
        assert status == 400
        assert data["error"] == "INVALID_INPUT"
        assert any("Systolic BP" in err and "between 50 and 300" in err for err in data["details"])


def test_api_predict_rejects_invalid_diastolic_bp():
    """P0-8 Validation: Diastolic BP must be between 30 and 200 mmHg."""
    for bad_lo in [20.0, 220.0]:
        p = dict(VALID_TEST_PATIENT)
        p["ap_lo"] = bad_lo
        payload = {"model": "catboost", "patient": p}
        status, data, _ = make_request("POST", "/api/predict", body=payload)
        assert status == 400
        assert data["error"] == "INVALID_INPUT"
        assert any("Diastolic BP" in err and "between 30 and 200" in err for err in data["details"])


def test_api_predict_rejects_equal_bp():
    """P0-8 Validation: ap_hi == ap_lo must be rejected with HTTP 400 (ap_hi must be > ap_lo)."""
    p = dict(VALID_TEST_PATIENT)
    p["ap_hi"] = 120.0
    p["ap_lo"] = 120.0
    payload = {"model": "catboost", "patient": p}
    status, data, _ = make_request("POST", "/api/predict", body=payload)
    assert status == 400
    assert data["error"] == "INVALID_INPUT"
    assert any("strictly greater than diastolic BP" in err for err in data["details"])


def test_api_predict_rejects_missing_required_clinical_fields():
    """P0-1 Validation: Verify HTTP 400 when any required clinical field is missing (no silent default fabrication)."""
    required_fields = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active"
    ]
    for field in required_fields:
        p = dict(VALID_TEST_PATIENT)
        del p[field]
        payload = {"model": "catboost", "patient": p}
        status, data, _ = make_request("POST", "/api/predict", body=payload)
        assert status == 400
        assert data["error"] == "INVALID_INPUT"
        assert any(f"'{field}'" in err for err in data["details"]), (
            f"Expected error for missing field '{field}', got details: {data['details']}"
        )


def test_api_stateless_modular_loading_without_monolithic_pipeline_bundle():
    """P0-2 Validation: Verify server resolves and predicts from modular decoupled artifacts even if pipeline.joblib is deleted."""
    import tempfile
    from pathlib import Path
    from src.models.serialization import save_production_pipeline, reconstruct_production_pipeline

    _ensure_test_server_models()
    base_pipe = ClinicalPlatformHandler.server_models["catboost"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir) / "artifacts" / "models" / "test_modular_cb"
        save_production_pipeline(base_pipe, model_dir)

        # Delete monolithic pipeline.joblib to guarantee stateless reconstruction
        (model_dir / "pipeline.joblib").unlink()
        assert not (model_dir / "pipeline.joblib").is_file()
        assert (model_dir / "preprocessing.joblib").is_file()
        assert (model_dir / "model.joblib").is_file()

        orig_resolve = ClinicalPlatformHandler._resolve_model
        try:
            def mock_resolve(self_handler, model_key):
                if model_key == "test_modular_cb":
                    return reconstruct_production_pipeline(model_dir)
                return orig_resolve(self_handler, model_key)

            ClinicalPlatformHandler._resolve_model = mock_resolve

            payload = {
                "model": "test_modular_cb",
                "patient": dict(VALID_TEST_PATIENT),
            }
            status, data, _ = make_request("POST", "/api/predict", body=payload)
            assert status == 200
            assert "risk_probability" in data
            assert 0.0 <= data["risk_probability"] <= 1.0
            assert data["applied_threshold"] == 0.485
        finally:
            ClinicalPlatformHandler._resolve_model = orig_resolve


