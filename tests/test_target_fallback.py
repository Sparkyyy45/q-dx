"""
Unit tests verifying the complete elimination of implicit target fallback
for user-uploaded / custom datasets (Fix #3).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

from src.api.server import ClinicalPlatformHandler


class MockSocket:
    """Mock socket for hermetic API handler testing."""

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
    status_code = int(lines[0].split(" ")[1])
    header_end = response_raw.find("\r\n\r\n")
    body_str = response_raw[header_end + 4 :] if header_end != -1 else ""
    try:
        data = json.loads(body_str)
    except Exception:
        data = {}
    return status_code, data, body_str


def _generate_synthetic_csv(last_col_name: str = "custom_label", target_col_name: str = "custom_label") -> str:
    """Generate minimal valid CSV for training tests."""
    lines = ["feature_age,feature_bp," + (f"{target_col_name},{last_col_name}" if target_col_name != last_col_name else last_col_name)]
    import numpy as np
    np.random.seed(42)
    for i in range(20):
        age = 40 + i
        bp = 110 + (i * 2)
        if target_col_name != last_col_name:
            t = 1 if i % 2 == 0 else 0
            last = 0 if i % 2 == 0 else 1
            lines.append(f"{age},{bp},{t},{last}")
        else:
            t = 1 if i % 2 == 0 else 0
            lines.append(f"{age},{bp},{t}")
    return "\n".join(lines) + "\n"


def test_custom_dataset_requires_explicit_target():
    """Rule 11A: Upload a valid custom CSV and call /api/train without a target -> HTTP 400 TARGET_REQUIRED."""
    csv_content = _generate_synthetic_csv()
    upload_res = make_request("POST", "/api/datasets/upload", body={"filename": "test_require_target.csv", "content": csv_content})
    assert upload_res[0] == 200
    ds_id = upload_res[1]["dataset_id"]

    # 1. Target completely omitted
    status, data, _ = make_request("POST", "/api/train", body={"dataset_id": ds_id, "models": ["logistic_regression"]})
    assert status == 400
    assert data["error"] == "TARGET_REQUIRED"
    assert "target" in data["message"].lower()

    # 2. Target explicitly None
    status, data, _ = make_request("POST", "/api/train", body={"dataset_id": ds_id, "target_column": None, "models": ["logistic_regression"]})
    assert status == 400
    assert data["error"] == "TARGET_REQUIRED"

    # 3. Target empty string / whitespace
    status, data, _ = make_request("POST", "/api/train", body={"dataset_id": ds_id, "target_column": "   ", "models": ["logistic_regression"]})
    assert status == 400
    assert data["error"] == "TARGET_REQUIRED"


def test_custom_dataset_never_uses_last_column_as_target():
    """Rule 11B: Create dataset where last column is binary but NOT desired target. Verify rejection rather than using last column."""
    # Last column is 'unrelated_flag', while desired target is 'actual_risk_outcome'
    csv_content = _generate_synthetic_csv(last_col_name="unrelated_flag", target_col_name="actual_risk_outcome")
    upload_res = make_request("POST", "/api/datasets/upload", body={"filename": "test_no_last_col.csv", "content": csv_content})
    assert upload_res[0] == 200
    ds_id = upload_res[1]["dataset_id"]

    status, data, _ = make_request("POST", "/api/train", body={"dataset_id": ds_id, "models": ["logistic_regression"]})
    assert status == 400
    assert data["error"] == "TARGET_REQUIRED"
    # Verify no job executed
    assert "job" not in data


def test_explicit_custom_target_is_used():
    """Rule 11C: Provide a valid target and verify that exact column is used throughout pipeline and artifacts."""
    csv_content = _generate_synthetic_csv(last_col_name="unrelated_flag", target_col_name="actual_risk_outcome")
    upload_res = make_request("POST", "/api/datasets/upload", body={"filename": "test_explicit_target.csv", "content": csv_content})
    assert upload_res[0] == 200
    ds_id = upload_res[1]["dataset_id"]

    status, data, _ = make_request("POST", "/api/train", body={
        "dataset_id": ds_id,
        "target_column": "actual_risk_outcome",
        "models": ["logistic_regression"],
        "random_seed": 42,
    })
    assert status == 200
    assert data["status"] == "completed"
    job = data["job"]
    assert job["target_column"] == "actual_risk_outcome"
    assert "unrelated_flag" in job["feature_columns"]
    assert "actual_risk_outcome" not in job["feature_columns"]

    # Verify job artifacts on disk
    job_id = job["job_id"]
    job_dir = Path("artifacts/training_jobs") / job_id
    assert job_dir.is_dir()

    # Verify job_summary.json
    with open(job_dir / "job_summary.json", "r") as f:
        job_summary = json.load(f)
    assert job_summary["target_column"] == "actual_risk_outcome"

    # Verify training_config.json
    with open(job_dir / "training_config.json", "r") as f:
        training_config = json.load(f)
    assert training_config["target_column"] == "actual_risk_outcome"

    # Verify model metadata.json
    with open(job_dir / "logistic_regression" / "metadata.json", "r") as f:
        m_meta = json.load(f)
    assert m_meta["target_column"] == "actual_risk_outcome"


def test_canonical_dataset_target_is_explicit():
    """Rule 11D: Verify canonical cardio dataset explicitly uses 'cardio' and Framingham uses 'TenYearCHD'."""
    # 1. Check catalog
    status, datasets, _ = make_request("GET", "/api/datasets")
    assert status == 200
    cardio_ds = next((d for d in datasets if d["dataset_id"] == "canonical_cardio_train"), None)
    assert cardio_ds is not None
    assert cardio_ds["target"] == "cardio"

    fram_ds = next((d for d in datasets if d["dataset_id"] == "canonical_framingham"), None)
    assert fram_ds is not None
    assert fram_ds["target"] == "TenYearCHD"

    # 2. Train on canonical without specifying target uses 'cardio'
    status, data, _ = make_request("POST", "/api/train", body={
        "dataset_id": "canonical_cardio_train",
        "models": ["logistic_regression"],
        "random_seed": 42,
    })
    assert status == 200
    assert data["job"]["target_column"] == "cardio"


def test_candidate_targets_are_not_auto_selected():
    """Rule 11E: Verify candidate-target detection in audit does not mutate the actual selected target."""
    csv_content = (
        "feature_1,binary_opt_a,binary_opt_b,binary_opt_c\n"
        "10,0,1,0\n"
        "20,1,0,1\n"
        "30,0,1,0\n"
        "40,1,0,1\n"
        "50,0,1,0\n"
        "60,1,0,1\n"
    )
    status, upload_res, _ = make_request("POST", "/api/datasets/upload", body={
        "filename": "multi_candidates.csv",
        "content": csv_content,
    })
    assert status == 200
    ds_id = upload_res["dataset_id"]
    audit = upload_res["audit"]

    # Candidate targets detected
    assert "binary_opt_a" in audit["candidate_targets"]
    assert "binary_opt_b" in audit["candidate_targets"]
    assert "binary_opt_c" in audit["candidate_targets"]

    # Dataset entry in /api/datasets MUST NOT auto-select any candidate target
    status, datasets, _ = make_request("GET", "/api/datasets")
    assert status == 200
    entry = next(d for d in datasets if d["dataset_id"] == ds_id)
    assert entry["target"] is None, "Uploaded dataset entry in catalog must have target=None"
    assert len(entry["candidate_targets"]) >= 3

    # Attempting to train without explicit target MUST be rejected
    status, train_res, _ = make_request("POST", "/api/train", body={
        "dataset_id": ds_id,
        "models": ["logistic_regression"],
    })
    assert status == 400
    assert train_res["error"] == "TARGET_REQUIRED"
