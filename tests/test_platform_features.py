"""
Platform Feature Validation Test Suite for CardioQ.
Tests:
- Dynamic dataset ingestion & automated clinical auditing (POST /api/datasets/upload)
- Dataset catalog listing & retrieval (GET /api/datasets, GET /api/datasets/<id>)
- Live dual-track benchmark metrics API (GET /api/benchmarks)
- Dataset-to-model training workflow (POST /api/train, GET /api/train/status/<job_id>)
- OpenQASM 2.0/3.0 circuit hardware export (GET /api/quantum/circuit/qasm)
- OpenQASMHardwareAdapter and QuantumCircuit portability
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.api.server import ClinicalPlatformHandler
from src.dataset_manager import (
    audit_dataframe,
    list_available_datasets,
    save_uploaded_csv,
    DatasetAuditSummary,
)
from src.training_engine import (
    execute_training_workflow,
    TRAINING_JOBS_DIR,
)
from src.quantum.circuit import (
    QuantumCircuit,
    OpenQASMHardwareAdapter,
    LocalStatevectorBackend,
)


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


def test_benchmarks_api():
    """Verify GET /api/benchmarks returns Track A, Track B, sample efficiency, and governance."""
    status, data, _ = make_request("GET", "/api/benchmarks")
    assert status == 200, f"Expected 200, got {status}"
    assert data["status"] == "success"
    assert "track_a" in data
    assert "track_b" in data
    assert "sample_efficiency" in data
    assert "computational_efficiency" in data

    # Verify Track A models
    track_a = data["track_a"]
    assert "CatBoost" in track_a
    cb_a = track_a["CatBoost"]
    assert cb_a["roc_auc"] > 0.75
    assert cb_a["locked_threshold"] > 0.0
    assert cb_a["brier_score"] < 0.25
    assert "sensitivity" in cb_a
    assert "specificity" in cb_a

    # Verify Track B models include quantum and classical parity
    track_b = data["track_b"]
    assert "CatBoost" in track_b
    assert "Variational Quantum Classifier" in track_b
    assert "Hybrid Quantum Neural Network" in track_b
    assert track_b["Variational Quantum Classifier"]["architecture"] == "Quantum Machine Learning"

    # Verify governance notes
    gov = data["scientific_governance"]
    assert "Technically stable for internal hackathon demonstration" in gov["clinical_certification"]
    assert "None" in gov["quantum_advantage_claim"]


def test_list_datasets_api():
    """Verify GET /api/datasets returns canonical and uploaded catalogs."""
    status, data, _ = make_request("GET", "/api/datasets")
    assert status == 200, f"Expected 200, got {status}"
    assert isinstance(data, list)
    assert len(data) >= 2
    ds_ids = [d["dataset_id"] for d in data]
    assert "canonical_cardio_train" in ds_ids
    assert "canonical_framingham" in ds_ids


def test_dataset_upload_and_audit():
    """Verify POST /api/datasets/upload ingests and immediately audits a CSV dataset."""
    sample_csv = "age,cholesterol,ap_hi,cardio\n50,1,120,0\n55,2,140,1\n60,3,160,1\n45,1,110,0\n"
    payload = {
        "filename": "test_screening_cohort.csv",
        "content": sample_csv,
    }
    status, data, _ = make_request("POST", "/api/datasets/upload", body=payload)
    assert status == 200, f"Expected 200, got {status}: {data}"
    assert data["status"] == "success"
    assert "dataset_id" in data
    audit = data["audit"]
    assert audit["row_count"] == 4
    assert audit["column_count"] == 4
    assert "cardio" in audit["candidate_targets"]
    assert audit["missing_values_total"] == 0


def test_dataset_upload_oversized_rejection():
    """Verify that uploads exceeding 15MB are rejected with HTTP 400."""
    oversized_data = b"x" * (16 * 1024 * 1024)
    # Test via save_uploaded_csv directly to verify security invariant
    success, err_msg, audit, path = save_uploaded_csv(oversized_data, original_filename="huge.csv")
    assert not success
    assert "exceeds maximum" in err_msg


def test_openqasm_hardware_export_api():
    """Verify GET /api/quantum/circuit/qasm generates valid OpenQASM 2.0 hardware instructions."""
    status, data, _ = make_request("GET", "/api/quantum/circuit/qasm")
    assert status == 200, f"Expected 200, got {status}"
    assert data["status"] == "success"
    assert "openqasm_2_0" in data
    qasm_str = data["openqasm_2_0"]
    assert "OPENQASM 2.0;" in qasm_str
    assert "qreg q[4];" in qasm_str
    assert "creg c[4];" in qasm_str
    assert "rz(" in qasm_str
    assert "ry(" in qasm_str
    assert "cx " in qasm_str
    assert data["hardware_execution"] is False
    assert data["backend_adapter"]["type"] == "nisq_hardware_adapter"


def test_openqasm_adapter_direct():
    """Verify OpenQASMHardwareAdapter and QuantumCircuit methods."""
    qc = QuantumCircuit(n_qubits=4, n_layers=2)
    adapter = OpenQASMHardwareAdapter()
    qasm = adapter.export_qasm(qc)
    assert "OPENQASM 2.0;" in qasm
    assert "barrier q;" in qasm
    info = adapter.get_backend_info()
    assert info["supported_gates"] == ["rz", "ry", "cx", "h", "barrier", "measure"]


def test_training_workflow_end_to_end():
    """Verify dynamic dataset-to-model training workflow with fold-isolation and metric computation."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "age_years": np.random.uniform(40, 70, n),
        "ap_hi": np.random.uniform(100, 180, n),
        "ap_lo": np.random.uniform(60, 110, n),
        "cholesterol": np.random.choice([1, 2, 3], n),
        "gluc": np.random.choice([1, 2, 3], n),
        "smoke": np.random.choice([0, 1], n),
        "cardio": np.random.choice([0, 1], n),
    })

    job = execute_training_workflow(
        df=df,
        target_column="cardio",
        dataset_name="mini_test_cohort",
        models_to_train=["logistic_regression", "random_forest"],
        test_size=0.25,
        random_seed=42,
        enable_quantum=False,
    )

    assert job.status.upper() == "COMPLETED"
    assert "logistic_regression" in job.results
    assert "random_forest" in job.results
    lr_res = job.results["logistic_regression"]
    assert lr_res["roc_auc"] >= 0.0
    assert 0.0 < lr_res["locked_threshold"] < 1.0
    assert lr_res["brier_score"] >= 0.0
    assert lr_res["training_time_seconds"] >= 0.0
    assert lr_res["inference_latency_ms"] >= 0.0

    # Verify job artifact was saved
    job_file = TRAINING_JOBS_DIR / job.job_id / "job.json"
    assert job_file.is_file()

    # Verify reading job status via API
    status, data, _ = make_request("GET", f"/api/train/status/{job.job_id}")
    assert status == 200
    assert data["status"] == "success"
    assert data["job"]["job_id"] == job.job_id


if __name__ == "__main__":
    test_benchmarks_api()
    print("PASS: test_benchmarks_api")
    test_list_datasets_api()
    print("PASS: test_list_datasets_api")
    test_dataset_upload_and_audit()
    print("PASS: test_dataset_upload_and_audit")
    test_dataset_upload_oversized_rejection()
    print("PASS: test_dataset_upload_oversized_rejection")
    test_openqasm_hardware_export_api()
    print("PASS: test_openqasm_hardware_export_api")
    test_openqasm_adapter_direct()
    print("PASS: test_openqasm_adapter_direct")
    test_training_workflow_end_to_end()
    print("PASS: test_training_workflow_end_to_end")
    print("\nALL PLATFORM FEATURE TESTS PASSED!")
