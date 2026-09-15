"""
Unit tests for computational efficiency benchmark measurements, dynamic artifact flow,
and complete elimination of hardcoded timing constants.
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


def test_benchmarks_api_reads_computational_efficiency_from_artifact():
    """Verify /api/benchmarks obtains computational efficiency dynamically from metrics.json."""
    metrics_file = Path("artifacts/remediated_v2/metrics.json")
    assert metrics_file.is_file(), "metrics.json must exist"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    assert "computational_efficiency" in metrics_data, "metrics.json must have computational_efficiency"
    expected_comp = metrics_data["computational_efficiency"]
    assert len(expected_comp) > 0

    status, resp, _ = make_request("GET", "/api/benchmarks")
    assert status == 200
    assert resp["status"] == "success"
    assert "computational_efficiency" in resp

    api_comp = resp["computational_efficiency"]
    assert len(api_comp) == len(expected_comp)

    for exp_item, api_item in zip(expected_comp, api_comp):
        assert api_item["model"] == exp_item["model"]
        assert api_item["train_samples"] == exp_item["train_samples"]
        exp_time = exp_item.get("train_time_sec", exp_item.get("training_time_seconds"))
        assert api_item["train_time_sec"] == exp_time
        exp_lat = exp_item.get("inf_latency_ms", exp_item.get("inference_latency_ms_per_sample"))
        assert api_item["inf_latency_ms"] == exp_lat
        # Verify memory_mb is null per protocol (no fabricated RAM)
        assert api_item["memory_mb"] is None


def test_modifying_metrics_artifact_changes_api_dynamically():
    """Verify that modifying the metrics artifact dynamically changes the API response without server restart."""
    metrics_file = Path("artifacts/remediated_v2/metrics.json")
    with open(metrics_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    try:
        modified_data = dict(original_data)
        canary_entry = [
            {
                "model": "Canary Custom Dynamic Model",
                "family": "Canary Family",
                "track": "Track A",
                "train_samples": 99999,
                "test_samples": 12345,
                "training_time_seconds": 1.2345,
                "inference_latency_ms_per_sample": 0.0987,
                "memory_mb": None,
            }
        ]
        modified_data["computational_efficiency"] = canary_entry
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(modified_data, f, indent=2)

        status, resp, _ = make_request("GET", "/api/benchmarks")
        assert status == 200
        comp = resp["computational_efficiency"]
        assert len(comp) == 1
        assert comp[0]["model"] == "Canary Custom Dynamic Model"
        assert comp[0]["train_samples"] == 99999
        assert comp[0]["train_time_sec"] == 1.2345
        assert comp[0]["inf_latency_ms"] == 0.0987
    finally:
        # Restore original artifact
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(original_data, f, indent=2)

    # Verify restoration
    status, restored_resp, _ = make_request("GET", "/api/benchmarks")
    assert status == 200
    assert len(restored_resp["computational_efficiency"]) == len(original_data["computational_efficiency"])


def test_no_hardcoded_timing_constants_in_server():
    """Verify old hardcoded constants (4.12, 8.54, 45.10, 142.5, 210.0, 12.40, 18.25) do not exist in src/api/server.py."""
    server_path = Path("src/api/server.py")
    assert server_path.is_file()

    server_content = server_path.read_text(encoding="utf-8")
    forbidden_constants = ["4.12", "8.54", "45.10", "142.5", "210.0", "12.40", "18.25"]
    found = [c for c in forbidden_constants if c in server_content]
    assert len(found) == 0, f"Forbidden hardcoded constants found in src/api/server.py: {found}"


def test_track_b_description_never_claims_identical_computational_budget():
    """Verify that Track B is documented as identical N=1,000 training-data budget, never identical computational budget."""
    target_files = [
        Path("src/report.py"),
        Path("README.md"),
        Path("architecture.md"),
        Path("artifacts/remediated_v2/pipeline_report.md"),
    ]
    for p in target_files:
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            assert "uniform computational constraints" not in text, f"Found 'uniform computational constraints' in {p}"
            assert "identical computational budget" not in text, f"Found 'identical computational budget' in {p}"
