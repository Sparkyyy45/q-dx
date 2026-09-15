"""
Secure Dataset Management, Ingestion, Auditing, and Schema Validation for CardioQ Platform.
Enforces:
- File size bounds (max 15MB)
- Path traversal protection and filename sanitization
- Structural CSV integrity verification
- Automated dataset auditing (rows, columns, missing values, duplicates, distributions)
- Target and schema identification without synthetic data fabrication
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import settings

MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_BYTES
UPLOAD_DIR = settings.UPLOADS_DIR


@dataclass
class DatasetAuditSummary:
    dataset_id: str
    filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    column_types: Dict[str, str]
    missing_values_total: int
    missing_values_by_column: Dict[str, int]
    duplicate_rows_count: int
    candidate_targets: List[str]
    preview_rows: List[Dict[str, Any]]
    class_distributions: Dict[str, Dict[str, int]]
    numeric_columns: List[str]
    categorical_columns: List[str]
    is_cardio_schema: bool
    schema_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and shell injection."""
    base = os.path.basename(filename)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    if not safe or safe.startswith("."):
        safe = f"dataset_{uuid.uuid4().hex[:8]}.csv"
    return safe


def audit_dataframe(
    df: pd.DataFrame,
    dataset_id: str,
    filename: str,
    file_size_bytes: int = 0,
) -> DatasetAuditSummary:
    """Execute deep structural audit on ingested DataFrame."""
    row_count, col_count = df.shape
    columns = list(df.columns)

    col_types = {}
    num_cols = []
    cat_cols = []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            col_types[c] = "numeric"
            num_cols.append(c)
        else:
            col_types[c] = "categorical"
            cat_cols.append(c)

    missing_by_col = {c: int(df[c].isna().sum()) for c in columns}
    total_missing = sum(missing_by_col.values())
    dup_count = int(df.duplicated().sum())

    candidate_targets = []
    class_dist = {}
    for c in columns:
        uniques = df[c].dropna().unique()
        if len(uniques) in (2, 3, 4, 5):
            candidate_targets.append(c)
            val_counts = df[c].value_counts().to_dict()
            class_dist[c] = {str(k): int(v) for k, v in val_counts.items()}

    expected_cardio_cols = {"age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"}
    present_cols = set(columns)
    is_cardio_schema = expected_cardio_cols.issubset(present_cols)
    schema_status = "CARDIOQ_CANONICAL_SCHEMA" if is_cardio_schema else "USER_GENERAL_SCHEMA"

    preview = []
    for _, row in df.head(5).iterrows():
        r_dict = {}
        for c in columns:
            val = row[c]
            if pd.isna(val):
                r_dict[c] = None
            elif isinstance(val, (np.integer, int)):
                r_dict[c] = int(val)
            elif isinstance(val, (np.floating, float)):
                r_dict[c] = round(float(val), 4)
            else:
                r_dict[c] = str(val)
        preview.append(r_dict)

    return DatasetAuditSummary(
        dataset_id=dataset_id,
        filename=filename,
        file_size_bytes=file_size_bytes,
        row_count=row_count,
        column_count=col_count,
        columns=columns,
        column_types=col_types,
        missing_values_total=total_missing,
        missing_values_by_column=missing_by_col,
        duplicate_rows_count=dup_count,
        candidate_targets=candidate_targets,
        preview_rows=preview,
        class_distributions=class_dist,
        numeric_columns=num_cols,
        categorical_columns=cat_cols,
        is_cardio_schema=is_cardio_schema,
        schema_status=schema_status,
    )


def save_uploaded_csv(
    file_content: Union[str, bytes],
    original_filename: str,
    upload_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str], Optional[DatasetAuditSummary], Optional[Path]]:
    """
    Validate, safely parse, audit, and persist an uploaded CSV file.
    Returns: (success, error_message, audit_summary, stored_path)
    """
    out_dir = upload_dir or UPLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(file_content, str):
        content_bytes = file_content.encode("utf-8")
    else:
        content_bytes = file_content

    size_bytes = len(content_bytes)
    if size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes).", None, None
    if size_bytes > MAX_UPLOAD_BYTES:
        return False, f"Uploaded file exceeds maximum allowable size of {MAX_UPLOAD_BYTES // (1024*1024)} MB.", None, None

    safe_name = sanitize_filename(original_filename)
    if not safe_name.lower().endswith(".csv"):
        return False, "Invalid file format. Only standard CSV (.csv) files are supported.", None, None

    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    dest_path = out_dir / f"{dataset_id}_{safe_name}"

    try:
        df = pd.read_csv(io.BytesIO(content_bytes))
    except Exception as exc:
        return False, f"Failed to parse CSV file: {exc}", None, None

    if df.empty:
        return False, "CSV file contains no records (empty dataset).", None, None

    if df.shape[1] < 2:
        return False, f"Dataset must contain at least 2 columns (features and target), found {df.shape[1]}.", None, None

    audit_summary = audit_dataframe(
        df=df,
        dataset_id=dataset_id,
        filename=safe_name,
        file_size_bytes=size_bytes,
    )

    try:
        with open(dest_path, "wb") as f:
            f.write(content_bytes)

        audit_meta_path = out_dir / f"{dataset_id}_audit.json"
        with open(audit_meta_path, "w") as f:
            json.dump(audit_summary.to_dict(), f, indent=2)
    except Exception as exc:
        return False, f"Failed to persist dataset to disk: {exc}", None, None

    return True, None, audit_summary, dest_path


def list_available_datasets(upload_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List canonical platform datasets and any user-uploaded datasets."""
    out_dir = upload_dir or UPLOAD_DIR
    datasets = []

    cardio_path = Path("cardio_train_fixed (1).csv")
    if cardio_path.is_file():
        datasets.append({
            "dataset_id": "canonical_cardio_train",
            "name": "Canonical Cardiovascular Cohort (Kaggle)",
            "filename": "cardio_train_fixed (1).csv",
            "path": str(cardio_path),
            "records": 70000,
            "columns": 13,
            "target": "cardio",
            "type": "Primary Development Cohort",
            "is_cardio_schema": True,
        })

    fram_path = Path("framingham.csv")
    if fram_path.is_file():
        datasets.append({
            "dataset_id": "canonical_framingham",
            "name": "Framingham Heart Study Cohort",
            "filename": "framingham.csv",
            "path": str(fram_path),
            "records": 4240,
            "columns": 16,
            "target": "TenYearCHD",
            "type": "External OOD Transportability Cohort",
            "is_cardio_schema": False,
        })

    wdbc_path = Path("data/wdbc_cancer.csv")
    if wdbc_path.is_file():
        datasets.append({
            "dataset_id": "canonical_wdbc_cancer",
            "name": "Wisconsin Diagnostic Breast Cancer (WDBC - High-Dim Oncology)",
            "filename": "wdbc_cancer.csv",
            "path": str(wdbc_path),
            "records": 569,
            "columns": 31,
            "target": "diagnosis_malignant",
            "type": "High-Dimensional Early Oncology Cohort",
            "is_cardio_schema": False,
        })

    if out_dir.is_dir():
        for audit_file in sorted(out_dir.glob("*_audit.json"), reverse=True):
            try:
                with open(audit_file, "r") as f:
                    summary = json.load(f)
                ds_id = summary.get("dataset_id", "")
                csv_files = list(out_dir.glob(f"{ds_id}_*.csv"))
                if csv_files:
                    datasets.append({
                        "dataset_id": ds_id,
                        "name": f"Uploaded: {summary.get('filename')}",
                        "filename": summary.get("filename"),
                        "path": str(csv_files[0]),
                        "records": summary.get("row_count"),
                        "columns": summary.get("column_count"),
                        "target": None,  # User-uploaded datasets require explicit target selection
                        "candidate_targets": summary.get("candidate_targets", []),
                        "type": "User Ingested Dataset",
                        "is_cardio_schema": summary.get("is_cardio_schema", False),
                    })
            except Exception:
                continue

    return datasets
