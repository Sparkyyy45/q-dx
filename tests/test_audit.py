"""
Unit tests for pre-modeling data audit.
"""

import numpy as np
import pandas as pd
import pytest

from src.audit import audit_dataset
from src.dataset import DatasetContract


def test_audit_detects_identifiers_and_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cat_a": ["Normal", "High", "Normal", "High"],
        "cat_b": ["Normal", "High", "Normal", "High"],  # Exact duplicate of cat_a
        "constant_col": [1, 1, 1, 1],
        "cardio": [0, 1, 0, 1],
    })
    contract = DatasetContract(target_column="cardio", identifier_columns=["id"])
    report = audit_dataset(df, contract=contract)

    assert "id" in report.potential_identifier_columns
    assert "constant_col" in report.constant_features
    assert ("cat_a", "cat_b") in report.redundant_duplicate_columns
    assert len(report.findings) >= 3


def test_audit_detects_invalid_values():
    df = pd.DataFrame({
        "id": [1, 2],
        "ap_hi": [80, 120],  # First row ap_hi (80) < ap_lo (100)
        "ap_lo": [100, 80],
        "inf_col": [1.0, np.inf],
        "cardio": [0, 1],
    })
    contract = DatasetContract(target_column="cardio", identifier_columns=["id"])
    report = audit_dataset(df, contract=contract)

    assert "inf_col" in report.invalid_or_infinite_counts
    assert any("systolic BP" in flag for flag in report.biological_plausibility_flags)


def test_audit_markdown_generation():
    df = pd.DataFrame({
        "id": [1, 2],
        "cardio": [0, 1],
        "age": [50, 60],
    })
    report = audit_dataset(df, DatasetContract())
    md = report.to_markdown()
    assert "# Pre-Modeling Data Audit Report" in md
    assert "Dataset Dimensions" in md


def test_clean_dataset():
    from src.clean import clean_dataset

    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "age": [50, 60, 50, 45, 55, 65],
        "gender": [1, 2, 1, 2, 1, 2],
        "ap_hi": [120, 80, 120, 130, 140, 150],   # Row 2 has ap_hi (80) < ap_lo (100) -> drop
        "ap_lo": [80, 100, 80, 85, 90, 95],
        "bmi": [24.0, 26.0, 24.0, 5.0, 85.0, 28.0], # Row 4 has bmi=5.0 (<10) -> drop, Row 5 has bmi=85.0 (>70) -> drop
        "cardio": [0, 1, 0, 1, 0, 1],
    })
    # Note: Row 3 is an exact duplicate of Row 1 (excluding id) -> drop

    contract = DatasetContract(target_column="cardio", identifier_columns=["id"])
    cleaned_df, log = clean_dataset(df, contract=contract)

    assert log.initial_rows == 6
    assert log.duplicates_removed == 1
    assert log.inverted_bp_removed == 1
    assert log.extreme_bmi_removed == 2
    # Rows 1 and 6 remain -> 2 cleaned rows
    assert log.cleaned_rows == 2
    assert len(cleaned_df) == 2
    assert list(cleaned_df["id"]) == [1, 6]
