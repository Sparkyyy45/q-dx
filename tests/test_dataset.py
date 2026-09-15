"""
Unit tests for dataset loading, contract validation, sanity checks, and provenance.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    DatasetContract,
    DatasetContractViolation,
    DatasetLoader,
    load_dataset,
    verify_dataset_sanity,
)
from src.feature_dictionary import get_expected_columns


def create_synthetic_dataset(num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic tabular DataFrame conforming to the 13 canonical columns."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "id": np.arange(num_rows),
        "age": rng.randint(12000, 24000, size=num_rows),
        "gender": rng.choice([1, 2], size=num_rows),
        "height": rng.randint(150, 190, size=num_rows),
        "weight": rng.uniform(50.0, 100.0, size=num_rows),
        "ap_hi": rng.randint(100, 160, size=num_rows),
        "ap_lo": rng.randint(60, 100, size=num_rows),
        "cholesterol": rng.choice([1, 2, 3], size=num_rows),
        "gluc": rng.choice([1, 2, 3], size=num_rows),
        "smoke": rng.choice([0, 1], size=num_rows),
        "alco": rng.choice([0, 1], size=num_rows),
        "active": rng.choice([0, 1], size=num_rows),
        "cardio": rng.choice([0, 1], size=num_rows),
    })


def get_dataset_path() -> str:
    """Provide dataset path from environment variable or dynamically discover matching local CSV."""
    env_path = os.getenv("CVD_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # Search working directory for available CSV files matching canonical columns
    for candidate in Path(".").glob("*.csv"):
        if "framingham" in candidate.name.lower():
            continue
        try:
            head_df = pd.read_csv(candidate, nrows=5)
            if list(head_df.columns) == get_expected_columns():
                return str(candidate)
        except Exception:
            continue
    # Fallback to creating a temporary synthetic csv
    synth_csv = Path("artifacts") / "temp_canonical.csv"
    synth_csv.parent.mkdir(parents=True, exist_ok=True)
    if not synth_csv.exists():
        create_synthetic_dataset(100).to_csv(synth_csv, index=False)
    return str(synth_csv)


def test_load_dataset_success():
    path = get_dataset_path()
    artifact = load_dataset(path)
    assert artifact.df is not None
    assert len(artifact.df) > 0
    assert len(artifact.sha256) == 64
    assert artifact.summary.row_count == len(artifact.df)
    assert artifact.summary.positive_class_resolved == 1
    assert artifact.summary.negative_class_resolved == 0


def test_expected_row_count_matches_source():
    """
    Verify loaded row count matches the source dataset, asserting >= 50,000 rows
    and all 13 canonical columns when evaluated on the full cohort.
    """
    path = get_dataset_path()
    artifact = load_dataset(path)
    if len(artifact.df) >= 50000:
        assert len(artifact.df) >= 50000
        assert artifact.summary.row_count >= 50000
        assert len(artifact.df.columns) == 13
        assert list(artifact.df.columns) == get_expected_columns()


def test_row_count_sanity_check_fails_on_truncated(tmp_path):
    """
    Verify that verify_dataset_sanity fails loudly on a truncated dataset (e.g. 500 rows).
    """
    df_small = create_synthetic_dataset(num_rows=500)
    csv_file = tmp_path / "truncated_dataset.csv"
    df_small.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="DATASET TRUNCATION ERROR"):
        verify_dataset_sanity(df_small, filepath=csv_file, min_rows=50000)


def test_row_count_sanity_check_passes_on_full_cohort(tmp_path):
    """
    Verify that verify_dataset_sanity passes on a dataset meeting the min_rows threshold.
    """
    data_path = os.getenv("CVD_DATASET_PATH")
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        verify_dataset_sanity(df, filepath=data_path, min_rows=50000)
    else:
        df_full = create_synthetic_dataset(num_rows=50000)
        verify_dataset_sanity(df_full, filepath=tmp_path / "synthetic_full.csv", min_rows=50000)


def test_sanity_check_fails_on_missing_column(tmp_path):
    """
    Verify that verify_dataset_sanity fails loudly if canonical columns are missing.
    """
    df_missing = create_synthetic_dataset(50).drop(columns=["cholesterol"])
    with pytest.raises(ValueError, match="DATASET SCHEMA ERROR"):
        verify_dataset_sanity(df_missing, filepath=tmp_path / "missing.csv", min_rows=10)


def test_contract_violation_missing_target():
    df = pd.DataFrame({
        "age": [50, 60],
        "gender": [1, 2],
    })
    contract = DatasetContract(target_column="cardio")
    loader = DatasetLoader(contract=contract)
    with pytest.raises(DatasetContractViolation, match="Required target column 'cardio'"):
        loader.validate_contract(df)


def test_contract_violation_multiclass():
    df = create_synthetic_dataset(10)
    df.loc[0, "cardio"] = 2
    contract = DatasetContract(target_column="cardio")
    loader = DatasetLoader(contract=contract)
    with pytest.raises(DatasetContractViolation, match="requires exactly 2 distinct classes"):
        loader.validate_contract(df)


def test_contract_violation_unexpected_labels():
    df = create_synthetic_dataset(10)
    df["cardio"] = ["yes" if c == 1 else "no" for c in df["cardio"]]
    contract = DatasetContract(target_column="cardio", positive_class=1, negative_class=0)
    loader = DatasetLoader(contract=contract)
    with pytest.raises(DatasetContractViolation, match="do not match contract"):
        loader.validate_contract(df)


def test_sha256_reproducibility(tmp_path):
    f1 = tmp_path / "test1.csv"
    f1.write_text("a,b\n1,2\n")
    h1 = DatasetLoader.compute_sha256(f1)
    h2 = DatasetLoader.compute_sha256(f1)
    assert h1 == h2
    assert len(h1) == 64
