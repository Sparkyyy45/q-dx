"""
Dataset loading, contract validation, and provenance tracking.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


from src.feature_dictionary import (
    FEATURE_DICTIONARY,
    get_expected_columns,
    get_required_feature_columns,
)


class DatasetContractViolation(ValueError):
    """Raised when a dataset violates contract invariants."""
    pass


@dataclass(frozen=True)
class DatasetContract:
    """
    Formal contract defining schema requirements, target semantics,
    demographic indicators, and column exclusions for binary CVD classification.
    """
    target_column: str = "cardio"
    positive_class: int = 1
    negative_class: int = 0
    identifier_columns: List[str] = field(default_factory=lambda: ["id"])
    demographic_columns: List[str] = field(default_factory=lambda: ["age", "gender", "age_years"])
    group_column: Optional[str] = None
    drop_columns: List[str] = field(default_factory=list)
    required_feature_columns: List[str] = field(default_factory=get_required_feature_columns)


@dataclass
class DatasetSummary:
    """Statistical summary and metadata of the loaded dataset."""
    filepath: str
    sha256_hash: str
    row_count: int
    column_count: int
    feature_count: int
    target_distribution: Dict[Any, int]
    target_proportions: Dict[Any, float]
    positive_class_resolved: Any
    negative_class_resolved: Any
    missing_values_by_column: Dict[str, int]
    total_missing_values: int
    duplicate_rows_count: int
    column_dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    basic_statistics: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "sha256_hash": self.sha256_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "feature_count": self.feature_count,
            "target_distribution": {str(k): int(v) for k, v in self.target_distribution.items()},
            "target_proportions": {str(k): float(v) for k, v in self.target_proportions.items()},
            "positive_class_resolved": self.positive_class_resolved,
            "negative_class_resolved": self.negative_class_resolved,
            "missing_values_by_column": self.missing_values_by_column,
            "total_missing_values": self.total_missing_values,
            "duplicate_rows_count": self.duplicate_rows_count,
            "column_dtypes": self.column_dtypes,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "basic_statistics": self.basic_statistics,
        }


@dataclass
class DatasetArtifact:
    """Artifact containing the validated raw DataFrame and complete metadata."""
    df: pd.DataFrame
    contract: DatasetContract
    summary: DatasetSummary
    sha256: str


class DatasetLoader:
    """
    Dedicated dataset loader responsible for:
    - Verifying raw file integrity via SHA-256
    - Validating schema against the DatasetContract
    - Enforcing strict binary target semantics without silent assumptions
    - Calculating rich dataset profile and summary statistics
    """

    def __init__(self, contract: Optional[DatasetContract] = None):
        self.contract = contract or DatasetContract()

    @staticmethod
    def compute_sha256(filepath: str | Path) -> str:
        """Compute SHA-256 checksum of raw dataset file."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_contract(self, df: pd.DataFrame) -> None:
        """
        Validate dataset against contract. Raises DatasetContractViolation on any mismatch.
        """
        # 1. Target presence
        if self.contract.target_column not in df.columns:
            raise DatasetContractViolation(
                f"Contract violation: Required target column '{self.contract.target_column}' "
                f"is not present in dataset columns: {list(df.columns)}"
            )

        # 2. Target binary validation & distinct classes
        target_series = df[self.contract.target_column].dropna()
        unique_targets = set(target_series.unique())

        if len(unique_targets) != 2:
            raise DatasetContractViolation(
                f"Contract violation: Binary CVD classification requires exactly 2 distinct classes, "
                f"but found {len(unique_targets)}: {unique_targets}"
            )

        expected_classes = {self.contract.positive_class, self.contract.negative_class}
        if unique_targets != expected_classes:
            raise DatasetContractViolation(
                f"Contract violation: Target values {unique_targets} do not match contract "
                f"expected positive ({self.contract.positive_class}) and negative "
                f"({self.contract.negative_class}) classes. Never silently guess target semantics."
            )

        # 3. Required feature presence check
        missing_required = [
            c for c in self.contract.required_feature_columns if c not in df.columns
        ]
        if missing_required:
            raise DatasetContractViolation(
                f"Contract violation: Missing required clinical columns: {missing_required}"
            )

        # 4. Identifier columns check
        for id_col in self.contract.identifier_columns:
            if id_col in df.columns and df[id_col].nunique() != len(df):
                # Warning or notice, but let's ensure it exists
                pass

        # 5. Group column check if specified
        if self.contract.group_column and self.contract.group_column not in df.columns:
            raise DatasetContractViolation(
                f"Contract violation: Declared group column '{self.contract.group_column}' "
                f"not found in dataset."
            )

    def generate_summary(
        self, df: pd.DataFrame, filepath: str, sha256_hash: str
    ) -> DatasetSummary:
        """Compute full statistical summary of dataset."""
        target_col = self.contract.target_column
        target_counts = df[target_col].value_counts(dropna=False).to_dict()
        total_rows = len(df)
        target_props = {k: v / total_rows for k, v in target_counts.items()}

        non_feature_cols = set(self.contract.identifier_columns + [target_col])
        if self.contract.group_column:
            non_feature_cols.add(self.contract.group_column)

        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

        missing_by_col = df.isnull().sum().to_dict()
        total_missing = sum(missing_by_col.values())
        dup_rows = int(df.duplicated().sum())
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Basic statistics for numeric columns
        stats_dict = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            for col, col_stats in desc.items():
                stats_dict[col] = {
                    "mean": float(col_stats.get("mean", 0.0)),
                    "std": float(col_stats.get("std", 0.0)),
                    "min": float(col_stats.get("min", 0.0)),
                    "25%": float(col_stats.get("25%", 0.0)),
                    "50%": float(col_stats.get("50%", 0.0)),
                    "75%": float(col_stats.get("75%", 0.0)),
                    "max": float(col_stats.get("max", 0.0)),
                }

        return DatasetSummary(
            filepath=str(filepath),
            sha256_hash=sha256_hash,
            row_count=total_rows,
            column_count=len(df.columns),
            feature_count=len(feature_cols),
            target_distribution=target_counts,
            target_proportions=target_props,
            positive_class_resolved=self.contract.positive_class,
            negative_class_resolved=self.contract.negative_class,
            missing_values_by_column=missing_by_col,
            total_missing_values=total_missing,
            duplicate_rows_count=dup_rows,
            column_dtypes=dtypes_dict,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            basic_statistics=stats_dict,
        )

    def load(self, filepath: str | Path) -> DatasetArtifact:
        """
        Load dataset from disk, verify SHA-256, validate contract, and return DatasetArtifact.
        """
        path = Path(filepath)
        sha256 = self.compute_sha256(path)
        df = pd.read_csv(path)
        self.validate_contract(df)
        summary = self.generate_summary(df, str(path), sha256)

        return DatasetArtifact(
            df=df,
            contract=self.contract,
            summary=summary,
            sha256=sha256,
        )


def load_dataset(
    filepath: str | Path, contract: Optional[DatasetContract] = None
) -> DatasetArtifact:
    """Convenience functional interface for loading and validating dataset."""
    loader = DatasetLoader(contract=contract)
    return loader.load(filepath)


def verify_dataset_sanity(
    df: pd.DataFrame,
    filepath: Optional[str | Path] = None,
    sha256_hash: Optional[str] = None,
    min_rows: int = 50000,
) -> None:
    """
    Sanity check dataset integrity and schema completeness:
    1. Logs loaded row count and SHA-256 hash.
    2. Asserts row count is above sane minimum threshold (min_rows >= 50,000, NOT an exact 70,000).
       Warns loudly if below threshold and raises ValueError.
    3. Verifies all 13 canonical columns against official FEATURE_DICTIONARY.
    """
    row_count = len(df)
    col_count = len(df.columns)
    path_str = str(filepath) if filepath is not None else "in-memory dataframe"
    hash_str = sha256_hash or "N/A"

    print(f"  [Sanity Check] Dataset File: '{path_str}'")
    print(f"  [Sanity Check] Loaded Records: {row_count:,} across {col_count} columns")
    print(f"  [Sanity Check] SHA-256 Checksum: {hash_str}")

    # Check canonical schema
    expected_cols = get_expected_columns()
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DATASET SCHEMA ERROR: Missing expected canonical columns in '{path_str}': {missing_cols}"
        )

    # Check row count threshold
    if row_count < min_rows:
        warning_msg = (
            "\n" + "=" * 80 + "\n"
            "⚠️  DATASET TRUNCATION / UNDERSIZED ALERT!\n"
            f"Loaded {row_count:,} rows, which is below the minimum sane threshold of {min_rows:,} rows.\n"
            f"The expected canonical cohort contains ~70,000 records.\n"
            f"The file at '{path_str}' appears severely truncated or incomplete.\n"
            "Halting execution to prevent invalid model training on truncated data.\n"
            + "=" * 80 + "\n"
        )
        print(warning_msg, file=sys.stderr)
        raise ValueError(
            f"DATASET TRUNCATION ERROR: Loaded {row_count:,} rows from '{path_str}', "
            f"expected at least {min_rows:,} records (~70,000 canonical rows)."
        )

    print(f"  ✓ Sanity check passed: {row_count:,} rows (>= {min_rows:,}) and all {len(expected_cols)} canonical columns verified.")

