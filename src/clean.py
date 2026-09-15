"""
Pre-split data cleaning module for cardiovascular dataset.
Removes duplicate records, inverted blood pressures, and non-biological BMI outliers
prior to train/test partitioning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset import DatasetContract
from src.audit import AuditReport


@dataclass
class CleaningLog:
    """Summary record of pre-split data cleaning actions."""
    initial_rows: int
    cleaned_rows: int
    total_removed: int
    duplicates_removed: int
    inverted_bp_removed: int
    extreme_bmi_removed: int
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_dataset(
    df: pd.DataFrame,
    contract: Optional[DatasetContract] = None,
    audit_report: Optional[AuditReport] = None,
) -> Tuple[pd.DataFrame, CleaningLog]:
    """
    Clean dataset prior to train/test splitting:
    1. Drops duplicate rows when ignoring the identifier column.
    2. Drops physiologically impossible inverted blood pressure rows (ap_hi < ap_lo).
    3. Drops extreme non-biological BMI outliers (< 10 or > 70).

    Returns:
        (cleaned_df, cleaning_log)
    """
    contract = contract or DatasetContract()
    initial_len = len(df)
    id_cols = contract.identifier_columns

    # 1. Detect duplicates excluding ID
    non_id_cols = [c for c in df.columns if c not in id_cols]
    dup_mask = df.duplicated(subset=non_id_cols)
    dup_count = int(dup_mask.sum())

    # 2. Detect inverted blood pressure (ap_hi < ap_lo)
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        inv_bp_mask = df["ap_hi"] < df["ap_lo"]
        inv_bp_count = int(inv_bp_mask.sum())
    else:
        inv_bp_mask = pd.Series(False, index=df.index)
        inv_bp_count = 0

    # 3. Detect extreme BMI outliers (< 10 or > 70)
    if "bmi" in df.columns:
        bmi_mask = (df["bmi"] < 10.0) | (df["bmi"] > 70.0)
        bmi_count = int(bmi_mask.sum())
    elif "height" in df.columns and "weight" in df.columns:
        # Compute BMI from raw clinical measurements: weight (kg) / (height (m))^2
        computed_bmi = df["weight"] / ((df["height"] / 100.0) ** 2)
        bmi_mask = (computed_bmi < 10.0) | (computed_bmi > 70.0)
        bmi_count = int(bmi_mask.sum())
    else:
        bmi_mask = pd.Series(False, index=df.index)
        bmi_count = 0

    # Combine all exclusion criteria
    drop_mask = dup_mask | inv_bp_mask | bmi_mask
    total_dropped = int(drop_mask.sum())

    cleaned_df = df[~drop_mask].copy().reset_index(drop=True)
    cleaned_len = len(cleaned_df)

    reasons = [
        f"Dropped {dup_count} exact duplicate rows (excluding '{', '.join(id_cols)}').",
        f"Dropped {inv_bp_count} records with impossible inverted blood pressures (systolic ap_hi < diastolic ap_lo).",
        f"Dropped {bmi_count} records with non-biological BMI values (< 10 or > 70 kg/m^2).",
    ]

    log = CleaningLog(
        initial_rows=initial_len,
        cleaned_rows=cleaned_len,
        total_removed=total_dropped,
        duplicates_removed=dup_count,
        inverted_bp_removed=inv_bp_count,
        extreme_bmi_removed=bmi_count,
        reasons=reasons,
    )

    return cleaned_df, log
