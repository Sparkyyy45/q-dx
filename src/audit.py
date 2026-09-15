"""
Comprehensive pre-training data audit and data quality report.
Runs prior to modeling to identify leakage, identifiers, invalid values, and anomalies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset import DatasetContract
from src.feature_dictionary import FEATURE_DICTIONARY


@dataclass
class AnomalyFinding:
    """Represents a specific anomaly or risk identified during audit."""
    category: str  # "leakage", "identifier", "constant", "invalid_values", "redundancy"
    column: str
    severity: str  # "HIGH", "MEDIUM", "LOW", "INFO"
    detail: str
    action_recommended: str


@dataclass
class AuditReport:
    """Structured audit report containing all pre-modeling data checks."""
    dimensions: Tuple[int, int]
    row_count: int
    column_count: int
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    missing_value_counts: Dict[str, int]
    missing_value_percentages: Dict[str, float]
    duplicate_rows_total: int
    duplicate_rows_excluding_id: int
    constant_features: List[str]
    near_constant_features: Dict[str, float]  # col -> dominant value ratio
    categorical_cardinality: Dict[str, int]
    target_class_counts: Dict[str, int]
    class_imbalance_ratio: float
    potential_identifier_columns: List[str]
    potential_leakage_columns: List[str]
    redundant_duplicate_columns: List[Tuple[str, str]]
    invalid_or_infinite_counts: Dict[str, int]
    biological_plausibility_flags: List[str]
    suspicious_target_related_columns: List[str]
    findings: List[AnomalyFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        d = asdict(self)
        d["dimensions"] = list(self.dimensions)
        return d

    def to_markdown(self) -> str:
        """Generate a formatted markdown audit summary."""
        lines = [
            "# Pre-Modeling Data Audit Report",
            "",
            f"- **Dataset Dimensions**: {self.row_count} rows × {self.column_count} columns",
            f"- **Numeric Features ({len(self.numeric_features)})**: {', '.join(self.numeric_features)}",
            f"- **Categorical Features ({len(self.categorical_features)})**: {', '.join(self.categorical_features)}",
            f"- **Total Exact Duplicates**: {self.duplicate_rows_total} (Excluding ID: {self.duplicate_rows_excluding_id})",
            f"- **Class Balance**: {self.target_class_counts} (Imbalance Ratio: {self.class_imbalance_ratio:.3f})",
            "",
            "## Key Findings & Risk Assessment",
            "",
            "| Category | Column | Severity | Detail | Recommended Action |",
            "|---|---|---|---|---|",
        ]
        if not self.findings:
            lines.append("| None | None | INFO | No critical anomalies detected. | Proceed to preprocessing. |")
        else:
            for f in self.findings:
                lines.append(f"| {f.category} | `{f.column}` | **{f.severity}** | {f.detail} | {f.action_recommended} |")

        lines.extend([
            "",
            "## Biological Plausibility Checks",
            "",
        ])
        if self.biological_plausibility_flags:
            for flag in self.biological_plausibility_flags:
                lines.append(f"- ⚠️ {flag}")
        else:
            lines.append("- ✅ All inspected clinical ranges pass standard boundary checks.")

        return "\n".join(lines)


class DatasetAudit:
    """
    Comprehensive auditor for tabular medical datasets.
    Audits dimensions, leakage, identifiers, redundancy, and biological plausibility
    without mutating or silently dropping any data.
    """

    def __init__(self, contract: Optional[DatasetContract] = None):
        self.contract = contract or DatasetContract()

    def audit(self, df: pd.DataFrame) -> AuditReport:
        """Execute all audit routines and return structured AuditReport."""
        target_col = self.contract.target_column
        id_cols = self.contract.identifier_columns
        findings: List[AnomalyFinding] = []

        # 1. Dimensions and column types
        rows, cols = df.shape
        non_target_cols = [c for c in df.columns if c != target_col]

        numeric_cols = df[non_target_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[non_target_cols].select_dtypes(exclude=[np.number]).columns.tolist()

        # 2. Missing values & Infinites
        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = (df.isnull().sum() / rows * 100).to_dict()

        invalid_counts: Dict[str, int] = {}
        for col in numeric_cols:
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                invalid_counts[col] = inf_count
                findings.append(AnomalyFinding(
                    category="invalid_values",
                    column=col,
                    severity="HIGH",
                    detail=f"Contains {inf_count} infinite values (+/- inf).",
                    action_recommended="Impute or clip prior to scaling."
                ))

        # 3. Duplicate checks
        dup_total = int(df.duplicated().sum())
        cols_without_id = [c for c in df.columns if c not in id_cols]
        dup_no_id = int(df.duplicated(subset=cols_without_id).sum()) if cols_without_id else 0
        if dup_no_id > 0:
            findings.append(AnomalyFinding(
                category="redundancy",
                column="[multiple_features]",
                severity="MEDIUM",
                detail=f"Found {dup_no_id} duplicate rows when ignoring ID columns.",
                action_recommended="Retain for training if representing distinct clinical encounters, or record provenance."
            ))

        # 4. Constant & near-constant features
        constant_features = []
        near_constant_features = {}
        for col in non_target_cols:
            val_counts = df[col].value_counts(dropna=False, normalize=True)
            if len(val_counts) <= 1:
                constant_features.append(col)
                findings.append(AnomalyFinding(
                    category="constant",
                    column=col,
                    severity="HIGH",
                    detail="Zero variance feature (only 1 unique value).",
                    action_recommended="Exclude from model training."
                ))
            elif val_counts.iloc[0] > 0.99:
                near_constant_features[col] = float(val_counts.iloc[0])
                findings.append(AnomalyFinding(
                    category="constant",
                    column=col,
                    severity="MEDIUM",
                    detail=f"Near-constant feature ({val_counts.iloc[0]*100:.2f}% identical values).",
                    action_recommended="Monitor during feature selection."
                ))

        # 5. Categorical cardinality
        cat_cardinality = {col: int(df[col].nunique()) for col in categorical_cols}

        # 6. Target class distribution
        if target_col in df.columns:
            target_counts = {str(k): int(v) for k, v in df[target_col].value_counts().items()}
            counts_list = list(target_counts.values())
            imbalance_ratio = (min(counts_list) / max(counts_list)) if counts_list and max(counts_list) > 0 else 0.0
        else:
            target_counts = {}
            imbalance_ratio = 0.0

        # 7. Potential identifier columns
        potential_identifiers = []
        for col in df.columns:
            if col == target_col:
                continue
            # Check 100% uniqueness or typical naming
            is_unique = (df[col].nunique() == rows)
            name_indicates_id = any(term in col.lower() for term in ["id", "uuid", "index", "subject_id", "patient_id"])
            if is_unique or (name_indicates_id and df[col].nunique() > rows * 0.9):
                potential_identifiers.append(col)
                findings.append(AnomalyFinding(
                    category="identifier",
                    column=col,
                    severity="HIGH",
                    detail=f"Column exhibits identifier characteristics ({df[col].nunique()} unique values / {rows} rows).",
                    action_recommended="Strictly exclude from model features to prevent memorization leakage."
                ))

        # 8. Potential leakage & exact duplicate columns
        redundant_pairs: List[Tuple[str, str]] = []
        col_list = list(df.columns)
        for i in range(len(col_list)):
            for j in range(i + 1, len(col_list)):
                c1, c2 = col_list[i], col_list[j]
                if df[c1].dtype == df[c2].dtype:
                    try:
                        if (df[c1] == df[c2]).all():
                            redundant_pairs.append((c1, c2))
                            findings.append(AnomalyFinding(
                                category="leakage",
                                column=c2,
                                severity="HIGH",
                                detail=f"Column is an exact identical duplicate of '{c1}'.",
                                action_recommended=f"Drop redundant duplicate '{c2}' to avoid multicollinearity."
                            ))
                    except Exception:
                        pass

        potential_leakage = [pair[1] for pair in redundant_pairs]

        # Check correlation with target for numeric columns
        suspicious_target_cols = []
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            for col in numeric_cols:
                if col == target_col or col in potential_identifiers:
                    continue
                try:
                    if df[col].std() > 1e-6 and df[target_col].std() > 1e-6:
                        corr = float(df[col].corr(df[target_col]))
                        if abs(corr) > 0.95:
                            potential_leakage.append(col)
                            findings.append(AnomalyFinding(
                                category="leakage",
                                column=col,
                                severity="HIGH",
                                detail=f"Extremely high correlation with target (Pearson r = {corr:.4f}).",
                                action_recommended="Audit clinical provenance to verify whether this column leaks future diagnosis."
                            ))
                except Exception:
                    pass

        # Check column names with target-like semantics
        for col in df.columns:
            if col != target_col and any(term in col.lower() for term in ["cardio", "disease", "diagnosis", "target", "label"]):
                suspicious_target_cols.append(col)
                findings.append(AnomalyFinding(
                    category="leakage",
                    column=col,
                    severity="MEDIUM",
                    detail=f"Column name '{col}' shares diagnostic naming with target.",
                    action_recommended="Confirm clinical role and isolate from predictive features if derived post-outcome."
                ))

        # 9. Biological plausibility checks (Domain-specific for Cardiovascular data)
        bio_flags: List[str] = []
        if "ap_hi" in df.columns and "ap_lo" in df.columns:
            inverted_bp = int((df["ap_hi"] < df["ap_lo"]).sum())
            if inverted_bp > 0:
                bio_flags.append(f"Found {inverted_bp} rows where systolic BP (ap_hi) < diastolic BP (ap_lo).")
                findings.append(AnomalyFinding(
                    category="invalid_values",
                    column="ap_hi, ap_lo",
                    severity="HIGH",
                    detail=f"{inverted_bp} instances where systolic BP < diastolic BP.",
                    action_recommended="Apply clinical boundary correction or IQR clipping."
                ))

        if "height" in df.columns:
            extreme_heights = int(((df["height"] < 100) | (df["height"] > 220)).sum())
            if extreme_heights > 0:
                bio_flags.append(f"Found {extreme_heights} height measurements outside [100cm, 220cm].")

        if "weight" in df.columns:
            extreme_weights = int(((df["weight"] < 30) | (df["weight"] > 200)).sum())
            if extreme_weights > 0:
                bio_flags.append(f"Found {extreme_weights} weight measurements outside [30kg, 200kg].")

        if "bmi" in df.columns:
            extreme_bmi = int(((df["bmi"] < 10) | (df["bmi"] > 70)).sum())
            if extreme_bmi > 0:
                bio_flags.append(f"Found {extreme_bmi} BMI measurements outside [10, 70]. Max BMI observed: {df['bmi'].max():.1f}.")
                findings.append(AnomalyFinding(
                    category="invalid_values",
                    column="bmi",
                    severity="MEDIUM",
                    detail=f"{extreme_bmi} extreme BMI values detected (min: {df['bmi'].min():.1f}, max: {df['bmi'].max():.1f}).",
                    action_recommended="Apply robust IQR clipping to eliminate physiological measurement errors."
                ))
        elif "height" in df.columns and "weight" in df.columns:
            computed_bmi = df["weight"] / ((df["height"] / 100.0) ** 2)
            extreme_bmi = int(((computed_bmi < 10) | (computed_bmi > 70)).sum())
            if extreme_bmi > 0:
                bio_flags.append(f"Found {extreme_bmi} calculated BMI values outside [10, 70]. Max BMI observed: {computed_bmi.max():.1f}.")
                findings.append(AnomalyFinding(
                    category="invalid_values",
                    column="bmi (calculated)",
                    severity="MEDIUM",
                    detail=f"{extreme_bmi} extreme BMI values detected (min: {computed_bmi.min():.1f}, max: {computed_bmi.max():.1f}).",
                    action_recommended="Apply robust IQR clipping or pre-split cleaning to eliminate physiological measurement errors."
                ))

        # 10. Unit consistency audit against FEATURE_DICTIONARY
        if "age" in df.columns:
            age_min = float(df["age"].min())
            age_max = float(df["age"].max())
            if age_max < 150.0:
                findings.append(AnomalyFinding(
                    category="unit_mismatch",
                    column="age",
                    severity="HIGH",
                    detail=(
                        f"Age values appear to be in calendar years (min: {age_min:.1f}, max: {age_max:.1f}) instead of "
                        f"days as defined in FEATURE_DICTIONARY ('int, days')."
                    ),
                    action_recommended="Convert age to days (years * 365.25) to conform to FEATURE_DICTIONARY specification."
                ))
            else:
                bio_flags.append(
                    f"Age unit confirmed in days per FEATURE_DICTIONARY: observed range {age_min:,.0f}–{age_max:,.0f} days "
                    f"(~{age_min/365.25:.1f}–{age_max/365.25:.1f} years)."
                )

        return AuditReport(
            dimensions=(rows, cols),
            row_count=rows,
            column_count=cols,
            feature_names=non_target_cols,
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
            missing_value_counts=missing_counts,
            missing_value_percentages=missing_pcts,
            duplicate_rows_total=dup_total,
            duplicate_rows_excluding_id=dup_no_id,
            constant_features=constant_features,
            near_constant_features=near_constant_features,
            categorical_cardinality=cat_cardinality,
            target_class_counts=target_counts,
            class_imbalance_ratio=imbalance_ratio,
            potential_identifier_columns=potential_identifiers,
            potential_leakage_columns=list(set(potential_leakage)),
            redundant_duplicate_columns=redundant_pairs,
            invalid_or_infinite_counts=invalid_counts,
            biological_plausibility_flags=bio_flags,
            suspicious_target_related_columns=suspicious_target_cols,
            findings=findings,
        )


def audit_dataset(
    df: pd.DataFrame, contract: Optional[DatasetContract] = None
) -> AuditReport:
    """Functional interface to audit dataset."""
    auditor = DatasetAudit(contract=contract)
    return auditor.audit(df)
