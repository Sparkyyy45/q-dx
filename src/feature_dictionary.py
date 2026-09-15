"""
Official Feature Dictionary for Cardiovascular Disease Risk Prediction Dataset.
Single source-of-truth metadata defining canonical schema, types, units, and clinical descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FeatureMetadata:
    """Canonical metadata defining schema, domain category, units, and clinical meaning."""
    column_name: str
    feature_type: str  # "Identifier", "Objective", "Examination", "Subjective", "Target"
    unit_or_encoding: str
    description: str
    is_target: bool = False
    is_identifier: bool = False


# Authoritative clinical disclaimer on measurement reliability
EXAMINATION_VS_SUBJECTIVE_NOTE = (
    "Examination features (measured systolic/diastolic blood pressure, laboratory cholesterol, and glucose) "
    "provide significantly more objective and reliable physiological signals than Subjective features "
    "(self-reported smoking, alcohol intake, and physical activity), which are inherently vulnerable to "
    "under-reporting, recall bias, and social desirability effects."
)


# Official 13-column canonical feature dictionary
FEATURE_DICTIONARY: Dict[str, FeatureMetadata] = {
    "id": FeatureMetadata(
        column_name="id",
        feature_type="Identifier",
        unit_or_encoding="int",
        description="Unique patient encounter identifier (must be excluded from predictive features)",
        is_identifier=True,
    ),
    "age": FeatureMetadata(
        column_name="age",
        feature_type="Objective",
        unit_or_encoding="int, days",
        description="Chronological age recorded in elapsed days (biological range ~10,000–25,000 days)",
    ),
    "gender": FeatureMetadata(
        column_name="gender",
        feature_type="Objective",
        unit_or_encoding="categorical code",
        description="Biological sex (1 = female, 2 = male)",
    ),
    "height": FeatureMetadata(
        column_name="height",
        feature_type="Objective",
        unit_or_encoding="int, cm",
        description="Body height measured in centimeters",
    ),
    "weight": FeatureMetadata(
        column_name="weight",
        feature_type="Objective",
        unit_or_encoding="float, kg",
        description="Body weight measured in kilograms",
    ),
    "ap_hi": FeatureMetadata(
        column_name="ap_hi",
        feature_type="Examination",
        unit_or_encoding="int (systolic BP)",
        description="Systolic blood pressure measured at clinical examination encounter (mmHg)",
    ),
    "ap_lo": FeatureMetadata(
        column_name="ap_lo",
        feature_type="Examination",
        unit_or_encoding="int (diastolic BP)",
        description="Diastolic blood pressure measured at clinical examination encounter (mmHg)",
    ),
    "cholesterol": FeatureMetadata(
        column_name="cholesterol",
        feature_type="Examination",
        unit_or_encoding="1=normal, 2=above normal, 3=well above normal",
        description="Total serum cholesterol ordinal clinical lab category",
    ),
    "gluc": FeatureMetadata(
        column_name="gluc",
        feature_type="Examination",
        unit_or_encoding="1=normal, 2=above normal, 3=well above normal",
        description="Serum fasting glucose ordinal clinical lab category",
    ),
    "smoke": FeatureMetadata(
        column_name="smoke",
        feature_type="Subjective",
        unit_or_encoding="binary",
        description="Active self-reported tobacco smoking status (0 = non-smoker, 1 = active smoker)",
    ),
    "alco": FeatureMetadata(
        column_name="alco",
        feature_type="Subjective",
        unit_or_encoding="binary",
        description="Self-reported regular alcohol intake (0 = no alcohol, 1 = regular intake)",
    ),
    "active": FeatureMetadata(
        column_name="active",
        feature_type="Subjective",
        unit_or_encoding="binary",
        description="Self-reported physical activity habit (0 = inactive, 1 = physically active)",
    ),
    "cardio": FeatureMetadata(
        column_name="cardio",
        feature_type="Target",
        unit_or_encoding="binary",
        description="Prevalent cardiovascular disease clinical diagnosis (0 = absence, 1 = presence)",
        is_target=True,
    ),
}


def get_expected_columns() -> List[str]:
    """Return all 13 canonical raw dataset column names in defined order."""
    return list(FEATURE_DICTIONARY.keys())


def get_required_feature_columns() -> List[str]:
    """Return raw clinical feature column names (excluding target and identifiers)."""
    return [
        col for col, meta in FEATURE_DICTIONARY.items()
        if not meta.is_target and not meta.is_identifier
    ]


def get_feature_type(column_name: str) -> str:
    """
    Resolve the high-level feature category for any raw or engineered feature.
    Maps engineered features to their underlying domain category.
    """
    if column_name in FEATURE_DICTIONARY:
        return FEATURE_DICTIONARY[column_name].feature_type

    # Map engineered features based on primary clinical driver
    name_lower = column_name.lower()
    if any(k in name_lower for k in ["bp", "ap_hi", "ap_lo", "pressure", "hypertensive", "cholesterol", "gluc"]):
        return "Examination (Derived)"
    elif any(k in name_lower for k in ["age", "bmi", "height", "weight", "gender"]):
        return "Objective (Derived)"
    elif any(k in name_lower for k in ["smoke", "alco", "active", "lifestyle", "health"]):
        return "Subjective (Derived)"
    return "Engineered / Latent"


def to_markdown_table() -> str:
    """Format the official feature dictionary as a standard GitHub-flavored Markdown table."""
    lines = [
        "| Column | Category | Unit / Encoding | Clinical Description |",
        "|---|---|---|---|",
    ]
    for col, meta in FEATURE_DICTIONARY.items():
        lines.append(
            f"| `{meta.column_name}` | **{meta.feature_type}** | `{meta.unit_or_encoding}` | {meta.description} |"
        )
    return "\n".join(lines)
