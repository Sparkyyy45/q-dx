"""
Leakage-safe clinical feature engineering for cardiovascular disease risk prediction.
Every engineered feature has a documented mathematical formula and clinical justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class FeatureDefinition:
    """Documented specification of an engineered clinical feature."""
    name: str
    formula: str
    clinical_reason: str
    required_columns: List[str]


# Clinical dictionary of all supported tabular transformations
CLINICAL_FEATURE_CATALOG: Dict[str, FeatureDefinition] = {
    "bmi": FeatureDefinition(
        name="bmi",
        formula="weight_kg / (height_m ** 2)",
        clinical_reason="WHO standard body mass index; quantifies adiposity and metabolic cardiovascular load.",
        required_columns=["weight", "height"],
    ),
    "pulse_pressure": FeatureDefinition(
        name="pulse_pressure",
        formula="ap_hi - ap_lo",
        clinical_reason="Key hemodynamic indicator of large-artery stiffness, vascular aging, and left ventricular pulsatile workload.",
        required_columns=["ap_hi", "ap_lo"],
    ),
    "mean_arterial_pressure": FeatureDefinition(
        name="mean_arterial_pressure",
        formula="ap_lo + (ap_hi - ap_lo) / 3.0",
        clinical_reason="Perfusion pressure seen by vital organs; reflects both cardiac output and systemic vascular resistance.",
        required_columns=["ap_hi", "ap_lo"],
    ),
    "age_years": FeatureDefinition(
        name="age_years",
        formula="age_days / 365.25",
        clinical_reason="Chronological age standardized in calendar years; the dominant non-modifiable cardiovascular risk factor.",
        required_columns=["age"],
    ),
    "age_squared": FeatureDefinition(
        name="age_squared",
        formula="(age_years) ** 2",
        clinical_reason="Models the accelerating, non-linear incidence of atherosclerotic cardiovascular events in older cohorts.",
        required_columns=["age_years"],
    ),
    "age_bp_interaction": FeatureDefinition(
        name="age_bp_interaction",
        formula="age_years * ap_hi",
        clinical_reason="Synergistic interaction between advanced chronological vascular age and elevated systolic blood pressure.",
        required_columns=["age_years", "ap_hi"],
    ),
    "cholesterol_gluc_ratio": FeatureDefinition(
        name="cholesterol_gluc_ratio",
        formula="cholesterol / gluc",
        clinical_reason="Task-specific engineered interaction: ordinal ratio of cholesterol to glucose capturing relative dyslipidemia/dysglycemia interaction, not an established biomarker.",
        required_columns=["cholesterol", "gluc"],
    ),
    "metabolic_synergy": FeatureDefinition(
        name="metabolic_synergy",
        formula="1 if (cholesterol >= 2 and gluc >= 2) else 0",
        clinical_reason="Task-specific engineered binary interaction: flag for concurrent elevation of ordinal cholesterol and glucose (both >= 2); heuristic cardiometabolic marker, not a clinical diagnostic criterion.",
        required_columns=["cholesterol", "gluc"],
    ),
    "is_hypertensive_stage2": FeatureDefinition(
        name="is_hypertensive_stage2",
        formula="1 if (ap_hi >= 140 or ap_lo >= 90) else 0",
        clinical_reason="ACC/AHA clinical threshold for Stage 2 hypertension, strongly associated with myocardial infarction and stroke.",
        required_columns=["ap_hi", "ap_lo"],
    ),
    "high_risk_lifestyle": FeatureDefinition(
        name="high_risk_lifestyle",
        formula="1 if (smoke == 1 and alco == 1) else 0",
        clinical_reason="Task-specific engineered interaction indicating co-occurrence of self-reported smoking and alcohol intake.",
        required_columns=["smoke", "alco"],
    ),
    "log_pulse_pressure": FeatureDefinition(
        name="log_pulse_pressure",
        formula="log1p(max(pulse_pressure, 0))",
        clinical_reason="Logarithmic transformation of pulse pressure to stabilize positive skewness and linearize risk.",
        required_columns=["pulse_pressure"],
    ),
    "health_index": FeatureDefinition(
        name="health_index",
        formula="active - 0.5 * smoke - 0.5 * alco",
        clinical_reason="Task-specific engineered composite lifestyle score combining physical activity (+1.0) and behavioral toxic exposures (-0.5 smoking, -0.5 alcohol); exploratory tabular heuristic, not a clinically validated index.",
        required_columns=["active", "smoke", "alco"],
    ),
}


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe clinical feature generator.
    Evaluates available columns during fit() and safely applies documented transformations
    during transform(). Skips unavailable features without failing and logs provenance.
    """

    def __init__(self, selected_features: Optional[List[str]] = None, age_unit: str = "days"):
        self.selected_features = selected_features or list(CLINICAL_FEATURE_CATALOG.keys())
        self.age_unit = age_unit.lower()
        if self.age_unit not in ("days", "years", "auto"):
            raise ValueError(f"Invalid age_unit: '{age_unit}'. Must be 'days', 'years', or 'auto'.")
        self.available_engineered_features_: List[str] = []
        self.unavailable_features_: Dict[str, str] = {}
        self.catalog_: Dict[str, FeatureDefinition] = CLINICAL_FEATURE_CATALOG

    def fit(self, X: pd.DataFrame, y: Any = None) -> ClinicalFeatureEngineer:
        """Inspect available columns and determine feasible transformations."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer requires a pandas DataFrame input.")

        self.available_engineered_features_ = []
        self.unavailable_features_ = {}

        cols = set(X.columns)

        # Iteratively determine which features can be engineered
        # (Handling potential inter-engineered dependencies like age_years -> age_squared)
        simulated_cols = set(cols)
        for feat_name in self.selected_features:
            if feat_name not in self.catalog_:
                continue
            definition = self.catalog_[feat_name]
            # Check if raw input columns exist or will be generated
            missing_reqs = [req for req in definition.required_columns if req not in simulated_cols]
            if not missing_reqs:
                self.available_engineered_features_.append(feat_name)
                simulated_cols.add(feat_name)
            else:
                self.unavailable_features_[feat_name] = f"Missing required columns: {missing_reqs}"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features and return enriched DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer requires a pandas DataFrame input.")

        if not getattr(self, "available_engineered_features_", None):
            self.fit(X)

        out = X.copy()

        for feat_name in self.available_engineered_features_:
            # 1. BMI (if not already present or if we recalculate)
            if feat_name == "bmi" and "bmi" not in out.columns:
                if "height" in out.columns and "weight" in out.columns:
                    height_m = out["height"].clip(lower=50) / 100.0
                    out["bmi"] = out["weight"] / (height_m ** 2)

            # 2. Pulse pressure
            elif feat_name == "pulse_pressure":
                if "ap_hi" in out.columns and "ap_lo" in out.columns:
                    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]

            # 3. Mean arterial pressure
            elif feat_name == "mean_arterial_pressure":
                if "ap_hi" in out.columns and "ap_lo" in out.columns:
                    pp = out["ap_hi"] - out["ap_lo"]
                    out["mean_arterial_pressure"] = out["ap_lo"] + (pp / 3.0)

            # 4. Age in years (Row-independent conversion; no batch conditioning)
            elif feat_name == "age_years" and "age_years" not in out.columns:
                if "age" in out.columns:
                    age_numeric = pd.to_numeric(out["age"], errors="coerce")
                    out["age_years"] = np.where(
                        age_numeric.isna(),
                        np.nan,
                        np.where(age_numeric > 120.0, age_numeric / 365.25, age_numeric)
                    )

            # 5. Age squared
            elif feat_name == "age_squared":
                if "age_years" in out.columns:
                    age_y = out["age_years"]
                elif "age" in out.columns:
                    age_numeric = pd.to_numeric(out["age"], errors="coerce")
                    age_y = np.where(
                        age_numeric.isna(),
                        np.nan,
                        np.where(age_numeric > 120.0, age_numeric / 365.25, age_numeric)
                    )
                else:
                    continue
                out["age_squared"] = age_y ** 2

            # 6. Age BP interaction
            elif feat_name == "age_bp_interaction":
                if "ap_hi" not in out.columns:
                    continue
                if "age_years" in out.columns:
                    age_y = out["age_years"]
                elif "age" in out.columns:
                    age_numeric = pd.to_numeric(out["age"], errors="coerce")
                    age_y = np.where(
                        age_numeric.isna(),
                        np.nan,
                        np.where(age_numeric > 120.0, age_numeric / 365.25, age_numeric)
                    )
                else:
                    continue
                out["age_bp_interaction"] = age_y * out["ap_hi"]

            # 7. Cholesterol / Glucose ratio
            elif feat_name == "cholesterol_gluc_ratio":
                if "cholesterol" in out.columns and "gluc" in out.columns:
                    out["cholesterol_gluc_ratio"] = out["cholesterol"] / out["gluc"].clip(lower=1)

            # 8. Metabolic synergy flag
            elif feat_name == "metabolic_synergy":
                if "cholesterol" in out.columns and "gluc" in out.columns:
                    out["metabolic_synergy"] = (
                        (out["cholesterol"] >= 2) & (out["gluc"] >= 2)
                    ).astype(float)

            # 9. Stage 2 hypertension flag
            elif feat_name == "is_hypertensive_stage2":
                if "ap_hi" in out.columns and "ap_lo" in out.columns:
                    out["is_hypertensive_stage2"] = (
                        (out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)
                    ).astype(float)

            # 10. High risk lifestyle
            elif feat_name == "high_risk_lifestyle":
                if "smoke" in out.columns and "alco" in out.columns:
                    out["high_risk_lifestyle"] = (
                        (out["smoke"] == 1) & (out["alco"] == 1)
                    ).astype(float)

            # 11. Log pulse pressure
            elif feat_name == "log_pulse_pressure":
                if "pulse_pressure" in out.columns:
                    pp = out["pulse_pressure"]
                elif "ap_hi" in out.columns and "ap_lo" in out.columns:
                    pp = out["ap_hi"] - out["ap_lo"]
                else:
                    continue
                out["log_pulse_pressure"] = np.log1p(np.maximum(pp, 0.0))

            # 12. Health index (composite lifestyle score)
            elif feat_name == "health_index":
                if "active" in out.columns and "smoke" in out.columns and "alco" in out.columns:
                    out["health_index"] = (
                        out["active"].astype(float)
                        - 0.5 * out["smoke"].astype(float)
                        - 0.5 * out["alco"].astype(float)
                    )

        return out

    def get_feature_catalog(self) -> Dict[str, Dict[str, str]]:
        """Return clinical definitions for reporting."""
        return {
            name: {
                "name": defn.name,
                "formula": defn.formula,
                "clinical_reason": defn.clinical_reason,
                "required_columns": ", ".join(defn.required_columns),
            }
            for name, defn in self.catalog_.items()
        }
