"""
Unit tests for clinical feature engineering and safe column dependency handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import CLINICAL_FEATURE_CATALOG, ClinicalFeatureEngineer


def test_clinical_feature_math():
    df = pd.DataFrame({
        "height": [170.0],
        "weight": [68.0],
        "ap_hi": [140.0],
        "ap_lo": [80.0],
        "age": [18262],  # ~50 years
        "cholesterol": [2],
        "gluc": [2],
        "smoke": [1],
        "alco": [1],
    })

    fe = ClinicalFeatureEngineer()
    fe.fit(df)
    res = fe.transform(df)

    # BMI = 68 / (1.7^2) = 68 / 2.89 = 23.529
    assert np.isclose(res["bmi"].iloc[0], 68.0 / (1.7 ** 2), atol=1e-2)
    # Pulse pressure = 140 - 80 = 60
    assert np.isclose(res["pulse_pressure"].iloc[0], 60.0)
    # MAP = 80 + 60 / 3 = 100.0
    assert np.isclose(res["mean_arterial_pressure"].iloc[0], 100.0)
    # High risk lifestyle = 1 & 1 = 1.0
    assert res["high_risk_lifestyle"].iloc[0] == 1.0
    # Metabolic synergy = 2 & 2 = 1.0
    assert res["metabolic_synergy"].iloc[0] == 1.0
    # Hypertension Stage 2 = ap_hi >= 140 -> 1.0
    assert res["is_hypertensive_stage2"].iloc[0] == 1.0


def test_safe_skip_unavailable_features():
    df = pd.DataFrame({
        "ap_hi": [120],
        "ap_lo": [80],
    })
    fe = ClinicalFeatureEngineer()
    fe.fit(df)
    res = fe.transform(df)

    assert "pulse_pressure" in res.columns
    assert "bmi" not in res.columns
    assert "bmi" in fe.unavailable_features_
    assert "pulse_pressure" in fe.available_engineered_features_


def test_clinical_catalog_completeness():
    catalog = CLINICAL_FEATURE_CATALOG
    assert len(catalog) >= 10
    for name, defn in catalog.items():
        assert defn.formula
        assert defn.clinical_reason
        assert len(defn.required_columns) > 0


def test_feature_dictionary_matches_dataset_columns():
    """
    Verify bidirectional integrity of FEATURE_DICTIONARY:
    1. Contains all 13 canonical columns.
    2. Every category matches the official specification:
       Identifier, Objective, Examination, Subjective, Target.
    3. Bidirectional mapping: get_expected_columns() matches FEATURE_DICTIONARY.keys().
    """
    from src.feature_dictionary import (
        FEATURE_DICTIONARY,
        EXAMINATION_VS_SUBJECTIVE_NOTE,
        get_expected_columns,
        get_feature_type,
        get_required_feature_columns,
        to_markdown_table,
    )

    expected_13 = [
        "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
    ]
    assert len(FEATURE_DICTIONARY) == 13
    assert list(FEATURE_DICTIONARY.keys()) == expected_13
    assert get_expected_columns() == expected_13

    valid_categories = {"Identifier", "Objective", "Examination", "Subjective", "Target"}
    for col_name, meta in FEATURE_DICTIONARY.items():
        assert meta.feature_type in valid_categories, f"Invalid category {meta.feature_type} for {col_name}"
        assert meta.unit_or_encoding, f"Missing unit/encoding for {col_name}"
        assert meta.description, f"Missing description for {col_name}"

    assert FEATURE_DICTIONARY["id"].feature_type == "Identifier"
    assert FEATURE_DICTIONARY["id"].is_identifier is True

    for obj_col in ["age", "gender", "height", "weight"]:
        assert FEATURE_DICTIONARY[obj_col].feature_type == "Objective"

    for exam_col in ["ap_hi", "ap_lo", "cholesterol", "gluc"]:
        assert FEATURE_DICTIONARY[exam_col].feature_type == "Examination"

    for subj_col in ["smoke", "alco", "active"]:
        assert FEATURE_DICTIONARY[subj_col].feature_type == "Subjective"

    assert FEATURE_DICTIONARY["cardio"].feature_type == "Target"
    assert FEATURE_DICTIONARY["cardio"].is_target is True

    # Check helper functions
    req_features = get_required_feature_columns()
    assert "id" not in req_features
    assert "cardio" not in req_features
    assert len(req_features) == 11

    # Check derived feature category resolution
    assert get_feature_type("is_hypertensive_stage2") == "Examination (Derived)"
    assert get_feature_type("pulse_pressure") == "Examination (Derived)"
    assert get_feature_type("bmi") == "Objective (Derived)"
    assert get_feature_type("high_risk_lifestyle") == "Subjective (Derived)"
    assert get_feature_type("health_index") == "Subjective (Derived)"

    # Check markdown table generation
    md_table = to_markdown_table()
    assert "| Column | Category | Unit / Encoding | Clinical Description |" in md_table
    for col in expected_13:
        assert f"`{col}`" in md_table

    # Check authoritative clinical note
    assert "Examination features" in EXAMINATION_VS_SUBJECTIVE_NOTE
    assert "Subjective features" in EXAMINATION_VS_SUBJECTIVE_NOTE


def test_health_index_composite_lifestyle_feature():
    """
    Verify health_index composite lifestyle score:
    formula: active - 0.5 * smoke - 0.5 * alco
    Evaluates across all 8 combinations of active (0/1), smoke (0/1), and alco (0/1).
    """
    df = pd.DataFrame({
        "active": [1, 0, 1, 0, 1, 0],
        "smoke":  [0, 0, 1, 1, 1, 0],
        "alco":   [0, 0, 0, 0, 1, 1],
    })
    # Expected scores:
    # 1. Active, non-smoker, non-drinker: 1.0 - 0 - 0 = 1.0
    # 2. Inactive, non-smoker, non-drinker: 0.0 - 0 - 0 = 0.0
    # 3. Active, smoker, non-drinker: 1.0 - 0.5 - 0 = 0.5
    # 4. Inactive, smoker, non-drinker: 0.0 - 0.5 - 0 = -0.5
    # 5. Active, smoker, drinker: 1.0 - 0.5 - 0.5 = 0.0
    # 6. Inactive, non-smoker, drinker: 0.0 - 0 - 0.5 = -0.5
    fe = ClinicalFeatureEngineer(selected_features=["health_index"])
    fe.fit(df)
    res = fe.transform(df)

    expected = np.array([1.0, 0.0, 0.5, -0.5, 0.0, -0.5])
    np.testing.assert_allclose(res["health_index"].to_numpy(), expected, atol=1e-5)


def test_bmi_calculation_and_guard_against_double_unit_conversion():
    """
    Verify BMI calculation: bmi = weight_kg / (height_m ** 2),
    where height is converted from cm to meters exactly once (height_m = height_cm / 100.0).

    Guards against the unit-conversion bug found in external notebooks where height
    is converted to meters and then divided by 100 a second time:
    weight / ((height_in_meters / 100) ** 2), which produces spurious values like 228,571!
    """
    # 3 hand-computed reference patients:
    # Patient 1: 70.0 kg, 175.0 cm -> 70 / (1.75^2) = 70 / 3.0625 = 22.85714... (~22.86)
    # Patient 2: 64.0 kg, 160.0 cm -> 64 / (1.60^2) = 64 / 2.56 = 25.00
    # Patient 3: 81.0 kg, 180.0 cm -> 81 / (1.80^2) = 81 / 3.24 = 25.00
    df = pd.DataFrame({
        "weight": [70.0, 64.0, 81.0],
        "height": [175.0, 160.0, 180.0],
    })

    fe = ClinicalFeatureEngineer(selected_features=["bmi"])
    fe.fit(df)
    res = fe.transform(df)

    computed_bmi = res["bmi"].to_numpy()

    # 1. Verify against exact analytical values
    assert np.isclose(computed_bmi[0], 70.0 / (1.75 ** 2), atol=1e-3)
    assert np.isclose(computed_bmi[0], 22.8571, atol=1e-3)
    assert np.isclose(computed_bmi[1], 25.00, atol=1e-3)
    assert np.isclose(computed_bmi[2], 25.00, atol=1e-3)

    # 2. Guard: verify all BMIs reside within physiological human bounds [12.0, 60.0]
    assert np.all(computed_bmi >= 12.0)
    assert np.all(computed_bmi <= 60.0)

    # 3. Explicitly verify it is NOT subject to the double-conversion bug (~228,571)
    double_conversion_bmi = df["weight"].to_numpy() / (((df["height"].to_numpy() / 100.0) / 100.0) ** 2)
    assert not np.allclose(computed_bmi, double_conversion_bmi)
    assert np.all(computed_bmi < 1000.0), "BMI calculation has double division bug!"


def test_age_batch_independence_and_unit_determinism():
    """
    Forensic verification: Age conversion must be deterministic, unit-governed,
    and 100% batch-independent.
    A patient evaluated alone must have the exact same representation as when
    evaluated in a mixed batch.
    """
    # 1. Test single patient with age in days
    df_single_days = pd.DataFrame({"age": [18262.5]})
    fe_days = ClinicalFeatureEngineer(selected_features=["age_years"], age_unit="days")
    fe_days.fit(df_single_days)
    res_single_days = fe_days.transform(df_single_days)
    assert np.isclose(res_single_days["age_years"].iloc[0], 50.0, atol=1e-3)

    # 2. Test single patient with age in years
    df_single_years = pd.DataFrame({"age": [55.0]})
    fe_years = ClinicalFeatureEngineer(selected_features=["age_years"], age_unit="years")
    fe_years.fit(df_single_years)
    res_single_years = fe_years.transform(df_single_years)
    assert np.isclose(res_single_years["age_years"].iloc[0], 55.0, atol=1e-3)

    # 3. Batch invariance: 55-year-old patient evaluated alone vs in a multi-patient batch
    df_batch = pd.DataFrame({"age": [55.0, 60.0, 72.0]})
    fe_batch = ClinicalFeatureEngineer(selected_features=["age_years"], age_unit="years")
    fe_batch.fit(df_batch)
    res_batch = fe_batch.transform(df_batch)

    # The 55-year-old representation MUST be bitwise identical regardless of batch size or companion rows
    assert res_single_years["age_years"].iloc[0] == res_batch["age_years"].iloc[0]
    assert np.isclose(res_batch["age_years"].iloc[1], 60.0, atol=1e-3)
    assert np.isclose(res_batch["age_years"].iloc[2], 72.0, atol=1e-3)

    # 4. Reject invalid units
    with pytest.raises(ValueError):
        ClinicalFeatureEngineer(age_unit="invalid_unit")


def test_mixed_batch_age_row_independence():
    """
    ISSUE 3 Regression Test:
    Verify transform([55]) == transform([55, 60, 18262.5])[0],
    proving row-level interpretation never depends on companion rows.
    """
    fe = ClinicalFeatureEngineer(selected_features=["age_years"])
    res_single = fe.transform(pd.DataFrame({"age": [55.0]}))
    res_mixed = fe.transform(pd.DataFrame({"age": [55.0, 60.0, 18262.5, np.nan]}))

    # Row 0: 55 years evaluated in isolation == evaluated in mixed batch
    assert np.isclose(res_single["age_years"].iloc[0], res_mixed["age_years"].iloc[0], atol=1e-5)
    assert np.isclose(res_mixed["age_years"].iloc[0], 55.0, atol=1e-5)
    assert np.isclose(res_mixed["age_years"].iloc[1], 60.0, atol=1e-5)
    # Row 2: 18262.5 days converted to 50.0 years
    assert np.isclose(res_mixed["age_years"].iloc[2], 50.0, atol=1e-5)
    # Row 3: NaN correctly preserved
    assert np.isnan(res_mixed["age_years"].iloc[3])
