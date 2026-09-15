import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.external_validation import (
    DEFAULT_BROKEN_EXTERNAL_AUC,
    EXPECTED_HARMONIZED_COLUMNS,
    MissingExternalFeatureError,
    audit_external_feature_ranges,
    compute_pr_auc_lift,
    create_external_comparison_table,
    harmonize_framingham,
    load_framingham,
    map_cholesterol_to_category,
    run_external_validation,
)
from src.preprocess import PreprocessingArtifact


def _resolve_test_dataset_path() -> str:
    env_path = os.getenv("CVD_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in Path(".").glob("*.csv"):
        if "framingham" in candidate.name.lower():
            continue
        try:
            head = pd.read_csv(candidate, nrows=2)
            if "cardio" in head.columns:
                return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("No training dataset found for external validation tests.")


def test_load_framingham_dataset():
    """Verify loading real Framingham dataset if present, and handling missing files."""
    # Test valid load
    df = load_framingham("framingham.csv")
    assert len(df) == 4240
    assert "male" in df.columns
    assert "age" in df.columns
    assert "sysBP" in df.columns
    assert "diaBP" in df.columns
    assert "totChol" in df.columns
    assert "BMI" in df.columns
    assert "currentSmoker" in df.columns
    assert "TenYearCHD" in df.columns

    # Test missing file error
    with pytest.raises(FileNotFoundError):
        load_framingham("non_existent_framingham_file.csv")


def test_cholesterol_bucketing_cutoffs():
    """Verify continuous totChol mapping onto 1/2/3 categorical scale per clinical cutoffs."""
    # < 200 -> 1 (Normal / Desirable)
    assert map_cholesterol_to_category(150.0) == 1
    assert map_cholesterol_to_category(180.0) == 1
    assert map_cholesterol_to_category(199.9) == 1

    # 200 - 239.9 -> 2 (Above Normal / Borderline High)
    assert map_cholesterol_to_category(200.0) == 2
    assert map_cholesterol_to_category(215.0) == 2
    assert map_cholesterol_to_category(239.9) == 2

    # >= 240 -> 3 (Well Above Normal / High)
    assert map_cholesterol_to_category(240.0) == 3
    assert map_cholesterol_to_category(280.0) == 3
    assert map_cholesterol_to_category(400.0) == 3

    # Preserves NaN
    assert np.isnan(map_cholesterol_to_category(np.nan))

    # Series mapping
    s = pd.Series([180.0, 200.0, 240.0, np.nan])
    res = map_cholesterol_to_category(s)
    assert res.iloc[0] == 1.0
    assert res.iloc[1] == 2.0
    assert res.iloc[2] == 3.0
    assert np.isnan(res.iloc[3])


def test_harmonize_framingham_columns_and_no_nans():
    """Verify harmonize_framingham produces expected columns with zero introduced NaNs."""
    df_raw = pd.DataFrame({
        "male": [0, 1, 0, 1],
        "age": [39, 48, 55, 62],
        "currentSmoker": [0, 1, 0, 1],
        "cigsPerDay": [0, 20, 0, 10],
        "BPMeds": [0, 0, 1, 0],
        "prevalentStroke": [0, 0, 0, 0],
        "prevalentHyp": [0, 1, 1, 0],
        "diabetes": [0, 0, 0, 0],
        "totChol": [195.0, 220.0, 260.0, 185.0],
        "sysBP": [106.0, 135.0, 150.0, 120.0],
        "diaBP": [70.0, 85.0, 95.0, 80.0],
        "BMI": [26.97, 28.50, 24.30, 25.10],
        "heartRate": [80, 75, 68, 72],
        "glucose": [77, 85, 90, 70],
        "TenYearCHD": [0, 1, 0, 1],
    })

    df_harm = harmonize_framingham(df_raw, dropna=True)

    # 1. Expected column set (9 columns including age in days and age_years in years)
    assert set(df_harm.columns) == set(EXPECTED_HARMONIZED_COLUMNS)
    assert len(df_harm.columns) == 9

    # 2. No NaNs introduced
    assert df_harm.isna().sum().sum() == 0

    # 3. Gender mapping: 0 -> 1 (Female), 1 -> 2 (Male)
    assert df_harm["gender"].tolist() == [1, 2, 1, 2]

    # 4. Age: both days and years preserved with exact conversion
    assert df_harm["age_years"].tolist() == [39.0, 48.0, 55.0, 62.0]
    expected_days = [39.0 * 365.25, 48.0 * 365.25, 55.0 * 365.25, 62.0 * 365.25]
    np.testing.assert_allclose(df_harm["age"].tolist(), expected_days)

    # 5. BP mapping
    assert df_harm["ap_hi"].tolist() == [106.0, 135.0, 150.0, 120.0]
    assert df_harm["ap_lo"].tolist() == [70.0, 85.0, 95.0, 80.0]

    # 6. Cholesterol categorization
    assert df_harm["cholesterol"].tolist() == [1, 2, 3, 1]

    # 7. Target mapped directly from TenYearCHD
    assert df_harm["target"].tolist() == [0, 1, 0, 1]


def test_gender_mapping_correlation():
    """
    Verify mapped gender preserves demographic correlations and prevents silent inversions.
    In both Kaggle and Framingham cohorts, males (gender=2) have significantly higher
    smoking prevalence than females (gender=1).
    """
    from pathlib import Path
    framingham_path = Path("framingham.csv")
    if framingham_path.is_file():
        df_raw = load_framingham(framingham_path)
        df_harm = harmonize_framingham(df_raw, dropna=True)

        female_smoke_rate = df_harm.loc[df_harm["gender"] == 1, "smoke"].mean()
        male_smoke_rate = df_harm.loc[df_harm["gender"] == 2, "smoke"].mean()

        # Framingham: Male smoking (~60.8%) is significantly higher than female smoking (~40.9%)
        assert male_smoke_rate > female_smoke_rate, (
            f"Expected male smoking rate ({male_smoke_rate:.3f}) > female smoking rate ({female_smoke_rate:.3f})"
        )
        assert male_smoke_rate > 0.50
        assert female_smoke_rate < 0.50

    # Synthetic verification: ensure inverted mapping (0->2, 1->1) would fail correlation invariant
    syn_raw = pd.DataFrame({
        "male": [0, 0, 1, 1],  # 0 females (non-smokers), 1 males (smokers)
        "age": [45, 50, 55, 60],
        "currentSmoker": [0, 0, 1, 1],
        "sysBP": [120.0, 122.0, 130.0, 135.0],
        "diaBP": [80.0, 82.0, 85.0, 88.0],
        "BMI": [25.0, 26.0, 27.0, 28.0],
        "totChol": [190.0, 210.0, 220.0, 250.0],
        "TenYearCHD": [0, 0, 1, 1],
    })
    syn_harm = harmonize_framingham(syn_raw, dropna=True)
    syn_female_smoke = syn_harm.loc[syn_harm["gender"] == 1, "smoke"].mean()
    syn_male_smoke = syn_harm.loc[syn_harm["gender"] == 2, "smoke"].mean()
    assert syn_male_smoke > syn_female_smoke


def test_feature_ranges_within_bounds():
    """
    Verify external validation feature distributions do not get squashed by IQRClipper.
    Fails if >50% of any feature falls outside training IQR bounds (regression check for STEP 1 & 5).
    """
    from src.audit import audit_dataset
    from src.clean import clean_dataset
    from src.config import PipelineConfig
    from src.dataset import DatasetContract, load_dataset
    from src.features import ClinicalFeatureEngineer
    from src.preprocess import build_preprocessor, fit_preprocessor

    config = PipelineConfig(data_path=_resolve_test_dataset_path())
    contract = DatasetContract()
    data_art = load_dataset(config.data_path, contract)
    df = data_art.df
    audit = audit_dataset(df, contract)
    df_clean, _ = clean_dataset(df, contract, audit)

    exclude_cols = set(contract.identifier_columns + contract.drop_columns + [contract.target_column])
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    X = df_clean[feature_cols].copy()
    y = df_clean[contract.target_column].copy()

    fe = ClinicalFeatureEngineer().fit(X)
    X_fe = fe.transform(X)
    if "age" in X_fe.columns and "age_years" in X_fe.columns:
        X_fe = X_fe.drop(columns=["age"])

    num_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_fe.select_dtypes(exclude=[np.number]).columns.tolist()
    prep = build_preprocessor(config, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, X_fe, y, num_cols, cat_cols, config)

    df_fram = load_framingham("framingham.csv")
    df_harm = harmonize_framingham(df_fram, dropna=True)

    report_df = audit_external_feature_ranges(prep_art, df_harm, print_report=False)

    # 1. No feature should be flagged (>50% outside training bounds)
    flagged_features = report_df[report_df["Flagged (>50% Out)"].str.contains("YES")]["Feature"].tolist()
    assert len(flagged_features) == 0, f"Features squashed by IQRClipper: {flagged_features}"

    # 2. Parsed outside bounds % must be strictly < 50% for every feature
    for _, row in report_df.iterrows():
        pct_val = float(str(row["Outside Bounds %"]).replace("%", "").strip())
        assert pct_val < 50.0, f"Feature {row['Feature']} has {pct_val}% values outside training bounds!"


def test_run_external_validation_raises_error_if_required_feature_missing():
    """Verify run_external_validation raises a loud, clear error if a required feature is absent."""
    # Create mock PreprocessingArtifact that requires a missing feature
    mock_preprocessor = PreprocessingArtifact(
        preprocessor=None,  # Not reached
        input_numeric_features=["age", "ap_hi", "ap_lo", "gluc"],  # 'gluc' is missing in harmonized frame
        input_categorical_features=[],
        output_feature_names=[],
        scaling_strategy="robust",
    )

    df_harm = pd.DataFrame({
        "age": [50.0],
        "gender": [1],
        "ap_hi": [120.0],
        "ap_lo": [80.0],
        "bmi": [25.0],
        "smoke": [0],
        "cholesterol": [1],
        "target": [0],
    })

    # Must raise MissingExternalFeatureError highlighting the missing required feature
    with pytest.raises(MissingExternalFeatureError) as exc_info:
        run_external_validation(
            fitted_preprocessor=mock_preprocessor,
            fitted_models={},
            harmonized_df=df_harm,
        )

    assert "Required feature(s) missing" in str(exc_info.value)
    assert "gluc" in str(exc_info.value)


def test_full_feature_set_raises_clear_error_on_missing_columns():
    """
    Verify run_external_validation raises MissingExternalFeatureError with a clear error
    when the preprocessor was fitted on the full feature set (reduction='none', where
    height, weight, gluc, alco, active are required by the pipeline).
    """
    from src.audit import audit_dataset
    from src.clean import clean_dataset
    from src.config import PipelineConfig
    from src.dataset import DatasetContract, load_dataset
    from src.features import ClinicalFeatureEngineer
    from src.preprocess import build_preprocessor, fit_preprocessor

    # Full feature set: do NOT drop height, weight, gluc, alco, active
    config = PipelineConfig(data_path=_resolve_test_dataset_path(), reduction_strategy="none")
    config.drop_columns = ["bp_category_encoded", "bp_category"]

    contract = DatasetContract(drop_columns=config.drop_columns)
    data_art = load_dataset(config.data_path, contract)
    df = data_art.df
    audit = audit_dataset(df, contract)
    df_clean, _ = clean_dataset(df, contract, audit)

    exclude_cols = set(contract.identifier_columns + contract.drop_columns + [contract.target_column])
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    X = df_clean[feature_cols].copy()
    y = df_clean[contract.target_column].copy()

    fe = ClinicalFeatureEngineer().fit(X)
    X_fe = fe.transform(X)
    if "age" in X_fe.columns and "age_years" in X_fe.columns:
        X_fe = X_fe.drop(columns=["age"])

    num_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_fe.select_dtypes(exclude=[np.number]).columns.tolist()
    prep = build_preprocessor(config, num_cols, cat_cols)
    prep_art = fit_preprocessor(prep, X_fe, y, num_cols, cat_cols, config)

    df_fram = load_framingham("framingham.csv")
    df_harm = harmonize_framingham(df_fram, dropna=True)

    with pytest.raises(MissingExternalFeatureError) as exc_info:
        run_external_validation(
            fitted_preprocessor=prep_art,
            fitted_models={},
            harmonized_df=df_harm,
            feature_engineer=fe,
        )

    err_msg = str(exc_info.value)
    # Check that missing features are explicitly listed
    for col in ["gluc", "alco", "active", "height", "weight"]:
        assert col in err_msg, f"Expected '{col}' in error message, got: {err_msg}"
    assert "reduction" in err_msg.lower() or "f_classif" in err_msg or "imputation" in err_msg.lower()


def test_pr_auc_lift_calculation():
    """
    Verify PR-AUC lift calculation (pr_auc / prevalence) is computed correctly
    for both internal holdout and external Framingham cohorts using known synthetic values.
    """
    # 1. Direct mathematical verification with known synthetic inputs
    # Internal: PR-AUC 0.7809, prevalence 0.4947 -> lift ~1.5785
    int_pr = 0.7809
    int_prev = 0.4947
    int_lift = compute_pr_auc_lift(int_pr, int_prev)
    assert abs(int_lift - (0.7809 / 0.4947)) < 1e-6
    assert abs(int_lift - 1.5785) < 1e-3

    # External: PR-AUC 0.2408, prevalence 0.1498 -> lift ~1.6075
    ext_pr = 0.2408
    ext_prev = 0.1498
    ext_lift = compute_pr_auc_lift(ext_pr, ext_prev)
    assert abs(ext_lift - (0.2408 / 0.1498)) < 1e-6
    assert abs(ext_lift - 1.6075) < 1e-3

    # Invalid non-positive prevalence must raise ValueError
    with pytest.raises(ValueError):
        compute_pr_auc_lift(0.5, 0.0)

    # 2. Verify integration into create_external_comparison_table
    from src.evaluate import EvaluationMetrics
    mock_metric_int = EvaluationMetrics(
        roc_auc=0.795, pr_auc=0.780, accuracy=0.73, balanced_accuracy=0.73,
        sensitivity=0.67, specificity=0.78, precision=0.75, npv=0.71, f1=0.71,
        brier_score=0.18, expected_calibration_error=0.01,
        confusion_matrix=[[78, 22], [33, 67]], tn=78, fp=22, fn=33, tp=67,
    )
    mock_metric_ext = EvaluationMetrics(
        roc_auc=0.665, pr_auc=0.240, accuracy=0.52, balanced_accuracy=0.62,
        sensitivity=0.77, specificity=0.48, precision=0.20, npv=0.92, f1=0.32,
        brier_score=0.32, expected_calibration_error=0.15,
        confusion_matrix=[[48, 52], [23, 77]], tn=48, fp=52, fn=23, tp=77,
    )

    table = create_external_comparison_table(
        internal_metrics={"Logistic Regression": mock_metric_int},
        external_metrics={"Logistic Regression": mock_metric_ext},
        internal_prevalence=0.50,
        external_prevalence=0.15,
    )

    assert "Internal PR-AUC Lift" in table.columns
    assert "External PR-AUC Lift" in table.columns
    assert "Internal Prevalence" in table.columns
    assert "External Prevalence" in table.columns

    row = table.iloc[0]
    assert row["Internal Prevalence"] == "50.00%"
    assert row["External Prevalence"] == "15.00%"
    assert row["Internal PR-AUC Lift"] == f"{0.780 / 0.50:.2f}x"
    assert row["External PR-AUC Lift"] == f"{0.240 / 0.15:.2f}x"
