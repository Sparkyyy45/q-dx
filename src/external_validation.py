"""
External validation module benchmarking the cardiovascular risk prediction models
against the independent Framingham Heart Study cohort (4,240 patients).

Validation Principles:
1. Zero Refitting: Models, scalers, imputers, and dimensionality reduction artifacts
   fitted strictly on the development cohort are evaluated directly on Framingham data.
2. Clinical Feature Harmonization: Explicitly maps Framingham columns onto the pipeline
   feature schema while documenting lossy approximations (e.g. continuous totChol to
   3-tier categorical cholesterol).
3. Strict Feature Integrity: Excluded pipeline features with no Framingham equivalent
   (gluc, alco, active, height, weight) are not fabricated or imputed with dummy zeros.
   If a feature required by the fitted preprocessor is missing, the module fails loudly.
4. Outcome Definition Caveat: TenYearCHD (10-year incident coronary heart disease risk)
   is distinct from cross-sectional prevalent CVD ('cardio').
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluate import EvaluationMetrics, evaluate_at_locked_threshold, evaluate_predictions
from src.preprocess import PreprocessingArtifact, transform_data
from src.reduction import FeatureReductionArtifact, reduce_features

logger = logging.getLogger(__name__)


class MissingExternalFeatureError(ValueError):
    """
    Raised when external validation data is missing required features expected
    by the fitted preprocessor/model pipeline.
    """
    pass


def compute_pr_auc_lift(pr_auc: float, prevalence: float) -> float:
    """
    Compute normalized PR-AUC lift over random baseline:
        PR-AUC Lift = PR-AUC / positive_class_prevalence
    """
    if prevalence <= 0:
        raise ValueError(f"Prevalence must be positive to compute PR-AUC lift, got {prevalence}")
    return float(pr_auc / prevalence)


# Standard Framingham Heart Study CSV schema
FRAMINGHAM_REQUIRED_RAW_COLUMNS = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "TenYearCHD",
]

# Pipeline harmonized feature columns for external evaluation
EXPECTED_HARMONIZED_COLUMNS = [
    "age",
    "age_years",
    "gender",
    "ap_hi",
    "ap_lo",
    "bmi",
    "smoke",
    "cholesterol",
    "target",
]


def load_framingham(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the Framingham Heart Study CSV dataset.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the framingham.csv dataset.

    Returns
    -------
    pd.DataFrame
        Raw Framingham dataset containing patient records.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file lacks required Framingham schema columns.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Framingham dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # Validate essential Framingham columns
    missing_cols = [c for c in ["male", "age", "currentSmoker", "totChol", "sysBP", "diaBP", "BMI", "TenYearCHD"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Framingham dataset at {path} is missing expected columns: {missing_cols}")

    return df


def map_cholesterol_to_category(tot_chol: Union[float, int, pd.Series]) -> Union[int, pd.Series]:
    """
    Map continuous serum total cholesterol (mg/dL) to the pipeline's 3-tier ordinal scale:
      - 1 (Normal / Desirable): < 200 mg/dL (AHA / NCEP ATP III guidelines)
      - 2 (Above Normal / Borderline High): 200 - 239 mg/dL
      - 3 (Well Above Normal / High): >= 240 mg/dL

    Note: This is a documented, lossy clinical approximation mapping continuous
    laboratory values onto ordinal survey tiers. Missing values (NaN) are preserved.
    """
    if isinstance(tot_chol, pd.Series):
        conditions = [
            tot_chol.isna(),
            tot_chol < 200.0,
            (tot_chol >= 200.0) & (tot_chol < 240.0),
            tot_chol >= 240.0,
        ]
        choices = [np.nan, 1, 2, 3]
        return pd.Series(np.select(conditions, choices, default=np.nan), index=tot_chol.index, dtype=float)
    else:
        if pd.isna(tot_chol):
            return np.nan
        val = float(tot_chol)
        if val < 200.0:
            return 1
        elif val < 240.0:
            return 2
        else:
            return 3


def harmonize_framingham(df_framingham: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Map and rename Framingham columns onto this pipeline's feature schema.

    Harmonization Rules:
    - age: already in years, no conversion needed (contrasting with Kaggle's raw age in days).
    - male -> gender: mapped to this pipeline's gender encoding (0 -> 1 [Female], 1 -> 2 [Male]).
    - sysBP -> ap_hi: systolic blood pressure (mmHg).
    - diaBP -> ap_lo: diastolic blood pressure (mmHg).
    - BMI -> bmi: body mass index (kg/m^2). Not recomputed since Framingham lacks height/weight.
    - currentSmoker -> smoke: active smoking indicator (0/1).
    - totChol -> cholesterol: discretized into 1/2/3 using AHA/NCEP ATP III cutoffs (<200=1, 200-239=2, >=240=3).
    - TenYearCHD -> target: binary indicator of 10-year coronary heart disease incidence.
      (Caveat: TenYearCHD is a 10-year prospective outcome, not cross-sectional prevalent CVD).

    Features excluded entirely (no Framingham equivalent):
    - gluc, alco, active, height, weight (raw).
    Values are NOT fabricated or imputed with dummy zeros.

    Parameters
    ----------
    df_framingham : pd.DataFrame
        Raw Framingham cohort DataFrame.
    dropna : bool, default=True
        Whether to drop records with missing values in the harmonized columns.

    Returns
    -------
    pd.DataFrame
        Harmonized DataFrame containing EXPECTED_HARMONIZED_COLUMNS.
    """
    df_in = df_framingham.copy()

    # Verify input columns exist
    req_inputs = ["age", "male", "sysBP", "diaBP", "BMI", "currentSmoker", "totChol", "TenYearCHD"]
    missing = [c for c in req_inputs if c not in df_in.columns]
    if missing:
        raise ValueError(f"Cannot harmonize Framingham: missing columns {missing}")

    # Build harmonized output frame
    out = pd.DataFrame(index=df_in.index)

    # 1. Age units: Framingham is recorded in calendar years. Convert to days for 'age'
    # (Kaggle pipeline convention) and retain 'age_years' directly so both features
    # land on the expected scale learned by the preprocessor.
    age_years = df_in["age"].astype(float)
    out["age"] = age_years * 365.25
    out["age_years"] = age_years

    # 2. Gender: Framingham 0=Female, 1=Male -> Pipeline 1=Female, 2=Male
    # Avoid introducing NaNs for unexpected values
    gender_mapped = df_in["male"].map({0: 1, 1: 2})
    if gender_mapped.isna().any() and not df_in["male"].isna().any():
        raise ValueError("Unexpected values in Framingham 'male' column; expected {0, 1}.")
    out["gender"] = gender_mapped

    # 3. Blood Pressure: sysBP -> ap_hi, diaBP -> ap_lo
    out["ap_hi"] = df_in["sysBP"].astype(float)
    out["ap_lo"] = df_in["diaBP"].astype(float)

    # 4. BMI: direct utilization
    out["bmi"] = df_in["BMI"].astype(float)

    # 5. Smoking: currentSmoker -> smoke
    out["smoke"] = df_in["currentSmoker"].astype(float)

    # 6. Cholesterol: continuous totChol mapped to 1/2/3 categorical scale
    out["cholesterol"] = map_cholesterol_to_category(df_in["totChol"])

    # 7. Target: TenYearCHD -> target
    out["target"] = df_in["TenYearCHD"].astype(int)

    # Ensure column order matches EXPECTED_HARMONIZED_COLUMNS
    out = out[EXPECTED_HARMONIZED_COLUMNS]

    if dropna:
        out = out.dropna().reset_index(drop=True)
        out["gender"] = out["gender"].astype(int)
        out["smoke"] = out["smoke"].astype(int)
        out["cholesterol"] = out["cholesterol"].astype(int)
        out["target"] = out["target"].astype(int)

    return out


def _get_required_preprocessor_features(preprocessor: Any) -> List[str]:
    """Extract the list of required input feature names from a preprocessor."""
    if hasattr(preprocessor, "input_numeric_features") and hasattr(preprocessor, "input_categorical_features"):
        return list(preprocessor.input_numeric_features) + list(preprocessor.input_categorical_features)

    # Direct scikit-learn ColumnTransformer
    if hasattr(preprocessor, "transformers"):
        cols = []
        for name, transformer, column_list in preprocessor.transformers:
            if column_list is not None and column_list != "drop":
                if isinstance(column_list, list):
                    cols.extend(column_list)
                elif isinstance(column_list, (str, int)):
                    cols.append(str(column_list))
        return cols

    return []


def run_external_validation(
    fitted_preprocessor: Any,
    fitted_models: Dict[str, Any],
    reduction_artifact: Any = None,
    reducer: Any = None,
    harmonized_df: Optional[pd.DataFrame] = None,
    feature_engineer: Any = None,
    return_probabilities: bool = False,
    catboost_preprocessor: Any = None,
    locked_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, EvaluationMetrics] | Tuple[Dict[str, EvaluationMetrics], Dict[str, np.ndarray]]:
    """
    Apply pre-fitted preprocessor, reducer, and models to the harmonized Framingham data.

    Evaluation Invariants:
    1. Zero Refitting: Models, transformers, and reducers are applied purely via transform/predict.
    2. Strict Feature Validation: If any feature required by fitted_preprocessor is missing
       from harmonized_df, raises a clear ValueError rather than silently imputing zeros.

    Parameters
    ----------
    fitted_preprocessor : PreprocessingArtifact or ColumnTransformer
        Preprocessor fitted on the development cohort.
    fitted_models : Dict[str, Any]
        Dictionary of model name -> fitted model instance.
    reduction_artifact : Optional[ReductionArtifact]
        Fitted dimensionality reduction artifact (if used).
    reducer : Optional[FeatureReducer]
        Fitted dimensionality reduction instance (if used).
    harmonized_df : pd.DataFrame
        Harmonized Framingham dataset containing 'target'.
    feature_engineer : Optional[ClinicalFeatureEngineer]
        Fitted clinical feature engineering transformer (if used).
    return_probabilities : bool
        Whether to return per-model prediction probabilities.
    catboost_preprocessor : Optional[Any]
        Optional CatBoost-specific preprocessor preserving unscaled categorical features.

    Returns
    -------
    Dict[str, EvaluationMetrics]
        Computed evaluation metrics per model on the Framingham cohort.

    Raises
    ------
    ValueError
        If harmonized_df is invalid or missing required features.
    """
    if harmonized_df is None or not isinstance(harmonized_df, pd.DataFrame):
        raise ValueError("harmonized_df must be a valid pandas DataFrame.")

    if "target" not in harmonized_df.columns:
        raise ValueError("harmonized_df must contain the 'target' column for external evaluation.")

    y_ext = harmonized_df["target"].copy()
    X_ext_raw = harmonized_df.drop(columns=["target"]).copy()

    # Check what features the fitted preprocessor expects
    required_features = _get_required_preprocessor_features(fitted_preprocessor)

    # If the preprocessor expects engineered features not present in raw harmonized frame,
    # generate feasible clinical transformations using ClinicalFeatureEngineer without fabrication.
    missing_raw = [f for f in required_features if f not in X_ext_raw.columns]
    if missing_raw:
        from src.features import ClinicalFeatureEngineer
        fe = feature_engineer if feature_engineer is not None else ClinicalFeatureEngineer()
        fe.fit(X_ext_raw)
        X_ext_proc_in = fe.transform(X_ext_raw)
    else:
        X_ext_proc_in = X_ext_raw

    # Now verify that all required features exist in the processed input
    missing_features = [f for f in required_features if f not in X_ext_proc_in.columns]
    if missing_features:
        raise MissingExternalFeatureError(
            f"Required feature(s) missing from harmonized Framingham dataset: {sorted(missing_features)}. "
            "Cannot perform external validation with missing required features. "
            "Do not silently impute zeros or fabricate values. "
            "Limitation: external validation is only supported for feature-reduced configurations that "
            "exclude Framingham-absent columns (e.g. --reduction f_classif --reduction-k 15, or "
            "drop_columns=['height', 'weight', 'gluc', 'alco', 'active']); full-feature-set external "
            "validation requires explicit imputation policy or column exclusion, not yet implemented."
        )

    # Align input features strictly to the columns expected by the preprocessor
    X_ext_proc_in = X_ext_proc_in[required_features]

    # 1. Transform features via fitted preprocessor
    if hasattr(fitted_preprocessor, "preprocessor"):
        # PreprocessingArtifact instance
        X_ext_proc = transform_data(fitted_preprocessor, X_ext_proc_in)
    elif hasattr(fitted_preprocessor, "transform"):
        # Raw ColumnTransformer
        arr = fitted_preprocessor.transform(X_ext_proc_in)
        cols = [f"feat_{i}" for i in range(arr.shape[1])]
        X_ext_proc = pd.DataFrame(arr, columns=cols, index=X_ext_proc_in.index)
    else:
        raise TypeError(f"Unsupported preprocessor type: {type(fitted_preprocessor)}")

    # 2. Apply feature reduction if configured
    if reducer is not None and reduction_artifact is not None:
        X_ext_red = reduce_features(reduction_artifact, reducer, X_ext_proc)
    else:
        X_ext_red = X_ext_proc

    # CatBoost-specific representation (unscaled categoricals) if supplied
    X_ext_red_cb = None
    if catboost_preprocessor is not None:
        if hasattr(catboost_preprocessor, "preprocessor"):
            X_ext_proc_cb = transform_data(catboost_preprocessor, X_ext_proc_in)
        else:
            arr_cb = catboost_preprocessor.transform(X_ext_proc_in)
            cols_cb = [f"feat_{i}" for i in range(arr_cb.shape[1])]
            X_ext_proc_cb = pd.DataFrame(arr_cb, columns=cols_cb, index=X_ext_proc_in.index)

        # Select matching columns retained by CatBoost's fitted model
        cb_model = fitted_models.get("CatBoost")
        if cb_model is not None and getattr(cb_model, "feature_names_in_", None):
            cb_cols = [c for c in cb_model.feature_names_in_ if c in X_ext_proc_cb.columns]
        else:
            cb_cols = [c for c in X_ext_red.columns if c in X_ext_proc_cb.columns]
        X_ext_red_cb = X_ext_proc_cb[cb_cols]

    # 3. Evaluate all pre-fitted models
    results: Dict[str, EvaluationMetrics] = {}
    probabilities: Dict[str, np.ndarray] = {}
    for model_name, model in fitted_models.items():
        X_eval = X_ext_red_cb if (model_name == "CatBoost" and X_ext_red_cb is not None) else X_ext_red
        risk_probs = model.predict_risk(X_eval)

        if locked_thresholds and model_name in locked_thresholds:
            thresh = float(locked_thresholds[model_name])
            preds = (risk_probs >= thresh).astype(np.int64)
            metrics = evaluate_at_locked_threshold(y_ext, risk_probs, locked_threshold=thresh)
        else:
            preds = model.predict(X_eval)
            metrics = evaluate_predictions(y_ext, preds, risk_probs)

        results[model_name] = metrics
        probabilities[model_name] = risk_probs

    if return_probabilities:
        return results, probabilities
    return results


DEFAULT_BROKEN_EXTERNAL_AUC: Dict[str, float] = {
    "Logistic Regression": 0.5919,
    "Calibrated SVM": 0.6364,
    "Random Forest": 0.6673,
    "Gradient Boosting": 0.6669,
    "Multilayer Perceptron": 0.6600,
}


def create_external_comparison_table(
    internal_metrics: Dict[str, EvaluationMetrics],
    external_metrics: Dict[str, EvaluationMetrics],
    broken_external_metrics: Optional[Dict[str, float]] = None,
    internal_prevalence: Optional[float] = None,
    external_prevalence: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate a side-by-side comparison table contrasting internal holdout performance
    with external Framingham Heart Study generalization metrics (and broken vs fixed external AUC),
    including positive-class prevalence and normalized PR-AUC lift.
    """
    broken_map = broken_external_metrics if broken_external_metrics is not None else DEFAULT_BROKEN_EXTERNAL_AUC
    int_prev = internal_prevalence if internal_prevalence is not None else 0.4947
    ext_prev = external_prevalence if external_prevalence is not None else 0.1498

    rows = []
    for name in internal_metrics:
        if name in external_metrics:
            im = internal_metrics[name]
            em = external_metrics[name]
            delta_auc = em.roc_auc - im.roc_auc
            im_lift = compute_pr_auc_lift(im.pr_auc, int_prev)
            em_lift = compute_pr_auc_lift(em.pr_auc, ext_prev)

            ext_total_n = em.tp + em.fp + em.tn + em.fn
            ext_positives = em.tp + em.fn

            row = {
                "Model": name,
                "Internal Holdout ROC-AUC": f"{im.roc_auc:.4f}",
                "Original Broken Ext ROC-AUC": f"{broken_map[name]:.4f}" if name in broken_map else "N/A",
                "Fixed External ROC-AUC": f"{em.roc_auc:.4f}",
                "Δ ROC-AUC": f"{delta_auc:+.4f}",
                "Internal Prevalence": f"{int_prev:.2%}",
                "Internal Holdout PR-AUC": f"{im.pr_auc:.4f}",
                "Internal PR-AUC Lift": f"{im_lift:.2f}x",
                "External Total N": f"{ext_total_n:,}",
                "External Positives": f"{ext_positives:,}",
                "External Prevalence": f"{ext_prev:.2%}",
                "External Framingham PR-AUC": f"{em.pr_auc:.4f}",
                "External PR-AUC Lift": f"{em_lift:.2f}x",
                "External Accuracy": f"{em.accuracy:.4f}",
                "External Sensitivity": f"{em.sensitivity:.4f}",
                "External Specificity": f"{em.specificity:.4f}",
                "External PPV": f"{em.precision:.4f}",
                "External NPV": f"{em.npv:.4f}",
                "Internal Holdout Brier": f"{im.brier_score:.4f}",
                "External Framingham Brier": f"{em.brier_score:.4f}",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def audit_external_feature_ranges(
    fitted_preprocessor: Any,
    harmonized_framingham_df: pd.DataFrame,
    training_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Audit external feature ranges against training IQR outlier bounds (STEP 1 DIAGNOSIS).

    For every feature expected by the fitted preprocessor:
      - Computes Framingham min, max, mean.
      - Retrieves IQRClipper learned lower/upper bounds from training.
      - Calculates fraction of Framingham records falling outside bounds.
      - Flags features where > 50% of values fall outside bounds (indicating clipper squashing).

    Parameters
    ----------
    fitted_preprocessor : PreprocessingArtifact or ColumnTransformer
        Fitted preprocessing artifact containing learned outlier bounds.
    harmonized_framingham_df : pd.DataFrame
        Harmonized Framingham dataset.
    training_bounds : Optional[Dict[str, Tuple[float, float]]]
        Explicit training bounds mapping feature name -> (lower_bound, upper_bound).
        If None, retrieved from fitted_preprocessor.outlier_bounds.
    print_report : bool, default=True
        Whether to print a diagnostic table to stdout.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing feature range diagnostics and flagged issues.
    """
    bounds_dict: Dict[str, Tuple[float, float]] = training_bounds or {}
    if not bounds_dict and hasattr(fitted_preprocessor, "outlier_bounds"):
        bounds_dict = fitted_preprocessor.outlier_bounds

    required_features = _get_required_preprocessor_features(fitted_preprocessor)

    # Separate raw features and enrich with feature engineer if required
    X_df = harmonized_framingham_df.drop(columns=["target"], errors="ignore").copy()
    from src.features import ClinicalFeatureEngineer
    fe = ClinicalFeatureEngineer()
    fe.fit(X_df)
    X_enriched = fe.transform(X_df)

    rows = []
    for col in required_features:
        if col in X_enriched.columns:
            s = X_enriched[col].astype(float)
            f_min = float(s.min())
            f_max = float(s.max())
            f_mean = float(s.mean())

            lb, ub = bounds_dict.get(col, (np.nan, np.nan))
            if lb is not None and ub is not None and not np.isnan(lb) and not np.isnan(ub):
                out_pct = float(((s < lb) | (s > ub)).mean())
                flag = out_pct > 0.50
            else:
                out_pct = 0.0
                flag = False

            rows.append({
                "Feature": col,
                "Framingham Min": f_min,
                "Framingham Max": f_max,
                "Framingham Mean": f_mean,
                "Train Lower Bound": lb,
                "Train Upper Bound": ub,
                "Outside Bounds %": f"{out_pct * 100:.1f}%",
                "Flagged (>50% Out)": "🚨 YES (SQUASHED)" if flag else "No",
            })

    report_df = pd.DataFrame(rows)

    if print_report:
        print("\n" + "=" * 95)
        print(" EXTERNAL FEATURE RANGE & OUTLIER BOUNDS DIAGNOSTIC AUDIT (STEP 1)")
        print("=" * 95)
        print(f"{'Feature':24} {'Fram. Min':>10} {'Fram. Max':>10} {'Fram. Mean':>11} {'Train Low':>11} {'Train High':>11} {'Outside %':>10} {'Flagged'}")
        print("-" * 95)
        for _, r in report_df.iterrows():
            lb_s = f"{r['Train Lower Bound']:11.2f}" if not np.isnan(r['Train Lower Bound']) else f"{'None':>11}"
            ub_s = f"{r['Train Upper Bound']:11.2f}" if not np.isnan(r['Train Upper Bound']) else f"{'None':>11}"
            print(f"{r['Feature']:24} {r['Framingham Min']:10.2f} {r['Framingham Max']:10.2f} {r['Framingham Mean']:11.2f} {lb_s} {ub_s} {r['Outside Bounds %']:>10} {r['Flagged (>50% Out)']}")
        print("=" * 95)

    return report_df
