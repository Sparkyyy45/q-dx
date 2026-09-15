"""
Clinical risk explanation, feature attribution, and patient-level risk interpretation.
Strictly presents predictions as 'Estimated model probability' and enforces clinical disclaimers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.models.base import BaseCardioModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.feature_dictionary import EXAMINATION_VS_SUBJECTIVE_NOTE, get_feature_type

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: The output represents an estimated statistical model probability and is "
    "NOT a medical diagnosis. Clinical prediction tools are decision-support aids designed "
    "to assist licensed healthcare providers, not replace professional clinical judgment."
)

CLINICAL_INTERPRETATION_GUIDE: Dict[str, str] = {
    "ap_hi": "Systolic Blood Pressure: Peak arterial pressure during ventricular contraction; primary mechanical driver of vascular remodeling and coronary wall stress.",
    "ap_lo": "Diastolic Blood Pressure: Resting systemic vascular resistance between contractions; high values indicate continuous microvascular tension.",
    "pulse_pressure": "Pulse Pressure (ap_hi - ap_lo): Primary hemodynamic marker of arterial wall stiffening and diminished aortic compliance.",
    "mean_arterial_pressure": "Mean Arterial Pressure (MAP): Mean systemic perfusion pressure determining end-organ perfusion and systemic vascular strain.",
    "age_years": "Standardized Age: Cumulative biological aging of vascular endothelium; strongest non-modifiable cardiovascular risk factor.",
    "age": "Chronological Age (days): Total elapsed days of life representing cumulative vascular exposure.",
    "age_squared": "Accelerated Vascular Age: Non-linear acceleration in atherosclerotic plaque burden observed in older adults.",
    "age_bp_interaction": "Vascular Age × Systolic Burden: Synergistic cardiovascular hazard where advanced arterial age amplifies the harm of hypertension.",
    "cholesterol": "Serum Total Cholesterol: Circulating atherogenic lipoproteins contributing to arterial intimal plaque formation and luminal narrowing.",
    "gluc": "Blood Glucose Level: Biomarker of glycemic dysregulation and insulin resistance promoting vascular endothelial inflammation.",
    "cholesterol_gluc_ratio": "Task-Specific Engineered Ratio: Ordinal ratio of cholesterol to glucose capturing relative dyslipidemia/dysglycemia interaction, not an established biomarker.",
    "metabolic_synergy": "Task-Specific Engineered Interaction: Binary flag for concurrent elevation of ordinal cholesterol and glucose (both >= 2); heuristic marker, not a clinical diagnostic criterion.",
    "bmi": "Body Mass Index: Anthropometric estimate of adiposity; correlates with left ventricular hypertrophy and systemic inflammation.",
    "height": "Patient Height: Component of body surface area and hemodynamic vascular impedance.",
    "weight": "Patient Weight: Gross body mass influencing total blood volume and cardiac cardiac output demands.",
    "smoke": "Active Tobacco Smoking: Direct oxidant chemical injury inducing endothelial dysfunction and hypercoagulability.",
    "alco": "Alcohol Consumption: Excess intake promotes neurohormonal activation, cardiac arrhythmia, and secondary hypertension.",
    "active": "Physical Activity: Regular aerobic activity preserves endothelial nitric oxide availability and lowers resting vascular tone.",
    "gender": "Biological Sex: Demographic covariate capturing sex-specific cardiovascular risk trajectories.",
    "is_hypertensive_stage2": "ACC/AHA Stage 2 Hypertension: Clinically severe blood pressure elevation (>=140 SBP or >=90 DBP).",
    "high_risk_lifestyle": "Task-Specific Engineered Interaction: Co-occurrence of smoking and alcohol intake compounding oxidative vascular stress.",
    "log_pulse_pressure": "Log Pulse Pressure: Skew-stabilized representation of large-artery pulsatile load.",
    "health_index": "Task-Specific Engineered Lifestyle Score: Linear heuristic (+1.0 active, -0.5 smoke, -0.5 alco); exploratory interaction, not a validated biomarker.",
}


@dataclass
class PatientRiskExplanation:
    """Individual patient risk explanation with clinical context."""
    patient_id: Optional[Any]
    estimated_risk_probability: float
    risk_tier: str  # "Low (<20%)", "Moderate (20-50%)", "High (>50%)"
    top_risk_increasing_factors: List[Dict[str, Any]]
    top_risk_reducing_factors: List[Dict[str, Any]]
    clinical_summary: str
    disclaimer: str = MEDICAL_DISCLAIMER
    attribution_method: str = "cohort_feature_deviation"
    is_model_attribution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


from sklearn.linear_model import LinearRegression, LogisticRegression


def compute_vif(
    X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all features to diagnose multicollinearity.
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is the coefficient of determination when feature j
    is regressed on all other remaining features.
    """
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    n_features = X_arr.shape[1]
    rows = []

    for i in range(n_features):
        y_i = X_arr[:, i]
        X_rest = np.delete(X_arr, i, axis=1)

        if np.std(y_i) < 1e-8:
            vif = 1.0
        else:
            lr = LinearRegression().fit(X_rest, y_i)
            r2 = lr.score(X_rest, y_i)
            if r2 >= 0.99999:
                vif = 100000.0  # Perfect or near-perfect collinearity
            elif r2 <= 0.0:
                vif = 1.0
            else:
                vif = 1.0 / (1.0 - r2)

        severity = (
            "Severe (VIF > 10)" if vif > 10.0 else
            "Moderate (5 < VIF <= 10)" if vif > 5.0 else
            "Low (Acceptable)"
        )
        rows.append({
            "Feature": names[i],
            "Feature Type": get_feature_type(names[i]),
            "VIF": float(vif),
            "Collinearity Severity": severity,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def prune_collinear_features(
    X: pd.DataFrame, max_vif: float = 10.0
) -> Tuple[List[str], pd.DataFrame]:
    """
    Iteratively eliminate the feature with the highest VIF until all remaining
    features have VIF <= max_vif.
    """
    remaining_cols = list(X.columns)
    while len(remaining_cols) > 1:
        current_vif_df = compute_vif(X[remaining_cols])
        max_row = current_vif_df.iloc[0]
        if max_row["VIF"] <= max_vif:
            break
        # Drop highest VIF feature
        drop_feat = max_row["Feature"]
        remaining_cols.remove(drop_feat)

    final_vif = compute_vif(X[remaining_cols])
    return remaining_cols, final_vif


def explain_logistic_regression(
    model: LogisticRegressionModel,
    feature_names: Optional[List[str]] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate transparent clinical table of Logistic Regression coefficients and Odds Ratios.
    Also computes full VIF table and fits a de-collinearized explanation model (VIF <= 10)
    when X_train and y_train are provided to eliminate coefficient inflation.

    Returns:
        (odds_ratios_df, full_vif_df)
    """
    names = feature_names or model.feature_names_in_

    # 1. Compute full feature space VIF
    if X_train is not None:
        full_vif_df = compute_vif(X_train[names] if all(c in X_train.columns for c in names) else X_train)
    else:
        full_vif_df = pd.DataFrame(columns=["Feature", "VIF", "Collinearity Severity"])

    # 2. If training data provided, fit de-collinearized explanation LR
    if X_train is not None and y_train is not None:
        sub_X = X_train[names].copy() if all(c in X_train.columns for c in names) else X_train.copy()
        clean_cols, clean_vif_df = prune_collinear_features(sub_X, max_vif=10.0)

        exp_lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        exp_lr.fit(sub_X[clean_cols], y_train)

        coefs = {col: float(c) for col, c in zip(clean_cols, exp_lr.coef_[0])}
        vif_lookup = dict(zip(clean_vif_df["Feature"], clean_vif_df["VIF"]))
        target_names = clean_cols
    else:
        coefs = model.get_coefficients()
        vif_lookup = dict(zip(full_vif_df["Feature"], full_vif_df["VIF"])) if not full_vif_df.empty else {}
        target_names = names

    rows = []
    for f in target_names:
        if f in coefs:
            beta = coefs[f]
            odds_ratio = float(np.exp(beta))
            pct_change = (odds_ratio - 1.0) * 100.0
            direction = "Increases Risk" if beta > 0 else "Protective (Reduces Risk)"
            clinical_text = CLINICAL_INTERPRETATION_GUIDE.get(
                f, "Clinical tabular feature associated with cardiovascular risk profile."
            )
            vif_val = vif_lookup.get(f, 1.0)
            rows.append({
                "Feature": f,
                "Feature Type": get_feature_type(f),
                "VIF": round(vif_val, 2),
                "Log-Odds Beta (β)": round(beta, 4),
                "Odds Ratio (e^β)": round(odds_ratio, 4),
                "% Change in Odds": round(pct_change, 1),
                "Clinical Direction": direction,
                "Clinical Interpretation": clinical_text,
            })

    odds_df = pd.DataFrame(rows)
    if not odds_df.empty:
        odds_df = odds_df.sort_values(by="Odds Ratio (e^β)", ascending=False).reset_index(drop=True)

    return odds_df, full_vif_df


def explain_tree_feature_importance(
    model: RandomForestModel, feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Extract impurity-based (MDI) feature importances for tree models."""
    names = feature_names or model.feature_names_in_
    importances = model.get_feature_importances()

    rows = []
    for f in names:
        if f in importances:
            imp = importances[f]
            rows.append({
                "Feature": f,
                "Gini Importance": imp,
                "Clinical Interpretation": CLINICAL_INTERPRETATION_GUIDE.get(
                    f, "Clinical cardiovascular indicator."
                ),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Gini Importance", ascending=False).reset_index(drop=True)
    return df


def compute_model_permutation_importance(
    model: BaseCardioModel,
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute model-agnostic permutation feature importance strictly on held-out validation data.
    Quantifies the reduction in ROC-AUC when a feature's values are permuted.
    """
    if isinstance(X_val, pd.DataFrame):
        names = list(X_val.columns)
        X_arr = X_val.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X_val, dtype=np.float64)
        names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y_val, dtype=np.int64)

    # Use a scikit-learn compatible wrapper or direct estimator if available
    estimator = model.model if hasattr(model, "model") and model.model is not None else model

    perm = permutation_importance(
        estimator=estimator,
        X=X_arr,
        y=y_arr,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    rows = []
    for i, f in enumerate(names):
        mean_imp = float(perm.importances_mean[i])
        std_imp = float(perm.importances_std[i])
        rows.append({
            "Feature": f,
            "Permutation Importance (Mean ΔAUC)": mean_imp,
            "Std Dev": std_imp,
            "Clinical Interpretation": CLINICAL_INTERPRETATION_GUIDE.get(
                f, "Clinical indicator evaluated for prognostic utility."
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Permutation Importance (Mean ΔAUC)", ascending=False).reset_index(drop=True)
    return df


def explain_patient_risk(
    model: BaseCardioModel,
    patient_features: Union[pd.Series, pd.DataFrame, np.ndarray],
    cohort_mean_features: Union[pd.Series, np.ndarray],
    feature_names: List[str],
    patient_id: Optional[Any] = None,
) -> PatientRiskExplanation:
    """
    Provide patient-level estimated probability and feature-level risk contributors.
    Identifies top clinical factors pushing estimated probability above or below cohort average.
    """
    if isinstance(patient_features, pd.DataFrame):
        feat_vector = patient_features.iloc[0].to_numpy(dtype=np.float64)
    elif isinstance(patient_features, pd.Series):
        feat_vector = patient_features.to_numpy(dtype=np.float64)
    else:
        feat_vector = np.asarray(patient_features, dtype=np.float64).flatten()

    if isinstance(cohort_mean_features, pd.Series):
        mean_vector = cohort_mean_features.to_numpy(dtype=np.float64)
    else:
        mean_vector = np.asarray(cohort_mean_features, dtype=np.float64).flatten()

    # Predict risk probability
    X_input = feat_vector.reshape(1, -1)
    risk_prob = float(model.predict_risk(X_input)[0])

    # Stratify into clinical risk tiers
    if risk_prob < 0.20:
        tier = "Low Estimated Risk (<20%)"
    elif risk_prob <= 0.50:
        tier = "Moderate Estimated Risk (20% - 50%)"
    else:
        tier = "High Estimated Risk (>50%)"

    # Compute deviation from cohort reference
    delta = feat_vector - mean_vector

    # If linear/logistic regression, use model weights for logit attribution;
    # otherwise use unweighted cohort deviation (explicitly flagged as non-model attribution)
    is_model_attribution = False
    attribution_method = "cohort_feature_deviation"
    if hasattr(model, "model") and hasattr(model.model, "coef_"):
        weights = model.model.coef_[0]
        is_model_attribution = True
        attribution_method = "linear_model_logit_attribution"
    else:
        weights = np.ones_like(delta)
        is_model_attribution = False
        attribution_method = "cohort_feature_deviation"

    contributions = delta * weights

    increasing = []
    reducing = []

    for i, name in enumerate(feature_names):
        contrib = float(contributions[i])
        val = float(feat_vector[i])
        mean_val = float(mean_vector[i])
        clinical_note = CLINICAL_INTERPRETATION_GUIDE.get(name, "Clinical variable.")

        item = {
            "feature": name,
            "patient_value": val,
            "cohort_mean": mean_val,
            "attribution_score": contrib,
            "attribution_type": attribution_method,
            "is_model_attribution": is_model_attribution,
            "clinical_meaning": clinical_note,
        }
        if contrib > 0:
            increasing.append(item)
        else:
            reducing.append(item)

    increasing.sort(key=lambda x: x["attribution_score"], reverse=True)
    reducing.sort(key=lambda x: x["attribution_score"])

    summary = (
        f"Patient exhibits an estimated model probability of {risk_prob*100:.1f}% for cardiovascular "
        f"disease risk, categorizing into the {tier} band. "
        f"Top positive risk driver: {increasing[0]['feature'] if increasing else 'None'}."
    )

    return PatientRiskExplanation(
        patient_id=patient_id,
        estimated_risk_probability=risk_prob,
        risk_tier=tier,
        top_risk_increasing_factors=increasing[:5],
        top_risk_reducing_factors=reducing[:5],
        clinical_summary=summary,
        disclaimer=MEDICAL_DISCLAIMER,
        attribution_method=attribution_method,
        is_model_attribution=is_model_attribution,
    )


def explain_quantum_risk(
    model: BaseCardioModel,
    patient_features: Union[pd.Series, pd.DataFrame, np.ndarray],
    feature_names: List[str],
) -> Any:
    """
    Explain risk predictions from a quantum machine learning model (e.g. VQCModel)
    via analytical parameter-shift gradient and expectation deviation.
    """
    from src.quantum.explain import explain_vqc_circuit
    return explain_vqc_circuit(model, patient_features, feature_names)

