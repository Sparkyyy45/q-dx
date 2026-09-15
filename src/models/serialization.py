"""
Production Pipeline Artifact and Serialization Engine.
Encapsulates feature engineering, preprocessing, feature reduction, model prediction,
threshold locking, and local explainability for production deployment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.features import ClinicalFeatureEngineer
from src.models.base import BaseCardioModel
from src.preprocess import transform_data
from src.reduction import reduce_features

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Production-grade pipeline bundle encapsulating the entire inference stack:
    raw features -> clinical engineering -> preprocessing -> reduction ->
    calibrated prediction -> locked threshold.
    """

    def __init__(
        self,
        model: BaseCardioModel,
        feature_engineer: ClinicalFeatureEngineer,
        preprocessor_artifact: Any,
        preprocessor: Optional[Any] = None,
        reducer: Optional[Any] = None,
        reduction_artifact: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_catboost: bool = False,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.preprocessor_artifact = preprocessor_artifact
        self.preprocessor = preprocessor
        self.reducer = reducer
        self.reduction_artifact = reduction_artifact
        self.metadata = metadata or {}
        self.is_catboost = is_catboost
        self.locked_threshold = float(self.metadata.get("locked_threshold", 0.50))
        self.model_name = self.metadata.get("model_name", getattr(model, "name", "Model"))
        self.cohort_mean_: Optional[pd.Series] = None

    def transform_raw_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Transform raw patient inputs through feature engineering and preprocessing."""
        df_in = df_raw.copy()

        # Handle age / age_years alignment
        if "age_years" in df_in.columns and "age" not in df_in.columns:
            # If only age_years is provided, compute age in days for engineering
            df_in["age"] = df_in["age_years"] * 365.25

        # 1. Feature Engineering
        df_fe = self.feature_engineer.transform(df_in)

        # Standardize root collinearity: drop raw 'age' (days) retaining standardized 'age_years'
        if "age" in df_fe.columns and "age_years" in df_fe.columns:
            df_fe = df_fe.drop(columns=["age"])

        # 2. Preprocessing
        df_proc = transform_data(self.preprocessor_artifact, df_fe)

        # 3. Feature Reduction
        if self.reducer is not None and self.reduction_artifact is not None:
            df_red = reduce_features(self.reduction_artifact, self.reducer, df_proc)
        else:
            df_red = df_proc

        # 4. CatBoost specific categorical formatting
        if self.is_catboost:
            cb_cat_cols = [c for c in ["gender", "cholesterol", "gluc"] if c in df_red.columns]
            for col in cb_cat_cols:
                df_red[col] = df_red[col].round().astype(np.int64)

        # 5. Enforce column order and completeness matching training estimator / metadata
        expected = None
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_:
            expected = self.model.feature_names_in_
        elif "features" in self.metadata and self.metadata["features"]:
            expected = self.metadata["features"]

        if expected:
            missing = [c for c in expected if c not in df_red.columns]
            if missing:
                raise ValueError(f"Missing expected features after preprocessing: {missing}")
            df_red = df_red[expected]

        if self.is_catboost:
            cb_cat_cols = [c for c in ["gender", "cholesterol", "gluc"] if c in df_red.columns]
            for col in cb_cat_cols:
                df_red[col] = df_red[col].round().astype(np.int64)

        return df_red

    def predict_risk(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Return calibrated 1D array of CVD risk probabilities P(cardio = 1)."""
        df_ready = self.transform_raw_dataframe(df_raw)
        return self.model.predict_risk(df_ready)

    def predict_proba(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Return (N, 2) array of class probabilities."""
        df_ready = self.transform_raw_dataframe(df_raw)
        return self.model.predict_proba(df_ready)

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Return binary predictions strictly using the locked operational threshold."""
        probas = self.predict_risk(df_raw)
        return (probas >= self.locked_threshold).astype(np.int64)

    def explain(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Compute genuine model-specific local attribution for a patient."""
        df_ready = self.transform_raw_dataframe(df_raw)
        prob = float(self.predict_risk(df_raw)[0])
        pred = 1 if prob >= self.locked_threshold else 0

        if prob < 0.20:
            tier = "Low Estimated Risk (<20%)"
        elif prob <= 0.50:
            tier = "Moderate Estimated Risk (20-50%)"
        else:
            tier = "High Estimated Risk (>50%)"

        feature_names = list(df_ready.columns)
        patient_vector = df_ready.iloc[0].to_numpy(dtype=np.float64)

        from src.explain_risk import CLINICAL_INTERPRETATION_GUIDE, MEDICAL_DISCLAIMER

        top_factors: List[Dict[str, Any]] = []
        explanation_method = "unknown"
        explanation_available = True
        shap_output_space = "unknown"
        output_space = "unknown"

        base_val: Optional[float] = None

        # A. VQC Model: Exact PyTorch autograd input Jacobian (local input sensitivity)
        if self.model.__class__.__name__ == "VQCModel" or "vqc" in self.metadata.get("model_key", ""):
            explanation_method = "quantum_autograd_local_input_sensitivity"
            shap_output_space = "circuit_expectation_gradient"
            output_space = "circuit_expectation_gradient"
            from src.quantum.explain import explain_vqc_circuit
            vqc_exp = explain_vqc_circuit(self.model, patient_vector, feature_names)
            exp_dict = vqc_exp.to_dict() if hasattr(vqc_exp, "to_dict") else (vqc_exp if isinstance(vqc_exp, dict) else {})
            for item in exp_dict.get("feature_attributions", []):
                score = float(item.get("attribution", 0.0))
                f_name = item.get("feature", "unknown")
                dir_text = "increased local circuit expectation" if score > 0 else "decreased local circuit expectation"
                top_factors.append({
                    "feature": f_name,
                    "feature_label": f_name.replace("_", " ").title(),
                    "score": f"{score:+.4f}",
                    "attribution": score,
                    "attribution_type": "local_input_sensitivity_jacobian",
                    "is_risk_increasing": score > 0,
                    "contribution_direction": dir_text,
                    "clinical_meaning": CLINICAL_INTERPRETATION_GUIDE.get(f_name, ""),
                })

        # B. Linear Models (Logistic Regression / Calibrated SVM): Exact logit contribution beta_j * z_j
        elif hasattr(self.model, "model") and hasattr(self.model.model, "coef_"):
            explanation_method = "linear_model_logit_attribution"
            shap_output_space = "margin_log_odds"
            output_space = "raw_logit"
            coefs = self.model.model.coef_[0]
            contributions = patient_vector * coefs

            for i, f_name in enumerate(feature_names):
                c_val = float(contributions[i])
                dir_text = "contributed positively to the model prediction" if c_val > 0 else "contributed negatively to the model prediction"
                top_factors.append({
                    "feature": f_name,
                    "feature_label": f_name.replace("_", " ").title(),
                    "score": f"{c_val:+.4f}",
                    "attribution": c_val,
                    "attribution_type": "linear_logit_component",
                    "is_risk_increasing": c_val > 0,
                    "contribution_direction": dir_text,
                    "clinical_meaning": CLINICAL_INTERPRETATION_GUIDE.get(f_name, ""),
                })
            top_factors.sort(key=lambda x: abs(x["attribution"]), reverse=True)

        # C. Tree-based Models (CatBoost, Random Forest, XGBoost, LightGBM, HistGB): Exact local TreeSHAP
        elif self.model.__class__.__name__ in ("CatBoostModel", "XGBoostModel", "LightGBMModel", "RandomForestModel", "GradientBoostingModel") or (hasattr(self.model, "model") and hasattr(self.model.model, "predict_proba")):
            explanation_method = "treeshap"
            is_rf = self.model.__class__.__name__ == "RandomForestModel"
            shap_output_space = "probability" if is_rf else "margin_log_odds"
            output_space = "probability" if is_rf else "raw_logit"
            try:
                import shap
                underlying = getattr(self.model, "model", self.model)
                explainer = shap.TreeExplainer(underlying)
                res = explainer(df_ready)
                raw_vals = res.values
                if hasattr(res, "base_values"):
                    bv_arr = np.asarray(res.base_values)
                    if bv_arr.ndim > 1 and bv_arr.shape[1] > 1:
                        base_val = float(bv_arr[0, 1])
                    elif bv_arr.size > 0:
                        base_val = float(bv_arr.flatten()[0])

                if raw_vals.ndim == 3:
                    shap_vals = raw_vals[0, :, 1]
                elif raw_vals.ndim == 2:
                    shap_vals = raw_vals[0, :]
                else:
                    shap_vals = raw_vals.flatten()

                for i, f_name in enumerate(feature_names):
                    s_val = float(shap_vals[i])
                    dir_text = "contributed positively to the model prediction" if s_val > 0 else "contributed negatively to the model prediction"
                    top_factors.append({
                        "feature": f_name,
                        "feature_label": f_name.replace("_", " ").title(),
                        "score": f"{s_val:+.4f}",
                        "attribution": s_val,
                        "attribution_type": "treeshap",
                        "is_risk_increasing": s_val > 0,
                        "contribution_direction": dir_text,
                        "clinical_meaning": CLINICAL_INTERPRETATION_GUIDE.get(f_name, ""),
                    })
                top_factors.sort(key=lambda x: abs(x["attribution"]), reverse=True)
            except Exception as exc:
                logger.warning(f"TreeSHAP calculation failed: {exc}")
                explanation_available = False
                explanation_method = "treeshap_unavailable"
                shap_output_space = "none"
                output_space = "none"
                top_factors = []

        else:
            explanation_available = False
            explanation_method = "cohort_feature_deviation"
            shap_output_space = "none"
            output_space = "none"
            top_factors = []

        resp = {
            "model": self.model_name,
            "probability": round(prob, 4),
            "risk_probability": round(prob, 4),
            "threshold": self.locked_threshold,
            "applied_threshold": self.locked_threshold,
            "threshold_source": "OOF_Youden",
            "prediction": pred,
            "risk_tier": tier,
            "explanation_available": explanation_available,
            "method": explanation_method,
            "explanation_method": explanation_method,
            "output_space": output_space,
            "shap_output_space": shap_output_space,
            "top_factors": top_factors[:6],
            "clinical_summary": f"Estimated model probability of {prob*100:.1f}% placing patient in {tier}.",
            "disclaimer": MEDICAL_DISCLAIMER,
        }
        if base_val is not None:
            resp["base_value"] = round(base_val, 4)
            resp["expected_value"] = round(base_val, 4)
        return resp

    def save(self, target_dir: Union[str, Path]) -> Path:
        """Save pipeline bundle and decoupled modular artifacts to disk."""
        import torch

        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Monolithic pipeline for backward compatibility
        pipeline_file = out_dir / "pipeline.joblib"
        joblib.dump(self, pipeline_file)

        # 2. Decoupled preprocessing
        preproc_bundle = {
            "feature_engineer": self.feature_engineer,
            "preprocessor_artifact": self.preprocessor_artifact,
            "preprocessor": self.preprocessor,
            "reducer": self.reducer,
            "reduction_artifact": self.reduction_artifact,
            "is_catboost": self.is_catboost,
            "cohort_mean": self.cohort_mean_,
        }
        joblib.dump(preproc_bundle, out_dir / "preprocessing.joblib")

        # 3. Decoupled model / weights
        if hasattr(self.model, "parameters_") and hasattr(self.model, "projection"):
            state_dict = {
                "parameters": self.model.parameters_.detach().cpu(),
                "projection_weight": self.model.projection.weight.detach().cpu() if self.model.projection is not None else None,
                "projection_bias": self.model.projection.bias.detach().cpu() if self.model.projection is not None and self.model.projection.bias is not None else None,
                "calibrator": getattr(self.model, "calibrator_", None),
            }
            torch.save(state_dict, out_dir / "weights.pt")
        else:
            joblib.dump(self.model, out_dir / "model.joblib")

        # 4. Decoupled threshold metadata
        threshold_data = {
            "locked_threshold": self.locked_threshold,
            "threshold_source": "OOF_Youden",
            "metric": "Youden_J",
            "oof_youden_j": self.metadata.get("oof_youden_j"),
        }
        with open(out_dir / "threshold.json", "w") as f:
            json.dump(threshold_data, f, indent=2)

        # 5. Decoupled schema
        schema_data = {
            "features": self.metadata.get("features", []),
            "n_features": len(self.metadata.get("features", [])),
        }
        with open(out_dir / "schema.json", "w") as f:
            json.dump(schema_data, f, indent=2)

        # 6. Provenance metadata
        meta_file = out_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # 7. Circuit configuration
        if hasattr(self.model, "get_circuit_summary"):
            circuit_file = out_dir / "circuit_config.json"
            with open(circuit_file, "w") as f:
                json.dump(self.model.get_circuit_summary(), f, indent=2)

        return out_dir

    @classmethod
    def load(cls, target_dir: Union[str, Path]) -> ProductionPipeline:
        """Load serialized pipeline bundle from disk."""
        in_dir = Path(target_dir)
        pipeline_file = in_dir / "pipeline.joblib"
        if not pipeline_file.is_file():
            raise FileNotFoundError(f"Pipeline bundle not found at '{pipeline_file}'")

        pipeline = joblib.load(pipeline_file)

        meta_file = in_dir / "metadata.json"
        if meta_file.is_file():
            with open(meta_file, "r") as f:
                pipeline.metadata = json.load(f)
            pipeline.locked_threshold = float(pipeline.metadata.get("locked_threshold", 0.50))
            pipeline.model_name = pipeline.metadata.get("model_name", getattr(pipeline.model, "name", "Model"))

        return pipeline


def reconstruct_production_pipeline(target_dir: Union[str, Path]) -> ProductionPipeline:
    """
    Reconstruct ProductionPipeline statelessly from decoupled artifacts:
    preprocessing.joblib + (model.joblib | (weights.pt + circuit_config.json)) +
    threshold.json + schema.json + metadata.json.
    Eliminates monolithic runtime bindings.
    """
    import torch

    in_dir = Path(target_dir)
    preproc_file = in_dir / "preprocessing.joblib"
    if not preproc_file.is_file():
        raise FileNotFoundError(f"Decoupled preprocessing not found at '{preproc_file}'")

    preproc_bundle = joblib.load(preproc_file)

    meta_file = in_dir / "metadata.json"
    metadata = {}
    if meta_file.is_file():
        with open(meta_file, "r") as f:
            metadata = json.load(f)

    threshold_file = in_dir / "threshold.json"
    if threshold_file.is_file():
        with open(threshold_file, "r") as f:
            t_data = json.load(f)
            metadata["locked_threshold"] = t_data.get("locked_threshold", 0.50)

    weights_file = in_dir / "weights.pt"
    model_file = in_dir / "model.joblib"

    if weights_file.is_file():
        # Reconstruct PyTorch / Quantum model from weights.pt and circuit_config.json
        circuit_config_file = in_dir / "circuit_config.json"
        n_qubits = 4
        n_layers = 2
        if circuit_config_file.is_file():
            with open(circuit_config_file, "r") as f:
                cc = json.load(f)
                n_qubits = cc.get("n_qubits", 4)
                n_layers = cc.get("n_layers", 2)

        from src.models.vqc_model import VQCModel
        model = VQCModel(n_qubits=n_qubits, n_layers=n_layers)
        state_dict = torch.load(weights_file, weights_only=False)
        model.parameters_ = torch.nn.Parameter(state_dict["parameters"])
        if state_dict.get("projection_weight") is not None:
            pw = state_dict["projection_weight"]
            d_in = pw.shape[1]
            model.projection = torch.nn.Linear(d_in, n_qubits)
            model.projection.weight.data = pw
            if state_dict.get("projection_bias") is not None:
                model.projection.bias.data = state_dict["projection_bias"]
        model.calibrator_ = state_dict.get("calibrator")
        model.is_fitted_ = True
    elif model_file.is_file():
        model = joblib.load(model_file)
    else:
        return load_production_pipeline(target_dir)

    schema_file = in_dir / "schema.json"
    if schema_file.is_file():
        try:
            with open(schema_file, "r") as f:
                s_data = json.load(f)
                if "features" in s_data and s_data["features"]:
                    metadata.setdefault("features", s_data["features"])
        except Exception:
            pass

    if "features" in metadata and metadata["features"] and hasattr(model, "feature_names_in_"):
        if not model.feature_names_in_:
            model.feature_names_in_ = list(metadata["features"])

    pipeline = ProductionPipeline(
        model=model,
        feature_engineer=preproc_bundle["feature_engineer"],
        preprocessor_artifact=preproc_bundle["preprocessor_artifact"],
        preprocessor=preproc_bundle.get("preprocessor"),
        reducer=preproc_bundle.get("reducer"),
        reduction_artifact=preproc_bundle.get("reduction_artifact"),
        metadata=metadata,
        is_catboost=preproc_bundle.get("is_catboost", False),
    )
    pipeline.cohort_mean_ = preproc_bundle.get("cohort_mean")
    pipeline.locked_threshold = float(metadata.get("locked_threshold", 0.50))
    pipeline.model_name = metadata.get("model_name", getattr(model, "name", "Model"))
    return pipeline


def save_production_pipeline(pipeline: ProductionPipeline, target_dir: Union[str, Path]) -> Path:
    """Save production pipeline bundle."""
    return pipeline.save(target_dir)


def load_production_pipeline(target_dir: Union[str, Path]) -> ProductionPipeline:
    """Load production pipeline bundle."""
    return ProductionPipeline.load(target_dir)


__all__ = [
    "ProductionPipeline",
    "save_production_pipeline",
    "load_production_pipeline",
    "reconstruct_production_pipeline",
]
