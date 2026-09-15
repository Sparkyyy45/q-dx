"""
Leakage-safe tabular preprocessing pipeline with IQR outlier clipping,
imputation, scaling, and categorical encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.config import PipelineConfig


def compute_iqr_bounds(
    X: Union[np.ndarray, pd.DataFrame],
    factor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute strictly unsupervised IQR lower and upper bounds across pooled samples.
    By design, this function does NOT accept a target/label argument, making per-class
    or label-conditioned bound fitting structurally impossible.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=np.float64, copy=False)
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    q1 = np.nanpercentile(X_arr, 25, axis=0)
    q3 = np.nanpercentile(X_arr, 75, axis=0)
    iqr = q3 - q1

    lower_bounds = q1 - factor * iqr
    upper_bounds = q3 + factor * iqr

    # If IQR is zero or near-zero (e.g. discrete/ordinal features),
    # do not squash categories into a degenerate single-point constant.
    zero_iqr = iqr <= 1e-6
    lower_bounds[zero_iqr] = -np.inf
    upper_bounds[zero_iqr] = np.inf

    return lower_bounds, upper_bounds


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Interquartile Range (IQR) outlier clipper.
    Computes Q1, Q3, and clipping bounds strictly during fit() on training data.
    Transform() applies the learned bounds without updating internal state.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> IQRClipper:
        """
        Fit learned bounds strictly from X across all pooled samples.
        Note: y is accepted solely for scikit-learn Pipeline API compatibility,
        and is completely ignored. Bound computation is delegated to compute_iqr_bounds(X, factor).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)

        self.lower_bounds_, self.upper_bounds_ = compute_iqr_bounds(X, factor=self.factor)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("IQRClipper has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64, copy=True)
        else:
            X_arr = np.array(X, dtype=np.float64, copy=True)

        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return learned clipping bounds per feature."""
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            return {}
        names = self.feature_names_in_ or [f"feature_{i}" for i in range(len(self.lower_bounds_))]
        return {
            names[i]: (float(self.lower_bounds_[i]), float(self.upper_bounds_[i]))
            for i in range(len(names))
        }


def _get_scaler(scaling_strategy: str) -> BaseEstimator:
    """Return configured scaler instance."""
    strategy = (scaling_strategy or "none").lower()
    if strategy == "standard":
        return StandardScaler()
    elif strategy == "robust":
        return RobustScaler()
    elif strategy == "minmax":
        return MinMaxScaler()
    elif strategy == "none":
        return FunctionTransformer(func=None, validate=False)
    else:
        raise ValueError(
            f"Unsupported scaling_strategy: '{scaling_strategy}'. "
            f"Choose from ['standard', 'robust', 'minmax', 'none']."
        )


@dataclass
class PreprocessingArtifact:
    """
    Holds the fitted preprocessor, feature names, learned outlier bounds,
    and provenance metadata.
    """
    preprocessor: ColumnTransformer
    input_numeric_features: List[str]
    input_categorical_features: List[str]
    output_feature_names: List[str]
    scaling_strategy: str
    outlier_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def build_preprocessor(
    config: PipelineConfig,
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Construct an unfitted scikit-learn ColumnTransformer for numeric and categorical features.
    """
    # 1. Numeric pipeline
    num_steps: List[Tuple[str, BaseEstimator]] = []
    if config.outlier_clipping:
        num_steps.append(("iqr_clipper", IQRClipper(factor=config.outlier_factor)))

    num_steps.append((
        "imputer",
        SimpleImputer(strategy=config.numeric_imputation_strategy)
    ))
    num_steps.append(("scaler", _get_scaler(config.scaling_strategy)))
    numeric_pipeline = Pipeline(steps=num_steps)

    # 2. Categorical pipeline
    cat_steps: List[Tuple[str, BaseEstimator]] = [
        (
            "imputer",
            SimpleImputer(strategy=config.categorical_imputation_strategy, fill_value="missing")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
    ]
    categorical_pipeline = Pipeline(steps=cat_steps)

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


CATBOOST_CATEGORICAL_COLUMNS: List[str] = ["gender", "cholesterol", "gluc"]


def build_catboost_preprocessor(
    config: PipelineConfig,
    numeric_features: List[str],
    categorical_features: List[str],
    bypass_categorical_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct an unfitted scikit-learn ColumnTransformer specifically for CatBoost.
    Guarantees that specified categorical columns (gender, cholesterol, gluc) bypass
    RobustScaler/StandardScaler and IQRClipper, retaining their exact discrete integer encodings
    for CatBoost's native ordered target statistics.
    """
    cat_bypass = bypass_categorical_columns or CATBOOST_CATEGORICAL_COLUMNS
    bypass_cols = [c for c in cat_bypass if c in numeric_features]
    pure_numeric = [c for c in numeric_features if c not in bypass_cols]

    # 1. Pure numeric pipeline (clipped, imputed, scaled)
    num_steps: List[Tuple[str, BaseEstimator]] = []
    if config.outlier_clipping:
        num_steps.append(("iqr_clipper", IQRClipper(factor=config.outlier_factor)))

    num_steps.append((
        "imputer",
        SimpleImputer(strategy=config.numeric_imputation_strategy)
    ))
    num_steps.append(("scaler", _get_scaler(config.scaling_strategy)))
    numeric_pipeline = Pipeline(steps=num_steps)

    # 2. Native categorical bypass pipeline (imputed if missing, but NO clipper, NO scaler)
    cat_bypass_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # 3. Categorical pipeline (OneHotEncoder for any object/string features if present)
    cat_steps: List[Tuple[str, BaseEstimator]] = [
        (
            "imputer",
            SimpleImputer(strategy=config.categorical_imputation_strategy, fill_value="missing")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
    ]
    categorical_pipeline = Pipeline(steps=cat_steps)

    transformers = []
    if pure_numeric:
        transformers.append(("num", numeric_pipeline, pure_numeric))
    if bypass_cols:
        transformers.append(("cat_native", cat_bypass_pipeline, bypass_cols))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Store attribute for fallback feature name retrieval
    transformer._cat_bypass_cols = bypass_cols
    return transformer


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
) -> PreprocessingArtifact:
    """
    Fit preprocessor strictly on training data and generate PreprocessingArtifact.
    Never pass validation or test data to this function.
    """
    num_cols = numeric_features or []
    cat_cols = categorical_features or []

    # Fit ColumnTransformer on train data only
    preprocessor.fit(X_train, y_train)

    # Resolve output feature names safely across scikit-learn versions
    try:
        out_names = list(preprocessor.get_feature_names_out())
    except Exception:
        out_names = []
        if "num" in preprocessor.named_transformers_:
            pure_numeric = [c for c in num_cols if c not in getattr(preprocessor, "_cat_bypass_cols", [])]
            out_names.extend(pure_numeric)
        if "cat_native" in preprocessor.named_transformers_:
            out_names.extend(getattr(preprocessor, "_cat_bypass_cols", []))
        if "cat" in preprocessor.named_transformers_:
            encoder = preprocessor.named_transformers_["cat"].named_steps.get("encoder")
            if encoder and hasattr(encoder, "get_feature_names_out"):
                out_names.extend(list(encoder.get_feature_names_out(cat_cols)))

    # Extract outlier bounds if IQRClipper was fitted
    outlier_bounds = {}
    if "num" in preprocessor.named_transformers_:
        num_steps = dict(preprocessor.named_transformers_["num"].named_steps)
        if "iqr_clipper" in num_steps:
            outlier_bounds = num_steps["iqr_clipper"].get_bounds()

    provenance = {
        "train_rows": len(X_train),
        "scaling_strategy": config.scaling_strategy if config else "unknown",
        "outlier_clipping": config.outlier_clipping if config else True,
        "input_feature_count": len(num_cols) + len(cat_cols),
        "output_feature_count": len(out_names),
    }

    return PreprocessingArtifact(
        preprocessor=preprocessor,
        input_numeric_features=num_cols,
        input_categorical_features=cat_cols,
        output_feature_names=out_names,
        scaling_strategy=config.scaling_strategy if config else "standard",
        outlier_bounds=outlier_bounds,
        provenance=provenance,
    )


def transform_data(
    artifact: PreprocessingArtifact, X: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform input data using an already-fitted PreprocessingArtifact.
    Returns transformed DataFrame with aligned feature names.
    """
    transformed_arr = artifact.preprocessor.transform(X)
    cols = artifact.output_feature_names or [
        f"feature_{i}" for i in range(transformed_arr.shape[1])
    ]
    return pd.DataFrame(transformed_arr, columns=cols, index=X.index)


def prepare_fold(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: pd.DataFrame,
    config: PipelineConfig,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessingArtifact]:
    """
    Strictly isolated fold preparation:
    1. Builds fresh unfitted ColumnTransformer
    2. Fits transformer ONLY on X_train
    3. Transforms X_train using fitted artifact
    4. Transforms X_val using the fitted artifact (NO fit on validation)
    """
    preprocessor = build_preprocessor(config, numeric_features, categorical_features)
    artifact = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )
    X_train_proc = transform_data(artifact, X_train)
    X_val_proc = transform_data(artifact, X_val)
    return X_train_proc, X_val_proc, artifact
