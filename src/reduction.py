"""
Leakage-safe feature reduction and selection module.
Supports univariate ANOVA F-test, Mutual Information, PCA, and identity pass-through.
Fitted exclusively on training partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


@dataclass
class FeatureReductionArtifact:
    """
    Metadata artifact capturing fitted feature reduction state, selected features,
    and clinical interpretability warnings.
    """
    reducer: Optional[BaseEstimator]
    strategy: str
    input_feature_names: List[str]
    output_feature_names: List[str]
    scores: Optional[Dict[str, float]] = None
    p_values: Optional[Dict[str, float]] = None
    explained_variance_ratio: Optional[List[float]] = None
    is_latent: bool = False
    clinical_notice: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Feature Reducer.
    Supports:
    - 'none': Identity pass-through preserving all features
    - 'f_classif': Univariate ANOVA F-test
    - 'mutual_info': Non-parametric Mutual Information
    - 'pca': Principal Component Analysis (latent components)
    """

    def __init__(
        self,
        strategy: str = "none",
        k: int = 15,
        pca_components: int = 10,
        random_state: int = 42,
    ):
        self.strategy = (strategy or "none").lower()
        self.k = k
        self.pca_components = pca_components
        self.random_state = random_state

        self.fitted_model_: Optional[BaseEstimator] = None
        self.input_feature_names_: List[str] = []
        self.output_feature_names_: List[str] = []
        self.scores_: Optional[Dict[str, float]] = None
        self.p_values_: Optional[Dict[str, float]] = None
        self.explained_variance_ratio_: Optional[List[float]] = None
        self.is_latent_: bool = False
        self.clinical_notice_: str = ""

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> FeatureReducer:
        if isinstance(X, pd.DataFrame):
            self.input_feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.input_feature_names_ = [f"feature_{i}" for i in range(X_arr.shape[1])]

        n_features = X_arr.shape[1]

        if self.strategy == "none":
            self.fitted_model_ = None
            self.output_feature_names_ = list(self.input_feature_names_)
            self.is_latent_ = False
            self.clinical_notice_ = "Preserved original clinical features without reduction."

        elif self.strategy == "f_classif":
            if y is None:
                raise ValueError("Supervised feature selection 'f_classif' requires target labels y.")
            actual_k = min(self.k, n_features)
            selector = SelectKBest(score_func=f_classif, k=actual_k)
            selector.fit(X_arr, y)
            self.fitted_model_ = selector

            support = selector.get_support()
            self.output_feature_names_ = [
                name for name, sel in zip(self.input_feature_names_, support) if sel
            ]
            self.scores_ = {
                name: float(score) for name, score in zip(self.input_feature_names_, selector.scores_)
            }
            if selector.pvalues_ is not None:
                self.p_values_ = {
                    name: float(p) for name, p in zip(self.input_feature_names_, selector.pvalues_)
                }
            self.is_latent_ = False
            self.clinical_notice_ = (
                f"Selected top {actual_k} features using univariate ANOVA F-test."
            )

        elif self.strategy == "mutual_info":
            if y is None:
                raise ValueError("Supervised feature selection 'mutual_info' requires target labels y.")
            actual_k = min(self.k, n_features)

            def mi_func(X_in, y_in):
                return mutual_info_classif(X_in, y_in, random_state=self.random_state)

            selector = SelectKBest(score_func=mi_func, k=actual_k)
            selector.fit(X_arr, y)
            self.fitted_model_ = selector

            support = selector.get_support()
            self.output_feature_names_ = [
                name for name, sel in zip(self.input_feature_names_, support) if sel
            ]
            self.scores_ = {
                name: float(score) for name, score in zip(self.input_feature_names_, selector.scores_)
            }
            self.is_latent_ = False
            self.clinical_notice_ = (
                f"Selected top {actual_k} features using Mutual Information."
            )

        elif self.strategy == "pca":
            actual_n = min(self.pca_components, n_features)
            pca = PCA(n_components=actual_n, random_state=self.random_state)
            pca.fit(X_arr)
            self.fitted_model_ = pca

            self.output_feature_names_ = [f"PC_{i+1}" for i in range(actual_n)]
            self.explained_variance_ratio_ = [float(v) for v in pca.explained_variance_ratio_]
            self.is_latent_ = True
            self.clinical_notice_ = (
                f"Dimensionality reduced using PCA to {actual_n} latent components. "
                "WARNING: Latent variables represent orthogonal mathematical projections and "
                "do NOT correspond 1-to-1 with individual clinical measurements."
            )

        else:
            raise ValueError(
                f"Unknown reduction strategy: '{self.strategy}'. "
                "Choose from ['none', 'f_classif', 'mutual_info', 'pca']."
            )

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.input_feature_names_:
                missing = [c for c in self.input_feature_names_ if c not in X.columns]
                if missing:
                    raise ValueError(
                        f"Input DataFrame is missing required features expected by FeatureReducer: {missing}"
                    )
                X_arr = X[self.input_feature_names_].to_numpy(dtype=np.float64)
            else:
                X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if self.input_feature_names_ and X_arr.shape[1] != len(self.input_feature_names_):
                raise ValueError(
                    f"Input array has {X_arr.shape[1]} features, but FeatureReducer "
                    f"was fitted on {len(self.input_feature_names_)} features."
                )

        if self.strategy == "none" or self.fitted_model_ is None:
            return X_arr

        return self.fitted_model_.transform(X_arr)

    def get_artifact(self) -> FeatureReductionArtifact:
        """Return structured FeatureReductionArtifact."""
        return FeatureReductionArtifact(
            reducer=self.fitted_model_,
            strategy=self.strategy,
            input_feature_names=self.input_feature_names_,
            output_feature_names=self.output_feature_names_,
            scores=self.scores_,
            p_values=self.p_values_,
            explained_variance_ratio=self.explained_variance_ratio_,
            is_latent=self.is_latent_,
            clinical_notice=self.clinical_notice_,
            provenance={
                "strategy": self.strategy,
                "input_features": len(self.input_feature_names_),
                "output_features": len(self.output_feature_names_),
            },
        )


def select_features(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    strategy: str = "none",
    k: int = 15,
    pca_components: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, FeatureReductionArtifact, FeatureReducer]:
    """
    Fit feature reduction strictly on X_train, y_train.
    Returns transformed X_train_red DataFrame, artifact, and fitted reducer.
    """
    reducer = FeatureReducer(
        strategy=strategy,
        k=k,
        pca_components=pca_components,
        random_state=random_state,
    )
    reducer.fit(X_train, y_train)
    artifact = reducer.get_artifact()
    X_train_red = reduce_features(artifact, reducer, X_train)
    return X_train_red, artifact, reducer


def reduce_features(
    artifact: FeatureReductionArtifact,
    reducer: FeatureReducer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform data X using an already-fitted FeatureReducer.
    Preserves index and assigns proper output feature names.
    """
    X_red_arr = reducer.transform(X)
    cols = artifact.output_feature_names
    return pd.DataFrame(X_red_arr, columns=cols, index=X.index)
