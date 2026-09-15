"""
Pipeline configuration and experiment reproducibility settings.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings


@dataclass
class ModelHyperparameters:
    """Hyperparameters for all classical ML models in the pipeline."""
    # Logistic Regression
    lr_c: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"

    # SVM with train-only Platt sigmoid calibration
    svm_c: float = 1.0
    svm_max_iter: int = 2000
    svm_calibration_method: str = "sigmoid"
    svm_calibration_cv: int = 3

    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 12
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 4
    rf_n_jobs: int = -1

    # Gradient Boosting (HistGradientBoosting)
    gb_max_iter: int = 100
    gb_learning_rate: float = 0.08
    gb_max_depth: Optional[int] = 6
    gb_min_samples_leaf: int = 20

    # Multilayer Perceptron
    mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32)
    mlp_activation: str = "relu"
    mlp_alpha: float = 0.001
    mlp_max_iter: int = 100
    mlp_early_stopping: bool = True
    mlp_n_iter_no_change: int = 10

    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # LightGBM
    lgb_n_estimators: int = 200
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8

    # CatBoost
    cat_iterations: int = 200
    cat_depth: int = 6
    cat_learning_rate: float = 0.05
    cat_features: List[str] = field(default_factory=lambda: ["gender", "cholesterol", "gluc"])


FRAMINGHAM_ABSENT_COLUMNS: List[str] = ["height", "weight", "gluc", "alco", "active"]


@dataclass
class PipelineConfig:
    """
    Global configuration object governing the entire cardiovascular ML pipeline.
    All parameters needed for reproducible execution are captured here.
    """
    random_seed: int = 42
    data_path: Optional[str] = field(default_factory=lambda: os.getenv("CVD_DATASET_PATH"))
    target_column: str = "cardio"
    positive_class: int = 1
    negative_class: int = 0
    identifier_columns: List[str] = field(default_factory=lambda: ["id"])
    drop_columns: List[str] = field(default_factory=lambda: ["bp_category_encoded", "bp_category"])
    group_column: Optional[str] = None

    # Validation Strategy
    test_size: float = 0.20
    n_splits: int = 5
    shuffle_folds: bool = True

    # Preprocessing
    scaling_strategy: str = "robust"  # "standard", "robust", "minmax", "none"
    outlier_clipping: bool = True
    outlier_factor: float = 1.5
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"

    # Feature Engineering
    enable_feature_engineering: bool = True

    # Feature Reduction
    reduction_strategy: str = "none"  # "none", "f_classif", "mutual_info", "pca"
    reduction_k: int = 15
    pca_components: int = 10

    # Output paths
    artifacts_dir: str = field(default_factory=lambda: str(settings.ARTIFACTS_DIR))
    external_data_path: str = field(default_factory=lambda: str(settings.EXTERNAL_DATA_PATH))

    # Hyperparameter Tuning
    tune_hyperparameters: bool = True
    tuning_iter: int = 10
    tuning_cv: int = 3

    # Hyperparameters
    model_params: ModelHyperparameters = field(default_factory=ModelHyperparameters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert tuple to list for clean JSON serialization
        if "model_params" in data and "mlp_hidden_layer_sizes" in data["model_params"]:
            data["model_params"]["mlp_hidden_layer_sizes"] = list(
                data["model_params"]["mlp_hidden_layer_sizes"]
            )
        return data

    def save(self, filepath: str | Path) -> None:
        """Persist configuration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Instantiate configuration from dictionary."""
        cfg_data = data.copy()
        if "model_params" in cfg_data and isinstance(cfg_data["model_params"], dict):
            mp_dict = cfg_data["model_params"]
            if "mlp_hidden_layer_sizes" in mp_dict and isinstance(
                mp_dict["mlp_hidden_layer_sizes"], list
            ):
                mp_dict["mlp_hidden_layer_sizes"] = tuple(mp_dict["mlp_hidden_layer_sizes"])
            cfg_data["model_params"] = ModelHyperparameters(**mp_dict)
        return cls(**cfg_data)

    @classmethod
    def load(cls, filepath: str | Path) -> PipelineConfig:
        """Load configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
