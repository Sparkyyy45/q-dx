"""
Model registry and factory for cardiovascular risk prediction models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from src.config import PipelineConfig
from src.models.base import BaseCardioModel
from src.models.catboost_model import CatBoostModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.hybrid_qnn import HybridQNNModel
from src.models.lightgbm_model import LightGBMModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.mlp import MLPModel
from src.models.qsvm_model import QSVMModel
from src.models.random_forest import RandomForestModel
from src.models.serialization import (
    ProductionPipeline,
    load_production_pipeline,
    save_production_pipeline,
)
from src.models.svm_calibrated import CalibratedSVMModel
from src.models.vqc_model import VQCModel
from src.models.xgboost_model import XGBoostModel

CLASSICAL_MODELS: Dict[str, Type[BaseCardioModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "svm_calibrated": CalibratedSVMModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}

QUANTUM_MODELS: Dict[str, Type[BaseCardioModel]] = {
    "vqc": VQCModel,
    "qsvm": QSVMModel,
    "hybrid_qnn": HybridQNNModel,
}

MODEL_REGISTRY: Dict[str, Type[BaseCardioModel]] = {
    **CLASSICAL_MODELS,
    **QUANTUM_MODELS,
}

# Explicitly designated production champion selected strictly from development OOF cross-validation
PRODUCTION_CHAMPION_MODEL: str = "catboost"


def create_model(name: str, config: Optional[PipelineConfig] = None) -> BaseCardioModel:
    """Instantiate a specific model by key."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config=config)


def get_classical_models(config: Optional[PipelineConfig] = None) -> List[BaseCardioModel]:
    """Return instances of all eight classical baseline models."""
    return [cls(config=config) for cls in CLASSICAL_MODELS.values()]


def get_quantum_models(config: Optional[PipelineConfig] = None) -> List[BaseCardioModel]:
    """Return instances of all three quantum models (VQC, QSVM, Hybrid QNN)."""
    return [cls(config=config) for cls in QUANTUM_MODELS.values()]


def get_all_models(
    config: Optional[PipelineConfig] = None, include_quantum: bool = False
) -> List[BaseCardioModel]:
    """
    Return instances of models initialized with config.
    By default returns the 8 classical models for backward compatibility and speed.
    Pass include_quantum=True to include all 11 classical + quantum models.
    """
    if include_quantum:
        return [cls(config=config) for cls in MODEL_REGISTRY.values()]
    return get_classical_models(config=config)


__all__ = [
    "BaseCardioModel",
    "LogisticRegressionModel",
    "CalibratedSVMModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "MLPModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "VQCModel",
    "QSVMModel",
    "HybridQNNModel",
    "create_model",
    "get_all_models",
    "get_classical_models",
    "get_quantum_models",
    "CLASSICAL_MODELS",
    "QUANTUM_MODELS",
    "MODEL_REGISTRY",
    "PRODUCTION_CHAMPION_MODEL",
    "ProductionPipeline",
    "save_production_pipeline",
    "load_production_pipeline",
]
