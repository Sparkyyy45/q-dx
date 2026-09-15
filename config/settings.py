"""
Centralized Application Settings and Environment Configuration for CardioQ.
Eliminates all hardcoded values, magic strings, IP addresses, ports, and paths.
Supports dynamic override via environment variables or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Base repository root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Enterprise application settings with environment variable priority."""

    def __init__(self):
        # Server Configuration
        self.HOST: str = os.getenv("CARDIOQ_HOST", os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"))
        self.PORT: int = int(os.getenv("CARDIOQ_PORT", os.getenv("PORT", "8080")))
        self.DEBUG: bool = os.getenv("CARDIOQ_DEBUG", "false").lower() in ("true", "1", "yes")
        self.APP_NAME: str = "CardioQ: Hybrid Quantum-Classical CVD Diagnostic Platform"
        self.APP_VERSION: str = "2.0.0"
        self.DEFAULT_MODEL: str = os.getenv("CARDIOQ_DEFAULT_MODEL", "catboost")
        self.CORS_ORIGINS: List[str] = [
            origin.strip()
            for origin in os.getenv("CARDIOQ_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]

        # File System & Persistence Paths
        self.ARTIFACTS_DIR: Path = Path(os.getenv("CARDIOQ_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
        self.DATABASE_PATH: Path = Path(os.getenv("CARDIOQ_DATABASE_PATH", str(BASE_DIR / "artifacts" / "cardioq.db")))
        self.UPLOADS_DIR: Path = Path(os.getenv("CARDIOQ_UPLOADS_DIR", str(BASE_DIR / "artifacts" / "uploads")))
        self.MODELS_DIR: Path = Path(os.getenv("CARDIOQ_MODELS_DIR", str(BASE_DIR / "artifacts" / "models")))
        self.TRAINING_JOBS_DIR: Path = Path(os.getenv("CARDIOQ_TRAINING_JOBS_DIR", str(BASE_DIR / "artifacts" / "training_jobs")))
        self.DATA_PATH: Path = Path(os.getenv("CARDIOQ_DATA_PATH", str(BASE_DIR / "cardio_train_fixed (1).csv")))
        self.EXTERNAL_DATA_PATH: Path = Path(os.getenv("CARDIOQ_EXTERNAL_DATA_PATH", str(BASE_DIR / "framingham.csv")))

        # Clinical & Physiological Validation Bounds
        self.AGE_YEARS_MIN: float = float(os.getenv("CARDIOQ_AGE_MIN", "18.0"))
        self.AGE_YEARS_MAX: float = float(os.getenv("CARDIOQ_AGE_MAX", "120.0"))
        self.AP_HI_MIN: float = float(os.getenv("CARDIOQ_AP_HI_MIN", "50.0"))
        self.AP_HI_MAX: float = float(os.getenv("CARDIOQ_AP_HI_MAX", "300.0"))
        self.AP_LO_MIN: float = float(os.getenv("CARDIOQ_AP_LO_MIN", "30.0"))
        self.AP_LO_MAX: float = float(os.getenv("CARDIOQ_AP_LO_MAX", "200.0"))
        self.HEIGHT_CM_MIN: float = float(os.getenv("CARDIOQ_HEIGHT_MIN", "80.0"))
        self.HEIGHT_CM_MAX: float = float(os.getenv("CARDIOQ_HEIGHT_MAX", "250.0"))
        self.WEIGHT_KG_MIN: float = float(os.getenv("CARDIOQ_WEIGHT_MIN", "20.0"))
        self.WEIGHT_KG_MAX: float = float(os.getenv("CARDIOQ_WEIGHT_MAX", "350.0"))
        self.BMI_MIN: float = float(os.getenv("CARDIOQ_BMI_MIN", "10.0"))
        self.BMI_MAX: float = float(os.getenv("CARDIOQ_BMI_MAX", "70.0"))

        # Upload & Ingestion Constraints
        self.MAX_UPLOAD_MB: int = int(os.getenv("CARDIOQ_MAX_UPLOAD_MB", "15"))
        self.MAX_UPLOAD_BYTES: int = int(os.getenv("CARDIOQ_MAX_UPLOAD_BYTES", str(self.MAX_UPLOAD_MB * 1024 * 1024)))
        self.MAX_CSV_PREVIEW_ROWS: int = int(os.getenv("CARDIOQ_MAX_PREVIEW_ROWS", "8"))

        # Quantum Circuit Parameters
        self.DEFAULT_N_QUBITS: int = int(os.getenv("CARDIOQ_N_QUBITS", "4"))
        self.DEFAULT_N_LAYERS: int = int(os.getenv("CARDIOQ_N_LAYERS", "2"))
        self.QISKIT_SIMULATOR_SHOTS: int = int(os.getenv("CARDIOQ_QISKIT_SHOTS", "1024"))
        self.IBM_QUANTUM_TOKEN: Optional[str] = os.getenv("IBM_QUANTUM_TOKEN")
        self.IBM_QUANTUM_INSTANCE: str = os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
        self.TARGET_QUANTUM_BACKEND: str = os.getenv("CARDIOQ_QUANTUM_BACKEND", "ibm_superconducting")

        # National Health Stack (ABDM / MoHFW)
        self.ORGANIZATION_NAME: str = os.getenv("CARDIOQ_ORGANIZATION", "National Non-Communicable Disease Registry (MoHFW/ICMR)")
        self.HEALTH_FACILITY_ID: str = os.getenv("CARDIOQ_FACILITY_ID", "IN-DL-AIIMS-001")
        self.DEFAULT_LANGUAGE: str = os.getenv("CARDIOQ_DEFAULT_LANG", "en")


# Aliases for backward compatibility and test access
CardioQSettings = Settings
AppSettings = Settings

# Global singleton instance
settings = Settings()

# Ensure required persistent directories exist
for d in [settings.ARTIFACTS_DIR, settings.UPLOADS_DIR, settings.MODELS_DIR, settings.TRAINING_JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

