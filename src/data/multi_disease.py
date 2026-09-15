"""
Multi-Disease Biomedical Dataset Integration for Early Disease Detection.
Provides canonical high-dimensional cohorts for:
1. Cardiovascular Disease (CardioQ Kaggle Cohort, 70k samples)
2. Longitudinal Ischemic Heart Disease (Framingham Heart Study, 4.2k samples)
3. Early Oncology / Breast Cancer Diagnostic (WDBC, 569 samples, 30 continuous features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer


def ensure_wdbc_dataset(output_path: Optional[Path] = None) -> Path:
    """
    Ensure the Wisconsin Diagnostic Breast Cancer (WDBC) high-dimensional
    early detection dataset exists locally on disk as a CSV.
    """
    dest = output_path or Path("data/wdbc_cancer.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.is_file():
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        # Rename target column to 'diagnosis_malignant' (1 = Malignant, 0 = Benign)
        # Note: in sklearn target=0 is malignant, target=1 is benign; invert for clinical positive
        df["target"] = (df["target"] == 0).astype(int)
        df.rename(columns={"target": "diagnosis_malignant"}, inplace=True)
        df.to_csv(dest, index=False)

    return dest


if __name__ == "__main__":
    p = ensure_wdbc_dataset()
    print(f"WDBC dataset ensured at {p} with shape {pd.read_csv(p).shape}")
