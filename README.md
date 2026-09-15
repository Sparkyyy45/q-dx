# CardioQ: Hybrid Quantum-Classical Cardiovascular Risk ML Platform
### Scientifically Defensible, Forensic-Remediated Clinical Decision-Support Benchmark

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-99%2F99%20Passed-brightgreen.svg)](run_tests.py)
[![Architecture: Dual-Track Benchmark](https://img.shields.io/badge/Architecture-Dual--Track%20Benchmark-8b5cf6.svg)](architecture.md)
[![Certification: Hackathon Only](https://img.shields.io/badge/Certification-Technically%20Stable%20(Hackathon%20Only)-orange.svg)](README.md)

**CardioQ** is an open, auditable, and leakage-safe machine learning and quantum computing research prototype for **Cardiovascular Disease (CVD)** risk estimation. The platform implements rigorous methodological standards across classical gradient boosted decision trees, regularized linear models, deep neural networks, and statevector quantum architectures (**VQC**, **QSVM**, **Hybrid QNN**).

> **CRITICAL METHODOLOGICAL, CLINICAL & EPIDEMIOLOGICAL SPECIFICATION**:
> - **Platform Certification**: **Technically stable for internal hackathon demonstration; clinical deployment is not claimed.**
> - **Target Endpoint**: Models are trained and evaluated on **cross-sectional prevalent cardiovascular disease** (a binary clinical diagnosis at the time of examination). The platform does **not** predict 10-year prospective incident coronary heart disease or future event time.
> - **Operational Thresholds**: Decision thresholds ($\tau^*$) are locked **strictly on development out-of-fold (OOF) cross-validation predictions** using Youden's $J$ statistic. Client requests **cannot override or alter** the operational threshold (enforced via HTTP 400 `CLIENT_THRESHOLD_PROHIBITED`).
> - **Holdout Test Isolation**: The 13,741-sample holdout test partition is held out untouched upfront; test labels are never accessed during feature engineering, preprocessing, model tuning, or threshold selection.
> - **Framingham Cohort Framing**: The Framingham Heart Study ($N=4,240$) is framed strictly as an **Independent Out-of-Distribution (OOD) Transportability Stress Test** to investigate domain shift and endpoint mismatch, not as equivalent-target external validation.
> - **Zero Synthetic Data Fabrication**: The API never hallucinates patient biomarkers. All required physiological and clinical fields must be provided by the client, or HTTP 400 is returned.
> - **Medical Disclaimer**: Model outputs represent estimated statistical probabilities for research and algorithmic benchmarking; they do **not** constitute a medical diagnosis.

---

## 1. Architectural Highlights & Forensic Remediation

1. **Dual-Track Benchmarking (Parity vs Utility)**:
   - **Track A (Clinical Utility Benchmark)**: 8 classical models trained on the full available development cohort ($N_{\text{dev}} = 54,961$) and evaluated on the untouched 13,741 holdout test set with 1,000-iteration bootstrap 95% confidence intervals.
   - **Track B (Algorithmic Parity Benchmark)**: Paired classical and quantum architectures evaluated on an identical deterministic stratified sample manifest ($N = 1,000$, SHA-256: `af67bbd6...`) to scientifically evaluate sample efficiency and quantum representations under an identical $N=1,000$ training-data budget.

2. **Elimination of Silent Quantum Feature Loss & In-Sample Calibration**:
   - Replaced silent `[:, :4]` feature slicing with supervised linear projection (`nn.Linear(22, 4)`) in VQC/QNN and `StandardScaler -> PCA(4)` in QSVM, ensuring all 22 clinical variables inform quantum states.
   - Replaced in-sample calibration with strict 3-fold internal cross-validation out-of-fold Platt calibration.

3. **Leakage-Safe Clinical Preprocessing & Feature Engineering**:
   - Pre-split data sanitation: drops duplicates, eliminates physiologically inverted blood pressure records ($ap\_hi \le ap\_lo$), and filters extreme non-biological BMI outliers ($< 10$ or $> 70$).
   - Vectorized row-independent age conversion: patient age standardization is strictly batch-independent (`np.where(age > 120.0, age / 365.25, age)`).
   - Strict fold isolation: `IQRClipper`, scalers, and imputers are fitted exclusively on training fold partitions.

4. **Production Serving Parity & Hardened REST API (`src/api/server.py`)**:
   - Zero synthetic fallback: if a model artifact is missing from `artifacts/models/`, the API returns **HTTP 503** `MODEL_NOT_LOADED`. It never synthesizes dummy patients or trains on startup.
   - Strict physiological input validation: enforces biological bounds and rejects inverted blood pressures ($ap\_hi \le ap\_lo$) with **HTTP 400**.
   - Client threshold rejection: any client request attempting to supply a custom `"threshold"` is rejected with **HTTP 400** `CLIENT_THRESHOLD_PROHIBITED`.
   - Standalone reconstructible model artifacts: modular bundles (`preprocessing.joblib`, `model.joblib`/`weights.pt`, `threshold.json`, `schema.json`, `metadata.json`).

5. **Mathematical Local Explainability (`POST /api/explain`)**:
   - Tree models: Genuine local **TreeSHAP** attributions via `shap.TreeExplainer` with verified additivity (`base_value + sum(shap) == model_output`).
   - Linear models: Exact logit contributions ($\beta_j \cdot z_j$).
   - Quantum models: Exact PyTorch autograd input Jacobian ($\frac{\partial \langle Z \rangle}{\partial x_j}$) verified against numerical finite differences.

---

## 2. Empirical Benchmark Results

### Track A: Clinical Utility Benchmark (Full Cohort $N=56,000$, Holdout $N=13,741$)
*Evaluated at prospective locked thresholds $\tau^*$ with 1,000-iteration bootstrap 95% CIs:*

| Model | Architecture | Locked $\tau^*$ | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity @ $\tau^*$ | Specificity @ $\tau^*$ | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost** | Ordered Trees (Categorical Bypass) | 0.4836 | **0.8025** [0.7958, 0.8094] | 0.7839 [0.7725, 0.7950] | **73.48%** | 0.7019 | 0.7670 | **0.1798** |
| **LightGBM** | Leaf-Wise Gradient Boosting | 0.5053 | **0.8025** [0.7954, 0.8096] | **0.7852** [0.7740, 0.7966] | 73.41% | 0.6847 | 0.7825 | 0.1801 |
| **Random Forest** | Bagged Decision Trees | 0.5186 | 0.8019 [0.7946, 0.8090] | 0.7802 [0.7684, 0.7919] | 73.24% | 0.6631 | **0.8003** | 0.1802 |
| **Gradient Boosting** | HistGradientBoosting (scikit-learn) | 0.4857 | 0.8019 [0.7950, 0.8090] | 0.7829 [0.7716, 0.7937] | 73.28% | 0.7022 | 0.7627 | 0.1802 |
| **XGBoost** | Regularized Tree Boosting | 0.4910 | 0.8014 [0.7944, 0.8084] | 0.7801 [0.7683, 0.7918] | 73.45% | 0.6996 | 0.7688 | 0.1804 |
| **Multilayer Perceptron** | Deep Neural Network (64, 32) | 0.5090 | 0.7996 [0.7927, 0.8068] | 0.7818 [0.7707, 0.7931] | 73.34% | 0.6893 | 0.7765 | 0.1813 |
| **Logistic Regression** | L2 Linear Model | 0.4620 | 0.7958 [0.7885, 0.8037] | 0.7758 [0.7642, 0.7869] | 72.91% | **0.7075** | 0.7503 | 0.1838 |
| **Calibrated SVM** | Linear Margin (Platt Calibrated) | 0.4563 | 0.7956 [0.7883, 0.8034] | 0.7751 [0.7634, 0.7861] | 72.88% | 0.7069 | 0.7502 | 0.1838 |

---

### Track B: Algorithmic Parity Benchmark ($N=1,000$ Manifest, Holdout $N=13,741$)
*Paired classical vs quantum models trained on the identical 1,000-sample budget and evaluated at locked $\tau_B^*$:*

| Model | Architecture | Locked $\tau_B^*$ | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity | Specificity | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Classical | 0.4826 | **0.7885** [0.7810, 0.7963] | **0.7716** [0.7596, 0.7826] | **72.53%** | 0.6751 | 0.7744 | **0.1872** |
| **Calibrated SVM** | Classical | 0.4646 | **0.7879** [0.7804, 0.7958] | 0.7711 [0.7592, 0.7827] | 72.35% | **0.7088** | 0.7378 | 0.1879 |
| **CatBoost** | Classical | 0.5178 | 0.7834 [0.7759, 0.7912] | 0.7650 [0.7532, 0.7757] | 72.00% | 0.6866 | 0.7528 | 0.1901 |
| **Random Forest** | Classical | 0.4880 | 0.7810 [0.7732, 0.7881] | 0.7632 [0.7515, 0.7744] | 71.68% | 0.6965 | 0.7368 | 0.1902 |
| **Multilayer Perceptron** | Classical | 0.5189 | 0.7748 [0.7675, 0.7826] | 0.7587 [0.7470, 0.7703] | 70.79% | 0.6765 | 0.7387 | 0.1963 |
| **Gradient Boosting** | Classical | 0.5486 | 0.7602 [0.7521, 0.7683] | 0.7430 [0.7313, 0.7539] | 70.07% | 0.6434 | 0.7570 | 0.2050 |
| **XGBoost** | Classical | 0.4497 | 0.7589 [0.7504, 0.7668] | 0.7421 [0.7299, 0.7539] | 69.41% | **0.7131** | 0.6756 | 0.2091 |
| **Hybrid Quantum NN** | **Quantum** | 0.4480 | **0.7519** [0.7440, 0.7598] | **0.7423** [0.7301, 0.7550] | 70.18% | 0.6965 | 0.7070 | 0.2017 |
| **LightGBM** | Classical | 0.4226 | 0.7468 [0.7384, 0.7546] | 0.7318 [0.7199, 0.7432] | 68.14% | 0.7121 | 0.6513 | 0.2235 |
| **Variational Quantum (VQC)**| **Quantum** | 0.5983 | **0.7350** [0.7269, 0.7438] | 0.7108 [0.6984, 0.7233] | 69.99% | 0.5549 | **0.8421** | 0.2059 |
| **Quantum SVM (QSVM)** | **Quantum** | 0.5707 | **0.7113** [0.7019, 0.7196] | 0.6926 [0.6789, 0.7054] | 66.54% | 0.5568 | 0.7718 | 0.2171 |

---

## 3. Getting Started & Running the Platform

### Environment Setup
```bash
conda activate ml
```

### Launch Interactive Web Dashboard & REST API
```bash
python app.py --port 8080
```
Open **`http://127.0.0.1:8080/`** to access the clinician interface:
- Multi-parametric physiological inputs (Age, Systolic/Diastolic BP, Cholesterol, Glucose, BMI, Smoking).
- Head-to-head model comparison across all 11 architectures.
- Transparent display of model's locked threshold and genuine TreeSHAP/Linear/Quantum attributions.

#### REST API Endpoints
- **`GET /health`**: Server status, registered models, and loaded pipelines.
- **`GET /api/models`**: Full catalog of available classical and quantum architectures.
- **`POST /api/predict`**: Real-time inference using the model's locked threshold.
  ```json
  {
    "model": "catboost",
    "patient": {
      "age_years": 56.0,
      "gender": 2,
      "height": 172.0,
      "weight": 78.0,
      "ap_hi": 142.0,
      "ap_lo": 90.0,
      "cholesterol": 2,
      "gluc": 1,
      "smoke": 1,
      "alco": 0,
      "active": 1
    }
  }
  ```
  *Note: Client-submitted `"threshold"` fields are strictly prohibited and return HTTP 400 (`CLIENT_THRESHOLD_PROHIBITED`). Missing clinical fields return HTTP 400 (`INVALID_INPUT`).*
- **`POST /api/explain`**: Computes genuine mathematical feature attributions (TreeSHAP, Linear logit decomposition, or Quantum input Jacobian) using the identical production pipeline.

---

## 4. Running the Machine Learning Pipeline

### Production Dual-Track Run
```bash
python run_all.py \
  --data "cardio_train_fixed (1).csv" \
  --external-data framingham.csv \
  --quantum
```

### Fast Smoke Test (~60 seconds)
```bash
python run_all.py \
  --data "cardio_train_fixed (1).csv" \
  --quick-run \
  --quantum
```

---

## 5. Automated Regression Test Suite

Execute the complete 98-test test suite:
```bash
python run_tests.py
```

**Result: 98/98 Passed (100% Pass Rate across 12 Modules)**:
- `tests.test_api` (21 tests): Health, Models, Predict, Missing Model (HTTP 503), Input Validation (HTTP 400 on inverted BP, out-of-bounds inputs, or missing clinical fields), Client Threshold Rejection (HTTP 400), Explainability Parity, Stateless Modular Reconstruction.
- `tests.test_audit` (4 tests): Duplicates, physiological range audits, data cleaning.
- `tests.test_dataset` (9 tests): Target binary invariants, row-count sanity ($\ge 50\text{k}$), SHA-256 reproducibility.
- `tests.test_evaluate` (8 tests): Youden's J, threshold locking, bootstrap confidence intervals, cost-weighted thresholding, external label optimization prohibition, development OOF champion selection.
- `tests.test_external_validation` (8 tests): Framingham harmonization, cholesterol bucketing, missing feature error handling, PR-AUC lift.
- `tests.test_features` (8 tests): Clinical feature math, row-independent age conversion, batch invariance, catalog completeness.
- `tests.test_leakage` (4 tests): Preprocessor isolation, feature selection train-isolation, SVM train-isolation.
- `tests.test_models` (10 tests): Uniform interface protocol, CatBoost unscaled categorical passthrough, odds ratios, VIF.
- `tests.test_preprocess` (3 tests): Learned IQR clipping bounds, scaling strategies.
- `tests.test_quantum` (10 tests): Statevector simulation, unitary gates, parameter-shift rule, quantum kernels, VQC, QSVM, Hybrid QNN, autograd feature Jacobian, finite-difference gradient verification.
- `tests.test_reduction` (5 tests): ANOVA F-test, PCA latent representations, column alignment.
- `tests.test_serialization` (8 tests): Classical serialization round-trip, decoupled modular artifact creation, stateless reconstruction without monolithic bundle, training-inference preprocessing parity, TreeSHAP local explainability, TreeSHAP additivity in margin log-odds space, TreeSHAP unavailable explicit handling.

---

## 6. Cryptographic Provenance & Artifacts

All post-remediation outputs are cryptographically signed in [artifacts/remediated_v2/manifest.json](artifacts/remediated_v2/manifest.json):
- Canonical Dataset: `ba6523e1d823640d117c1ce8cbdb356240780f8cc376a42302d97545e0a507d4`
- Track B Manifest: `af67bbd61c02b59b9c7fc93eb2fb4d324384667801a1bc8bc52f96ec16288aa9`
- Immutable Pre-Remediation Baseline: `artifacts/baseline_v1/`
- Serialized Production Models: `artifacts/models/` and `artifacts/remediated_v2/models/`

