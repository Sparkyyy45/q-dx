# CardioQ: Forensic Remediation & Project Progress Summary

**Project Name**: CardioQ — Hybrid Quantum-Classical Cardiovascular Risk ML Platform  
**Repository Path**: `/Users/dhruvmakadiya/Documents/heartdisease`  
**Current Date/Time**: 2026-09-07T18:08:00+05:30  
**Environment**: Python 3.12, PyTorch 2.13, Scikit-Learn 1.4, CatBoost, LightGBM, XGBoost  
**Regression Test Status**: **99 / 99 Passed (100% Pass Rate across 12 Modules in ~18 seconds)**  
**Platform Certification**: **Technically stable for internal hackathon demonstration; clinical deployment is not claimed.**

---

## 1. Executive Summary

Over a series of rigorous forensic remediation cycles, this codebase has been transformed from an experimental academic prototype into an **auditable, leakage-safe, and scientifically defensible research platform**.

### Core Non-Negotiable Invariants Enforced:
1. **Zero Data Fabrication**: Missing clinical features or patient biomarkers are **never** synthesized, imputed with zeros, or filled with dummy constants.
2. **Zero Evaluation Peeking**: The 13,741-sample holdout test partition is isolated upfront. Operational decision cutoffs ($\tau^*$) and production model selection are derived **strictly from development out-of-fold (OOF) cross-validation**.
3. **No Hidden Model Training in API**: The serving API (`src/api/server.py`, `app.py`) **never** trains models on startup. If an artifact is missing, it fails loudly with HTTP 503 (`MODEL_NOT_LOADED`).
4. **Client Threshold Overrides Prohibited**: Client API payloads attempting to submit custom `"threshold"` values are rejected with HTTP 400 (`CLIENT_THRESHOLD_PROHIBITED`).
5. **Decoupled Modular Artifacts**: Production models are serialized into independent components (`preprocessing.joblib`, `model.joblib`/`weights.pt`, `threshold.json`, `schema.json`, `metadata.json`) allowing stateless pipeline reconstruction without monolithic bundle dependency.
6. **Genuine Mathematical Explainability**: Local explainability (`POST /api/explain`) computes genuine **TreeSHAP** attributions (with verified additivity in margin log-odds space for tree models), exact linear logit decomposition ($\beta_j \cdot z_j$), and exact PyTorch autograd input feature Jacobians ($\partial \langle Z \rangle / \partial x_j$) for quantum circuits.

---

## 2. Key Forensic Remediations & Scientific Audit

### P0-1 — Resolving the Framingham Provenance & State A / State B Contradiction
* **Cohort Design Divergence**:
  * Primary Training Cohort (`cardio_train`): Cross-sectional **prevalent** cardiovascular disease diagnosis at the time of examination ($49.47\%$ prevalence).
  * External Cohort (`framingham.csv`): 10-year prospective **incident** coronary heart disease (`TenYearCHD`, $14.98\%$ prevalence).
* **Attrition Trace**:
  * Raw Framingham Cohort: **4,240 records** across 16 columns.
  * Missing Values: `glucose`: 388, `education`: 105, `BPMeds`: 53, `totChol`: 50, `cigsPerDay`: 29, `BMI`: 19, `heartRate`: 1.
  * Pipeline Harmonization: The pipeline maps 8 clinical features (`age`, `gender`, `ap_hi`, `ap_lo`, `bmi`, `smoke`, `cholesterol`) and `target`. Dropping incomplete cases on required features (`totChol` $N=50$, `BMI` $N=19$, intersection $N=1$) leaves **4,172 evaluated cases** (or 3,978 in classic risk factor attrition where non-glucose risk factors are filtered).
* **Resolving the Evaluation Contradiction**:
  * **State A (Full 22-Feature Production Model — Default)**:
    > **"Full-feature production model cannot be evaluated on Framingham because required predictors are unavailable."**  
    The full production pipeline requires 22 predictors (including `gluc`, `alco`, `active`, `height`, `weight`, and derived interaction terms). Because Framingham lacks these features, `run_external_validation()` raises `MissingExternalFeatureError`. In canonical production artifacts, `external_validation` is recorded as `null`. Missing predictors are never fabricated.
  * **State B (Reduced-Feature Transportability Benchmark — CLI `--drop-absent`)**:
    > **"Framingham was evaluated as an OOD transportability stress test. Because the cohort has a different endpoint and lacks required predictors, the external analysis uses a separate, explicitly defined reduced-feature configuration and is not equivalent-target external validation."**  
    When trained on the 7 common features without absent columns, models achieve $\approx 0.6625 - 0.6684$ ROC-AUC (PR-AUC lift of $1.6\times - 1.7\times$ over random chance).
  * **Explicit Non-Claim**: The full production CatBoost model is **not** claimed to be externally validated on Framingham.

---

### P0-2 — CatBoost Champion Selection Verification
* **Audit Question**: Was CatBoost designated champion based on development OOF or did holdout test performance influence the choice?
* **Empirical Code Proof**:
  In `run_all.py`, the execution sequence strictly conforms to:
  $$\text{development/OOF} \longrightarrow \text{model selection} \longrightarrow \text{CatBoost chosen} \longrightarrow \text{threshold lock} \longrightarrow \text{holdout evaluation}$$
* **OOF Selection Metrics (5-Fold CV on $N_{\text{dev}}=54,961$)**:
  1. **ROC-AUC**: **CatBoost = 0.8015 (Rank #1)**; LightGBM = 0.8008; XGBoost = 0.8006; HistGB = 0.8004.
  2. **PR-AUC**: **CatBoost = 0.7832 (Rank #1)**; XGBoost = 0.7830; LightGBM = 0.7821.
  3. **Brier Score**: **CatBoost = 0.1804 (Rank #1, lowest calibration error)**; LightGBM = 0.1807.
  4. **Categorical Handling**: Native ordered target statistics on ordinal survey categories (`gender`, `cholesterol`, `gluc`) avoiding scaling distortions.
* **Code Implementation**:
  `OutOfFoldStore.select_champion_model()` in `src/evaluate.py` programmatically selects the champion at Step 4.7 before Step 5 holdout evaluation is executed.
* **Holdout Set Isolation**:
  The 13,741 holdout test set was evaluated purely post-hoc. CatBoost and LightGBM tied at 0.8025 ROC-AUC on holdout, confirming CatBoost was chosen because it won on development OOF, not because it won on holdout.

---

### P0-3 — TreeSHAP Output Space & Mathematical Additivity
* **Gradient Boosted Decision Trees (CatBoost, XGBoost, LightGBM)**:
  Empirically verified on serialized production pipelines:
  $$\text{base} + \sum_{i=1}^M \phi_i = f(\mathbf{x}) = \text{logit}(P(Y=1)) = \ln\left(\frac{P(Y=1)}{1 - P(Y=1)}\right)$$
  * Base value: $-0.01418$
  * Sum of SHAP attributions: $+0.22270$
  * Sum: $+0.20851396222476412$
  * Model margin (`RawFormulaVal`): $+0.20851396222476430$
  * $|\text{Base} + \sum \phi_i - \text{margin}| = \mathbf{1.66 \times 10^{-16}}$
  * Predicted Probability: $P(Y=1) = \sigma(f(\mathbf{x})) = \mathbf{0.5519404376}$
  * Difference between $\text{base} + \sum \phi_i$ and $\text{logit}(P(Y=1))$: $\mathbf{2.78 \times 10^{-17}}$ (machine precision).
* **Random Forest**:
  $$\text{base} + \sum_{i=1}^M \phi_i = f(\mathbf{x}) = P(Y=1) \quad (\text{Probability Space, error } 1.11 \times 10^{-16})$$

---

### P0-4 — Removal of Overclaims
* Replaced: `"intrinsic biological and data noise floor"`  
  $\longrightarrow$ **`"diminishing returns within the evaluated feature/model space."`**
* Replaced: `"quantum models act as regularized approximators that resist small-sample over-splitting"`  
  $\longrightarrow$ **`"quantum models achieved competitive but lower discrimination under the constrained N=1,000 benchmark."`**

---

### P0-5 — Platform Certification Notice
Updated across `README.md`, `architecture.md`, `limitations.md`, and `walkthrough.md`:
> **"Technically stable for internal hackathon demonstration; clinical deployment is not claimed."**

---

### P0-6 — REST API Hardening & Error Handling (`src/api/server.py`)
* **Elimination of Startup Fallback**: Removed legacy 5-row toy synthetic patient DataFrame. If model is absent from `artifacts/models/`, API responds with **HTTP 503** `MODEL_NOT_LOADED`.
* **Prohibition of Client Thresholds**: Any client payload attempting to provide a `"threshold"` field (at payload root or inside patient object) is rejected with **HTTP 400** `CLIENT_THRESHOLD_PROHIBITED`.
* **Strict Hemodynamic & Physiological Validation**:
  * Inverted blood pressure ($ap\_hi \le ap\_lo$) returns **HTTP 400**.
  * Out-of-bounds parameters (Age $<18$ or $>120$, Height $<80$ or $>250$ cm, Weight $<20$ or $>350$ kg, Systolic $<50$ or $>300$ mmHg, Diastolic $<30$ or $>200$ mmHg) return **HTTP 400**.
  * Missing required clinical fields return **HTTP 400** `INVALID_INPUT` detailing missing field names.
* **Stateless Modular Reconstruction**: API primarily loads modular components (`reconstruct_production_pipeline`), eliminating single-point failure of monolithic pickles.

---

### P0-7 — Quantum Pipeline Remediation
* **Elimination of Silent Feature Slicing**:
  * Replaced `[:, :4]` silent slicing with supervised linear projection (`nn.Linear(22, 4)`) in VQC and Hybrid QNN, and `PCA(4)` in QSVM. All 22 clinical variables inform quantum states.
* **Elimination of In-Sample Calibration Leakage**:
  * Replaced training-readout calibration with 3-fold internal cross-validation out-of-fold Platt scaling.
* **Quantum Explainability**:
  * Implemented exact PyTorch autograd input feature Jacobian ($\partial \langle Z \rangle / \partial x_j$) verified by finite differences ($|\nabla_{\text{autograd}} - \nabla_{\text{fd}}| < 1.5 \times 10^{-5}$).

---

## 3. Empirical Performance Leaderboards

### Track A: Clinical Utility Benchmark (Full Cohort $N=56,000$, Holdout $N=13,741$)
*Evaluated at prospective locked thresholds $\tau^*$ with 1,000-iteration bootstrap 95% CIs:*

| Model | Architecture | Locked $\tau^*$ | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity @ $\tau^*$ | Specificity @ $\tau^*$ | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost** (Champion) | Ordered Trees (Categorical Bypass) | 0.4836 | **0.8025** [0.7958, 0.8094] | 0.7839 [0.7725, 0.7950] | **73.48%** | 0.7019 | 0.7670 | **0.1798** |
| **LightGBM** | Leaf-Wise Gradient Boosting | 0.5053 | **0.8025** [0.7954, 0.8096] | **0.7852** [0.7740, 0.7966] | 73.41% | 0.6847 | 0.7825 | 0.1801 |
| **Random Forest** | Bagged Decision Trees | 0.5186 | 0.8019 [0.7946, 0.8090] | 0.7802 [0.7684, 0.7919] | 73.24% | 0.6631 | **0.8003** | 0.1802 |
| **Gradient Boosting** | HistGradientBoosting (sklearn) | 0.4857 | 0.8019 [0.7950, 0.8090] | 0.7829 [0.7716, 0.7937] | 73.28% | 0.7022 | 0.7627 | 0.1802 |
| **XGBoost** | Regularized Tree Boosting | 0.4910 | 0.8014 [0.7944, 0.8084] | 0.7801 [0.7683, 0.7918] | 73.45% | 0.6996 | 0.7688 | 0.1804 |
| **Multilayer Perceptron** | Deep Neural Network (64, 32) | 0.5090 | 0.7996 [0.7927, 0.8068] | 0.7818 [0.7707, 0.7931] | 73.34% | 0.6893 | 0.7765 | 0.1813 |
| **Logistic Regression** | L2 Linear Model | 0.4620 | 0.7958 [0.7885, 0.8037] | 0.7758 [0.7642, 0.7869] | 72.91% | **0.7075** | 0.7503 | 0.1838 |
| **Calibrated SVM** | Linear Margin (Platt Calibrated) | 0.4563 | 0.7956 [0.7883, 0.8034] | 0.7751 [0.7634, 0.7861] | 72.88% | 0.7069 | 0.7502 | 0.1838 |

---

### Track B: Algorithmic Parity Benchmark ($N=1,000$ Manifest, Holdout $N=13,741$)
*Paired classical vs quantum models trained on the identical 1,000-sample budget and evaluated at locked $\tau_B^*$:*

| Model | Family | Locked $\tau_B^*$ | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity | Specificity | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Classical Linear | 0.4826 | **0.7885** [0.7810, 0.7963] | **0.7716** [0.7596, 0.7826] | **72.53%** | 0.6751 | 0.7744 | **0.1872** |
| **Calibrated SVM** | Classical Margin | 0.4646 | **0.7879** [0.7804, 0.7958] | 0.7711 [0.7592, 0.7827] | 72.35% | **0.7088** | 0.7378 | 0.1879 |
| **CatBoost** | Classical Trees | 0.5178 | 0.7834 [0.7759, 0.7912] | 0.7650 [0.7532, 0.7757] | 72.00% | 0.6866 | 0.7528 | 0.1901 |
| **Random Forest** | Classical Trees | 0.4880 | 0.7810 [0.7732, 0.7881] | 0.7632 [0.7515, 0.7744] | 71.68% | 0.6965 | 0.7368 | 0.1902 |
| **Multilayer Perceptron** | Classical Neural | 0.5189 | 0.7748 [0.7675, 0.7826] | 0.7587 [0.7470, 0.7703] | 70.79% | 0.6765 | 0.7387 | 0.1963 |
| **Gradient Boosting** | Classical Trees | 0.5486 | 0.7602 [0.7521, 0.7683] | 0.7430 [0.7313, 0.7539] | 70.07% | 0.6434 | 0.7570 | 0.2050 |
| **XGBoost** | Classical Trees | 0.4497 | 0.7589 [0.7504, 0.7668] | 0.7421 [0.7299, 0.7539] | 69.41% | **0.7131** | 0.6756 | 0.2091 |
| **Hybrid Quantum NN** | **Quantum Hybrid** | 0.4480 | **0.7519** [0.7440, 0.7598] | **0.7423** [0.7301, 0.7550] | 70.18% | 0.6965 | 0.7070 | 0.2017 |
| **LightGBM** | Classical Trees | 0.4226 | 0.7468 [0.7384, 0.7546] | 0.7318 [0.7199, 0.7432] | 68.14% | 0.7121 | 0.6513 | 0.2235 |
| **Variational Quantum (VQC)**| **Quantum Variational** | 0.5983 | **0.7350** [0.7269, 0.7438] | 0.7108 [0.6984, 0.7233] | 69.99% | 0.5549 | **0.8421** | 0.2059 |
| **Quantum SVM (QSVM)** | **Quantum Kernel** | 0.5707 | **0.7113** [0.7019, 0.7196] | 0.6926 [0.6789, 0.7054] | 66.54% | 0.5568 | 0.7718 | 0.2171 |

---

## 4. Automated Regression Test Suite

Execution command:
```bash
python run_tests.py
```

**Result: 98 / 98 Passed across 12 Modules (16.90s)**:
* `tests.test_api` (21 tests): Server health, model catalog, input range enforcement, BP inversion rejection, threshold override prohibition, stateless modular reconstruction, TreeSHAP explainability endpoint.
* `tests.test_audit` (4 tests): Duplicates, physiological range checks, data cleaning.
* `tests.test_dataset` (9 tests): Target binary invariants, row-count sanity ($\ge 50\text{k}$), SHA-256 reproducibility.
* `tests.test_evaluate` (8 tests): Youden's J, threshold locking, bootstrap confidence intervals, cost-weighted thresholding, external label optimization prohibition, **development OOF champion selection**.
* `tests.test_external_validation` (8 tests): Framingham harmonization, cholesterol bucketing, missing feature error handling, PR-AUC lift.
* `tests.test_features` (8 tests): Clinical feature math, row-independent age conversion, batch invariance, catalog completeness.
* `tests.test_leakage` (4 tests): Preprocessor isolation, feature selection train-isolation, SVM train-isolation.
* `tests.test_models` (10 tests): Uniform interface protocol, CatBoost unscaled categorical passthrough, odds ratios, VIF.
* `tests.test_preprocess` (3 tests): Learned IQR clipping bounds, scaling strategies.
* `tests.test_quantum` (10 tests): Statevector simulation, unitary gates, parameter-shift rule, quantum kernels, VQC, QSVM, Hybrid QNN, autograd feature Jacobian, finite-difference gradient verification.
* `tests.test_reduction` (5 tests): ANOVA F-test, PCA latent representations, column alignment.
* `tests.test_serialization` (8 tests): Classical serialization round-trip, decoupled modular artifact creation, stateless reconstruction without monolithic bundle, training-inference preprocessing parity, TreeSHAP local explainability, TreeSHAP additivity in margin log-odds space, TreeSHAP unavailable explicit handling.

---

## 5. Repository File Map

| Path | Purpose |
|---|---|
| `run_all.py` | Master pipeline runner executing Track A, Track B, OOF threshold locking, champion selection, and holdout evaluation. |
| `run_tests.py` | Automated test runner executing the 98-test test suite across all 12 modules. |
| `app.py` | Web UI application entry point for clinician dashboard and REST API. |
| `src/api/server.py` | Hardened zero-dependency HTTP server (`/health`, `/api/models`, `/api/predict`, `/api/explain`). |
| `src/models/serialization.py` | ProductionPipeline serialization engine creating decoupled modular bundles. |
| `src/models/` | Classical & Quantum models implementing uniform `BaseCardioModel` protocol. |
| `src/evaluate.py` | Clinical evaluation metrics, `OutOfFoldStore`, threshold locking, and champion selection. |
| `src/external_validation.py` | Framingham harmonization, cholesterol discretization, and OOD stress test module. |
| `src/clean.py` | Pre-split data sanitation (duplicate removal, inverted BP drops, extreme BMI filtering). |
| `src/features.py` | Row-independent feature engineering and clinical biomarker transformations. |
| `src/quantum/` | Statevector simulator, parameter-shift rule, quantum circuits, and kernels. |
| `artifacts/models/` | Production model artifacts (CatBoost, LightGBM, XGBoost, etc.) with decoupled components. |
| `artifacts/remediated_v2/` | Immutable audit artifacts, `pipeline_report.md`, `metrics.json`, and `manifest.json`. |
| `README.md` | Primary platform documentation with architecture, leaderboards, and quickstart. |
| `architecture.md` | System design, mathematical formulations, and component diagrams. |
| `limitations.md` | Clinical boundary disclaimers, domain shift analysis, and non-claims. |
| `PROJECT_PROGRESS.md` | **This document**: Comprehensive handoff of progress, audit findings, and verification. |

---

## 6. How to Run for Hackathon Demonstration

### Step 1: Activate Environment
```bash
conda activate ml
```

### Step 2: Verify Test Suite
```bash
python run_tests.py
```

### Step 3: Launch Web Application & REST API
```bash
python app.py --port 8080
```
Open **`http://127.0.0.1:8080/`** to interact with:
* Patient clinical attribute sliders (Age, BP, Cholesterol, Smoking, Glucose, BMI).
* Head-to-head comparison across all 11 models (8 Classical + 3 Quantum).
* Locked operational thresholds and genuine TreeSHAP/Linear/Quantum feature attributions.

### Step 4 (Optional): Execute Fast Dual-Track ML Pipeline
```bash
python run_all.py --data "cardio_train_fixed (1).csv" --external-data framingham.csv --quick-run --quantum
```

---

## 7. Future Post-Hackathon Roadmap

1. **Longitudinal Incident Modeling**: Transition from cross-sectional binary classification to time-to-event survival models (Cox Proportional Hazards, Fine-Gray competing risk) for true multi-year prospective prognosis.
2. **Laboratory Biomarker Integration**: Replace ordinal categorical survey bins (`cholesterol` 1/2/3, `gluc` 1/2/3) with continuous serum lipid subfractions (LDL-C, HDL-C, triglycerides, ApoB, HbA1c).
3. **Physical NISQ Hardware Deployment**: Execute quantum circuits on physical hardware backends (IBM Quantum Heron/Eagle or IonQ) incorporating Zero-Noise Extrapolation (ZNE) and readout error mitigation.
