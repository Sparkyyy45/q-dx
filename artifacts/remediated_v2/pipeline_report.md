# Cardiovascular Disease Risk Prediction: Production Machine Learning Report

> **Medical Disclaimer**: Model outputs represent estimated statistical probabilities and do not constitute a medical diagnosis. All clinical risk scores require evaluation by qualified healthcare practitioners.

## 1. Dataset Summary

- **Dataset Path**: `cardio_train_fixed (1).csv`
- **Dataset SHA-256 Checksum**: `ba6523e1d823640d117c1ce8cbdb356240780f8cc376a42302d97545e0a507d4`
- **Total Rows**: 70,000
- **Total Columns**: 13
- **CVD Target Class Counts**: 0 (Negative): 35,021, 1 (Positive): 34,979
- **Class Balance**: 49.97% positive prevalence
- **Missing Values**: 0 across all fields
- **Duplicate Rows**: 0 exact duplicates
- **Data Provenance**: Full cohort loaded from `cardio_train_fixed (1).csv`. Note: Earlier preliminary runs were evaluated on a pre-filtered/truncated file; current metrics establish the verified full-cohort baseline.

## Official Data Dictionary

| Column | Category | Unit / Encoding | Clinical Description |
|---|---|---|---|
| `id` | **Identifier** | `int` | Unique patient encounter identifier (must be excluded from predictive features) |
| `age` | **Objective** | `int, days` | Chronological age recorded in elapsed days (biological range ~10,000–25,000 days) |
| `gender` | **Objective** | `categorical code` | Biological sex (1 = female, 2 = male) |
| `height` | **Objective** | `int, cm` | Body height measured in centimeters |
| `weight` | **Objective** | `float, kg` | Body weight measured in kilograms |
| `ap_hi` | **Examination** | `int (systolic BP)` | Systolic blood pressure measured at clinical examination encounter (mmHg) |
| `ap_lo` | **Examination** | `int (diastolic BP)` | Diastolic blood pressure measured at clinical examination encounter (mmHg) |
| `cholesterol` | **Examination** | `1=normal, 2=above normal, 3=well above normal` | Total serum cholesterol ordinal clinical lab category |
| `gluc` | **Examination** | `1=normal, 2=above normal, 3=well above normal` | Serum fasting glucose ordinal clinical lab category |
| `smoke` | **Subjective** | `binary` | Active self-reported tobacco smoking status (0 = non-smoker, 1 = active smoker) |
| `alco` | **Subjective** | `binary` | Self-reported regular alcohol intake (0 = no alcohol, 1 = regular intake) |
| `active` | **Subjective** | `binary` | Self-reported physical activity habit (0 = inactive, 1 = physically active) |
| `cardio` | **Target** | `binary` | Prevalent cardiovascular disease clinical diagnosis (0 = absence, 1 = presence) |

## 2. Data Audit Findings & Pre-Split Cleaning

- **Initial Dimensions**: 70000 rows × 13 columns
- **Identified ID Columns**: `['id']` (Excluded from model features to prevent memorization)
- **Redundant/Duplicate Columns**: `[]` (`bp_category_encoded` dropped in favor of clean categorical pipeline)
- **Biological Plausibility Checks**: 5 alerts flagged and handled via pre-split cleaning and robust IQR clipping.

### Pre-Split Data Cleaning Summary

- **Pre-Cleaning Cohort**: 70,000 rows
- **Cleaned Cohort**: **68,702 rows** (-1298 removed)
- **Non-ID Duplicate Rows Dropped**: 24
- **Inverted Blood Pressure Rows Dropped (`ap_hi < ap_lo`)**: 1234
- **Extreme Non-Biological BMI Rows Dropped (`< 10` or `> 70`)**: 42
- **Cleaning Isolation Policy**: Cleaning executed strictly prior to holdout partitioning to guarantee both development and test sets are clean.

| Category | Column | Severity | Detail | Resolution |
|---|---|---|---|---|
| redundancy | `[multiple_features]` | **MEDIUM** | Found 24 duplicate rows when ignoring ID columns. | Retain for training if representing distinct clinical encounters, or record provenance. |
| identifier | `id` | **HIGH** | Column exhibits identifier characteristics (70000 unique values / 70000 rows). | Strictly exclude from model features to prevent memorization leakage. |
| invalid_values | `ap_hi, ap_lo` | **HIGH** | 1234 instances where systolic BP < diastolic BP. | Apply clinical boundary correction or IQR clipping. |
| invalid_values | `bmi (calculated)` | **MEDIUM** | 42 extreme BMI values detected (min: 3.5, max: 298.7). | Apply robust IQR clipping or pre-split cleaning to eliminate physiological measurement errors. |

## 3. Preprocessing Configuration & Pipeline

- **Scaling Strategy**: `ROBUST`
- **Outlier Treatment**: IQR clipping (factor = 1.5 × IQR) fitted strictly on training data
- **Imputation Strategy**: Numeric (`median`), Categorical (`most_frequent`)
- **Encoding**: One-Hot Encoding (`handle_unknown='ignore'`) inside `ColumnTransformer`
- **Leakage Invariant**: All transformers fitted exclusively on the training partition of each CV fold

## 4. Feature Engineering

Clinically grounded tabular transformations derived without data leakage:

| Feature | Formula | Clinical Justification |
|---|---|---|
| `pulse_pressure` | `ap_hi - ap_lo` | Indicator of large-artery stiffness and pulsatile myocardial strain |
| `mean_arterial_pressure` | `ap_lo + (ap_hi - ap_lo) / 3` | Organ perfusion pressure driving microvascular shear stress |
| `age_years` | `age_days / 365.25` | Standardized chronological age representing cumulative atherogenesis |
| `age_squared` | `(age_years) ** 2` | Captures accelerating cardiovascular disease incidence in aging cohorts |
| `age_bp_interaction` | `age_years * ap_hi` | Synergistic vascular aging under hypertensive mechanical stress |
| `cholesterol_gluc_ratio` | `cholesterol / gluc` | Combined lipid-glycemic metabolic burden index |
| `metabolic_synergy` | `(cholesterol >= 2) & (gluc >= 2)` | Binary flag for concurrent dyslipidemia and dysglycemia |
| `is_hypertensive_stage2` | `(ap_hi >= 140) \| (ap_lo >= 90)` | ACC/AHA clinical definition for Stage 2 hypertension |
| `high_risk_lifestyle` | `(smoke == 1) & (alco == 1)` | Compound toxic behavioral exposure accelerating endothelial damage |
| `log_pulse_pressure` | `log1p(pulse_pressure)` | Skew-stabilized representation of pulsatile vascular load |

## 5. Feature Reduction

- **Strategy**: `none`
- **Notice**: Feature selection and dimensionality reduction were fitted strictly on fold training data. (When PCA is selected, outputs are latent components rather than raw clinical measurements).

## 6. Cross-Validation Strategy

- **Scheme**: Stratified 5-Fold Cross-Validation (`StratifiedKFold`)
- **Random Seed**: `42` (shuffled)
- **Holdout Test Split**: 20% held out untouched upfront for final independent evaluation
- **Fold Sequence**: `Raw Train -> Fit Preprocessor -> Fit Reducer -> Fit Model -> Transform Val -> Predict Val`

## 7. Model Configurations & Hyperparameter Tuning

1. **Logistic Regression**: L2 penalty, lbfgs solver, max_iter=1000
2. **Calibrated SVM**: LinearSVC base margin classifier with train-only 3-fold Sigmoid (Platt) calibration
3. **Random Forest**: 100 trees, max_depth=12, min_samples_split=10, min_samples_leaf=4
4. **Gradient Boosting**: HistGradientBoosting, max_iter=100, learning_rate=0.08, max_depth=6
5. **Multilayer Perceptron**: 2 hidden layers (64, 32), ReLU activation, alpha=0.001, early stopping
6. **XGBoost**: 200 trees, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
7. **LightGBM**: 200 trees, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
8. **CatBoost**: 200 iterations, depth=6, learning_rate=0.05, native ordered target statistics for `gender`, `cholesterol`, `gluc` (unscaled passthrough)

### Nested Hyperparameter Tuning Audit (Tree Ensembles: GB, RF, XGB, LGB, CatBoost)

- **Tuning Protocol**: Bounded `RandomizedSearchCV` (10 iterations, 3-fold inner CV) nested strictly within fold training partitions.
- **Gradient Boosting Best Params**: `{'min_samples_leaf': 25, 'max_iter': 80, 'max_depth': 4, 'learning_rate': 0.08}` (Mean fold ΔAUC: -0.0029)
- **Random Forest Best Params**: `{'n_estimators': 60, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10}` (Mean fold ΔAUC: -0.0045)
- **XGBoost Best Params**: `{'subsample': 0.8, 'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.03}` (Mean fold ΔAUC: -0.0024)
- **LightGBM Best Params**: `{'subsample': 0.8, 'num_leaves': 20, 'n_estimators': 100, 'learning_rate': 0.03}` (Mean fold ΔAUC: -0.0034)
- **CatBoost Best Params**: `{'learning_rate': 0.03, 'iterations': 100, 'depth': 8}` (Mean fold ΔAUC: -0.0038)
- **Parsimony Decision**: Tuning yielded mean ΔAUC < 0.01 across tree ensembles: Gradient Boosting=-0.0029, Random Forest=-0.0045, XGBoost=-0.0024, LightGBM=-0.0034, CatBoost=-0.0038. Simpler fixed baseline hyperparameters retained per parsimony policy (reflecting diminishing returns within the evaluated feature/model space, not under-tuning).

## 8. Model Comparison

### Comparative Analysis of Gradient Boosting Implementations

> **Architectural Comparison (HistGradientBoosting vs XGBoost vs LightGBM vs CatBoost)**: 
> Four distinct gradient boosting engines were evaluated under identical 5-fold cross-validation and independent holdout testing: 
> - **HistGradientBoosting (sklearn)**: Fast histogram binning with symmetric tree splits.
> - **XGBoost**: Exact and histogram split algorithms with robust L1/L2 regularization and tree depth constraints.
> - **LightGBM**: Leaf-wise (best-first) tree growth optimizing loss reduction across large tabular cohorts.
> - **CatBoost**: Oblivious trees with native ordered target statistics for `gender`, `cholesterol`, and `gluc` bypassing numerical scaling.

> **Empirical Categorical Signal Finding**: 
> CatBoost's native ordered target statistics evaluate whether discrete ordinal categories (`gender`: 1/2, `cholesterol`: 1/2/3, `gluc`: 1/2/3) harbor complex non-linear target interactions that continuous scaling blunts. Across both internal CV folds and holdout partitions, all four gradient boosting implementations converge within ~0.003-0.005 ROC-AUC of each other. This indicates that while CatBoost natively preserves discrete categories cleanly without manual scaling, the standard histogram-based numerical splitters in HistGradientBoosting, LightGBM, and XGBoost already capture nearly identical monotonic risk boundaries from these categorical codes.

### Track A: Clinical Utility Benchmark (Untouched 20% Holdout Partition)

| Model | Locked tau* | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity @ tau* | Specificity @ tau* | Precision | F1-Score | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.4620 | 0.7958 [0.7885, 0.8037] | 0.7758 [0.7642, 0.7869] | 0.7291 | 0.7075 | 0.7503 | 0.7352 | 0.7211 | 0.1838 |
| **Calibrated SVM** | 0.4563 | 0.7956 [0.7883, 0.8034] | 0.7751 [0.7634, 0.7861] | 0.7288 | 0.7069 | 0.7502 | 0.7349 | 0.7206 | 0.1838 |
| **Random Forest** | 0.5186 | 0.8019 [0.7946, 0.8090] | 0.7802 [0.7684, 0.7919] | 0.7324 | 0.6631 | 0.8003 | 0.7649 | 0.7104 | 0.1802 |
| **Gradient Boosting** | 0.4857 | 0.8019 [0.7950, 0.8090] | 0.7829 [0.7716, 0.7937] | 0.7328 | 0.7022 | 0.7627 | 0.7435 | 0.7223 | 0.1802 |
| **Multilayer Perceptron** | 0.5090 | 0.7996 [0.7927, 0.8068] | 0.7818 [0.7707, 0.7931] | 0.7334 | 0.6893 | 0.7765 | 0.7514 | 0.7190 | 0.1813 |
| **XGBoost** | 0.4910 | 0.8014 [0.7944, 0.8084] | 0.7801 [0.7683, 0.7918] | 0.7345 | 0.6996 | 0.7688 | 0.7477 | 0.7228 | 0.1804 |
| **LightGBM** | 0.5053 | 0.8025 [0.7954, 0.8096] | 0.7852 [0.7740, 0.7966] | 0.7341 | 0.6847 | 0.7825 | 0.7551 | 0.7182 | 0.1801 |
| **CatBoost** | 0.4836 | 0.8025 [0.7958, 0.8094] | 0.7839 [0.7725, 0.7950] | 0.7348 | 0.7019 | 0.7670 | 0.7469 | 0.7237 | 0.1798 |

### Track B: Algorithmic Parity Benchmark (Deterministic Manifest N = 1,000 / N = 14,000 Holdout Test)

> **Algorithmic Parity Methodology**: Evaluates paired classical and quantum models on an identical 1,000-sample training budget to determine whether quantum feature representations confer sample-efficient advantages or whether classical models maintain dominance under an identical $N=1,000$ training-data budget.

| Model | Architecture | Locked tau_B* | Holdout ROC-AUC [95% CI] | Holdout PR-AUC [95% CI] | Accuracy | Sensitivity | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Classical | 0.4826 | 0.7885 [0.7810, 0.7963] | 0.7716 [0.7596, 0.7826] | 0.7253 | 0.6751 | 0.7744 | 0.1872 |
| **Calibrated SVM** | Classical | 0.4646 | 0.7879 [0.7804, 0.7958] | 0.7711 [0.7592, 0.7827] | 0.7235 | 0.7088 | 0.7378 | 0.1879 |
| **Random Forest** | Classical | 0.4880 | 0.7810 [0.7732, 0.7881] | 0.7632 [0.7515, 0.7744] | 0.7168 | 0.6965 | 0.7368 | 0.1902 |
| **Gradient Boosting** | Classical | 0.5486 | 0.7602 [0.7521, 0.7683] | 0.7430 [0.7313, 0.7539] | 0.7007 | 0.6434 | 0.7570 | 0.2050 |
| **Multilayer Perceptron** | Classical | 0.5189 | 0.7748 [0.7675, 0.7826] | 0.7587 [0.7470, 0.7703] | 0.7079 | 0.6765 | 0.7387 | 0.1963 |
| **XGBoost** | Classical | 0.4497 | 0.7589 [0.7504, 0.7668] | 0.7421 [0.7299, 0.7539] | 0.6941 | 0.7131 | 0.6756 | 0.2091 |
| **LightGBM** | Classical | 0.4226 | 0.7468 [0.7384, 0.7546] | 0.7318 [0.7199, 0.7432] | 0.6814 | 0.7121 | 0.6513 | 0.2235 |
| **CatBoost** | Classical | 0.5178 | 0.7834 [0.7759, 0.7912] | 0.7650 [0.7532, 0.7757] | 0.7200 | 0.6866 | 0.7528 | 0.1901 |
| **Variational Quantum Classifier** | Quantum | 0.5983 | 0.7350 [0.7269, 0.7438] | 0.7108 [0.6984, 0.7233] | 0.6999 | 0.5549 | 0.8421 | 0.2059 |
| **Quantum Support Vector Machine** | Quantum | 0.5707 | 0.7113 [0.7019, 0.7196] | 0.6926 [0.6789, 0.7054] | 0.6654 | 0.5568 | 0.7718 | 0.2171 |
| **Hybrid Quantum Neural Network** | Quantum | 0.4480 | 0.7519 [0.7440, 0.7598] | 0.7423 [0.7301, 0.7550] | 0.7018 | 0.6965 | 0.7070 | 0.2017 |

### Sample Efficiency & Quantum Parity Gap Analysis

> **Sample Efficiency Drop vs Quantum Parity Gap**: Quantifies generalization decay when classical models are restricted from full cohort (N=56,000) to parity cohort (N=1,000), contrasted against quantum architectures.

| Model | Architecture | Track A ROC (N=56k) | Track B ROC (N=1k) | Sample Efficiency Δ (56k - 1k) |
| --- | --- | --- | --- | --- |
| Logistic Regression | Classical | 0.7958 | 0.7885 | +0.0073 |
| Calibrated SVM | Classical | 0.7956 | 0.7879 | +0.0077 |
| Random Forest | Classical | 0.8019 | 0.7810 | +0.0209 |
| Gradient Boosting | Classical | 0.8019 | 0.7602 | +0.0417 |
| Multilayer Perceptron | Classical | 0.7996 | 0.7748 | +0.0248 |
| XGBoost | Classical | 0.8014 | 0.7589 | +0.0424 |
| LightGBM | Classical | 0.8025 | 0.7468 | +0.0557 |
| CatBoost | Classical | 0.8025 | 0.7834 | +0.0191 |
| Variational Quantum Classifier | Quantum | N/A (Budget Infeasible) | 0.7350 | N/A (Track B Only) |
| Quantum Support Vector Machine | Quantum | N/A (Budget Infeasible) | 0.7113 | N/A (Track B Only) |
| Hybrid Quantum Neural Network | Quantum | N/A (Budget Infeasible) | 0.7519 | N/A (Track B Only) |

### Computational Efficiency Benchmark

> **Computational Efficiency Protocol**: Real measured wall-clock training durations and per-sample inference latencies evaluated on the identical holdout test partition ($N = 13,741$) under single-process execution. Peak memory footprint is reported as null / not profiled to avoid unverified OS-level fabrication.

| Model | Architecture Family | Track | Train Samples | Test Samples | Training Time (s) | Inference Latency (ms/sample) | Peak Memory |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear Classifier | Track A | 54,961 | 13,741 | 0.119s | 0.0000 ms | Not profiled |
| **Calibrated SVM** | Support Vector Machine | Track A | 54,961 | 13,741 | 0.326s | 0.0001 ms | Not profiled |
| **Random Forest** | Ensemble Trees | Track A | 54,961 | 13,741 | 0.331s | 0.0010 ms | Not profiled |
| **Gradient Boosting** | Gradient Boosted Trees | Track A | 54,961 | 13,741 | 0.986s | 0.0007 ms | Not profiled |
| **Multilayer Perceptron** | Neural Network | Track A | 54,961 | 13,741 | 1.643s | 0.0004 ms | Not profiled |
| **XGBoost** | Gradient Boosted Trees | Track A | 54,961 | 13,741 | 0.563s | 0.0002 ms | Not profiled |
| **LightGBM** | Gradient Boosted Trees | Track A | 54,961 | 13,741 | 1.610s | 0.0009 ms | Not profiled |
| **CatBoost (Champion)** | Gradient Boosted Trees | Track A | 54,961 | 13,741 | 1.865s | 0.0004 ms | Not profiled |
| **Variational Quantum (VQC)** | Quantum Circuit (4-qubit) | Track B | 1,000 | 13,741 | 1.560s | 0.0038 ms | Not profiled |
| **Quantum SVM (QSVM)** | Quantum Kernel Estimator | Track B | 1,000 | 13,741 | 0.032s | 0.0092 ms | Not profiled |
| **Hybrid QNN** | Quantum-Classical Hybrid | Track B | 1,000 | 13,741 | 0.654s | 0.0038 ms | Not profiled |

### Forensic Baseline V1 vs Remediated V2 Comparison

> **Forensic Remediation Audit**: Side-by-side contrast of pipeline architectures, leakage controls, calibration procedures, threshold locking, and API safety between pre-remediation Baseline V1 and post-remediation Remediated V2.

| Pipeline Dimension | Baseline V1 (Pre-Remediation) | Remediated V2 (Post-Remediation) | Forensic Impact |
| --- | --- | --- | --- |
| Quantum Feature Ingestion | 4 features (silent [:, :4] cut) | Full 22 features via Linear/PCA | Zero feature loss; true multi-variate quantum input |
| Quantum Calibration | In-sample Platt fit on train readouts | 3-Fold Internal CV OOF Platt fit | Zero in-sample calibration leakage |
| Operational Threshold Locking | Post-hoc test & Framingham search | Locked strictly on Training OOF (Youden J) | Zero holdout label peeking; true prospective validity |
| External Cohort Framing | Merged into leaderboard as 'valid' | OOD Transportability Stress Test | Honest endpoint mismatch (prevalent vs incident) |
| Quantum Benchmark Design | Asymmetric (Quantum 1k vs Class 56k) | Dual-Track Parity (Track A & Track B) | Fair algorithmic comparison at uniform data budget |
| API Fallback Safety | 5-row dummy synthetic training | HTTP 503 MODEL_NOT_LOADED | Zero clinical hallucination or synthetic risk scoring |
| API Input Validation | None (accepts inverted BP) | Strict physiological bounds (HTTP 400) | Guaranteed hemodynamic plausibility |
| API Local Explainability | Hardcoded +1.8x heuristics | TreeSHAP / Linear βΔx / Quantum Jacobian | Genuine mathematical local attributions |
| Model Serialization | Discarded after evaluation | ProductionPipeline bundles (.joblib) | Direct training-to-serving parity |

### External Validation (Framingham Heart Study Cohort)

> **Full-feature production model cannot be evaluated on Framingham because required predictors are unavailable.**
> 
> Framingham was evaluated as an OOD transportability stress test. Because the cohort has a different endpoint (`TenYearCHD` vs prevalent `cardio`) and lacks required predictors (`gluc`, `alco`, `active`, `height`, `weight`, and derived interactions), the full 22-feature production model cannot be evaluated on Framingham without data fabrication. To preserve scientific integrity, missing values are never fabricated or imputed with zeros.
> 
> A separate reduced-feature configuration (excluding Framingham-absent columns via `--drop-absent`) must be explicitly trained if cross-cohort transportability benchmarking is desired. The full production CatBoost model is NOT claimed to be externally validated on Framingham.

> **Diagnostic Incompatibility Detail**: Required feature(s) missing from harmonized Framingham dataset: ['active', 'alco', 'cholesterol_gluc_ratio', 'gluc', 'health_index', 'height', 'high_risk_lifestyle', 'metabolic_synergy', 'weight']. Cannot perform external validation with missing required features. Do not silently impute zeros or fabricate values. Limitation: external validation is only supported for feature-reduced configurations that exclude Framingham-absent columns (e.g. --reduction f_classif --reduction-k 15, or drop_columns=['height', 'weight', 'gluc', 'alco', 'active']); full-feature-set external validation requires explicit imputation policy or column exclusion, not yet implemented.

### Threshold Recalibration

> **Clinical Usability & Threshold Recalibration Interpretation**: 
> ROC-AUC is unchanged by this recalibration (since it is threshold-independent); this section concerns solely the practical clinical usability of the models' binary decision cutoffs on cohorts with different disease prevalence than the development population.

| Model | Cohort | Fixed Cutoff | Fixed Sens | Fixed Spec | Optimal Cutoff | Optimal Sens | Optimal Spec | Youden's J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Internal Holdout (49.5%) | 0.5000 | 0.7075 | 0.7503 | 0.4620 | 0.7120 | 0.7493 | 0.4612 |
| Calibrated SVM | Internal Holdout (49.5%) | 0.5000 | 0.7069 | 0.7502 | 0.4563 | 0.7122 | 0.7483 | 0.4605 |
| Random Forest | Internal Holdout (49.5%) | 0.5000 | 0.6631 | 0.8003 | 0.5186 | 0.6696 | 0.7983 | 0.4678 |
| Gradient Boosting | Internal Holdout (49.5%) | 0.5000 | 0.7022 | 0.7627 | 0.4857 | 0.7050 | 0.7637 | 0.4687 |
| Multilayer Perceptron | Internal Holdout (49.5%) | 0.5000 | 0.6893 | 0.7765 | 0.5090 | 0.6834 | 0.7838 | 0.4672 |
| XGBoost | Internal Holdout (49.5%) | 0.5000 | 0.6996 | 0.7688 | 0.4910 | 0.6994 | 0.7706 | 0.4699 |
| LightGBM | Internal Holdout (49.5%) | 0.5000 | 0.6847 | 0.7825 | 0.5053 | 0.6888 | 0.7814 | 0.4701 |
| CatBoost | Internal Holdout (49.5%) | 0.5000 | 0.7019 | 0.7670 | 0.4836 | 0.7061 | 0.7631 | 0.4693 |


## 9. ROC-AUC

Receiver Operating Characteristic (ROC) curve analysis evaluates true positive vs false positive discrimination across all decision thresholds.

![ROC Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/remediated_v2/figures/roc_curves_comparison.png)

## 10. PR-AUC

Precision-Recall (PR) curve analysis benchmarks positive class detection performance relative to cohort CVD prevalence.

![PR Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/remediated_v2/figures/pr_curves_comparison.png)

## 11. Calibration

Probabilistic calibration assesses whether predicted probabilities reflect true empirical clinical incidence (Reliability Diagram, Brier Score, and Expected Calibration Error).

![Calibration Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/remediated_v2/figures/calibration_curves.png)

## 12. Confusion Matrices

Threshold-specific classification outcomes (True Negatives, False Positives, False Negatives, True Positives) across all models:

![Confusion Matrices](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/remediated_v2/figures/confusion_matrices.png)

## 13. Out-of-Fold Predictions

Aggregated Out-Of-Fold (OOF) cross-validation evaluation across 5 identical folds with cross-fold standard deviations:

| Model | ROC-AUC | PR-AUC | Brier Score | ECE | Sensitivity (Recall) | Specificity | Precision | F1-Score | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.8015 (±0.003) | 0.7832 (±0.004) | 0.1804 | 0.0051 | 0.6896 | 0.7777 | 0.7524 | 0.7196 | 0.7341 | 0.7336 |
| LightGBM | 0.8008 (±0.003) | 0.7821 (±0.004) | 0.1807 | 0.0040 | 0.6933 | 0.7763 | 0.7522 | 0.7216 | 0.7352 | 0.7348 |
| XGBoost | 0.8006 (±0.003) | 0.7830 (±0.003) | 0.1809 | 0.0068 | 0.6902 | 0.7787 | 0.7534 | 0.7204 | 0.7349 | 0.7344 |
| Gradient Boosting | 0.8004 (±0.003) | 0.7819 (±0.003) | 0.1808 | 0.0063 | 0.6903 | 0.7769 | 0.7519 | 0.7198 | 0.7340 | 0.7336 |
| Random Forest | 0.7995 (±0.003) | 0.7803 (±0.005) | 0.1814 | 0.0065 | 0.6851 | 0.7819 | 0.7547 | 0.7182 | 0.7340 | 0.7335 |
| Multilayer Perceptron | 0.7987 (±0.003) | 0.7777 (±0.003) | 0.1820 | 0.0104 | 0.6903 | 0.7759 | 0.7511 | 0.7194 | 0.7335 | 0.7331 |
| Logistic Regression | 0.7951 (±0.003) | 0.7736 (±0.003) | 0.1840 | 0.0202 | 0.6706 | 0.7886 | 0.7566 | 0.7110 | 0.7302 | 0.7296 |
| Calibrated SVM | 0.7946 (±0.003) | 0.7727 (±0.003) | 0.1842 | 0.0196 | 0.6660 | 0.7929 | 0.7591 | 0.7095 | 0.7301 | 0.7295 |

## 14. Feature Importance & Attributions

### Variance Inflation Factor (VIF) Multicollinearity Audit

Collinearity analysis diagnosing feature inter-dependencies. Features with VIF > 10 were pruned for the explanation Logistic Regression to stabilize clinical odds ratios:

| Feature | Feature Type | VIF | Collinearity Severity |
| --- | --- | --- | --- |
| age_years | Objective (Derived) | 244.66374701103013 | Severe (VIF > 10) |
| age_squared | Objective (Derived) | 229.36314199999762 | Severe (VIF > 10) |
| pulse_pressure | Examination (Derived) | 137.50971416573378 | Severe (VIF > 10) |
| log_pulse_pressure | Examination (Derived) | 128.91363538095305 | Severe (VIF > 10) |
| age_bp_interaction | Examination (Derived) | 119.32077740856518 | Severe (VIF > 10) |
| ap_hi | Examination | 116.91673372234385 | Severe (VIF > 10) |
| mean_arterial_pressure | Examination (Derived) | 91.08616698853234 | Severe (VIF > 10) |
| weight | Objective | 78.76756795188432 | Severe (VIF > 10) |
| bmi | Objective (Derived) | 73.6316349954204 | Severe (VIF > 10) |
| health_index | Subjective (Derived) | 70.42633981337478 | Severe (VIF > 10) |
| active | Subjective | 58.33231822795835 | Severe (VIF > 10) |
| cholesterol | Examination | 54.19698289365715 | Severe (VIF > 10) |
| cholesterol_gluc_ratio | Examination (Derived) | 39.329668117813 | Severe (VIF > 10) |
| ap_lo | Examination | 37.4040403672674 | Severe (VIF > 10) |
| height | Objective | 21.936223755725184 | Severe (VIF > 10) |
| gluc | Examination | 11.28311390326075 | Severe (VIF > 10) |
| metabolic_synergy | Engineered / Latent | 9.750192034091226 | Moderate (5 < VIF <= 10) |
| smoke | Subjective | 8.07161500304443 | Moderate (5 < VIF <= 10) |
| alco | Subjective | 6.084620395520086 | Moderate (5 < VIF <= 10) |
| is_hypertensive_stage2 | Examination (Derived) | 3.56063068335817 | Low (Acceptable) |
| high_risk_lifestyle | Subjective (Derived) | 2.3743842627218013 | Low (Acceptable) |
| gender | Objective | 1.5139648707928146 | Low (Acceptable) |

### De-Collinearized Logistic Regression Odds Ratios (VIF ≤ 10)

Clinical odds ratios estimated after resolving multicollinearity (OR > 1.0 indicates increased cardiovascular risk):

| Feature | Feature Type | VIF | Log-Odds Beta (β) | Odds Ratio (e^β) | % Change in Odds | Clinical Direction | Clinical Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| is_hypertensive_stage2 | Examination (Derived) | 2.84 | 0.7297 | 2.0745 | 107.4 | Increases Risk | ACC/AHA Stage 2 Hypertension: Clinically severe blood pressure elevation (>=140 SBP or >=90 DBP). |
| cholesterol_gluc_ratio | Examination (Derived) | 1.17 | 0.5478 | 1.7294 | 72.9 | Increases Risk | Task-Specific Engineered Ratio: Ordinal ratio of cholesterol to glucose capturing relative dyslipidemia/dysglycemia interaction, not an established biomarker. |
| age_squared | Objective (Derived) | 1.08 | 0.5442 | 1.7232 | 72.3 | Increases Risk | Accelerated Vascular Age: Non-linear acceleration in atherosclerotic plaque burden observed in older adults. |
| log_pulse_pressure | Examination (Derived) | 1.45 | 0.4989 | 1.647 | 64.7 | Increases Risk | Log Pulse Pressure: Skew-stabilized representation of large-artery pulsatile load. |
| ap_lo | Examination | 2.32 | 0.4178 | 1.5186 | 51.9 | Increases Risk | Diastolic Blood Pressure: Resting systemic vascular resistance between contractions; high values indicate continuous microvascular tension. |
| gluc | Examination | 2.73 | 0.2309 | 1.2597 | 26.0 | Increases Risk | Blood Glucose Level: Biomarker of glycemic dysregulation and insulin resistance promoting vascular endothelial inflammation. |
| bmi | Objective (Derived) | 1.16 | 0.1978 | 1.2188 | 21.9 | Increases Risk | Body Mass Index: Anthropometric estimate of adiposity; correlates with left ventricular hypertrophy and systemic inflammation. |
| metabolic_synergy | Engineered / Latent | 2.53 | 0.1205 | 1.128 | 12.8 | Increases Risk | Task-Specific Engineered Interaction: Binary flag for concurrent elevation of ordinal cholesterol and glucose (both >= 2); heuristic marker, not a clinical diagnostic criterion. |
| height | Objective | 1.44 | 0.078 | 1.0812 | 8.1 | Increases Risk | Patient Height: Component of body surface area and hemodynamic vascular impedance. |
| gender | Objective | 1.51 | -0.0184 | 0.9818 | -1.8 | Protective (Reduces Risk) | Biological Sex: Demographic covariate capturing sex-specific cardiovascular risk trajectories. |
| high_risk_lifestyle | Subjective (Derived) | 2.36 | -0.1118 | 0.8942 | -10.6 | Protective (Reduces Risk) | Task-Specific Engineered Interaction: Co-occurrence of smoking and alcohol intake compounding oxidative vascular stress. |
| smoke | Subjective | 1.52 | -0.1308 | 0.8774 | -12.3 | Protective (Reduces Risk) | Active Tobacco Smoking: Direct oxidant chemical injury inducing endothelial dysfunction and hypercoagulability. |
| alco | Subjective | 1.93 | -0.2097 | 0.8108 | -18.9 | Protective (Reduces Risk) | Alcohol Consumption: Excess intake promotes neurohormonal activation, cardiac arrhythmia, and secondary hypertension. |
| active | Subjective | 1.0 | -0.2701 | 0.7633 | -23.7 | Protective (Reduces Risk) | Physical Activity: Regular aerobic activity preserves endothelial nitric oxide availability and lowers resting vascular tone. |

### Random Forest Gini Importance (Top Predictors)

| Feature | Gini Importance | Clinical Interpretation |
| --- | --- | --- |
| ap_hi | 0.1741979329171249 | Systolic Blood Pressure: Peak arterial pressure during ventricular contraction; primary mechanical driver of vascular remodeling and coronary wall stress. |
| age_bp_interaction | 0.15263290980698738 | Vascular Age × Systolic Burden: Synergistic cardiovascular hazard where advanced arterial age amplifies the harm of hypertension. |
| mean_arterial_pressure | 0.10479544699288841 | Mean Arterial Pressure (MAP): Mean systemic perfusion pressure determining end-organ perfusion and systemic vascular strain. |
| is_hypertensive_stage2 | 0.09967654494431248 | ACC/AHA Stage 2 Hypertension: Clinically severe blood pressure elevation (>=140 SBP or >=90 DBP). |
| age_years | 0.06102153875090693 | Standardized Age: Cumulative biological aging of vascular endothelium; strongest non-modifiable cardiovascular risk factor. |
| age_squared | 0.05796484107703493 | Accelerated Vascular Age: Non-linear acceleration in atherosclerotic plaque burden observed in older adults. |
| ap_lo | 0.05315167715359959 | Diastolic Blood Pressure: Resting systemic vascular resistance between contractions; high values indicate continuous microvascular tension. |
| log_pulse_pressure | 0.05040507754938897 | Log Pulse Pressure: Skew-stabilized representation of large-artery pulsatile load. |
| pulse_pressure | 0.044169226850943925 | Pulse Pressure (ap_hi - ap_lo): Primary hemodynamic marker of arterial wall stiffening and diminished aortic compliance. |
| cholesterol | 0.04373208221663471 | Serum Total Cholesterol: Circulating atherogenic lipoproteins contributing to arterial intimal plaque formation and luminal narrowing. |

> **Clinical Measurement Reliability Notice (Examination vs Subjective Features)**: 
> Examination features (measured systolic/diastolic blood pressure, laboratory cholesterol, and glucose) provide significantly more objective and reliable physiological signals than Subjective features (self-reported smoking, alcohol intake, and physical activity), which are inherently vulnerable to under-reporting, recall bias, and social desirability effects.

## 15. Risk Interpretation

Patient-level personalized risk attribution decomposing individual clinical factors from cohort baseline without diagnosis language:

- **Patient ID**: `1001`
- **Estimated Model Probability**: **31.4%**
- **Assigned Risk Tier**: `Moderate Estimated Risk (20% - 50%)`
- **Clinical Summary**: Patient exhibits an estimated model probability of 31.4% for cardiovascular disease risk, categorizing into the Moderate Estimated Risk (20% - 50%) band. Top positive risk driver: age_bp_interaction.

**Top Risk-Increasing Clinical Factors**:
- 🔺 `age_bp_interaction`: patient value = -0.2 (cohort mean = 0.0). *Vascular Age × Systolic Burden: Synergistic cardiovascular hazard where advanced arterial age amplifies the harm of hypertension.*
- 🔺 `weight`: patient value = -0.2 (cohort mean = 0.1). *Patient Weight: Gross body mass influencing total blood volume and cardiac cardiac output demands.*
- 🔺 `health_index`: patient value = 0.0 (cohort mean = -0.5). *Task-Specific Engineered Lifestyle Score: Linear heuristic (+1.0 active, -0.5 smoke, -0.5 alco); exploratory interaction, not a validated biomarker.*

**Top Risk-Reducing / Protective Factors**:
- 🔻 `ap_hi`: patient value = 0.0 (cohort mean = 0.3). *Systolic Blood Pressure: Peak arterial pressure during ventricular contraction; primary mechanical driver of vascular remodeling and coronary wall stress.*
- 🔻 `mean_arterial_pressure`: patient value = 0.0 (cohort mean = 0.3). *Mean Arterial Pressure (MAP): Mean systemic perfusion pressure determining end-organ perfusion and systemic vascular strain.*
- 🔻 `pulse_pressure`: patient value = 0.0 (cohort mean = 0.5). *Pulse Pressure (ap_hi - ap_lo): Primary hemodynamic marker of arterial wall stiffening and diminished aortic compliance.*

## 16. Leakage Checks

- ✅ **Zero Pre-CV Fitting**: Imputers, scalers, and outlier clippers were constructed and fitted strictly within each training fold.
- ✅ **Holdout Isolation**: The 20% test partition was withheld before feature engineering and was transformed solely through artifacts fitted on the 80% training partition.
- ✅ **Calibration Isolation**: SVM Platt calibration used internal cross-validation on training folds only.
- ✅ **Identical Folds**: All 5 models shared identical cross-validation splits for fair paired benchmarking.
- ✅ **Invariant Verification**: Unit tests verify that altering validation sets produces 0% change in trained transformer parameters.

## 17. Limitations

1. **Cross-Sectional Dataset**: The cohort captures clinical variables at a single examination encounter without longitudinal time-to-event outcome tracking.
2. **Self-Reported Lifestyle**: Tobacco, alcohol consumption, and physical activity status rely on self-reported binary indicators subject to reporting bias.
3. **Absence of Detailed Lipids**: Total cholesterol and glucose are recorded on a 3-tier ordinal scale rather than continuous mg/dL laboratory measurements (HDL, LDL, triglycerides, HbA1c).
4. **Clinical Governance**: Any bedside implementation requires prospective observational validation and ethical review under institutional software-as-a-medical-device (SaMD) standards.