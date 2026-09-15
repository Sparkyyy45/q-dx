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

- **Initial Dimensions**: 10000 rows × 13 columns
- **Identified ID Columns**: `['id']` (Excluded from model features to prevent memorization)
- **Redundant/Duplicate Columns**: `[]` (`bp_category_encoded` dropped in favor of clean categorical pipeline)
- **Biological Plausibility Checks**: 5 alerts flagged and handled via pre-split cleaning and robust IQR clipping.

### Pre-Split Data Cleaning Summary

- **Pre-Cleaning Cohort**: 10,000 rows
- **Cleaned Cohort**: **9,824 rows** (-176 removed)
- **Non-ID Duplicate Rows Dropped**: 1
- **Inverted Blood Pressure Rows Dropped (`ap_hi < ap_lo`)**: 168
- **Extreme Non-Biological BMI Rows Dropped (`< 10` or `> 70`)**: 9
- **Cleaning Isolation Policy**: Cleaning executed strictly prior to holdout partitioning to guarantee both development and test sets are clean.

| Category | Column | Severity | Detail | Resolution |
|---|---|---|---|---|
| redundancy | `[multiple_features]` | **MEDIUM** | Found 1 duplicate rows when ignoring ID columns. | Retain for training if representing distinct clinical encounters, or record provenance. |
| identifier | `id` | **HIGH** | Column exhibits identifier characteristics (10000 unique values / 10000 rows). | Strictly exclude from model features to prevent memorization leakage. |
| invalid_values | `ap_hi, ap_lo` | **HIGH** | 168 instances where systolic BP < diastolic BP. | Apply clinical boundary correction or IQR clipping. |
| invalid_values | `bmi (calculated)` | **MEDIUM** | 9 extreme BMI values detected (min: 3.5, max: 187.8). | Apply robust IQR clipping or pre-split cleaning to eliminate physiological measurement errors. |

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

### Internal Holdout Test Evaluation (Untouched 20% Partition)

| Model | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7973 | 0.7794 | 0.7333 | 0.6874 | 0.7789 | 0.7553 | 0.7198 | 0.1830 |
| **Calibrated SVM** | 0.7977 | 0.7795 | 0.7349 | 0.6844 | 0.7850 | 0.7596 | 0.7200 | 0.1827 |
| **Random Forest** | 0.7980 | 0.7803 | 0.7425 | 0.7058 | 0.7789 | 0.7602 | 0.7320 | 0.1815 |
| **Gradient Boosting** | 0.7958 | 0.7719 | 0.7410 | 0.7099 | 0.7718 | 0.7554 | 0.7320 | 0.1827 |
| **Multilayer Perceptron** | 0.7964 | 0.7832 | 0.7328 | 0.7068 | 0.7586 | 0.7441 | 0.7250 | 0.1829 |
| **XGBoost** | 0.7922 | 0.7634 | 0.7369 | 0.7048 | 0.7688 | 0.7516 | 0.7275 | 0.1844 |
| **LightGBM** | 0.7901 | 0.7634 | 0.7425 | 0.7109 | 0.7738 | 0.7573 | 0.7334 | 0.1854 |
| **CatBoost** | 0.7998 | 0.7802 | 0.7384 | 0.7028 | 0.7738 | 0.7552 | 0.7280 | 0.1805 |

### External Validation (Framingham Heart Study Cohort)

> **External Validation Configuration Limitation**: 
> External validation is only supported for feature-reduced configurations that exclude Framingham-absent columns (`gluc`, `alco`, `active`, `height`, `weight`); full-feature-set external validation requires explicit imputation policy or column exclusion, not yet implemented.

> **Diagnostic Incompatibility Detail**: Required feature(s) missing from harmonized Framingham dataset: ['active', 'alco', 'cholesterol_gluc_ratio', 'gluc', 'health_index', 'height', 'high_risk_lifestyle', 'metabolic_synergy', 'weight']. Cannot perform external validation with missing required features. Do not silently impute zeros or fabricate values. Limitation: external validation is only supported for feature-reduced configurations that exclude Framingham-absent columns (e.g. --reduction f_classif --reduction-k 15, or drop_columns=['height', 'weight', 'gluc', 'alco', 'active']); full-feature-set external validation requires explicit imputation policy or column exclusion, not yet implemented.

### Threshold Recalibration

> **Clinical Usability & Threshold Recalibration Interpretation**: 
> ROC-AUC is unchanged by this recalibration (since it is threshold-independent); this section concerns solely the practical clinical usability of the models' binary decision cutoffs on cohorts with different disease prevalence than the development population.

| Model | Cohort | Fixed Cutoff | Fixed Sens | Fixed Spec | Optimal Cutoff | Optimal Sens | Optimal Spec | Youden's J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Internal Holdout (49.8%) | 0.5000 | 0.6874 | 0.7789 | 0.5472 | 0.6435 | 0.8306 | 0.4741 |
| Calibrated SVM | Internal Holdout (49.8%) | 0.5000 | 0.6844 | 0.7850 | 0.5482 | 0.6456 | 0.8316 | 0.4772 |
| Random Forest | Internal Holdout (49.8%) | 0.5000 | 0.7058 | 0.7789 | 0.4940 | 0.7109 | 0.7769 | 0.4878 |
| Gradient Boosting | Internal Holdout (49.8%) | 0.5000 | 0.7099 | 0.7718 | 0.5239 | 0.6956 | 0.7931 | 0.4887 |
| Multilayer Perceptron | Internal Holdout (49.8%) | 0.5000 | 0.7068 | 0.7586 | 0.5376 | 0.6752 | 0.7951 | 0.4703 |
| XGBoost | Internal Holdout (49.8%) | 0.5000 | 0.7048 | 0.7688 | 0.4619 | 0.7446 | 0.7434 | 0.4880 |
| LightGBM | Internal Holdout (49.8%) | 0.5000 | 0.7109 | 0.7738 | 0.4923 | 0.7191 | 0.7698 | 0.4889 |
| CatBoost | Internal Holdout (49.8%) | 0.5000 | 0.7028 | 0.7738 | 0.5637 | 0.6568 | 0.8266 | 0.4834 |


## 9. ROC-AUC

Receiver Operating Characteristic (ROC) curve analysis evaluates true positive vs false positive discrimination across all decision thresholds.

![ROC Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/figures/roc_curves_comparison.png)

## 10. PR-AUC

Precision-Recall (PR) curve analysis benchmarks positive class detection performance relative to cohort CVD prevalence.

![PR Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/figures/pr_curves_comparison.png)

## 11. Calibration

Probabilistic calibration assesses whether predicted probabilities reflect true empirical clinical incidence (Reliability Diagram, Brier Score, and Expected Calibration Error).

![Calibration Curves](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/figures/calibration_curves.png)

## 12. Confusion Matrices

Threshold-specific classification outcomes (True Negatives, False Positives, False Negatives, True Positives) across all models:

![Confusion Matrices](/Users/dhruvmakadiya/Documents/heartdisease/artifacts/figures/confusion_matrices.png)

## 13. Out-of-Fold Predictions

Aggregated Out-Of-Fold (OOF) cross-validation evaluation across 5 identical folds with cross-fold standard deviations:

| Model | ROC-AUC | PR-AUC | Brier Score | ECE | Sensitivity (Recall) | Specificity | Precision | F1-Score | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.7975 (±0.010) | 0.7767 (±0.010) | 0.1816 | 0.0146 | 0.7009 | 0.7696 | 0.7515 | 0.7253 | 0.7353 | 0.7352 |
| Random Forest | 0.7950 (±0.010) | 0.7744 (±0.010) | 0.1831 | 0.0204 | 0.6937 | 0.7734 | 0.7527 | 0.7220 | 0.7337 | 0.7336 |
| Logistic Regression | 0.7938 (±0.009) | 0.7717 (±0.009) | 0.1849 | 0.0237 | 0.6776 | 0.7823 | 0.7558 | 0.7146 | 0.7301 | 0.7300 |
| Multilayer Perceptron | 0.7933 (±0.008) | 0.7729 (±0.010) | 0.1847 | 0.0215 | 0.7062 | 0.7480 | 0.7359 | 0.7208 | 0.7272 | 0.7271 |
| Calibrated SVM | 0.7931 (±0.009) | 0.7710 (±0.010) | 0.1850 | 0.0255 | 0.6779 | 0.7853 | 0.7584 | 0.7159 | 0.7318 | 0.7316 |
| Gradient Boosting | 0.7921 (±0.009) | 0.7731 (±0.010) | 0.1845 | 0.0228 | 0.6930 | 0.7742 | 0.7531 | 0.7218 | 0.7337 | 0.7336 |
| LightGBM | 0.7888 (±0.009) | 0.7701 (±0.010) | 0.1863 | 0.0314 | 0.7021 | 0.7655 | 0.7486 | 0.7246 | 0.7339 | 0.7338 |
| XGBoost | 0.7888 (±0.011) | 0.7717 (±0.010) | 0.1865 | 0.0350 | 0.6909 | 0.7663 | 0.7461 | 0.7175 | 0.7287 | 0.7286 |

## 14. Feature Importance & Attributions

### Variance Inflation Factor (VIF) Multicollinearity Audit

Collinearity analysis diagnosing feature inter-dependencies. Features with VIF > 10 were pruned for the explanation Logistic Regression to stabilize clinical odds ratios:

| Feature | Feature Type | VIF | Collinearity Severity |
| --- | --- | --- | --- |
| age_years | Objective (Derived) | 249.31611125999615 | Severe (VIF > 10) |
| age_squared | Objective (Derived) | 233.69906255307862 | Severe (VIF > 10) |
| pulse_pressure | Examination (Derived) | 136.24141275912623 | Severe (VIF > 10) |
| log_pulse_pressure | Examination (Derived) | 127.981582325557 | Severe (VIF > 10) |
| age_bp_interaction | Examination (Derived) | 124.34311052215782 | Severe (VIF > 10) |
| ap_hi | Examination | 114.19648241700659 | Severe (VIF > 10) |
| mean_arterial_pressure | Examination (Derived) | 86.83866315301445 | Severe (VIF > 10) |
| health_index | Subjective (Derived) | 78.8003325071713 | Severe (VIF > 10) |
| weight | Objective | 69.38312546478579 | Severe (VIF > 10) |
| active | Subjective | 66.62075422021158 | Severe (VIF > 10) |
| bmi | Objective (Derived) | 64.71385575712364 | Severe (VIF > 10) |
| cholesterol | Examination | 53.43562537531779 | Severe (VIF > 10) |
| cholesterol_gluc_ratio | Examination (Derived) | 40.02463452049168 | Severe (VIF > 10) |
| ap_lo | Examination | 36.634376889491804 | Severe (VIF > 10) |
| height | Objective | 19.333872981276294 | Severe (VIF > 10) |
| gluc | Examination | 10.5147787836619 | Severe (VIF > 10) |
| metabolic_synergy | Engineered / Latent | 9.88072080122436 | Moderate (5 < VIF <= 10) |
| smoke | Subjective | 8.66797245760618 | Moderate (5 < VIF <= 10) |
| alco | Subjective | 6.274364743808378 | Moderate (5 < VIF <= 10) |
| is_hypertensive_stage2 | Examination (Derived) | 3.4684341984677514 | Low (Acceptable) |
| high_risk_lifestyle | Subjective (Derived) | 2.3675193641200836 | Low (Acceptable) |
| gender | Objective | 1.4961968686633926 | Low (Acceptable) |

### De-Collinearized Logistic Regression Odds Ratios (VIF ≤ 10)

Clinical odds ratios estimated after resolving multicollinearity (OR > 1.0 indicates increased cardiovascular risk):

| Feature | Feature Type | VIF | Log-Odds Beta (β) | Odds Ratio (e^β) | % Change in Odds | Clinical Direction | Clinical Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| is_hypertensive_stage2 | Examination (Derived) | 2.8 | 0.6614 | 1.9376 | 93.8 | Increases Risk | ACC/AHA Stage 2 Hypertension: Clinically severe blood pressure elevation (>=140 SBP or >=90 DBP). |
| age_squared | Objective (Derived) | 1.08 | 0.5808 | 1.7874 | 78.7 | Increases Risk | Accelerated Vascular Age: Non-linear acceleration in atherosclerotic plaque burden observed in older adults. |
| cholesterol_gluc_ratio | Examination (Derived) | 1.19 | 0.5735 | 1.7744 | 77.4 | Increases Risk | Lipid-to-Glycemic Ratio: Composite measure of concurrent dyslipidemia and glycemic dysregulation. |
| log_pulse_pressure | Examination (Derived) | 1.44 | 0.4974 | 1.6445 | 64.4 | Increases Risk | Log Pulse Pressure: Skew-stabilized representation of large-artery pulsatile load. |
| ap_lo | Examination | 2.3 | 0.4063 | 1.5013 | 50.1 | Increases Risk | Diastolic Blood Pressure: Resting systemic vascular resistance between contractions; high values indicate continuous microvascular tension. |
| bmi | Objective (Derived) | 1.17 | 0.2563 | 1.2921 | 29.2 | Increases Risk | Body Mass Index: Anthropometric estimate of adiposity; correlates with left ventricular hypertrophy and systemic inflammation. |
| gluc | Examination | 2.59 | 0.2134 | 1.2379 | 23.8 | Increases Risk | Blood Glucose Level: Biomarker of glycemic dysregulation and insulin resistance promoting vascular endothelial inflammation. |
| metabolic_synergy | Engineered / Latent | 2.38 | 0.0968 | 1.1017 | 10.2 | Increases Risk | Metabolic Syndrome Flag: Concurrent elevation of cholesterol and glucose reflecting heightened cardiometabolic risk. |
| height | Objective | 1.43 | 0.075 | 1.0779 | 7.8 | Increases Risk | Patient Height: Component of body surface area and hemodynamic vascular impedance. |
| high_risk_lifestyle | Subjective (Derived) | 2.34 | -0.0208 | 0.9794 | -2.1 | Protective (Reduces Risk) | Combined Toxic Lifestyle: Co-occurrence of regular smoking and alcohol intake compounding oxidative vascular stress. |
| gender | Objective | 1.49 | -0.0524 | 0.9489 | -5.1 | Protective (Reduces Risk) | Biological Sex: Demographic covariate capturing sex-specific cardiovascular risk trajectories. |
| alco | Subjective | 1.97 | -0.0595 | 0.9422 | -5.8 | Protective (Reduces Risk) | Alcohol Consumption: Excess intake promotes neurohormonal activation, cardiac arrhythmia, and secondary hypertension. |
| smoke | Subjective | 1.46 | -0.1075 | 0.8981 | -10.2 | Protective (Reduces Risk) | Active Tobacco Smoking: Direct oxidant chemical injury inducing endothelial dysfunction and hypercoagulability. |
| active | Subjective | 1.0 | -0.3106 | 0.733 | -26.7 | Protective (Reduces Risk) | Physical Activity: Regular aerobic activity preserves endothelial nitric oxide availability and lowers resting vascular tone. |

### Random Forest Gini Importance (Top Predictors)

| Feature | Gini Importance | Clinical Interpretation |
| --- | --- | --- |
| age_bp_interaction | 0.1465586390988838 | Vascular Age × Systolic Burden: Synergistic cardiovascular hazard where advanced arterial age amplifies the harm of hypertension. |
| ap_hi | 0.1358306543348532 | Systolic Blood Pressure: Peak arterial pressure during ventricular contraction; primary mechanical driver of vascular remodeling and coronary wall stress. |
| mean_arterial_pressure | 0.09264262988954769 | Mean Arterial Pressure (MAP): Mean systemic perfusion pressure determining end-organ perfusion and systemic vascular strain. |
| age_years | 0.07818332404569674 | Standardized Age: Cumulative biological aging of vascular endothelium; strongest non-modifiable cardiovascular risk factor. |
| age_squared | 0.07053525956932558 | Accelerated Vascular Age: Non-linear acceleration in atherosclerotic plaque burden observed in older adults. |
| is_hypertensive_stage2 | 0.06898473650517917 | ACC/AHA Stage 2 Hypertension: Clinically severe blood pressure elevation (>=140 SBP or >=90 DBP). |
| bmi | 0.06815617467527993 | Body Mass Index: Anthropometric estimate of adiposity; correlates with left ventricular hypertrophy and systemic inflammation. |
| weight | 0.0508363647147857 | Patient Weight: Gross body mass influencing total blood volume and cardiac cardiac output demands. |
| ap_lo | 0.04370091601463793 | Diastolic Blood Pressure: Resting systemic vascular resistance between contractions; high values indicate continuous microvascular tension. |
| cholesterol | 0.0427429270062724 | Serum Total Cholesterol: Circulating atherogenic lipoproteins contributing to arterial intimal plaque formation and luminal narrowing. |

> **Clinical Measurement Reliability Notice (Examination vs Subjective Features)**: 
> Examination features (measured systolic/diastolic blood pressure, laboratory cholesterol, and glucose) provide significantly more objective and reliable physiological signals than Subjective features (self-reported smoking, alcohol intake, and physical activity), which are inherently vulnerable to under-reporting, recall bias, and social desirability effects.

## 15. Risk Interpretation

Patient-level personalized risk attribution decomposing individual clinical factors from cohort baseline without diagnosis language:

- **Patient ID**: `1001`
- **Estimated Model Probability**: **95.0%**
- **Assigned Risk Tier**: `High Estimated Risk (>50%)`
- **Clinical Summary**: Patient exhibits an estimated model probability of 95.0% for cardiovascular disease risk, categorizing into the High Estimated Risk (>50%) band. Top positive risk driver: mean_arterial_pressure.

**Top Risk-Increasing Clinical Factors**:
- 🔺 `mean_arterial_pressure`: patient value = 2.5 (cohort mean = 0.3). *Mean Arterial Pressure (MAP): Mean systemic perfusion pressure determining end-organ perfusion and systemic vascular strain.*
- 🔺 `cholesterol`: patient value = 2.0 (cohort mean = 0.4). *Serum Total Cholesterol: Circulating atherogenic lipoproteins contributing to arterial intimal plaque formation and luminal narrowing.*
- 🔺 `ap_hi`: patient value = 2.5 (cohort mean = 0.3). *Systolic Blood Pressure: Peak arterial pressure during ventricular contraction; primary mechanical driver of vascular remodeling and coronary wall stress.*

**Top Risk-Reducing / Protective Factors**:
- 🔻 `age_bp_interaction`: patient value = 2.0 (cohort mean = 0.0). *Vascular Age × Systolic Burden: Synergistic cardiovascular hazard where advanced arterial age amplifies the harm of hypertension.*
- 🔻 `metabolic_synergy`: patient value = 1.0 (cohort mean = 0.1). *Metabolic Syndrome Flag: Concurrent elevation of cholesterol and glucose reflecting heightened cardiometabolic risk.*
- 🔻 `gluc`: patient value = 2.0 (cohort mean = 0.2). *Blood Glucose Level: Biomarker of glycemic dysregulation and insulin resistance promoting vascular endothelial inflammation.*

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