# Platform Limitations, Domain Shift & Clinical Boundary Disclaimers
## CardioQ: Hybrid Quantum-Classical Cardiovascular Disease Diagnostic Engine

---

## 1. Clinical Outcome Definition & Cohort Discrepancy

A fundamental caveat highlighted by CardioQ is the distinction between internal training targets and external validation outcomes:

| Dimension | Primary Training Cohort (Kaggle CVD) | External Validation Cohort (Framingham Heart Study) |
|---|---|---|
| **Target Variable** | `cardio` | `TenYearCHD` |
| **Outcome Semantics** | Cross-sectional prevalent cardiovascular disease diagnosis at the time of examination. | 10-year prospective incidence of coronary heart disease events (fatal/non-fatal MI, angina pectoris, coronary insufficiency) in a baseline-free cohort. |
| **Cohort Prevalence** | $49.47\%$ (balanced clinical survey) | $14.98\%$ (prospective population epidemiology) |
| **Temporal Horizon** | Contemporaneous (same day examination) | 10-year prospective longitudinal follow-up |
| **Key Implication** | Models predict current presence of vascular disease. | Models are benchmarked on predicting future first incident cardiac event. |

> [!IMPORTANT]
> **Cohort & Model Incompatibility Specification**:
> - **State A (Full Production Model)**: The full-feature production model (22 predictors) **cannot be evaluated on Framingham** because required predictors (`gluc`, `alco`, `active`, `height`, `weight`, and derived interactions) are unmeasured in Framingham. The platform refuses to fabricate data or impute dummy zeros, raising `MissingExternalFeatureError`.
> - **State B (Reduced-Feature Transportability Benchmark)**: When a separate, explicitly defined reduced-feature model is trained (excluding the 5 Framingham-absent columns via `--drop-absent`), external ROC-AUC on Framingham is $\approx 0.66 - 0.67$. Because `cardio` (cross-sectional prevalent CVD) and `TenYearCHD` (10-year prospective incident CHD) represent distinct pathophysiological endpoints, this $\approx 0.13$ AUC delta is driven by outcome divergence and feature restriction, not pipeline over-fitting.
> - **Non-Claim**: The full production CatBoost model is **not** claimed to be externally validated on Framingham.

---

## 2. Prevalence Shift & Prospective Decision Thresholds

A model trained on a balanced cohort ($49.5\%$ prevalence) implicitly learns decision boundaries suited to a high pre-test probability. When deployed into lower-prevalence settings ($15.0\%$ in Framingham, or primary care screening):
* Applying an unadjusted $0.50$ threshold depresses sensitivity and alters false-positive tradeoffs.
* **Prospective Policy**: Operational thresholds $\tau^*$ are locked **strictly on development out-of-fold cross-validation** using Youden's $J$ ($J = \text{TPR} - \text{FPR}$) and frozen prior to external evaluation. Post-hoc threshold tuning against external evaluation labels is strictly prohibited to prevent data leakage and label peeking.

---

## 3. Strict Feature Integrity vs. Zero Imputation Policy

The Framingham Heart Study does not capture certain lifestyle and clinical examination variables present in the primary cohort:
* Absent variables: `gluc` (blood glucose categories), `alco` (alcohol intake), `active` (physical activity), `height` (cm), and `weight` (kg).
* **CardioQ Strict Zero-Fabrication Invariant**:
  - Excluded features with no direct Framingham equivalent are **never** filled with synthetic zeros or arbitrary dummy constants.
  - When evaluating on the full feature space without column exclusions, the external validation module explicitly fails loudly (`MissingExternalFeatureError`), forcing researchers to consciously declare feature exclusion or explicit domain transfer policies rather than silently contaminating test distributions.
  - The production 22-feature pipeline remains uncompromised; external stress testing is restricted to explicitly configured reduced-feature runs.

---

## 4. Quantum Simulation Scale & NISQ Hardware Boundaries

* **Current Architecture**: The platform utilizes an exact statevector simulation engine implemented in PyTorch/NumPy, operating on $N=4$ to $N=8$ qubits.
* **Computational Complexity**: Classical statevector simulation scales exponentially as $\mathcal{O}(2^N)$ in memory and runtime. Circuits with $>25$ qubits cannot be simulated on standard server hardware without distributed HPC tensor contractions.
* **Dimensionality Reduction**: Input feature vectors are projected into low-dimensional representations (via ANOVA $F$-test, PCA, or classical neural encoder) prior to quantum angle encoding.
* **Physical Hardware Mapping**: The circuits are designed with 1-qubit rotations ($R_x, R_y, R_z$) and 2-qubit CNOT gates compatible with modern Noisy Intermediate-Scale Quantum (NISQ) superconducting and trapped-ion quantum processors (e.g. IBM Eagle/Heron, IonQ, Rigetti). Real physical hardware execution would incur quantum gate infidelity, decoherence ($T_1, T_2$ relaxation), and readout measurement noise, necessitating error mitigation techniques (Zero-Noise Extrapolation, Readout Calibration).

---

## 5. Regulatory & Clinical Decision Support Notice

```
================================================================================
                    CLINICAL DECISION SUPPORT NOTICE
================================================================================
The algorithmic outputs, risk probabilities, and feature attributions produced
by CardioQ are intended strictly for investigational, research, and clinical
decision-support use. This platform does NOT constitute a software as a medical
device (SaMD) diagnostic apparatus and is not certified for autonomous clinical
decision making. All risk assessments must be interpreted by licensed medical
professionals within the context of comprehensive patient clinical workups.
================================================================================
```
