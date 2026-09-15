# CardioQ: Full Technical Architecture Specification
## Hybrid Quantum-Classical Cardiovascular Disease Prediction Platform
### SIH Problem Statement 3 — Technical Document 02

**Version**: 1.0 — Production  
**Platform**: CardioQ  
**Status**: All 11 models verified, 64/64 tests passing

---

## 1. High-Level Architecture

CardioQ is a **dual-track hybrid quantum-classical ML platform** structured around a strict data-flow invariant: a single immutable 80/20 stratified holdout partition is created once at system initialization and never altered.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CARDIOQ PLATFORM ARCHITECTURE                    │
│                                                                       │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐  │
│  │  DATA LAYER  │────►│  ML ENGINE   │────►│   API / PLATFORM     │  │
│  │             │     │              │     │                      │  │
│  │ CSV Upload  │     │ Track A:     │     │ REST API (Flask)     │  │
│  │ Audit       │     │  8 Classical │     │ Web Dashboard        │  │
│  │ Pre-Clean   │     │              │     │ FHIR R4 Export       │  │
│  │ 80/20 Split │     │ Track B:     │     │ ABHA Interop         │  │
│  │ SQLite      │     │  8+3 Hybrid  │     │ Explain Engine       │  │
│  └─────────────┘     └──────────────┘     └──────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   QUANTUM ENGINE (PyTorch/NumPy)              │    │
│  │  Statevector Sim  │  VQC  │  QSVM  │  Hybrid QNN  │  SHAP   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component-by-Component Breakdown

### 2.1 Data Layer (`src/audit.py`, `src/clean.py`, `src/dataset.py`)

#### Stage 1: Pre-Split Clinical Data Audit
Every uploaded dataset undergoes a deterministic audit **before** any train/test split:

| Audit Check | Implementation | Output |
|-------------|---------------|--------|
| Missing values per column | `df.isnull().sum()` | Missing % matrix |
| Duplicate rows | `df.duplicated(subset=[non-id cols])` | Deduplicated rows count |
| Target class balance | `value_counts(normalize=True)` | Imbalance ratio |
| Biological plausibility | Domain range checks per FEATURE_DICTIONARY | Flagged anomalies |
| Potential identifier columns | High-cardinality scan | ID column warnings |
| Potential leakage columns | Correlation > 0.95 with target | Leakage risk flags |

#### Stage 2: Pre-Split Data Cleaning (`src/clean.py`)
Applied to the **full raw dataset** before any partitioning:

```python
# Non-physiological blood pressure inversions
df = df[df['ap_hi'] > df['ap_lo']]

# Biological extreme BMI removal
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 70)]

# Exact duplicate removal (excluding patient ID)
df = df.drop_duplicates(subset=[c for c in df.columns if c != 'id'])
```

#### Stage 3: Stratified 80/20 Partition
```python
from sklearn.model_selection import train_test_split
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# N_dev = 54,961 | N_holdout = 13,741
```

**The holdout partition is locked and never accessed during preprocessing, training, or hyperparameter tuning.**

---

### 2.2 Classical Feature Engineering (`src/features.py`)

All feature transformations are **row-independent** — safe for single-patient inference:

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| BMI | weight(kg) / height(m)² | WHO obesity classification |
| Mean Arterial Pressure (MAP) | (ap_hi + 2·ap_lo) / 3 | Tissue perfusion pressure |
| Pulse Pressure | ap_hi - ap_lo | Arterial stiffness proxy |
| Age (normalized) | `np.where(age > 120, age / 365.25, age)` | Unified years scale |
| Age² | age² | Non-linear aging effect |
| Cardiovascular Strain Index | MAP × age | Cumulative vascular stress |
| Cholesterol-Glucose Ratio | cholesterol / gluc.clip(1) | Metabolic syndrome marker |

**Critical invariant**: `ClinicalFeatureEngineer.transform()` validates all required columns are present before computing derived features, safely skipping absent optional biomarkers (Framingham compatibility).

---

### 2.3 Leakage-Safe Preprocessing Pipeline (`src/preprocess.py`)

Each fold's preprocessing parameters are computed **strictly within that fold's training data**:

```
For each of 3 cross-validation folds:
  Train fold only → fit IQRClipper, RobustScaler, FeatureReducer
  Apply fitted parameters → transform train fold
  Apply same fitted parameters → transform validation fold
  (Test partition NEVER accessed during this loop)
```

**IQR Clipping**: `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` — bounds computed on training fold only.  
**Scaling**: `RobustScaler` (median/IQR) for continuous features; `StandardScaler` as alternative.  
**Feature Selection**: ANOVA F-test (`SelectKBest`) or PCA — fitted on training fold only.

---

### 2.4 Dual-Track Benchmarking Architecture

#### Track A — Clinical Utility Benchmark
- **N_train** = 54,961 (full development cohort)
- **Models**: 8 classical models
- **Evaluation**: 13,741 holdout patients with 1,000-iteration bootstrap 95% CIs
- **Decision threshold**: Locked via Youden's J on OOF cross-validation predictions

```
J* = argmax_τ [Sensitivity(τ) + Specificity(τ) - 1]
τ* = argmax_τ J*(τ)   [Derived from OOF only, never from test data]
```

#### Track B — Algorithmic Parity Benchmark
- **N_train** = 1,000 (deterministic stratified manifest, SHA-256: `af67bbd6...`)
- **Models**: 8 classical + 3 quantum (VQC, QSVM, Hybrid QNN)
- **Purpose**: Eliminates training data volume as confound — tests representational power
- **Feature space**: 22 clinical features → 4 quantum features (supervised LDA + PCA)

---

### 2.5 Quantum Computing Engine (`src/quantum/circuit.py`)

#### Quantum State Representation
An N-qubit system is modeled as a complex statevector of dimension 2^N:

```python
# N=4 qubits → 16-dimensional complex vector
psi = torch.zeros(2**N, dtype=torch.complex128)
psi[0] = 1.0  # |0000⟩ initial state
```

#### Quantum Gate Set (NISQ-compatible)
```
Single-Qubit Gates:
  H (Hadamard): Creates superposition
  Rx(θ): Rotation about X-axis
  Ry(θ): Rotation about Y-axis  ← Primary encoding gate
  Rz(ω): Rotation about Z-axis  ← Variational parameter gate

Two-Qubit Gates:
  CNOT: Controlled-NOT, creates entanglement
  CZ: Controlled-Z

Topology: Circular ring  q₀→q₁→q₂→q₃→q₀
```

Gate matrices:
```
Ry(θ) = [[cos(θ/2), -sin(θ/2)],
          [sin(θ/2),  cos(θ/2)]]

Rz(ω) = [[exp(-iω/2),         0],
          [0,          exp(iω/2)]]
```

#### Feature Encoding (Angle Encoding)
Each clinical feature `xⱼ` is encoded as a rotation angle:
```
φⱼ = 2·arctan(xⱼ)  ∈ [-π, π]
Applied via: Ry(φⱼ) on qubit j
```
This encoding is **bijective** (invertible) and naturally constrains angles to valid Bloch sphere rotations.

#### Measurement
Pauli-Z expectation value for each qubit:
```
⟨Zq⟩ = ⟨ψ|Zq|ψ⟩ = P(qubit q = |0⟩) - P(qubit q = |1⟩) ∈ [-1, 1]
```

#### Analytical Parameter-Shift Gradient
Exact gradient computation without finite differences:
```
∂⟨Z⟩/∂θₖ = [⟨Z⟩(θₖ + π/2) - ⟨Z⟩(θₖ - π/2)] / 2
```
This is the **exact quantum gradient** — hardware-compatible and verified against PyTorch autograd (error < 1×10⁻³).

---

### 2.6 Variational Quantum Classifier (`src/models/vqc_model.py`)

```
Architecture:
  Input: 4 clinical features (post-LDA+PCA reduction)
  ↓
  Angle Encoding: Ry(φⱼ) on qubits 0–3
  ↓
  L variational layers:
    For each layer l:
      Ry(θ_{l,q}) on each qubit q
      Rz(ω_{l,q}) on each qubit q
      CNOT ring: q₀→q₁, q₁→q₂, q₂→q₃, q₃→q₀
  ↓
  Measurement: ⟨Z₀⟩  (first qubit expectation)
  ↓
  Platt sigmoid scaling → P(CVD) ∈ [0,1]

Parameters: 2 × L × N_qubits variational angles
Optimizer: Adam (lr=0.01, β₁=0.9, β₂=0.999)
Loss: Binary Cross-Entropy
```

---

### 2.7 Quantum Support Vector Machine (`src/models/qsvm_model.py`)

```
ZZ-Feature Map:
  U_Φ(x) = exp(i·Σⱼ xⱼZⱼ + Σⱼ<ₖ (π-xⱼ)(π-xₖ)ZⱼZₖ) · H^⊗N

Quantum Kernel:
  K(xᵢ, xⱼ) = |⟨0^N|U†_Φ(xᵢ)·U_Φ(xⱼ)|0^N⟩|²

Properties:
  - Mercer's condition satisfied: K ≥ 0, K(x,x) = 1
  - Encodes non-linear feature correlations in Hilbert space
  - Classical SVM solved with precomputed kernel matrix
```

---

### 2.8 Hybrid Quantum-Classical Neural Network (`src/models/hybrid_qnn.py`)

```
Architecture:
  Input features (22 raw)
       ↓
  Classical Encoder: Linear(22 → 4) + Tanh
       ↓                 ← Backprop flows through here
  Quantum Circuit Layer (4 qubits, L layers)
       ↓                 ← Autograd Jacobian
  Measurement: [⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩]
       ↓
  Classical Decoder: Linear(4 → 1) + Sigmoid
       ↓
  P(CVD) ∈ [0, 1]

Training: End-to-end PyTorch autograd backpropagation
         Gradients flow through classical–quantum boundary
```

---

### 2.9 Mathematical Explainability Engine (`src/explain_risk.py`, `src/quantum/explain.py`)

| Model Type | Explainability Method | Mathematical Guarantee |
|---|---|---|
| CatBoost, LightGBM, XGBoost, RF, HistGB | **TreeSHAP** (Lundberg 2017) | Local accuracy: base + Σφⱼ = ŷ(x) |
| Logistic Regression, Calibrated SVM | **Exact logit attributions** | Δlogitⱼ = βⱼ·zⱼ(x) |
| VQC, Hybrid QNN | **PyTorch autograd Jacobian** | Jⱼ = ∂⟨Z⟩/∂xⱼ |
| QSVM | **Kernel gradient attribution** | ∂K(x,xᵢ)/∂xⱼ |

All quantum attributions are **verified against numerical finite differences** (error < 1×10⁻³).

---

### 2.10 REST API Server (`src/api/server.py`, `app.py`)

#### Endpoints

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/health` | Platform health, model inventory | None |
| GET | `/api/models` | Model registry, thresholds, metrics | None |
| POST | `/api/predict` | Single-patient CVD risk inference | None |
| POST | `/api/explain` | Feature attribution for prediction | None |
| POST | `/api/upload_dataset` | Upload CSV, trigger audit | None |
| GET | `/api/dataset_info` | Current dataset audit results | None |
| POST | `/api/train` | Launch training pipeline | None |
| GET | `/api/benchmark` | Return benchmark results table | None |

#### Input Validation (Physiological Bounds)

```json
{
  "age": {"min": 18, "max": 120, "unit": "years"},
  "height": {"min": 80, "max": 250, "unit": "cm"},
  "weight": {"min": 20, "max": 350, "unit": "kg"},
  "ap_hi": {"min": 50, "max": 300, "unit": "mmHg"},
  "ap_lo": {"min": 30, "max": 200, "unit": "mmHg"},
  "constraint": "ap_hi > ap_lo always required"
}
```

**Security**: Client-submitted thresholds are rejected with HTTP 400 (`CLIENT_THRESHOLD_PROHIBITED`). Only locked OOF thresholds are used.

---

### 2.11 National Health Interoperability Layer (`src/interop/`)

#### ABHA Integration
```python
def verify_abha_id(abha_id: str) -> bool:
    """Luhn-10 checksum verification for ABHA IDs (14-digit format XX-XXXX-XXXX-XXXX)"""
    digits = [int(d) for d in abha_id.replace('-', '')]
    # Luhn-10 standard validation algorithm
    ...
```

#### HL7 FHIR R4 Bundle Generation
```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resourceType": "Patient",
      "identifier": [{"system": "https://abha.nha.gov.in", "value": "14-digit-ABHA"}]
    },
    {
      "resourceType": "Observation",
      "code": {"coding": [{"system": "http://loinc.org", "code": "55284-4"}]},
      "valueQuantity": {"value": 140, "unit": "mmHg", "system": "http://unitsofmeasure.org"}
    },
    {
      "resourceType": "RiskAssessment",
      "method": {"coding": [{"code": "CardioQ-CatBoost-v1.0"}]},
      "prediction": [{"probabilityDecimal": 0.73, "outcome": {"text": "CVD Risk: HIGH"}}]
    }
  ]
}
```

LOINC codes used: `55284-4` (BP), `2085-9` (HDL), `2089-1` (LDL), `39156-5` (BMI), `35200-5` (Cholesterol)  
SNOMED codes used: `413350009` (CVD risk assessment finding), `73211009` (Diabetes mellitus)

---

## 3. Production Artifact Serialization

Each trained model is saved as **decoupled modular artifacts** (no monolithic pickle blobs):

```
artifacts/
├── catboost/
│   ├── preprocessing.joblib    ← IQRClipper + Scaler + FeatureReducer
│   ├── model.joblib            ← Fitted CatBoostClassifier
│   ├── threshold.json          ← {"tau_star": 0.4823, "youden_j": 0.4359}
│   ├── schema.json             ← {"features": [...], "bounds": {...}}
│   └── metadata.json           ← {"trained_at": "...", "framework_versions": {...}}
│
├── vqc/
│   ├── preprocessing.joblib
│   ├── weights.pt              ← PyTorch state_dict (circuit parameters)
│   ├── circuit_config.json     ← {"n_qubits": 4, "n_layers": 3, "ansatz": "ring"}
│   ├── threshold.json
│   └── metadata.json
│
└── hybrid_qnn/
    ├── preprocessing.joblib
    ├── weights.pt
    ├── threshold.json
    └── metadata.json
```

**Stateless reconstruction**: Any model can be reinstantiated from disk without running training:
```python
pipeline = reconstruct_production_pipeline(model_dir="artifacts/catboost/")
risk_prob = pipeline.predict_proba(patient_df)
```

---

## 4. Database Layer (`src/database/repository.py`)

- **Engine**: SQLite with WAL (Write-Ahead Logging) mode for concurrent read/write
- **ACID compliance**: Transactions for all writes
- **Schema**: Stores audit results, training run metadata, benchmark scores, and FHIR export logs
- **Access pattern**: `HybridMethod` decorator allows class-level and instance-level DB access

---

## 5. Testing Architecture (`tests/`)

| Test Module | Coverage Area | Tests |
|---|---|---|
| `test_audit.py` | Data quality audit engine | Schema detection, missing values, biological flags |
| `test_clean.py` | Pre-split data cleaning | BP inversions, BMI outliers, deduplication |
| `test_features.py` | Clinical feature engineering | BMI, MAP, age normalization, Framingham compat |
| `test_preprocess.py` | Preprocessing pipeline | IQR clipping, scaling, feature selection |
| `test_models.py` | All 8 classical models | fit/predict/predict_proba/evaluate contract |
| `test_quantum.py` | All 3 quantum models | Circuit unitarity, gradient accuracy, model contract |
| `test_api.py` | REST API endpoints | /health, /predict, /explain, input validation |
| `test_evaluate.py` | Evaluation engine | Bootstrap CIs, Youden J, threshold locking |
| `test_external_validation.py` | Framingham OOD stress | Zero-refit, domain shift handling |
| `test_serialization.py` | Artifact serialization | Save/load/reconstruct pipelines |
| `test_explain.py` | Explainability engine | SHAP additivity, Jacobian verification |

**Total: 64/64 tests passing (100%) in 4.91 seconds**

---

## 6. Security & Safety Architecture

| Risk | Mitigation |
|------|-----------|
| Client threshold manipulation | HTTP 400 rejection of any `threshold` key in request body |
| Model fabrication at startup | HTTP 503 if artifacts not found — never trains on demand |
| Data leakage via holdout | Physical partition created once, path never passed to training code |
| Feature scrambling | `_prepare_input` enforces strict column alignment by name |
| Silent external data imputation | `MissingExternalFeatureError` — loud failure, no zeros |
| Biological nonsense inputs | Hard physiological bounds on all API inputs |
