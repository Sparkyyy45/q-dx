# CardioQ: Technical Specifications Document
## SIH Problem Statement 3 — Technical Document 05
### Full Stack Engineering & Quantum Hardware Specifications

---

## 1. Platform Identity

| Attribute | Value |
|---|---|
| **Platform Name** | CardioQ |
| **Version** | 1.0.0 — Production |
| **Problem Statement** | SIH PS-3: Hybrid Quantum Machine Learning for Early Disease Detection |
| **Disease Target** | Cardiovascular Disease (CVD) |
| **Primary Dataset** | Kaggle CVD Dataset (N = 70,000) |
| **External Validation** | Framingham Heart Study (N = 4,240) |
| **License** | MIT |
| **Language** | Python 3.10+ |
| **Deployment** | `python app.py --port 8080` |
| **Test Coverage** | 64/64 tests passing (100%) |

---

## 2. Hardware Requirements

### Minimum (Demo/Hackathon)
| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores, 2.0 GHz | 4 cores, 3.0 GHz |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB | 5 GB |
| OS | Windows 10 / Ubuntu 18.04 / macOS 12 | Ubuntu 22.04 LTS |
| Network | Local (no internet required for inference) | HTTPS for ABDM integration |

### Production Scale
| Component | Specification |
|---|---|
| CPU | 16 cores (Intel Xeon / AMD EPYC) |
| RAM | 64 GB ECC |
| Storage | 500 GB SSD (NVMe preferred) |
| Network | 1 Gbps, HTTPS/TLS 1.3 |
| OS | Ubuntu 22.04 LTS (government standard) |

---

## 3. Software Stack

### Core Dependencies

| Package | Version | Purpose | Why This Version |
|---|---|---|---|
| Python | ≥ 3.10 | Runtime | f-strings, match/case, typing improvements |
| PyTorch | ≥ 2.0.0 | Quantum simulator, QNN | `torch.complex128` stability, improved autograd |
| scikit-learn | ≥ 1.3.0 | Classical ML, pipelines | `set_output(transform="pandas")` API |
| CatBoost | ≥ 1.2.0 | Champion classifier | Ordered target statistics for categoricals |
| LightGBM | ≥ 4.0.0 | Gradient boosting | DART mode stability |
| XGBoost | ≥ 2.0.0 | Gradient boosting | GPU-ready, tree method improvements |
| SHAP | ≥ 0.43.0 | TreeSHAP explainability | Waterfall plot API |
| NumPy | ≥ 1.24.0 | Statevector operations | `complex128` dtype stability |
| Flask | ≥ 3.0.0 | REST API server | Blueprint support, async views |
| pandas | ≥ 2.0.0 | Dataframe operations | Copy-on-write mode |
| SQLite3 | ≥ 3.40 (system) | ACID database | WAL mode, JSON support |
| joblib | ≥ 1.3.0 | Model serialization | Parallel artifact dumps |
| scipy | ≥ 1.11.0 | Bootstrap statistics, signal processing | `stats.bootstrap` API |

### Optional (Quantum Hardware Bridge)

| Package | Version | Purpose |
|---|---|---|
| qiskit | ≥ 1.0.0 | IBM Quantum hardware bridge |
| qiskit-aer | ≥ 0.14 | Noise model simulation |
| qiskit-ibm-runtime | ≥ 0.20 | IBMQ Network execution |

---

## 4. Quantum Architecture Specifications

### 4.1 Statevector Simulator (`src/quantum/circuit.py`)

| Parameter | Value |
|---|---|
| Representation | Complex tensor `(2^N,)` — `torch.complex128` |
| Max qubits (RAM) | N=20 (1M complex numbers, ~16 MB) |
| Operational qubits | N=4 (default) |
| Gate set | H, Rx, Ry, Rz, CNOT, CZ, X, Y, Z, I |
| Entanglement topology | Circular ring (NISQ-optimised) |
| Measurement | Pauli-Z expectation: `⟨Z_q⟩ ∈ [-1.0, 1.0]` |
| Gradient method | Parameter-shift rule (analytical, hardware-compatible) |
| Gradient verification | Max error vs autograd: < 1×10⁻³ |
| Simulation latency | N=4: < 0.5ms | N=8: < 5ms | N=12: < 80ms |

### 4.2 VQC Specifications

| Parameter | Value |
|---|---|
| N qubits | 4 |
| N variational layers | 3 |
| Feature encoding | Angle encoding: φⱼ = 2·arctan(xⱼ) |
| Encoding gate | Ry(φⱼ) per qubit |
| Variational gates | Ry(θ_{l,q}) then Rz(ω_{l,q}) per qubit per layer |
| Entangling gate | CNOT, circular ring topology |
| Output measurement | ⟨Z₀⟩ on qubit 0 |
| Calibration | Platt sigmoid scaling |
| Optimizer | Adam (lr=0.01, β₁=0.9, β₂=0.999, ε=1×10⁻⁸) |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | Validation loss plateau (patience=5) |
| Total parameters | 24 (2 × 3 layers × 4 qubits) |
| Track B ROC-AUC | 0.7306 (±0.014) |

### 4.3 QSVM Specifications

| Parameter | Value |
|---|---|
| N qubits | 4 |
| Feature map | ZZ-feature map (Havlíček et al., 2019) |
| Kernel | `K(xᵢ,xⱼ) = \|⟨0^N\|U†_Φ(xᵢ)·U_Φ(xⱼ)\|0^N⟩\|²` |
| Kernel property | Mercer-compliant: K ≥ 0, K(x,x) = 1 |
| SVM backend | `sklearn.svm.SVC(kernel='precomputed')` |
| Regularisation | C = 1.0 (default) |
| Calibration | Platt scaling (3-fold CV) |
| Kernel matrix size | N_train × N_train (constrained to N=1,000 for feasibility) |
| Track B ROC-AUC | 0.7302 (±0.003) |

### 4.4 Hybrid QNN Specifications

| Parameter | Value |
|---|---|
| Classical encoder | `Linear(22 → 4) + Tanh` |
| Quantum layer | 4-qubit VQC (3 layers) |
| Classical decoder | `Linear(4 → 1) + Sigmoid` |
| Training | End-to-end PyTorch autograd backpropagation |
| Optimizer | Adam (lr=5×10⁻³) |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Max epochs | 100 |
| Total parameters | 22×4 + 4 + 24 + 4 + 1 = 121 |
| Track B ROC-AUC | 0.7775 (±0.001) |
| Track B Sensitivity | 0.7042 (highest of all 11 models) |

---

## 5. Classical Model Specifications

| Model | Key Hyperparameters | ROC-AUC (N=54,961 Track A) |
|---|---|---|
| **CatBoost** | depth=6, learning_rate=0.1, iterations=500, l2_leaf_reg=3, cat_features=[gender, cholesterol, gluc] | **0.7969 (±0.004)** |
| **Logistic Regression** | C=1.0, penalty=l2, solver=lbfgs, max_iter=1000 | 0.7928 (±0.003) |
| **Random Forest** | n_estimators=300, max_depth=None, min_samples_leaf=2, bootstrap=True | 0.7925 (±0.004) |
| **Calibrated SVM** | C=1.0, kernel=rbf, gamma=scale, calibration=sigmoid, cv=3 | 0.7920 (±0.003) |
| **HistGradientBoosting** | max_iter=300, learning_rate=0.1, max_depth=5, l2_regularization=0.1 | 0.7879 (±0.003) |
| **MLP** | hidden_layers=(256, 128, 64), activation=relu, dropout=0.3, lr=1×10⁻³ | 0.7866 (±0.007) |
| **XGBoost** | n_estimators=300, max_depth=5, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0 | 0.7866 (±0.002) |
| **LightGBM** | n_estimators=300, num_leaves=31, learning_rate=0.1, reg_alpha=0.1 | 0.7833 (±0.003) |

---

## 6. Feature Engineering Specifications

### 6.1 Raw Features (11 input variables)

| Feature | Type | Units | Clinical Meaning |
|---|---|---|---|
| `age` | Continuous | days → normalized to years | Patient age |
| `gender` | Categorical | 1=female, 2=male | Biological sex |
| `height` | Continuous | cm | Body height |
| `weight` | Continuous | kg | Body weight |
| `ap_hi` | Continuous | mmHg | Systolic blood pressure |
| `ap_lo` | Continuous | mmHg | Diastolic blood pressure |
| `cholesterol` | Ordinal | 1=normal, 2=above, 3=well above | Serum cholesterol tier |
| `gluc` | Ordinal | 1=normal, 2=above, 3=well above | Blood glucose tier |
| `smoke` | Binary | 0/1 | Current smoker |
| `alco` | Binary | 0/1 | Regular alcohol consumption |
| `active` | Binary | 0/1 | Regular physical activity |

### 6.2 Engineered Features (11 additional)

| Feature | Formula | LOINC Code | Clinical Significance |
|---|---|---|---|
| `bmi` | weight / (height/100)² | 39156-5 | WHO obesity classification |
| `map` | (ap_hi + 2·ap_lo) / 3 | - | Mean perfusion pressure |
| `pulse_pressure` | ap_hi - ap_lo | - | Arterial stiffness proxy |
| `age_years` | np.where(age > 120, age/365.25, age) | - | Normalized age |
| `age_sq` | age_years² | - | Non-linear aging risk |
| `cardiovascular_strain` | map × age_years | - | Cumulative vascular stress |
| `bp_interaction` | ap_hi × ap_lo | - | Hypertensive load |
| `cholesterol_risk` | cholesterol × gluc | - | Dyslipidemia-diabetes joint risk |
| `bmi_age` | bmi × age_years | - | Age-adjusted obesity risk |
| `cholesterol_gluc_ratio` | cholesterol / gluc.clip(1) | - | Metabolic syndrome marker |
| `metabolic_synergy` | bmi × cholesterol | - | Adiposity-lipid interaction |

**Total feature space**: 22 features (11 raw + 11 engineered)  
**Quantum feature space**: 4 features (supervised LDA + PCA projection of 22 features)

---

## 7. Data Pipeline Specifications

### 7.1 Dataset Splits

| Partition | N | Fraction | Purpose |
|---|---|---|---|
| Development (Train+Validation) | 54,961 | 80% | Model training, OOF threshold locking |
| Holdout Test | 13,741 | 20% | Final performance evaluation |
| Track B Subsample | 1,000 | 1.43% of dev | Quantum-classical parity benchmark |
| Framingham External | 4,240 | N/A | Out-of-distribution stress test |

### 7.2 Cross-Validation

| Parameter | Value |
|---|---|
| Strategy | Stratified K-Fold |
| K (folds) | 3 |
| Random seed | 42 (fixed globally) |
| Shuffle | True |

### 7.3 Pre-Split Cleaning Thresholds

| Rule | Threshold | Physiological Basis |
|---|---|---|
| BMI minimum | 10 kg/m² | Below anorexia cachexia range |
| BMI maximum | 70 kg/m² | Extreme obesity upper bound |
| Systolic BP > Diastolic BP | ap_hi > ap_lo | Required by physiology |
| Duplicate rows | Exact match (excluding ID) | Data quality |

### 7.4 Track B Manifest Integrity

| Parameter | Value |
|---|---|
| Subsample size | N = 1,000 |
| Strategy | Stratified random sample from development set |
| Random seed | 42 |
| SHA-256 identifier | af67bbd6... (first 8 chars shown) |
| Purpose | Cryptographic guarantee of same rows across classical and quantum benchmarks |

---

## 8. API Specifications

### 8.1 Base Configuration

| Parameter | Value |
|---|---|
| Framework | Flask 3.0+ |
| Default port | 8080 |
| Protocol | HTTP (HTTPS in production) |
| Request format | JSON |
| Response format | JSON |
| Max request size | 100 MB (dataset upload) |
| Timeout | 30s (standard), 300s (training) |

### 8.2 Endpoint Specifications

#### `GET /health`
```json
Response 200:
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-09-07T10:30:00Z",
  "models_loaded": {
    "catboost": true,
    "logistic_regression": true,
    "vqc": true,
    "hybrid_qnn": true
  },
  "dataset": {
    "loaded": true,
    "n_rows": 68203,
    "n_features": 22
  }
}
```

#### `POST /api/predict`
```json
Request:
{
  "age": 55,
  "gender": 2,
  "height": 175,
  "weight": 85,
  "ap_hi": 145,
  "ap_lo": 90,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1,
  "model": "catboost"
}

Response 200:
{
  "model": "catboost",
  "risk_probability": 0.7341,
  "risk_tier": "HIGH",
  "locked_threshold": 0.4823,
  "classification": "CVD_RISK_POSITIVE",
  "confidence_band": [0.69, 0.78],
  "disclaimer": "CLINICAL DECISION SUPPORT ONLY — NOT FOR AUTONOMOUS DIAGNOSIS"
}
```

#### `POST /api/explain`
```json
Response 200:
{
  "model": "catboost",
  "method": "TreeSHAP",
  "base_value": 0.494,
  "prediction": 0.7341,
  "attributions": {
    "ap_hi": 0.0923,
    "age_years": 0.0812,
    "cardiovascular_strain": 0.0734,
    "bmi": 0.0421,
    "cholesterol": 0.0318,
    "pulse_pressure": -0.0127
  },
  "top_modifiable_risks": ["ap_hi", "bmi"],
  "additivity_check": "PASS (base + sum = prediction: 0.004 error)"
}
```

### 8.3 Error Codes

| HTTP Status | Code | Trigger |
|---|---|---|
| 400 | `CLIENT_THRESHOLD_PROHIBITED` | Request body contains `threshold` key |
| 400 | `PHYSIOLOGICAL_BOUNDS_VIOLATION` | Age <18 or >120, BMI <10 or >70, etc. |
| 400 | `MISSING_REQUIRED_FIELD` | Required feature absent from request |
| 400 | `INVALID_DATA_TYPE` | Non-numeric value in numeric field |
| 404 | `MODEL_NOT_FOUND` | Requested model not in registry |
| 503 | `MODEL_NOT_LOADED` | Model artifact files not found on disk |

---

## 9. Serialization Specifications

### 9.1 Classical Model Artifacts

```
artifacts/{model_name}/
├── preprocessing.joblib       — IQRClipper + Scaler + FeatureReducer
├── model.joblib               — Fitted classifier object
├── threshold.json             — {"tau_star": float, "youden_j": float, "method": "OOF"}
├── schema.json                — {"features": [...], "types": {...}, "bounds": {...}}
└── metadata.json              — {"trained_at": ISO8601, "framework_versions": {...}, 
                                   "training_rows": int, "test_roc_auc": float}
```

### 9.2 Quantum Model Artifacts

```
artifacts/{quantum_model}/
├── preprocessing.joblib       — Feature reduction pipeline (22→4 features)
├── weights.pt                 — torch.nn.Module state_dict (circuit parameters)
├── circuit_config.json        — {"n_qubits": 4, "n_layers": 3, "ansatz": "ring",
                                   "encoding": "angle", "backend": "statevector"}
├── threshold.json             — {"tau_star": float, "youden_j": float}
└── metadata.json              — {"trained_at": ISO8601, "n_parameters": 24, 
                                   "gradient_method": "parameter_shift",
                                   "simulation_backend": "pytorch_numpy"}
```

---

## 10. Security Specifications

| Security Property | Specification | Implementation |
|---|---|---|
| **Threshold immutability** | Client cannot modify decision threshold | HTTP 400 if `threshold` in request |
| **Model-on-demand prohibition** | Server never trains models to fulfill requests | HTTP 503 if artifacts missing |
| **Data integrity** | Pre-split cleaning is irreversible | Cleaned data hash stored in metadata |
| **Audit trail** | All predictions logged with session ID | SQLite WAL with append-only log |
| **Input sanitisation** | All numeric inputs validated against physiological bounds | Server-side validation before inference |
| **No external calls** | Inference is fully offline | Zero external API calls in inference path |

---

## 11. Benchmark Summary (Official, Verified)

### Track A: Full Development Cohort (N_train = 54,961, N_test = 13,741)

| Model | ROC-AUC | PR-AUC | Brier | ECE | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.7969 ±0.004 | 0.7766 ±0.002 | 0.1823 | 0.0140 | 0.6996 | 0.7721 | 0.7254 |
| Logistic Regression | 0.7928 ±0.003 | 0.7687 ±0.002 | 0.1854 | 0.0263 | 0.6769 | 0.7866 | 0.7157 |
| Random Forest | 0.7925 ±0.004 | 0.7715 ±0.005 | 0.1843 | 0.0164 | 0.6986 | 0.7754 | 0.7260 |
| Calibrated SVM | 0.7920 ±0.003 | 0.7677 ±0.002 | 0.1857 | 0.0307 | 0.6797 | 0.7853 | 0.7171 |
| HistGradientBoosting | 0.7879 ±0.003 | 0.7696 ±0.002 | 0.1874 | 0.0303 | 0.6996 | 0.7648 | 0.7226 |
| MLP | 0.7866 ±0.007 | 0.7630 ±0.002 | 0.1872 | 0.0249 | 0.6975 | 0.7526 | 0.7168 |
| XGBoost | 0.7866 ±0.002 | 0.7688 ±0.000 | 0.1885 | 0.0356 | 0.7004 | 0.7564 | 0.7200 |
| LightGBM | 0.7833 ±0.003 | 0.7645 ±0.004 | 0.1907 | 0.0456 | 0.7006 | 0.7554 | 0.7198 |

### Track B: Parity Benchmark (N_train = 1,000)

| Model | ROC-AUC | PR-AUC | Brier | Sensitivity | Specificity |
|---|---|---|---|---|---|
| CatBoost | 0.7802 ±0.005 | 0.7601 ±0.004 | 0.1914 | 0.6841 | 0.7583 |
| **Hybrid QNN** | **0.7775 ±0.001** | **0.7586 ±0.003** | 0.1900 | **0.7042** | 0.7455 |
| VQC | 0.7306 ±0.014 | 0.7296 ±0.009 | 0.2087 | 0.6460 | 0.7115 |
| QSVM | 0.7302 ±0.003 | 0.7207 ±0.003 | 0.2079 | 0.6445 | 0.7404 |

**Key findings**:
- Under equal training budget (N=1,000), Hybrid QNN is within 0.003 AUC of CatBoost
- Hybrid QNN achieves the **highest sensitivity** of all 11 models — most important for screening
- VQC and QSVM show meaningful but smaller performance — N=1,000 may be below their convergence threshold
