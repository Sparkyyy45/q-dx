# CardioQ: System Architecture Diagrams
## SIH Problem Statement 3 — Technical Document 06
### Visual Architecture Reference for Judges, Evaluators & Developers

---

## Diagram 1: Complete Platform Architecture (Mermaid)

```mermaid
flowchart TD
    subgraph "DATA LAYER"
        D1["Raw Patient Records (70,000 rows)"] --> A1["Dataset Audit & Sanity Verification\n(src/audit.py)"]
        A1 --> C1["Pre-Split Data Cleaning\nDrop Duplicates, Inverted BP, Extreme BMI\n(src/clean.py)"]
        C1 --> S1["Stratified 80/20 Partition\n54,961 Dev / 13,741 Holdout Test"]
    end

    subgraph "DUAL-TRACK ML ENGINE"
        S1 --> T_A["Track A: Full Dev Cohort N=54,961"]
        S1 --> T_B["Track B: Paired Manifest N=1,000\nSHA-256: af67bbd6..."]
        T_A --> FE_A["Clinical Feature Engineering + Fold-Isolated Preprocessing\n22 features | IQRClipper | RobustScaler"]
        T_B --> FE_B["Quantum Feature Projection\n22→4 features via LDA + PCA"]
        FE_A --> M_CLASSICAL["8 Classical Models\nCatBoost | LightGBM | XGBoost | HistGB\nRF | SVM | LR | MLP"]
        FE_B --> M_PARITY["Parity Suite\n8 Classical + 3 Quantum\nVQC | QSVM | Hybrid QNN"]
    end

    subgraph "QUANTUM ENGINE (src/quantum/)"
        M_PARITY --> QC["Statevector Simulator\nPyTorch complex128 | N=4 qubits"]
        QC --> VQC["VQC: Angle Encoding + Variational Ansatz\nParameter-Shift Gradients | Adam Optimizer"]
        QC --> QSVM["QSVM: ZZ-Feature Map\nQuantum Kernel Gram Matrix | Precomputed SVC"]
        QC --> QNN["Hybrid QNN: Classical Encoder + Quantum Layer\nEnd-to-end Backpropagation | PyTorch autograd"]
    end

    subgraph "THRESHOLD & EVALUATION"
        M_CLASSICAL --> OOF["OOF Cross-Validation Predictions\n3-Fold Stratified"]
        M_PARITY --> OOF
        OOF --> LOCK["Prospective Threshold Locking\nτ* = argmax_τ Youden's J on OOF only"]
        LOCK --> TEST["Holdout Test Evaluation N=13,741\n1,000-Bootstrap 95% CIs"]
        LOCK --> OOD["Framingham OOD Stress Test N=4,240\nZero-Refit | Zero-Retune"]
    end

    subgraph "SERIALIZATION"
        LOCK --> SERIAL["Decoupled Artifact Bundles\npreprocessing.joblib | model.joblib | weights.pt\nthreshold.json | schema.json | metadata.json"]
    end

    subgraph "PLATFORM & API (src/api/)"
        SERIAL --> API["Hardened REST API\nFlask | Physiological Input Validation\nHTTP 400: Client Threshold Prohibited\nHTTP 503: Model Not Loaded"]
        API --> EXP["Clinical Explainability Engine\nTreeSHAP | Linear Logit | Quantum Jacobian"]
        API --> FHIR["FHIR R4 Bundle Generator\nLOINC | SNOMED | ABHA Luhn-10"]
        API --> DASH["Web Dashboard\nGlassmorphism UI | Real-time Risk Gauge\nModel Comparison | Feature Attribution"]
    end

    subgraph "INTEROPERABILITY"
        FHIR --> ABHA["ABHA ID Validation\nLuhn-10 Checksum | NHA Standard"]
        FHIR --> HIS["HIS/EMR Export\nHL7 FHIR R4 Bundle | ABDM Compatible"]
    end
```

---

## Diagram 2: Quantum Circuit Architecture

```mermaid
flowchart LR
    subgraph "22 Clinical Features"
        F1["age, bmi, map, pulse_pressure,\ncardiovascular_strain, ..."]
    end

    subgraph "Feature Reduction"
        F1 --> LDA["Supervised LDA\n22 → 4\n(training only)"]
        LDA --> PCA["PCA Whitening\n4 → 4\n(preserves 95%+ variance)"]
    end

    subgraph "Angle Encoding (Ry gates)"
        PCA --> E0["qubit q₀: Ry(2·arctan(x₀))"]
        PCA --> E1["qubit q₁: Ry(2·arctan(x₁))"]
        PCA --> E2["qubit q₂: Ry(2·arctan(x₂))"]
        PCA --> E3["qubit q₃: Ry(2·arctan(x₃))"]
    end

    subgraph "Variational Layer (×L layers)"
        E0 --> V0["Ry(θ₀)·Rz(ω₀)"]
        E1 --> V1["Ry(θ₁)·Rz(ω₁)"]
        E2 --> V2["Ry(θ₂)·Rz(ω₂)"]
        E3 --> V3["Ry(θ₃)·Rz(ω₃)"]
        V0 --> CNOT1["CNOT q₀→q₁"]
        V1 --> CNOT2["CNOT q₁→q₂"]
        V2 --> CNOT3["CNOT q₂→q₃"]
        V3 --> CNOT4["CNOT q₃→q₀"]
    end

    subgraph "Measurement"
        CNOT1 --> M["⟨Z₀⟩ ∈ [-1, 1]"]
        CNOT4 --> M
        M --> PLATT["Platt Sigmoid Scaling"]
        PLATT --> P["P(CVD) ∈ [0, 1]"]
    end
```

---

## Diagram 3: Data Partition & Leakage Prevention

```mermaid
flowchart TD
    RAW["Raw Dataset\n70,000 rows"] --> CLEAN["Pre-Split Cleaning\n68,203 rows remain"]
    CLEAN --> SPLIT["Stratified 80/20 Split\nrandom_state=42"]
    SPLIT --> DEV["Development Set\n54,561 rows\n49.5% CVD"]
    SPLIT --> HOLD["Holdout Test Set\n13,741 rows\nNEVER ACCESSED DURING TRAINING"]

    DEV --> CV["3-Fold Stratified Cross-Validation"]

    CV --> F1["Fold 1\nTrain: 36,374\nVal: 18,187"]
    CV --> F2["Fold 2\nTrain: 36,374\nVal: 18,187"]
    CV --> F3["Fold 3\nTrain: 36,373\nVal: 18,188"]

    F1 --> PP1["Preprocessing fit on TRAIN fold only\nIQRClipper | RobustScaler | FeatureReducer"]
    F2 --> PP2["Preprocessing fit on TRAIN fold only"]
    F3 --> PP3["Preprocessing fit on TRAIN fold only"]

    PP1 --> OOF1["OOF Predictions\n(18,187 samples)"]
    PP2 --> OOF2["OOF Predictions\n(18,187 samples)"]
    PP3 --> OOF3["OOF Predictions\n(18,188 samples)"]

    OOF1 --> THRESH["Youden's J Threshold Locking\nτ* = argmax_τ (TPR-FPR) on OOF\nWRITTEN TO threshold.json"]
    OOF2 --> THRESH
    OOF3 --> THRESH

    THRESH --> FINAL_TRAIN["Final Model Trained on Full DEV\n54,561 rows"]
    FINAL_TRAIN --> EVAL["Holdout Evaluation\n13,741 rows @ locked τ*\n1,000-Bootstrap 95% CIs"]
    HOLD --> EVAL
```

---

## Diagram 4: API Request Flow

```mermaid
sequenceDiagram
    participant C as Clinician/Client
    participant API as CardioQ API Server
    participant DB as SQLite DB
    participant ML as ML Engine
    participant EXP as Explain Engine

    C->>API: POST /api/predict {"age": 55, "ap_hi": 145, ...}

    API->>API: Validate physiological bounds
    Note over API: age ∈ [18,120], ap_hi > ap_lo, etc.

    alt Invalid bounds
        API-->>C: 400 PHYSIOLOGICAL_BOUNDS_VIOLATION
    end

    API->>API: Check for "threshold" key in request
    alt Client threshold submitted
        API-->>C: 400 CLIENT_THRESHOLD_PROHIBITED
    end

    API->>ML: Load preprocessed features
    ML->>ML: ClinicalFeatureEngineer.transform(patient)
    ML->>ML: IQRClipper.transform | RobustScaler.transform
    ML->>ML: model.predict_proba(X)

    ML-->>API: risk_probability = 0.734

    API->>DB: Load locked_threshold from threshold.json
    DB-->>API: tau_star = 0.4823

    API->>API: Assign risk tier
    Note over API: P > 0.20 → HIGH tier (ICMR)

    API->>DB: Log prediction (session_id, timestamp)

    API-->>C: {"risk_probability": 0.734, "risk_tier": "HIGH",\n"locked_threshold": 0.4823, "disclaimer": "..."}

    C->>API: POST /api/explain {"patient": {...}, "model": "catboost"}
    API->>EXP: shap.TreeExplainer(model).shap_values(X)
    EXP-->>API: attributions = {"ap_hi": 0.092, "age": 0.081, ...}
    API-->>C: {"method": "TreeSHAP", "attributions": {...}, "additivity_check": "PASS"}
```

---

## Diagram 5: Hybrid QNN Forward Pass

```mermaid
flowchart LR
    X["Patient Features\n22-dimensional"] --> ENC["Classical Encoder\nLinear(22→4) + Tanh"]
    ENC --> QIN["Quantum Input\n4-dimensional z"]

    subgraph "Quantum Circuit (4 qubits)"
        QIN --> AE["Angle Encoding\nRy(2·arctan(zⱼ)) on each qubit"]
        AE --> VL1["Variational Layer 1\nRy(θ)Rz(ω) + CNOT ring"]
        VL1 --> VL2["Variational Layer 2\nRy(θ)Rz(ω) + CNOT ring"]
        VL2 --> VL3["Variational Layer 3\nRy(θ)Rz(ω) + CNOT ring"]
        VL3 --> MEAS["Measurement\n[⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩]"]
    end

    MEAS --> DEC["Classical Decoder\nLinear(4→1) + Sigmoid"]
    DEC --> PROB["P(CVD) ∈ [0,1]"]

    PROB --> LOSS["BCE Loss"]
    LOSS --> GRAD["PyTorch autograd\n∂L/∂θ through quantum-classical boundary"]
    GRAD --> ADAM["Adam Optimizer\nUpdate classical weights + circuit parameters"]
```

---

## Diagram 6: ABHA & FHIR Interoperability Flow

```mermaid
flowchart TD
    INPUT["Patient Assessment Input\n(Clinician enters vitals)"] --> ABHA_VAL["ABHA ID Validation\nLuhn-10 checksum verification"]

    ABHA_VAL --> |"Valid ABHA"| PREDICT["CardioQ Risk Prediction\nCatBoost | VQC | Hybrid QNN"]
    ABHA_VAL --> |"Invalid ABHA"| ERR["Error: Invalid ABHA ID format"]

    PREDICT --> FHIR_GEN["FHIR R4 Bundle Generation"]

    FHIR_GEN --> PAT["Patient Resource\n- ABHA identifier\n- Demographics"]

    FHIR_GEN --> OBS["Observation Resources\n- BP: LOINC 55284-4\n- BMI: LOINC 39156-5\n- Cholesterol: LOINC 35200-5"]

    FHIR_GEN --> RISK["RiskAssessment Resource\n- Method: CardioQ-CatBoost-v1.0\n- Probability: 0.734\n- Outcome: SNOMED 413350009\n- Risk tier: HIGH"]

    PAT --> BUNDLE["FHIR R4 Bundle\n(application/fhir+json)"]
    OBS --> BUNDLE
    RISK --> BUNDLE

    BUNDLE --> |"POST to HIS"| HIS_EMR["Hospital HIS/EMR\nDHIS2 | OpenMRS | NHA HCX"]
    BUNDLE --> |"Download"| PDF["Clinical Report PDF\n(for paper-based systems)"]
```

---

## Diagram 7: Explainability Architecture

```mermaid
flowchart TD
    MODEL_TYPE{"Model Type?"} --> TREE["Tree-Based Model\nCatBoost | LightGBM | XGBoost | RF | HistGB"]
    MODEL_TYPE --> LINEAR["Linear Model\nLogistic Regression | Calibrated SVM"]
    MODEL_TYPE --> QUANTUM["Quantum Model\nVQC | Hybrid QNN | QSVM"]

    TREE --> SHAP["TreeSHAP\nshap.TreeExplainer(model).shap_values(X)\nComplexity: O(TLD)"]
    SHAP --> SHAP_OUT["Local Attributions φⱼ\nGuarantee: base + Σφⱼ = ŷ(x)\nAdditivity verified per prediction"]

    LINEAR --> LOGIT["Exact Logit Attribution\nΔlogitⱼ = βⱼ · zⱼ(x)\n(scaled feature value × coefficient)"]
    LOGIT --> LOGIT_OUT["Signed Logit Contributions\n+: increases CVD risk\n-: decreases CVD risk"]

    QUANTUM --> JAC["PyTorch Autograd Jacobian\nJⱼ = ∂⟨Z⟩/∂xⱼ"]
    JAC --> JAC_VER["Finite Difference Verification\n|Jⱼ_autograd - Jⱼ_FD| < 1e-3"]
    JAC_VER --> JAC_OUT["Quantum Sensitivity Map\nCircuit output sensitivity to each input feature"]

    SHAP_OUT --> UI["Risk Driver Waterfall Chart\nTop 5 positive risk factors\nTop 3 protective factors\nModifiable vs non-modifiable"]
    LOGIT_OUT --> UI
    JAC_OUT --> UI
```
