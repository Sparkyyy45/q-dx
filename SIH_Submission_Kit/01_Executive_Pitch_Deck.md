# CardioQ — SIH Problem Statement 3
## Hybrid Quantum Machine Learning Platform for Early Disease Detection
### Smart India Hackathon 2025 — Official Submission Pitch Deck

---

## SLIDE 1 — Cover

```
╔══════════════════════════════════════════════════════════════════════╗
║               CARDIOQ: HYBRID QUANTUM-CLASSICAL                      ║
║          CARDIOVASCULAR DISEASE PREDICTION PLATFORM                  ║
║                                                                      ║
║  Smart India Hackathon 2025 | Problem Statement 3                    ║
║  Ministry of Health & Family Welfare | MoHFW                         ║
║                                                                      ║
║  Team: [Your Team Name]                                              ║
║  Institution: [Your College / University]                            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## SLIDE 2 — The Problem

### 17.9 Million People Die of CVD Every Year — Most Were Predictable

> India loses ₹6.2 lakh crore annually to cardiovascular disease (CVD) — yet **60–70% of cardiac events are preventable** if detected early enough.

**Why Existing Solutions Fall Short:**

| Challenge | Classical ML Limitation |
|-----------|------------------------|
| High-dimensional biomedical data (genomics, EHR, imaging) | Curse of dimensionality — performance degrades |
| Noisy, incomplete clinical records | Overfitting to spurious correlations |
| Unexplainable black-box outputs | Clinicians cannot trust or act on outputs |
| No interoperability with national health infrastructure | Siloed, non-deployable research prototypes |
| Classical algorithms plateau at pattern complexity | Cannot capture non-linear quantum correlations |

**Our Thesis:** Quantum computing offers a fundamentally new representational capacity through superposition and entanglement — enabling models that may identify biomedical patterns invisible to classical algorithms.

---

## SLIDE 3 — Our Solution: CardioQ

### CardioQ is a Production-Grade Hybrid Quantum-Classical CVD Risk Platform

**What it is:**
- A rigorously engineered dual-track machine learning platform
- 8 classical + 3 quantum models trained and benchmarked side-by-side
- Full clinical explainability — TreeSHAP, Quantum Jacobians, Odds Ratios
- An interactive web dashboard for clinicians
- REST API with physiological input validation
- ABHA / HL7 FHIR R4 interoperability
- Deployed, tested, and running now

**Built on N = 70,000 patient records** from the largest public CVD dataset, with **external validation against the Framingham Heart Study (N = 4,240)**.

---

## SLIDE 4 — Innovation Pillars

### Four Technical Pillars That Make CardioQ Different

```
     ┌─────────────────┐    ┌─────────────────┐
     │  QUANTUM ENGINE  │    │ CLASSICAL SUITE │
     │                  │    │                  │
     │ • VQC (4-qubit)  │◄──►│ • CatBoost       │
     │ • QSVM (ZZ-Map)  │    │ • LightGBM       │
     │ • Hybrid QNN     │    │ • XGBoost        │
     │ • Parameter-Shift│    │ • Random Forest  │
     │   Gradients      │    │ • Logistic Reg.  │
     └─────────────────┘    └─────────────────┘
              │                        │
              └──────────┬─────────────┘
                         ▼
           ┌─────────────────────────┐
           │  DUAL-TRACK BENCHMARKER  │
           │  Same N=1,000 manifest   │
           │  SHA-256 locked          │
           │  Bootstrap 95% CIs       │
           └─────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │   CLINICAL PLATFORM      │
           │  • REST API              │
           │  • Web Dashboard         │
           │  • FHIR R4 Export        │
           │  • ABHA Integration      │
           └─────────────────────────┘
```

**Pillar 1 — Quantum Computing**: Pure PyTorch/NumPy statevector simulator (no external C++ dependencies). Exact parameter-shift gradients. NISQ-hardware-compatible gate sets.

**Pillar 2 — Leakage-Safe ML Engineering**: Pre-split data cleaning, fold-isolated preprocessing, OOF threshold locking via Youden's J. Zero data fabrication policy.

**Pillar 3 — Mathematical Explainability**: TreeSHAP for tree models, exact linear logit attributions, PyTorch autograd Jacobians for quantum circuits — all verified against finite differences.

**Pillar 4 — National Health Interoperability**: ABHA ID Luhn-10 verification, HL7 FHIR R4 bundle generation with LOINC/SNOMED coding, ICMR clinical triage tiers.

---

## SLIDE 5 — Performance Results

### Head-to-Head: 11 Models on 13,741 Holdout Patients

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|-------|---------|--------|-------------|-------------|
| **CatBoost (Champion)** | **0.7969** | **0.7766** | 0.6996 | 0.7721 |
| Logistic Regression | 0.7928 | 0.7687 | 0.6769 | 0.7866 |
| Random Forest | 0.7925 | 0.7715 | 0.6986 | 0.7754 |
| Calibrated SVM | 0.7920 | 0.7677 | 0.6797 | 0.7853 |
| Gradient Boosting | 0.7879 | 0.7696 | 0.6996 | 0.7648 |
| MLP | 0.7866 | 0.7630 | 0.6975 | 0.7526 |
| XGBoost | 0.7866 | 0.7688 | 0.7004 | 0.7564 |
| LightGBM | 0.7833 | 0.7645 | 0.7006 | 0.7554 |
| **Hybrid QNN** | **0.7775** | **0.7586** | **0.7042** | **0.7455** |
| VQC | 0.7306 | 0.7296 | 0.6460 | 0.7115 |
| QSVM | 0.7302 | 0.7207 | 0.6445 | 0.7404 |

> **Key Finding**: Hybrid QNN achieves the **highest sensitivity (0.7042)** of all models — clinically critical for screening (minimize false negatives). Under equal N=1,000 training budget, quantum-classical hybrid narrows the gap to within 0.019 AUC of the full-data classical champion.

---

## SLIDE 6 — System Architecture (Overview)

```
Raw Data (70,000 rows)
       │
       ▼
┌─────────────────────────────────────────┐
│          PRE-SPLIT DATA AUDIT            │
│  Duplicates, BP inversions, BMI outliers │
└──────────────┬──────────────────────────┘
               │ Stratified 80/20 Split
       ┌───────┴───────┐
       ▼               ▼
  DEV (54,961)   HOLDOUT (13,741) -- Never touched until final eval
       │
       ├──── Track A ──► 8 Classical Models → OOF Threshold Lock → Test Eval
       │
       └──── Track B ──► 8 Classical + 3 Quantum (N=1000 budget, SHA-256 locked)
               │
               ▼
      ┌──────────────────────┐
      │  Quantum Feature Proj │ 22→4 features via supervised LDA + PCA
      │  Angle Encoding       │ φj = 2·arctan(xj)
      │  Variational Ansatz   │ Ry(θ)Rz(ω) + CNOT ring
      │  Measurement          │ <Z> in [-1, 1]
      │  Parameter-Shift Grad │ ∂<Z>/∂θk exact
      └──────────────────────┘
               │
               ▼
      ┌──────────────────────┐
      │  REST API / Dashboard │
      │  ABHA + FHIR R4       │
      └──────────────────────┘
```

---

## SLIDE 7 — Impact & Deployability

### Why This Can Actually Be Deployed in India's Healthcare System

| Factor | CardioQ Solution |
|--------|-----------------|
| **NHA Alignment** | ABHA ID integration, PM-JAY compatible risk tiers |
| **FHIR R4 Compliance** | HL7 standard bundles for HIS/EMR portability |
| **No Installation Required** | `python app.py --port 8080` — runs on any government server |
| **Explainability for Clinicians** | Feature attribution for every prediction, no black-box |
| **Offline-First** | Zero external API dependencies at inference time |
| **Quantum-Ready** | Circuit design is IBM Eagle / IonQ / Rigetti compatible |
| **Clinical Safety** | Prospective threshold locking — no post-hoc manipulation |

---

## SLIDE 8 — Team & Ask

### What We Built

- **11 complete ML models** (8 classical + 3 quantum) — all production-serialized
- **Full quantum statevector simulator** from scratch in PyTorch
- **Interactive web dashboard** with real-time risk prediction
- **64/64 automated tests passing** (100%)
- **Complete SIH documentation package**

### What We Are Asking For

Mentorship and resources to:
1. Run CardioQ on real IBM/IonQ quantum hardware via IBMQ Network
2. Expand to multi-disease detection (diabetes, kidney disease, cancer screening)
3. Pilot deployment at a government PHC/CHC for real-world clinical validation
4. Integration with NHA's ABHA ecosystem at scale
