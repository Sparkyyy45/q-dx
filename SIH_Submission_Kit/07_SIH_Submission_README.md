# CardioQ: SIH Submission — README & Navigation Guide
## How to Evaluate & Run This Platform

---

## Quick Start

```bash
# 1. Navigate to project root
cd "heartdisease"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the platform
python app.py --port 8080

# 4. Open browser
# → http://localhost:8080
```

**Platform is running when you see:**
```
╔═══════════════════════════════════════════╗
║  CardioQ Platform — Hybrid Quantum ML     ║
║  http://localhost:8080                    ║
║  11 models available | 64/64 tests pass   ║
╚═══════════════════════════════════════════╝
```

---

## Submission Kit Contents

| File | Purpose | Audience |
|---|---|---|
| [01_Executive_Pitch_Deck.md](./01_Executive_Pitch_Deck.md) | Problem, Solution, Performance Results, Impact | All judges, general audience |
| [02_Technical_Architecture.md](./02_Technical_Architecture.md) | Full platform engineering specification | Technical judges, ML scientists |
| [03_Methodology_and_Rationale.md](./03_Methodology_and_Rationale.md) | Why every design decision was made | AI/ML researchers, quantum engineers |
| [04_Judge_QA_Preparation.md](./04_Judge_QA_Preparation.md) | 25 expert Q&A covering ML, Quantum, Clinical, Deployment | All judges (preparation document) |
| [05_Technical_Specifications.md](./05_Technical_Specifications.md) | Hardware, software, API, and model specs | Technical reviewers, deployment team |
| [06_Architecture_Diagrams.md](./06_Architecture_Diagrams.md) | Full Mermaid visual diagrams of all subsystems | Presentation, visual learners |
| [07_SIH_Submission_README.md](./07_SIH_Submission_README.md) | This file — navigation and quick start | All |

---

## What This Platform Delivers Against SIH PS-3

SIH Problem Statement 3 required: *"Hybrid Quantum Machine Learning Platform for Early Disease Detection"*

| PS-3 Requirement | CardioQ Delivery | Status |
|---|---|---|
| Hybrid Quantum-Classical ML | VQC + QSVM + Hybrid QNN + 8 Classical models | ✅ Complete |
| Quantum circuit with feature encoding | Angle encoding + ZZ-feature map + variational ansatz | ✅ Complete |
| Classical ML baseline comparison | 8 models with Bootstrap 95% CIs on N=13,741 holdout | ✅ Complete |
| Explainability for clinical trust | TreeSHAP + Logit attribution + Quantum Jacobian | ✅ Complete |
| Early disease detection focus | CVD (primary) + Framingham external validation | ✅ Complete |
| Interactive software platform | Web dashboard + REST API + FHIR R4 | ✅ Complete |
| Medical data preprocessing | Clinical audit + leakage-safe pipeline | ✅ Complete |
| Decision threshold optimization | Youden's J on OOF, prospectively locked | ✅ Complete |
| National health interoperability | ABHA + HL7 FHIR R4 + ICMR triage tiers | ✅ Complete |
| External validation | Framingham Heart Study (N=4,240) OOD stress test | ✅ Complete |

---

## Running Tests

```bash
# Run full test suite (64 tests)
python run_tests.py

# Expected output:
# ====== 64 passed in 4.91s ======
# Pass rate: 100.0%
```

---

## Key Performance Numbers

| Metric | Value | Context |
|---|---|---|
| CatBoost ROC-AUC | 0.7969 ±0.004 | N=13,741 holdout, 1000-bootstrap CI |
| Hybrid QNN ROC-AUC | 0.7775 ±0.001 | N=1,000 budget parity benchmark |
| Hybrid QNN Sensitivity | 0.7042 | Highest of all 11 models |
| VQC gradient accuracy | < 1×10⁻³ | Parameter-shift vs autograd verification |
| Test pass rate | 64/64 (100%) | 11 test modules, 4.91s |
| API response time | < 50ms | Including FHIR R4 generation |

---

## File Structure

```
heartdisease/
├── app.py                          ← Entry point: python app.py --port 8080
├── run_all.py                      ← Full training pipeline (all 11 models)
├── run_tests.py                    ← Test runner
├── requirements.txt                ← Dependencies
│
├── src/
│   ├── api/
│   │   ├── server.py               ← Flask REST API
│   │   ├── templates/index.html    ← Web dashboard
│   │   └── static/                 ← JS, CSS assets
│   ├── quantum/
│   │   ├── circuit.py              ← Custom statevector simulator
│   │   ├── explain.py              ← Quantum Jacobian explainability
│   │   └── hardware_bridge.py      ← IBMQ integration scaffold
│   ├── models/
│   │   ├── vqc_model.py            ← Variational Quantum Classifier
│   │   ├── qsvm_model.py           ← Quantum Support Vector Machine
│   │   ├── hybrid_qnn.py           ← Hybrid QNN (PyTorch)
│   │   └── [8 classical models]
│   ├── interop/
│   │   └── fhir.py                 ← FHIR R4 + ABHA
│   ├── audit.py                    ← Clinical data audit
│   ├── clean.py                    ← Pre-split data cleaning
│   ├── features.py                 ← Clinical feature engineering
│   ├── preprocess.py               ← Leakage-safe preprocessing
│   ├── evaluate.py                 ← Benchmark + bootstrap CIs
│   ├── explain_risk.py             ← TreeSHAP + logit attributions
│   └── external_validation.py      ← Framingham OOD stress test
│
├── tests/                          ← 64 tests across 11 modules
├── artifacts/                      ← Serialized model artifacts
├── data/                           ← Datasets
├── SIH_Submission_Kit/             ← This folder
│
├── AUDIT.md                        ← Full codebase audit report
├── architecture.md                 ← Technical architecture (detailed)
└── limitations.md                  ← Known limitations + disclaimers
```

---

## Contact & Repository

**Team**: [Your Team Name]  
**Institution**: [Your College]  
**GitHub**: [Your Repository URL]  
**Demo URL**: http://localhost:8080 (local deployment)  

---

## Disclaimer

CardioQ is a research and clinical decision support platform. It does not constitute a Software as Medical Device (SaMD) diagnostic apparatus and is not certified for autonomous clinical decision making. All risk assessments must be interpreted by licensed medical professionals.
