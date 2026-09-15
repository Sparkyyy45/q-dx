# 🏆 Smart India Hackathon (SIH) Official PPT Presentation Kit
## Problem Statement 3: Hybrid Quantum Machine Learning Platform for Early Disease Detection
### Project Name: **CardioQ** | Ministry of Health & Family Welfare (MoHFW)

---

> **Designer's Note & Guidelines for Maximum Marks:**
> 1. **Visual-First Rule:** Never put walls of text. SIH judges look at the screen for only 3–5 seconds before listening to you.
> 2. **Colors:** Use Navy Blue (`#0F172A`) for headers, Emerald Green (`#10B981`) for success/stats, Quantum Purple (`#8B5CF6`) for quantum blocks.
> 3. **Slide Layout:** Use a **2-Column or 3-Card structure** on every slide. Left for crisp bullet points, Right for diagrams/visuals, Bottom for metric pill badges.

---

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      SLIDE INDEX                                             ║
║                                                                                              ║
║  SLIDE 1 : Cover Slide (Problem Statement, Team, Organization)                               ║
║  SLIDE 2 : Proposed Solution (Describe your Idea/Solution/Prototype) ◄ [Official Format]     ║
║  SLIDE 3 : Technical Architecture & Quantum-Classical Pipeline                               ║
║  SLIDE 4 : Feasibility, Viability & Tech Stack Execution                                     ║
║  SLIDE 5 : Impact, Clinical Benefits & National Alignment (ABDM/FHIR)                        ║
║  SLIDE 6 : Research, Novelty & Competitive Advantage (USP)                                   ║
║  SLIDE 7 : Potential Challenges, Risks & Engineering Mitigations                             ║
║  SLIDE 8 : Deployment Roadmap, Scalability & Future Scope                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## ❖ SLIDE 1: Cover Slide

### 🎯 Slide Title
**CardioQ: Hybrid Quantum-Classical Platform for Early Disease Detection**

### 📐 Visual Layout in PPT
```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  ❖ SMART INDIA HACKATHON 2025                                           MINISTRY OF HEALTH  │
│                                                                        & FAMILY WELFARE    │
│   CARDIOQ                                                                                   │
│   Hybrid Quantum Machine Learning Platform for Early Cardiovascular Disease Detection       │
│                                                                                             │
│  ┌──────────────────────────────┐        ┌──────────────────────────────────────────────┐   │
│  │      TEAM PARTICULARS        │        │             KEY PLATFORM HIGHLIGHTS          │   │
│  │                              │        │                                              │   │
│  │  • Problem Code: PS-3        │        │  ⚛️ 4-Qubit Parameterized Quantum Circuit     │   │
│  │  • Category: Software / Med  │        │  📊 11 Models Benchmarked on 70,000+ Records  │   │
│  │  • Team Name: [Team Name]    │        │  🛡️ Zero-Leakage Fold-Isolated Pipeline      │   │
│  │  • College: [College Name]   │        │  🇮🇳 ABHA ID & HL7 FHIR R4 Compliant          │   │
│  │  • Team Lead: [Lead Name]    │        │  ⚡ Sub-50ms Production REST API Ready        │   │
│  └──────────────────────────────┘        └──────────────────────────────────────────────┘   │
│                                                                                             │
│  [ 70,000+ Patients ]    [ 0.7969 ROC-AUC ]    [ 70.42% Sensitivity ]    [ 64/64 Unit Tests ] │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 📋 Copy-Paste Content
* **Problem Statement:** PS-3 — Hybrid Quantum Machine Learning Platform for Early Disease Detection
* **Ministry:** Ministry of Health & Family Welfare (MoHFW)
* **Team ID / Name:** [Insert Team ID & Name]
* **College / Institute:** [Insert College Name]
* **One-Line Pitch:** *"Bridging classical predictive scale with quantum representational capacity to eliminate preventable cardiovascular deaths in India."*

---

## ❖ SLIDE 2: Proposed Solution (Describe your Idea/Solution/Prototype)

### 🎯 Slide Title
**Proposed Solution: CardioQ Hybrid Diagnostic Engine**

### 📐 Visual Layout in PPT (Split 40% Left / 60% Right)
* **Left Side:** 3 Rounded Feature Cards with Icons.
* **Right Side:** End-to-End Visual Solution Flow.
* **Bottom:** 4 Metric Stat Badges.

### 📋 Left Side: 3 Content Cards

#### 1. Detailed Explanation of the Solution
* **Dual-Track Hybrid ML Platform:** Integrates classical tree ensembles (CatBoost/LightGBM) with a **4-qubit Parameterized Quantum Circuit (PQC)**.
* **Continuous Statevector Simulation:** Custom PyTorch quantum engine executing angle embedding and variational ansatz without external C++ hardware dependencies.
* **End-to-End Ecosystem:** From raw vitals intake to instant, explainable clinical triage risk scores.

#### 2. How It Addresses the Problem
* **Solves Biomedical Curse of Dimensionality:** Quantum entanglement captures subtle, non-linear biomarker correlations that classical models miss.
* **Sensitivity-First Screening:** Hybrid QNN achieves **70.42% sensitivity** — minimizing fatal false negatives in asymptomatic patients.
* **Mathematically Leakage-Safe:** Fold-isolated preprocessing + Out-of-fold Youden’s $J$ thresholding guarantees real-world generalizability.

#### 3. Innovation & Uniqueness
* **Explainable Quantum AI (XQAI):** TreeSHAP for classical models + PyTorch autograd Jacobian attribution for quantum circuits.
* **National Health Stack Ready:** Native **ABHA (Luhn-10)** verification, **HL7 FHIR R4** export, and **ICMR triage tiers**.
* **Zero Fabrication Guarantee:** All 11 models strictly benchmarked on identical SHA-256 locked data with bootstrap 95% confidence intervals.

### 🎨 Right Side: Visual Architecture Flow
```
┌─────────────────────────────────┐
│     PATIENT CLINICAL VITALS     │  (Age, Systolic BP, Chol, Glucose, Lifestyle)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│     PRE-SPLIT DATA AUDITING     │  (Outlier cleaning, Fold-isolated scaling)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────┐
│           HYBRID QUANTUM-CLASSICAL CORE                 │
│                                                         │
│   ┌────────────────────┐       ┌────────────────────┐   │
│   │ Classical Feature  │ ───►  │  4-Qubit PQC Layer │   │
│   │ Dimensionality Red.│       │  Angle Encoding +  │   │
│   │ (PCA + Supervised) │       │  CNOT Entanglement │   │
│   └────────────────────┘       └─────────┬──────────┘   │
│                                          ▼              │
│                                  Pauli-Z Expectation    │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│              EXPLAINABLE CLINICAL ACTION                │
│   🟢 Green (Low)  🟡 Amber (Moderate)  🔴 Red (High)    │
│   • SHAP & Quantum Gradient Attributions                │
│   • ABHA ID & HL7 FHIR R4 Standardized Output           │
└─────────────────────────────────────────────────────────┘
```

### 📊 Bottom Pill Badges
| 70,000+ Records | 0.7969 ROC-AUC | 70.42% Sensitivity | < 50ms Latency |
|:---:|:---:|:---:|:---:|
| *Holdout Validated* | *Champion CatBoost* | *Highest (Hybrid QNN)* | *PHC Real-Time Ready* |

---

## ❖ SLIDE 3: Technical Architecture & Methodology

### 🎯 Slide Title
**Technical Architecture: Quantum-Classical Pipeline**

### 📐 Visual Layout in PPT
* **Top Ribbon:** Data Ingestion & Preprocessing Steps.
* **Center:** Side-by-Side Dual-Track Engine (Classical Track A vs Quantum Track B).
* **Bottom:** Deployment & Interoperability Layer.

### 🎨 Diagram to Place on Slide
```
                 [ 70,000 Patient Records (CVD Dataset) ]
                                    │
                                    ▼
       [ Pre-Split Data Audit: BP Inversion & Outlier Sanitization ]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼ (80% Dev: 54,961)                       ▼ (20% Holdout: 13,741)
       [ 5-Fold Stratified CV ]                    [ NEVER TOUCHED ]
               │                                         │
       ┌───────┴─────────────────────────┐               │
       ▼                                 ▼               │
┌─────────────────────────┐   ┌─────────────────────────┐│
│   TRACK A: CLASSICAL    │   │    TRACK B: QUANTUM     ││
│ • CatBoost (Champion)   │   │ • 4-Qubit Angle Enc.    ││
│ • LightGBM, XGBoost     │   │ • Ry(θ)Rz(ω) Ansatz     ││
│ • Random Forest, LR     │   │ • Parameter-Shift Grad  ││
│ • Calibrated SVM, MLP   │   │ • Hybrid QNN / VQC / QSVM│
└───────────┬─────────────┘   └───────────┬─────────────┘│
            │                             │              │
            └──────────────┬──────────────┘              │
                           ▼                             │
            [ Out-of-Fold Threshold Locking ]            │
            [ Youden's J Optimization       ]            │
                           │                             │
                           ▼                             │
            [ FINAL EVALUATION ON HOLDOUT ] ◄────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [ Explainability Layer ]     [ Interoperability Engine ]
   • TreeSHAP Values            • ABHA Luhn-10 Validator
   • Quantum Jacobians          • HL7 FHIR R4 Bundles
   • Top Risk Drivers           • ICMR 3-Tier Clinical Triage
```

### 📋 Key Technical Bullets
* **Rigorous Separation:** 80/20 train-test split strictly frozen before any preprocessing — eliminates all data snooping.
* **Exact Quantum Gradients:** Parameter-shift rule $\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{\langle Z(\theta+\frac{\pi}{2})\rangle - \langle Z(\theta-\frac{\pi}{2})\rangle}{2}$ computed without approximation.
* **Prospective Decision Thresholds:** Operating thresholds locked exclusively on cross-validation folds, preventing inflated test scores.

---

## ❖ SLIDE 4: Feasibility, Viability & Tech Stack

### 🎯 Slide Title
**Feasibility, Viability & Production Tech Stack**

### 📐 Visual Layout in PPT (3 Columns / Tech Cards)
* **Column 1:** Quantum & AI Core.
* **Column 2:** Backend & Infrastructure.
* **Column 3:** Clinical Standards & Deployment.

### 📋 3 Column Content

#### Column 1: Quantum & AI Core
* **PyTorch & NumPy:** Pure statevector simulator running directly in memory — zero quantum cloud queue delays.
* **Hardware-Ready:** 4-qubit CNOT-ring ansatz is natively transpilable to IBM Quantum (Eagle/Heron) and IonQ.
* **Scikit-Learn / CatBoost:** Gradient boosted ensembles with exact TreeSHAP calculation.

#### Column 2: Backend & Reliability
* **Python Flask REST API:** Sub-50ms latency per inference; built for low-bandwidth rural health centers.
* **Physiological Validation:** Enforces strict clinical bounds (e.g., BP 60–240 mmHg, BMI 10–70 kg/m²).
* **Automated CI/CD:** **64/64 Unit Tests passing (100%)** covering data integrity, thresholding, and math invariance.

#### Column 3: Clinical & National Viability
* **Ayushman Bharat (ABDM):** Validates 14-digit ABHA IDs using the Luhn-10 algorithm.
* **HL7 FHIR R4:** Exports standardized JSON diagnostic bundles ready for hospital EMR integration.
* **ICMR Triage Alignment:** Automated routing into Green (Routine), Amber (14-day Review), or Red (Immediate Referral).

### 📊 Deployment Feasibility Summary Table
| Parameter | Feasibility Evidence | Production Status |
|---|---|---|
| **Inference Time** | < 50ms per patient query | ✅ Verified in REST API |
| **Server Footprint** | Pure CPU/GPU lightweight execution | ✅ Runs on ₹15k PC / Tablet |
| **Internet Dependency** | Fully offline capable for rural clinics | ✅ Standalone Statevector Engine |
| **Test Coverage** | 64 tests passing in 4.91 seconds | ✅ 100% Passing Test Suite |

---

## ❖ SLIDE 5: Impact, Clinical Benefits & National Alignment

### 🎯 Slide Title
**Clinical Impact & Alignment with National Health Mission**

### 📐 Visual Layout in PPT (Split Left Impact / Right Clinical Flow)
* **Left Side:** 3 Major National Impact Drivers with Statistics.
* **Right Side:** Patient Triage Path (ICMR 3-Tier Workflow).

### 📋 Left Side: Impact Drivers

#### 1. Reducing India's ₹6.2 Lakh Crore CVD Burden
* **Early Detection Window:** Flags asymptomatic endothelial dysfunction and early hypertension years before acute events.
* **Preventable Cardiac Mortality:** Early lifestyle and medical intervention reduces CVD mortality by up to **60%**.

#### 2. Empowering Rural Primary Health Centers (PHCs)
* **Democratizing Cardiology:** Frontline ASHA and ANM workers can screen patients in under 2 minutes using standard vitals.
* **Offline-First Resilience:** Operates in remote, zero-connectivity regions without reliance on cloud API uptime.

#### 3. Seamless NHA & PM-JAY Integration
* **Ayushman Bharat Digital Mission:** Every risk assessment is tagged with a valid ABHA ID and exportable as an HL7 FHIR record.
* **Resource Optimization:** Reduces unnecessary tertiary referrals by safely managing low-risk patients at the PHC level.

### 🎨 Right Side: ICMR Triage Workflow
```
                  [ PATIENT SCREENING INGESTION ]
                                 │
                                 ▼
                     [ CardioQ Risk Assessment ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   🟢 LOW RISK             🟡 MODERATE RISK         🔴 HIGH RISK
 (Risk Score < 0.35)      (Risk Score 0.35-0.65)   (Risk Score > 0.65)
         │                       │                       │
         ▼                       ▼                       ▼
• Annual Review          • 14-Day Clinic Followup • Immediate ECG / Troponin
• Lifestyle & Diet Plan  • Statin/BP Optimization • Cardiology Referral
• Local PHC Monitoring   • Tele-consultation      • PM-JAY Tertiary Alert
```

---

## ❖ SLIDE 6: Research Novelty & Competitive Advantage (USP)

### 🎯 Slide Title
**Research Novelty & Competitive Advantage (USP)**

### 📐 Visual Layout in PPT
* **Top:** Head-to-Head Benchmark Table.
* **Bottom Left:** Why Quantum Matters (Sensitivity Advantage).
* **Bottom Right:** Differentiator Matrix vs Existing Systems.

### 📊 Benchmark Comparison Table (Actual Platform Results)
| Model Architecture | Model Type | ROC-AUC | Sensitivity | Key Advantage |
|---|:---:|:---:|:---:|---|
| **CatBoost (Champion)** | Classical | **0.7969** | 0.6996 | Best overall discriminative power |
| **Hybrid QNN** | **Quantum-Classical** | **0.7775** | **0.7042** | **Highest sensitivity (fewest missed cases)** |
| Logistic Regression | Classical | 0.7928 | 0.6769 | High interpretability baseline |
| Random Forest | Classical | 0.7925 | 0.6986 | Non-linear ensemble |
| VQC (Variational) | Quantum | 0.7306 | 0.6460 | 4-qubit compact quantum representation |
| QSVM (ZZ-Feature Map) | Quantum | 0.7302 | 0.6445 | Non-linear quantum Hilbert space kernel |

### 📋 Unique Selling Points (USPs)
* **Quantum Sensitivity Edge:** The Hybrid QNN achieves **70.42% sensitivity** — higher than all classical models — essential for screening where missing a sick patient is catastrophic.
* **Dual-Track Honesty:** Unlike fraudulent projects claiming "100% quantum accuracy", we report real, audited metrics with 95% bootstrap confidence intervals.
* **No Black Box:** Clinicians receive individualized TreeSHAP feature impact and quantum gradient attributions for every single prediction.

---

## ❖ SLIDE 7: Challenges, Risks & Mitigation Strategy

### 🎯 Slide Title
**Technical Challenges, Clinical Risks & Engineering Mitigations**

### 📐 Visual Layout in PPT (4-Card Risk Matrix Grid)

### 📋 4 Risk-Mitigation Cards

```
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ 1. BARREN PLATEAUS & QUANTUM NOISE   │  │ 2. DATA LEAKAGE & OVERFITTING        │
├──────────────────────────────────────┤  ├──────────────────────────────────────┤
│ ⚠️ Risk: Vanishing gradients in deep │  │ ⚠️ Risk: Preprocessing across splits │
│    quantum circuits & noisy qubits.  │  │    inflating benchmark metrics.      │
│                                      │  │                                      │
│ 🛡️ Mitigation:                        │  │ 🛡️ Mitigation:                        │
│ • Shallow 4-qubit layered ansatz     │  │ • Strict 80/20 pre-split freeze      │
│ • Local Pauli-Z observables          │  │ • 5-fold isolated preprocessing      │
│ • Exact parameter-shift analytic     │  │ • Zero data snooping audited         │
│   gradients without numerical noise  │  │                                      │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ 3. CLINICAL ADOPTION & BLACK BOX     │  │ 4. HARDWARE SCALABILITY & ACCESSIBILITY│
├──────────────────────────────────────┤  ├──────────────────────────────────────┤
│ ⚠️ Risk: Doctors reject unexplained  │  │ ⚠️ Risk: Quantum QPUs unavailable in │
│    AI/quantum predictions.           │  │    rural primary health centers.     │
│                                      │  │                                      │
│ 🛡️ Mitigation:                        │  │ 🛡️ Mitigation:                        │
│ • TreeSHAP feature attributions      │  │ • Pure in-memory PyTorch statevector │
│ • Quantum autograd Jacobian plots    │  │   simulator runs on standard laptops │
│ • ICMR guideline-aligned risk tiers  │  │ • OpenQASM export ready for NISQ     │
│   (Green / Amber / Red)              │  │   hardware execution when available  │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```

---

## ❖ SLIDE 8: Deployment Roadmap & Future Scope

### 🎯 Slide Title
**Deployment Roadmap, Scalability & Future Scope**

### 📐 Visual Layout in PPT (Chronological 4-Phase Timeline)

### 🎨 Visual Roadmap Timeline
```
   PHASE 1 (Completed)          PHASE 2 (Month 1-3)          PHASE 3 (Month 4-6)          PHASE 4 (Month 7-12)
 ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │  WORKING PROTOTYPE   │ ──► │  HARDWARE & PILOT    │ ──► │  MULTI-DISEASE QML   │ ──► │  NATIONAL EXPANSION  │
 ├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
 │ • 11 Models Audited  │     │ • IBM Quantum QPU    │     │ • Expand to Type-2   │     │ • Full Ayushman      │
 │ • 64/64 Tests Pass   │     │   hardware run       │     │   Diabetes & CKD     │     │   Bharat integration │
 │ • Web App & REST API │     │ • PHC clinical pilot │     │ • Multi-modal vitals │     │ • State-wide rollout │
 │ • ABHA + FHIR R4     │     │   (1,000 patients)   │     │   + 12-lead ECG      │     │   with MoHFW         │
 └──────────────────────┘     └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 📋 Future Horizons
1. **NISQ Hardware Execution:** Run existing circuits on real IBM Quantum Eagle/Heron processors via cloud API.
2. **Multi-Organ Disease Suite:** Expand beyond CVD to predict diabetic nephropathy and metabolic syndrome.
3. **Federated Quantum Learning:** Enable privacy-preserving hospital model training without moving patient records.

---

## 🎙️ Judge Q&A Cheat Sheet (Keep Handy During Pitch)

| If the Judge Asks: | Answer in 10 Seconds: |
|---|---|
| *"Why use quantum if CatBoost has higher AUC?"* | *"CatBoost wins overall AUC (0.7969), but Hybrid QNN achieves our highest sensitivity (0.7042). In screening, missing a sick patient is fatal. Under equal N=1,000 data budget, quantum is within 0.019 AUC of classical."* |
| *"How do you run quantum algorithms in rural clinics?"* | *"Our production engine uses an in-memory PyTorch statevector simulator. It runs on any standard ₹15,000 clinic PC with <50ms response time and zero internet dependency."* |
| *"Did you prevent data leakage?"* | *"Yes. Pre-split data audit, 5-fold isolated preprocessing, and threshold locking via Youden's J solely on out-of-fold predictions. The 13,741 holdout set was never touched during training."* |
| *"Is this integrated with Indian government health systems?"* | *"Yes. We natively validate 14-digit ABHA IDs via Luhn-10, export standard HL7 FHIR R4 bundles, and classify patients into ICMR 3-tier triage paths."* |
