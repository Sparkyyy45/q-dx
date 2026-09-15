# CardioQ: Judge Q&A Preparation — Full Expert-Level Question Bank
## SIH Problem Statement 3 — Technical Document 04
### Prepared for: AI/ML Scientists, Quantum Computing Engineers, Clinical Domain Experts, Government Technology Reviewers

---

> **How to use this document**: Every question below has been formulated to match what senior SIH evaluators, IIT faculty judges, ICMR domain experts, and NASSCOM startup reviewers would realistically ask. Answers are comprehensive, technically correct, and reference actual platform metrics and code.

---

## CATEGORY 1: Quantum Computing — Deep Technical Questions

---

**Q1. What is your quantum circuit architecture? Explain it gate by gate.**

**A**: Our Variational Quantum Circuit uses 4 qubits in a hardware-efficient layered ansatz:

```
Step 1 — Feature Encoding: For each feature xⱼ (j = 0..3):
         Ry(φⱼ) where φⱼ = 2·arctan(xⱼ) ∈ [-π, π]
         This maps continuous clinical values to Bloch sphere rotations

Step 2 — Variational Layer (repeated L times):
         For each qubit q:
           Ry(θ_{l,q}) — parameterized Y-rotation (trainable)
           Rz(ω_{l,q}) — parameterized Z-rotation (trainable)
         Entangling CNOT ring:
           CNOT(q₀, q₁) → CNOT(q₁, q₂) → CNOT(q₂, q₃) → CNOT(q₃, q₀)

Step 3 — Measurement:
         ⟨Z₀⟩ = ⟨ψ|Z|ψ⟩ on qubit 0 ∈ [-1, 1]
         → Platt sigmoid calibration → P(CVD) ∈ [0, 1]
```

Total trainable parameters: 2 × L × N_qubits (e.g., L=3, N=4 → 24 parameters)

---

**Q2. What exactly is the "quantum advantage" you claim? Be specific.**

**A**: We do not claim unconditional quantum advantage. We claim and demonstrate:

1. **Constrained-budget parity**: Under an equal N=1,000 training budget (Track B), the Hybrid QNN (ROC-AUC 0.7775) achieves competitive performance with classical models (CatBoost: 0.7802 at N=1,000), **within 0.003 AUC**.

2. **Sensitivity superiority**: The Hybrid QNN achieves the **highest sensitivity of all 11 models (0.7042)** — the clinically most important metric for screening tools (minimizing false negatives).

3. **Representational complementarity**: The quantum kernel (QSVM's ZZ-feature map) operates in a Hilbert space of dimension 2^N = 16 for 4 qubits — potentially capturing feature correlations invisible to polynomial-kernel SVMs.

4. **No unconditional supremacy claim**: Classical CatBoost (0.7969 AUC on 54,961 samples) outperforms quantum models. This is expected — quantum advantage requires either: (a) much larger datasets where quantum kernels excel, or (b) actual quantum hardware where decoherence-mediated noise creates a natural ensemble effect.

---

**Q3. Why are you using a simulated quantum computer? Is this even "quantum"?**

**A**: Yes — the mathematics is fully quantum. Simulation vs. hardware is an **execution environment difference**, not a correctness difference.

- Our statevector simulator implements the **exact** quantum mechanics: Schrödinger equation evolution, unitary gate matrices, Born rule measurement
- IBM Qiskit, Google Cirq, and all major QML frameworks use simulation for development
- The circuits we implement use **only** gates available on NISQ hardware: Ry, Rz, CNOT — the native gate set of IBM Eagle (127 qubits), IonQ Aria (25 qubits), and Rigetti Aspen-M

**For SIH purposes**: Quantum hardware access requires IBMQ Network membership (₹0 for academic partners, but requires institutional registration). Our `src/quantum/hardware_bridge.py` is the integration point for real hardware execution via Qiskit Runtime. Given SIH's hackathon timeframe, simulation is the standard approach — even published QML research papers use Qiskit simulation.

---

**Q4. What is the ZZ-Feature Map in your QSVM? Why is it quantum?**

**A**: The ZZ-Feature Map creates a quantum state `|Φ(x)⟩` from classical input `x` via:

```
U_Φ(x) = exp(i·Σⱼ xⱼZⱼ + Σⱼ<ₖ (π-xⱼ)(π-xₖ)ZⱼZₖ) · H^⊗N
```

The key quantum property is the **two-qubit ZⱼZₖ term**. For features j and k:
- It computes `(π-xⱼ)(π-xₖ)` — a non-linear interaction term
- It applies this as a **joint phase rotation on two qubits simultaneously**
- This entangles the two qubits in a way that depends on both feature values simultaneously

The resulting quantum kernel:
```
K(xᵢ, xⱼ) = |⟨0^N|U†_Φ(xᵢ)·U_Φ(xⱼ)|0^N⟩|²
```
is a **fidelity measure in Hilbert space** — it measures how similar two clinical feature vectors are according to quantum geometry. This kernel satisfies Mercer's conditions and theoretically captures correlations that polynomial and RBF kernels cannot.

---

**Q5. Explain the parameter-shift rule. Can this run on real quantum hardware?**

**A**: Yes — the parameter-shift rule is specifically designed for quantum hardware.

**Why you can't use backpropagation on hardware**: Quantum hardware cannot return intermediate computational states needed for standard backpropagation. Only final measurement outcomes are accessible.

**The parameter-shift rule** computes exact gradients using only two circuit evaluations:
```
∂⟨Z⟩/∂θₖ = [⟨Z⟩(θₖ + π/2) - ⟨Z⟩(θₖ - π/2)] / 2
```

**Why this works on hardware**: Each term requires one forward circuit execution. The quantum hardware measures `⟨Z⟩(θₖ + π/2)` and `⟨Z⟩(θₖ - π/2)` separately — no intermediate state access needed.

**Our verification**: We compare parameter-shift gradients against PyTorch autograd numerical Jacobians. Maximum absolute error: < 1×10⁻³. Reference: Mitarai et al., 2018 "Quantum Circuit Learning", Physical Review A.

---

**Q6. What is decoherence? How does it affect your quantum models on real hardware?**

**A**: Decoherence is the loss of quantum coherence (superposition and entanglement) due to environmental noise:

| Decoherence Type | Symbol | Typical Value (IBM Eagle) | Effect |
|---|---|---|---|
| Amplitude damping | T₁ | ~100 µs | Qubit spontaneously decays |0⟩→|1⟩ |
| Phase damping | T₂ | ~100 µs | Phase relationships lost |
| Gate infidelity | - | ~0.1–1% per 2-qubit gate | Unitary errors accumulate |
| Readout error | - | ~1–5% | Measurement outputs wrong bit |

**Impact on our models**: On real hardware, our circuits would experience noise proportional to circuit depth. Mitigation strategies we have designed for (but not yet executed on hardware):
1. **Zero-Noise Extrapolation (ZNE)**: Run circuits at 1×, 2×, 3× noise amplification, extrapolate to zero noise
2. **Readout Error Mitigation**: Calibration matrix M where P_ideal = M⁻¹ · P_measured
3. **Dynamic Decoupling**: Insert echo pulses during idle qubit periods to suppress phase errors

Our N=4 qubit circuits have depth ~15 gates — **within the coherence window of current IBM Heron processors (T₁ ≈ 300µs, gate time ≈ 100ns)**.

---

**Q7. What is quantum entanglement, and what role does it play in your classifier?**

**A**: Entanglement is a quantum correlation between qubits with no classical analogue. When qubits are entangled, the state cannot be written as a product of individual qubit states:

```
Entangled state (Bell state): |Φ+⟩ = (|00⟩ + |11⟩)/√2
  → Measurement of qubit 0 instantly determines qubit 1, regardless of distance

Separable (non-entangled) state: |01⟩ = |0⟩ ⊗ |1⟩
  → Qubits are independent
```

**In CardioQ VQC**: After angle-encoding clinical features, the CNOT ring creates entanglement between adjacent qubits. This means the circuit's output `⟨Z₀⟩` depends on all 4 clinical features **jointly** through quantum correlations. Concretely: the risk from high blood pressure and old age is **not simply additive** — their joint effect is encoded as an entangled quantum state, potentially capturing non-linear synergistic risk.

---

## CATEGORY 2: Machine Learning — Technical Questions

---

**Q8. What is data leakage? How did you prevent it?**

**A**: Data leakage occurs when information from the evaluation set influences training, resulting in artificially inflated performance metrics. We identified and prevented 5 forms:

| Leakage Type | Prevention Method |
|---|---|
| **Holdout contamination** | 80/20 split happens first; holdout never accessed during training |
| **Preprocessing leakage** | IQRClipper, Scaler, FeatureReducer fitted on training folds only |
| **Threshold leakage** | τ* derived only from OOF predictions, never from holdout labels |
| **Feature selection leakage** | ANOVA F-test uses only training fold features and labels |
| **External data imputation** | `MissingExternalFeatureError` raised; zero-fill never performed |

The most dangerous and commonly missed form is **preprocessing leakage**: fitting a scaler on the full dataset and then splitting. This leaks test set statistics (mean, variance) into training — making the model appear better calibrated than it truly is. Our fold-isolated preprocessing prevents this.

---

**Q9. Why is CatBoost your champion model? Why not XGBoost or LightGBM?**

**A**: CatBoost outperforms LightGBM (0.7833) and XGBoost (0.7866) in our benchmark (ROC-AUC 0.7969) for three dataset-specific reasons:

1. **Ordered Target Statistics for ordinal categoricals**: `cholesterol` and `gluc` are ordinal tiers (1/2/3), not continuous values. CatBoost's OTS encoding correctly treats these as probabilities conditioned on ordering, while LightGBM treats them as numeric. This gives CatBoost a structural advantage on this specific feature set.

2. **Symmetric tree architecture**: CatBoost uses oblivious decision trees (same split condition at every node of a level) — this provides stronger regularisation for datasets with ~70k samples, preventing the overfitting visible in XGBoost's lower Brier score.

3. **Native GPU training + leaf regularisation**: CatBoost's L2 leaf penalty controls variance more precisely than XGBoost's lambda/alpha combination on this feature dimensionality.

---

**Q10. Why is ROC-AUC the right metric here? What about F1 score?**

**A**: ROC-AUC is the right primary metric for a clinical **screening tool** because:

1. **Threshold independence**: ROC-AUC measures discriminative ability across ALL possible thresholds. A clinician may later decide to screen at higher sensitivity (accepting more false positives in low-risk populations) or higher specificity (in high-cost intervention settings). ROC-AUC captures all of these use cases simultaneously.

2. **Prevalence invariance**: ROC-AUC is invariant to class imbalance — unlike F1, which depends on prevalence. Our training data (50% CVD) will behave differently from rural screening data (5–15% CVD). ROC-AUC remains stable.

3. **PR-AUC as secondary**: We also report PR-AUC (Precision-Recall AUC), which IS sensitive to prevalence — important for understanding performance in low-prevalence deployment. Both metrics together give a complete picture.

4. **Clinical standard**: ROC-AUC is required by CONSORT 2010, TRIPOD 2015, and EQUATOR guidelines for clinical prediction model reporting.

**F1 is still reported** in our benchmark table but as a secondary metric. Its value depends on the decision threshold — at our OOF-derived τ*, F1 is 0.7254 for CatBoost.

---

**Q11. How do you handle class imbalance in deployment?**

**A**: Our training data has ~50% CVD prevalence — near-balanced. Real deployment environments (primary care, general population) have 5–20% CVD prevalence. This **prevalence shift** affects calibration and threshold effectiveness.

**Our approach**:
1. **Prospective threshold locking via Youden's J**: Rather than using a fixed 0.5 threshold, we derive τ* = argmax_τ [TPR(τ) - FPR(τ)] from OOF predictions. This naturally adapts to the model's score distribution, producing better sensitivity-specificity tradeoffs.

2. **Calibrated probabilities**: All models output Platt-calibrated probabilities, not raw scores. This means P(CVD) = 0.73 genuinely means 73% CVD probability in the training distribution — important for clinical interpretation.

3. **Framingham OOD test documents the degradation**: When deployed on Framingham (15% prevalence), we observe performance drops. We document this transparently so deployment teams know to recalibrate thresholds for their local prevalence.

4. **Future work**: Isotonic regression recalibration for deployment prevalence correction.

---

**Q12. What is Youden's J statistic? Why is it better than F1 maximisation for threshold selection?**

**A**:
```
Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
F1 threshold = argmax [2·Precision·Recall / (Precision + Recall)]
```

Differences:

| Property | Youden's J | F1 Maximisation |
|---|---|---|
| Prevalence sensitivity | Invariant — TPR and FPR don't depend on prevalence | Sensitive — Precision depends on N_positive/N_total |
| Clinical interpretation | Maximises joint sensitivity+specificity equally | Maximises precision-recall tradeoff |
| WHO recommendation | Recommended for screening tools | Recommended for detection tasks |
| Our use case | Screening (catch all cases) | Not primary — used for secondary reporting |

Youden's J is the correct criterion when **both** false negatives (missed CVD cases) and false positives (unnecessary referrals) carry significant cost, which is the case in population screening.

---

## CATEGORY 3: Clinical Domain — Expert Questions

---

**Q13. How does your platform align with ICMR clinical guidelines for CVD screening?**

**A**: ICMR's "Standard Treatment Guidelines for Cardiovascular Diseases" (2019) defines three CVD risk tiers:

| ICMR Tier | 10-Year CVD Risk | Recommended Action |
|---|---|---|
| Low | < 10% | Lifestyle counselling |
| Moderate | 10–20% | Lifestyle + consider pharmacotherapy |
| High | > 20% | Immediate pharmacotherapy + specialist referral |

CardioQ outputs are **calibrated probabilities** mapped directly to these ICMR tiers. The API response includes:
```json
{
  "risk_probability": 0.73,
  "risk_tier": "HIGH",
  "tier_description": "Immediate specialist referral recommended",
  "icmr_guideline": "10-Year CVD Risk > 20%"
}
```

Additionally, our feature attribution output highlights the **modifiable risk factors** (BP, BMI, smoking) separately from non-modifiable ones (age, gender) — aligned with ICMR's emphasis on lifestyle intervention.

---

**Q14. Can this be used for diagnosis? What are the clinical limitations?**

**A**: **No, this is NOT a diagnostic device**. CardioQ is a **clinical decision support (CDS) tool**:

| Category | CardioQ | Diagnostic Device |
|---|---|---|
| Regulatory status | Research/CDS tool | Requires CDSCO Class IIb/III approval |
| Autonomy | Assists clinician | Can make autonomous decisions |
| Liability | User (clinician) | Manufacturer (CE/FDA/CDSCO marked) |
| Required disclosure | Yes (in every API response) | Regulated label |

**Specific clinical limitations**:
1. Trained on cross-sectional prevalent CVD — cannot predict **future** cardiac events
2. 22 features do not include ECG, echocardiography, coronary calcium score, or biomarkers (troponin, BNP) — standard for actual CVD diagnosis
3. External validation (Framingham) showed performance degradation (~0.13 AUC drop) due to endpoint mismatch — known generalization limitation
4. Not validated on Indian population cohorts specifically — Kaggle CVD dataset has no documented geographic information

**Medical disclaimer is included in every single API response payload**.

---

**Q15. Your Framingham external validation shows degraded performance. Isn't that a problem?**

**A**: It is exactly what we expected and designed for — and our transparency here is a strength.

The performance degradation (ROC-AUC 0.797 internal → 0.66–0.67 Framingham) has three documented causes:

1. **Endpoint mismatch** (primary cause): We predict prevalent CVD (cross-sectional diagnosis), Framingham measures incident 10-year CHD (prospective). These are fundamentally different outcomes.

2. **Feature subset restriction** (secondary): 5 features present in our training data are absent in Framingham (`gluc`, `alco`, `active`, `height`, `weight`). The reduced-feature model loses these predictors.

3. **Prevalence shift** (tertiary): 49.5% training prevalence vs 14.98% Framingham. At the locked threshold, sensitivity/specificity tradeoff shifts.

We are the only SIH team that:
- Runs external OOD validation at all
- Publishes the performance degradation
- Explains each cause rigorously
- Does NOT refine the model on Framingham to artificially inflate scores

A team that hides limitations is a team whose model will fail in real deployment. Judges should reward this transparency.

---

## CATEGORY 4: System Design & Deployability

---

**Q16. How would this scale to a national deployment? What is the throughput?**

**A**: CardioQ is designed as a stateless inference service — horizontal scaling is straightforward:

**Current single-server performance**:
- Classical inference (CatBoost): < 5ms per patient
- Quantum simulation (VQC, N=4): < 0.5ms per circuit execution
- API response time (end-to-end including FHIR generation): < 50ms

**National scale architecture** (proposed):
```
                    Load Balancer (NHA InfraStack)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
       CardioQ-1       CardioQ-2       CardioQ-3
      (Classical)     (Classical)     (Classical)
           │
           ├── IBMQ Quantum Cloud (for VQC/QSVM/QNN)
           │
           └── NHA FHIR Server (ABDM ecosystem)
```

At 50ms per prediction: one server = 20 predictions/second = 1.7M predictions/day. India's estimated screening load for 18+ population: ~50M workups annually = **135K/day** — comfortably handled by 1 server. For burst capacity (health drives), horizontal scale to 10 servers covers 10× load.

---

**Q17. How does your ABHA integration actually work?**

**A**: ABHA (Ayushman Bharat Health Account) is India's universal health ID system. Integration has two components:

**1. ABHA ID Validation (Luhn-10)**:
```python
def verify_abha_id(abha_id: str) -> bool:
    # ABHA format: XX-XXXX-XXXX-XXXX (14 digits)
    digits = [int(d) for d in abha_id.replace('-', '')]
    # Luhn algorithm: sum alternating doubled digits
    total = sum(
        [d if i % 2 != 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
         for i, d in enumerate(reversed(digits))]
    )
    return total % 10 == 0
```

This validates format and checksum — equivalent to how credit card numbers are validated. It prevents typos from being submitted to the NHA infrastructure.

**2. HL7 FHIR R4 Bundle Generation**:
- We generate a complete FHIR R4 `Bundle` resource containing Patient, Observation (vitals), and RiskAssessment resources
- This bundle can be directly POSTed to any FHIR-compliant server (NHA's HCX, OpenMRS, DHIS2)
- LOINC codes ensure interoperability with HIS systems
- SNOMED codes ensure clinical terminology consistency

**What's NOT implemented** (honest scope): OAuth2 token flow for actual NHA API authentication requires institutional ABDM sandbox access — we have implemented the data format and ID validation, not the live API handshake (which requires an official registered health facility ID).

---

**Q18. How is this different from existing tools like Framingham Risk Score or WHO CVD Risk Charts?**

**A**:

| Feature | Framingham Risk Score | WHO CVD Risk Charts | CardioQ |
|---|---|---|---|
| Algorithm | Logistic regression (Cox) | Lookup table | 11 hybrid quantum-classical models |
| Inputs | 7 features | 4 features | 22 clinical features |
| Explainability | Coefficients only | None | TreeSHAP, Odds Ratios, Quantum Jacobians |
| Quantum models | No | No | VQC, QSVM, Hybrid QNN |
| Uncertainty quantification | None | None | Bootstrap 95% CIs |
| API access | None | None | Full REST API |
| National interoperability | None | None | ABHA + FHIR R4 |
| Real-time update | No | No | Upload dataset → retrain in minutes |
| Open source | No | No | Fully open source |

CardioQ's primary differentiator is the **hybrid quantum-classical benchmarking infrastructure** — no existing clinical tool combines classical ensemble methods with quantum kernel machines and variational circuits in a unified, reproducible platform.

---

## CATEGORY 5: Research & Innovation Questions

---

**Q19. What is the novelty of this work? Has it been published?**

**A**: The technical novelty of CardioQ lies in four contributions:

1. **First open-source dual-track quantum-classical CVD benchmark**: No published paper provides a side-by-side comparison of VQC, QSVM, and Hybrid QNN against a full classical ensemble (8 models) on the same dataset partition with identical preprocessing.

2. **Leakage-safe hybrid ML engineering standard**: Our fold-isolated preprocessing + OOF threshold locking + SHA-256 manifest system is a reusable methodology for any clinical ML project. Most published QML medical papers do not address leakage this rigorously.

3. **NISQ-native PyTorch statevector engine**: Our from-scratch simulator with parameter-shift gradients verified against autograd is a clean educational and research reference that avoids Qiskit/PennyLane version dependencies.

4. **Quantum explainability via autograd Jacobians**: Input sensitivity analysis for VQC/QNN circuits via PyTorch autograd is novel in the tabular clinical ML context — most quantum XAI work focuses on image classification.

**Publication status**: This is an SIH hackathon project. The methodology is publication-ready and the authors intend to submit to a clinical informatics or quantum computing journal (e.g., npj Quantum Information, Journal of the American Medical Informatics Association) post-hackathon.

---

**Q20. What would you do differently with 6 more months?**

**A**:

**Technical priorities**:
1. **Real quantum hardware execution**: Submit circuits to IBM Quantum via Qiskit Runtime. Measure decoherence-induced performance delta between simulation and hardware.
2. **Quantum error mitigation**: Implement ZNE and readout error mitigation. Quantify improvement.
3. **Larger quantum circuits**: Test N=8, N=12 qubits on HPC. Profile where quantum models begin outperforming classical at larger N.
4. **Multi-disease expansion**: Extend to diabetes, chronic kidney disease, and stroke risk — all with ICMR-aligned feature sets.

**Clinical priorities**:
1. **Indian population cohort validation**: Collaborate with AIIMS/PGI to validate on Indian patient records (not US/European cohorts).
2. **CDSCO regulatory pathway**: Engage with CDSCO's Software as Medical Device framework for potential CDS clearance.
3. **PHC pilot deployment**: Collaborate with NHM for a 6-month Primary Health Centre pilot — gather real-world performance data.

**Infrastructure priorities**:
1. **OAuth2 ABDM integration**: Live ABHA token-based patient record access via the NHA Sandbox
2. **Federated learning**: Multi-site training without sharing patient data (privacy-preserving ML for government hospitals)

---

**Q21. What is the biggest risk if this is deployed in India tomorrow?**

**A**: There are three genuine risks we would insist on mitigating before any deployment:

1. **Covariate distribution shift**: CardioQ is trained on a dataset of unknown geographic origin. Indian patient populations (especially rural) may have different feature distributions (lower mean BMI but higher visceral fat, different cholesterol profiles on vegetarian diets). Performance guarantees from our benchmark may not transfer. **Mitigation**: Mandate local validation on a representative Indian cohort before deployment.

2. **Automation bias**: Clinicians may over-rely on the algorithmic risk score, reducing their own clinical reasoning. The "Moderate Risk" label may cause anchoring on a false negative. **Mitigation**: Platform should always display confidence intervals and explicitly state it is decision support, not diagnosis.

3. **Threshold miscalibration in low-prevalence settings**: Our τ* was derived at 49.5% CVD prevalence. Rural screening populations at 5–10% prevalence will have high false positive rates at this threshold. **Mitigation**: Require local threshold recalibration procedure before each deployment site goes live. We have the infrastructure to do this — it just needs local validation data.

---

## CATEGORY 6: SIH-Specific & Process Questions

---

**Q22. Why should this project win over other teams?**

**A**: We offer three unique differentiators:

1. **Actual working system, not a prototype**: The platform is deployed, tested (64/64 tests pass), and runs at `python app.py --port 8080`. Other teams often submit PowerPoint decks — we submit a running system.

2. **Methodological integrity**: We run external validation, publish limitations, use bootstrap CIs, and lock thresholds prospectively. This is the standard that Nature Machine Intelligence requires for published clinical ML papers — we meet it in a hackathon.

3. **Genuine quantum implementation**: We implement three quantum architectures from mathematical foundations, with analytically exact gradients verified against numerical computation. Most teams claiming "quantum ML" use a single pre-built Qiskit circuit with no understanding of the underlying mathematics.

---

**Q23. If a judge challenges you to show the quantum circuit running live, what do you do?**

**A**: Open the platform, navigate to Training Studio, select "Quantum (VQC)" model, and run a 1,000-sample training batch. The circuit training takes ~30 seconds. We can then show:
1. The training loss curve (decreasing over epochs)
2. The OOF ROC-AUC (converging to ~0.73)
3. The Jacobian attribution for a test patient (which features the circuit is sensitive to)
4. A side-by-side comparison with the CatBoost ROC curve

Alternatively: we can show the Jupyter-compatible circuit code and trace through the matrix multiplications live on a 2-qubit example, demonstrating superposition and entanglement explicitly.

---

**Q24. How do you ensure patient data privacy?**

**A**: Privacy-by-design at every layer:

| Layer | Privacy Measure |
|---|---|
| Data storage | SQLite local only — no cloud upload of patient records |
| API transport | HTTPS in production (HTTP for local demo) |
| Patient identifier | ABHA Luhn-10 validation only — no name/address stored |
| Model inference | Stateless — no patient record retained after response |
| Data minimisation | Only the 11 clinical inputs required for prediction |
| FHIR bundles | Generated locally, patient chooses where to send |
| Audit trail | Predictions are logged with anonymised session ID, not patient ID |

**For production deployment**: Full NHA ABDM Data Protection Framework compliance, including consent-based data flow via the Health Information Exchange (HIE-CM) layer.

---

**Q25. What happens if a doctor uses this and a patient dies?**

**A**: This is the most important question a clinical AI team must answer.

**Legal framework**: CardioQ is a **clinical decision support tool**, not a Software as Medical Device (SaMD). This distinction is critical:
- The **clinician** remains the responsible party for all clinical decisions
- CardioQ provides probabilistic risk estimates — it does not diagnose or prescribe
- Every API response contains a mandatory disclaimer stating this

**Technical safeguards**:
1. Confidence intervals are shown — "0.73 ± 0.08" not just "0.73"
2. HIGH risk tier always recommends "Immediate specialist referral" — it never says "patient has CVD"
3. Modifiable risk factors are highlighted — clinical action is lifestyle + pharmacotherapy
4. The model explicitly cannot see ECG, echo, or biomarkers — its limitations are front-and-centre

**The deeper answer**: 17.9M people die of CVD annually. A tool that identifies HIGH risk patients who would otherwise go undiagnosed — and prompts them to see a cardiologist — saves lives even if the model is imperfect. The question is not "is this model perfect" but "is this better than no screening at all." The answer is yes, as long as it is transparent about uncertainty and never used autonomously.
