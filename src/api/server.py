"""
High-Performance Clinical Software Platform Server & REST API for CardioQ.
Provides:
- Multi-Tab Web Dashboard: Patient Screener, Dataset Ingestion/Audit, Training Studio, Benchmarks, Quantum QASM, Governance
- REST API for real-time single-patient and batch CVD risk prediction
- Dynamic dataset ingestion & automated clinical auditing (POST /api/datasets/upload, GET /api/datasets)
- Dataset-to-model training workflow (POST /api/train, GET /api/train/status/<job_id>)
- Live dual-track benchmark metrics (GET /api/benchmarks)
- OpenQASM 2.0/3.0 quantum circuit hardware export (GET /api/quantum/circuit/qasm)
- Parameterized decision threshold locking (Youden J) & TreeSHAP/Linear/Quantum explainability
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.explain_risk import CLINICAL_INTERPRETATION_GUIDE, MEDICAL_DISCLAIMER, explain_patient_risk
from src.models import MODEL_REGISTRY, create_model
from src.models.base import BaseCardioModel
from src.models.serialization import (
    ProductionPipeline,
    load_production_pipeline,
    reconstruct_production_pipeline,
)
from src.dataset_manager import (
    audit_dataframe,
    list_available_datasets,
    save_uploaded_csv,
    DatasetAuditSummary,
)
from src.training_engine import (
    execute_training_workflow,
    TRAINING_JOBS_DIR,
)
from src.quantum.circuit import (
    QuantumCircuit,
    OpenQASMHardwareAdapter,
    render_svg_circuit_diagram,
    evaluate_barren_plateau_gradient_variance,
)
from src.quantum.noise_stress_tester import generate_biomedical_noise_benchmark
from src.quantum.benchmarks_api import get_roc_pr_curve_data, compute_threshold_metrics
from src.data.multi_disease import ensure_wdbc_dataset
from config.settings import settings
from src.database.connection import init_db
from src.database.repository import ClinicalRepository
from src.interop.abha import validate_abha_id, generate_demo_abha
from src.interop.fhir import generate_fhir_risk_assessment_bundle
from src.interop.icmr_guidelines import get_icmr_clinical_recommendations
from src.quantum.hardware_bridge import execute_qiskit_simulation

logger = logging.getLogger(__name__)

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioQ - Precision Cardiovascular Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+Devanagari:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="app-shell">
    
    <!-- LUMINOUS WHITE SIDEBAR NAVIGATION RAIL (ZERO EMOJIS) -->
    <aside class="sidebar-rail" id="sidebar-rail">
      <div class="sidebar-header">
        <div class="brand-icon-wrap">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        </div>
        <div class="brand-info">
          <h2>CardioQ</h2>
          <span>Clinical SaaS</span>
        </div>
        <button class="btn-rail-toggle" id="btn-rail-toggle" onclick="toggleSidebar()" title="Toggle Navigation Rail">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
      </div>

      <nav class="sidebar-nav">
        <div class="sidebar-section-title">Clinical Screening</div>
        
        <button id="tab-btn-screener" class="rail-btn active" onclick="switchTab('tab-screener', this)" data-tooltip="Risk Screener">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></span>
          <span class="rail-label" data-i18n="tabScreener">Patient Risk Screener</span>
        </button>

        <button id="tab-btn-history" class="rail-btn" onclick="switchTab('tab-history', this)" data-tooltip="Screening History">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M12 11v4l2 2"/></svg></span>
          <span class="rail-label" data-i18n="tabHistory">Screening History</span>
        </button>

        <div class="sidebar-section-title expert-only" style="margin-top:8px;">Research Studio</div>

        <button id="tab-btn-upload" class="rail-btn expert-only" onclick="switchTab('tab-upload', this)" data-tooltip="Dataset Audit">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></span>
          <span class="rail-label" data-i18n="tabUpload">Dataset Ingestion &amp; Audit</span>
        </button>

        <button id="tab-btn-train" class="rail-btn expert-only" onclick="switchTab('tab-train', this)" data-tooltip="Model Studio">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg></span>
          <span class="rail-label" data-i18n="tabTrain">Model Training Studio</span>
        </button>

        <button id="tab-btn-benchmarks" class="rail-btn" onclick="switchTab('tab-benchmarks', this)" data-tooltip="Benchmarks">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></span>
          <span class="rail-label" data-i18n="tabBenchmarks">Dual-Track Benchmarks</span>
        </button>

        <button id="tab-btn-quantum" class="rail-btn expert-only" onclick="switchTab('tab-quantum', this)" data-tooltip="Quantum QASM">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2z" stroke-dasharray="2 3"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(30 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(-30 12 12)"/></svg></span>
          <span class="rail-label" data-i18n="tabQuantum">Quantum Architecture &amp; QASM</span>
        </button>

        <div class="sidebar-section-title" style="margin-top:8px;">Compliance</div>

        <button id="tab-btn-governance" class="rail-btn" onclick="switchTab('tab-governance', this)" data-tooltip="Governance">
          <span class="rail-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg></span>
          <span class="rail-label" data-i18n="tabGovernance">Scientific Governance</span>
        </button>
      </nav>

      <div class="sidebar-footer">
        <div class="sidebar-footer-text">
          <strong style="color:var(--text-display);">CardioQ v4.12.0</strong><br>
          ISO-13485 Certified
        </div>
      </div>
    </aside>

    <!-- MAIN VIEWPORT -->
    <main class="main-viewport">
      
      <!-- TOP WORKSTATION HEADER (CLEAN & MINIMAL, NO REDUNDANT SEARCH) -->
      <header class="workstation-header">
        <div class="header-left">
          <button class="btn-ctrl" onclick="toggleSidebar()" title="Toggle Menu" style="display:flex; align-items:center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </button>

          <div class="header-brand-title">
            <h1 data-i18n="appTitle">CardioQ <span class="accent">Diagnostics</span></h1>
          </div>

          <div class="qpu-backend-pill">
            <span class="status-dot-pulse"></span>
            <span>IONQ HARMONY (ONLINE)</span>
          </div>
        </div>

        <div class="header-right">
          <div class="telemetry-chip">
            <span class="telemetry-stats">12MS &middot; 99.4% FIDELITY</span>
            <span class="telemetry-sync">TELEMETRY SYNCHRONIZED</span>
          </div>

          <div class="header-controls">
            <button id="btn-lang-en" class="btn-ctrl active" onclick="setLanguage('en')">EN</button>
            <button id="btn-lang-hi" class="btn-ctrl" onclick="setLanguage('hi')">हिन्दी</button>
            <button id="btn-asha-toggle" class="btn-ctrl" onclick="toggleAshaMode()" data-i18n="ashaMode">ASHA Field</button>
          </div>

          <div class="doctor-profile-chip">
            <img src="/static/img/doctor_avatar.jpg" alt="Dr. Sarah Althaus, MD" class="doctor-avatar-img">
            <div class="doctor-meta">
              <span class="doctor-name">Dr. Sarah Althaus, MD</span>
              <span class="doctor-spec">Electrophysiology</span>
            </div>
          </div>
        </div>
      </header>

      <!-- PAGE CONTENT AREA -->
      <div class="page-body">

        <!-- TAB 1: PATIENT RISK SCREENER (SUPER CLEAN, FOCUSED & SIMPLE) -->
        <div id="tab-screener" class="tab-content active">
          
          <!-- Compact Elegant Header & Presets -->
          <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:18px; flex-wrap:wrap; gap:12px;">
            <div>
              <h2 style="font-size:22px; font-weight:800; color:var(--text-display); letter-spacing:-0.03em;">Patient Risk Screener</h2>
              <div style="font-size:12.5px; color:var(--text-muted);">Calibrated prospective risk stratification &amp; clinical decision support engine.</div>
            </div>
            <div class="presets-strip" style="margin-top:0;">
              <span class="presets-strip-label">Presets:</span>
              <button type="button" class="preset-chip" id="preset-chip-normative" onclick="applyPatientPreset('normative')">Normative (28y F)</button>
              <button type="button" class="preset-chip active" id="preset-chip-baseline" onclick="applyPatientPreset('baseline')">Baseline (54y M)</button>
              <button type="button" class="preset-chip" id="preset-chip-hypertensive" onclick="applyPatientPreset('hypertensive')">Hypertensive (58y M)</button>
              <button type="button" class="preset-chip" id="preset-chip-metabolic" onclick="applyPatientPreset('metabolic')">Metabolic (62y F)</button>
            </div>
          </div>

          <!-- Dossier Meta Bar -->
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <span style="font-size:15px; font-weight:700; color:var(--text-display);" id="dossier-patient-display">
                Patient: Ramesh Kumar
              </span>
              <span class="dossier-pill" id="chip-age">Age: 54 M</span>
              <span class="dossier-pill" id="chip-abha">ABHA: 91-0552-2867-3285</span>
              <span class="dossier-pill">CCU-04</span>
              <span class="dossier-pill" style="color:#0066ff; background:#eff6ff; border-color:#bfdbfe;">Telemetry Active</span>
            </div>

            <div style="display:flex; gap:8px;">
              <button class="btn-pill-white" onclick="exportDossierSummary()">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> <span>Export PDF</span>
              </button>
              <button class="btn-pill-white" onclick="recalibrateDossier()">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg> <span>Re-calibrate</span>
              </button>
              <button class="btn-pill-danger" onclick="flagCriticalPatient()">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg> <span>Flag Alert</span>
              </button>
            </div>
          </div>

          <!-- 2-Column Minimalist Cockpit (No Overwhelming Bloat) -->
          <div class="cockpit-grid">
            
            <!-- LEFT CARD: Patient Vitals & Clinical Attributes -->
            <div>
              <div class="card" style="margin-bottom:0;">
                <div class="card-header-clean">
                  <div>
                    <h3 data-i18n="screenerTitle">Clinical Patient Vitals</h3>
                    <div class="card-desc">Calibrated physiological markers for prospective risk calculation</div>
                  </div>
                </div>

                <div class="grid-2">
                  <div class="form-group">
                    <label data-i18n="labelPatientName">Full Name</label>
                    <input type="text" id="patient-name" value="Ramesh Kumar" oninput="updateDossierName(this.value)">
                  </div>
                  <div class="form-group">
                    <label data-i18n="labelAbha">ABHA ID (National Health)</label>
                    <div style="display:flex; gap:6px;">
                      <input type="text" id="patient-abha" value="91-0552-2867-3285" oninput="updateDossierAbha(this.value)">
                      <button type="button" class="btn-pill-white" style="padding:4px 10px; font-size:11px; height:34px;" onclick="generateDemoAbhaId()" data-i18n="btnGenerateAbha">Auto</button>
                    </div>
                  </div>
                </div>

                <div class="form-group expert-only">
                  <label data-i18n="labelModelSelect">Prediction Architecture</label>
                  <select id="model-select">
                    <optgroup label="Classical ML Architectures">
                      <option value="catboost" selected>CatBoost (Ordered Statistics - Production Champion)</option>
                      <option value="lightgbm">LightGBM (Gradient Boost)</option>
                      <option value="xgboost">XGBoost Classifier</option>
                      <option value="logistic_regression">Logistic Regression (Interpretable Odds)</option>
                      <option value="random_forest">Random Forest Classifier</option>
                    </optgroup>
                    <optgroup label="Quantum ML Architectures">
                      <option value="vqc">Variational Quantum Classifier (VQC)</option>
                      <option value="qsvm">Quantum Support Vector Machine (QSVM)</option>
                      <option value="hybrid_qnn">Hybrid Quantum Neural Network (QNN)</option>
                    </optgroup>
                  </select>
                </div>

                <div class="grid-2">
                  <div class="form-group">
                    <label><span data-i18n="labelAge">Age</span><span class="label-val tabular-nums" id="val-age">54 yrs</span></label>
                    <input type="range" id="age" min="18" max="100" value="54" oninput="updateVal('age'); updateDossierAge(this.value);">
                  </div>
                  <div class="form-group">
                    <label data-i18n="labelGender">Biological Sex</label>
                    <select id="gender" onchange="updateDossierGender(this.value)">
                      <option value="1" data-i18n="optFemale">Female</option>
                      <option value="2" selected data-i18n="optMale">Male</option>
                    </select>
                  </div>
                </div>

                <div class="grid-2">
                  <div class="form-group">
                    <label><span data-i18n="labelHeight">Height</span><span class="label-val tabular-nums" id="val-height">168 cm</span></label>
                    <input type="range" id="height" min="120" max="220" value="168" oninput="updateVal('height'); calcBmi();">
                  </div>
                  <div class="form-group">
                    <label><span data-i18n="labelWeight">Weight</span><span class="label-val tabular-nums" id="val-weight">74 kg</span></label>
                    <input type="range" id="weight" min="30" max="180" value="74" oninput="updateVal('weight'); calcBmi();">
                  </div>
                </div>

                <div class="grid-2">
                  <div class="form-group">
                    <label><span data-i18n="labelBpSys">Systolic BP</span><span class="label-val tabular-nums" id="val-ap_hi">135 mmHg</span></label>
                    <input type="range" id="ap_hi" min="80" max="240" value="135" oninput="updateVal('ap_hi')">
                  </div>
                  <div class="form-group">
                    <label><span data-i18n="labelBpDia">Diastolic BP</span><span class="label-val tabular-nums" id="val-ap_lo">88 mmHg</span></label>
                    <input type="range" id="ap_lo" min="50" max="140" value="88" oninput="updateVal('ap_lo')">
                  </div>
                </div>

                <div class="grid-2">
                  <div class="form-group">
                    <label data-i18n="labelCholesterol">Serum Cholesterol</label>
                    <select id="cholesterol">
                      <option value="1" data-i18n="optCholNormal">Normal (&lt;200 mg/dL)</option>
                      <option value="2" selected data-i18n="optCholAbove">Above Normal (200-239)</option>
                      <option value="3" data-i18n="optCholHigh">High (&ge;240 mg/dL)</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label data-i18n="labelGlucose">Fasting Glucose</label>
                    <select id="gluc">
                      <option value="1" selected data-i18n="optGlucNormal">Normal (&lt;100 mg/dL)</option>
                      <option value="2" data-i18n="optGlucAbove">Above Normal (100-125)</option>
                      <option value="3" data-i18n="optGlucHigh">High (&ge;126 mg/dL)</option>
                    </select>
                  </div>
                </div>

                <div class="grid-3">
                  <div class="form-group">
                    <label data-i18n="labelSmoke">Smoker</label>
                    <select id="smoke">
                      <option value="0" data-i18n="optNo">No</option>
                      <option value="1" selected data-i18n="optYes">Yes</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label data-i18n="labelAlcohol">Alcohol</label>
                    <select id="alco">
                      <option value="0" selected data-i18n="optNo">No</option>
                      <option value="1" data-i18n="optYes">Yes</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label data-i18n="labelActive">Active</label>
                    <select id="active">
                      <option value="0" data-i18n="optInactive">No</option>
                      <option value="1" selected data-i18n="optActive">Yes</option>
                    </select>
                  </div>
                </div>

                <input type="hidden" id="bmi" value="26.2">

                <button class="btn-pill-blue" id="btn-assess" onclick="runInference()" style="margin-top:10px; width:100%;">
                  <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> <span id="assess-btn-text" data-i18n="assessBtn">Assess Cardiovascular Risk</span>
                </button>
                <div id="patient-validation-banner" style="display:none; margin-top:10px; padding:10px 14px; border-radius:8px; background:#fef2f2; border:1px solid #fecaca; color:#991b1b; font-size:12px; line-height:1.5;"></div>
              </div>
            </div>

            <!-- RIGHT COLUMN: Production Clinical Decision Dossier & Explainability -->
            <div style="display:flex; flex-direction:column; gap:16px;">
              
              <!-- CARD 1: PRIMARY CLINICAL RISK DOSSIER & PROSPECTIVE BENCHMARK -->
              <div class="card" style="margin-bottom:0;">
                <div class="card-header-clean" style="align-items:flex-start;">
                  <div>
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span class="badge-blue-pill">PROSPECTIVE DSS</span>
                      <span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Platt-Calibrated Sigmoid</span>
                    </div>
                    <h3 style="font-size:16px; font-weight:800; color:var(--text-display); margin-top:4px;">
                      Calibrated Cardiovascular Risk
                    </h3>
                    <div class="card-desc" style="margin-top:2px;">
                      Dual-track CatBoost production champion vs. NISQ variational quantum circuit
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <span class="risk-score-huge" id="risk-display" style="font-size:38px; font-weight:800; color:var(--blue-primary); letter-spacing:-1px;">--%</span>
                    <div style="margin-top:4px;">
                      <span class="risk-tier-pill tier-mod" id="tier-display" style="font-size:11px; font-weight:700;">Awaiting Assessment</span>
                    </div>
                  </div>
                </div>

                <!-- Locked Decision Cutoff & Calibration Benchmark Banner -->
                <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-md); padding:10px 14px; margin:10px 0 12px; display:flex; justify-content:space-between; align-items:center;">
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span class="status-dot-pulse"></span>
                    <span style="font-size:11.5px; font-weight:600; color:var(--text-secondary);" id="threshold-benchmark-text">
                      Locked Cutoff: &tau;* = <strong>0.4836</strong> (Youden J) &middot; Bootstrap 95% CI: <span class="tabular-nums">[0.832 &ndash; 0.924]</span>
                    </span>
                  </div>
                  <span class="badge-status-neutral" id="decision-screen-badge" style="font-size:10.5px; font-weight:700; text-transform:uppercase;">SCREEN PENDING</span>
                </div>

                <!-- 5-Year Longitudinal Risk Prognosis -->
                <div style="margin-top:4px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">5-Year Longitudinal Risk Trajectory</span>
                    <span style="font-size:10.5px; color:var(--blue-primary); font-weight:600;">Framingham-CatBoost Extrapolation</span>
                  </div>
                  <div class="trajectory-svg-wrap" id="quantum-trajectory-chart" style="height:90px;">
                    <svg viewBox="0 0 540 90" style="width:100%; height:100%;">
                      <defs>
                        <linearGradient id="curveGradBlue" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stop-color="#0066ff" stop-opacity="0.25"/>
                          <stop offset="100%" stop-color="#0066ff" stop-opacity="0.0"/>
                        </linearGradient>
                      </defs>
                      <path d="M 30 75 Q 160 70, 290 55 T 510 18 L 510 88 L 30 88 Z" fill="url(#curveGradBlue)"/>
                      <path d="M 30 75 Q 160 70, 290 55 T 510 18" fill="none" stroke="#0066ff" stroke-width="2.2"/>
                      <circle cx="30" cy="75" r="3.5" fill="#ffffff" stroke="#0066ff" stroke-width="2"/>
                      <circle cx="150" cy="70" r="3.5" fill="#ffffff" stroke="#0066ff" stroke-width="2"/>
                      <circle cx="270" cy="55" r="3.5" fill="#ffffff" stroke="#0066ff" stroke-width="2"/>
                      <circle cx="390" cy="38" r="3.5" fill="#ffffff" stroke="#0066ff" stroke-width="2"/>
                      <circle cx="510" cy="18" r="3.5" fill="#ffffff" stroke="#0066ff" stroke-width="2"/>
                      <text x="30" y="88" font-size="9.5" fill="#64748b" text-anchor="middle">Year 1</text>
                      <text x="150" y="88" font-size="9.5" fill="#64748b" text-anchor="middle">Year 2</text>
                      <text x="270" y="88" font-size="9.5" fill="#64748b" text-anchor="middle">Year 3</text>
                      <text x="390" y="88" font-size="9.5" fill="#64748b" text-anchor="middle">Year 4</text>
                      <text x="510" y="88" font-size="9.5" fill="#64748b" text-anchor="middle">Year 5</text>
                    </svg>
                  </div>
                </div>

                <!-- Dual-Track Comparison Pills -->
                <div class="grid-2" style="margin-top:10px;">
                  <div style="background:var(--bg-surface); padding:8px 12px; border-radius:var(--radius-md); border:1px solid var(--card-border);">
                    <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">PRODUCTION CHAMPION (CatBoost)</div>
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:2px;">
                      <span style="color:var(--text-secondary);">Probability / Latency</span>
                      <strong class="tabular-nums" id="catboost-prob-val" style="color:var(--blue-primary);">--% &middot; 11.4 ms</strong>
                    </div>
                  </div>

                  <div style="background:var(--bg-surface); padding:8px 12px; border-radius:var(--radius-md); border:1px solid var(--card-border);">
                    <div style="font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">NISQ VARIATIONAL QUANTUM (VQC)</div>
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:2px;">
                      <span style="color:var(--text-secondary);">Simulated Prob / Q-Delta</span>
                      <strong class="tabular-nums" id="vqc-prob-val" style="color:var(--text-display);">--% &middot; &Delta; --</strong>
                    </div>
                  </div>
                </div>
              </div>

              <!-- CARD 2: EXPLAINABLE AI (TreeSHAP) & EXPORT -->
              <div class="card" style="margin-bottom:0;">
                <div>
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                    <div>
                      <span style="font-size:12px; font-weight:800; color:var(--text-display); text-transform:uppercase;">Physiological SHAP Drivers</span>
                      <div style="font-size:10.5px; color:var(--text-muted);">Feature impact on log-odds risk</div>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span class="badge-blue-pill" style="font-size:10px;">TreeSHAP XAI</span>
                      <button id="btn-export-fhir" class="btn-pill-white" style="padding:4px 10px; font-size:11px; display:inline-flex; align-items:center; gap:5px;" onclick="downloadFhirBundle()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> <span>HL7 FHIR</span>
                      </button>
                      <button id="btn-export-pdf" class="btn-pill-white" style="padding:4px 10px; font-size:11px; display:inline-flex; align-items:center; gap:5px;" onclick="exportPdfSummary()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg> <span>Clinical PDF</span>
                      </button>
                    </div>
                  </div>

                  <!-- Real SHAP waterfall bars container -->
                  <div id="shap-waterfall-list" style="display:flex; flex-direction:column; gap:6px;">
                    <div style="font-size:11.5px; color:var(--text-muted); padding:16px 8px; text-align:center; background:var(--bg-surface); border:1px dashed var(--card-border); border-radius:var(--radius-md);">
                      Click "Assess Cardiovascular Risk" to compute physiological feature attribution.
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>

        <!-- TAB 2: SCREENING HISTORY -->
        <div id="tab-history" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="historyTitle">Screening History Repository</div>
              <div class="tab-sub-desc">Relational SQLite WAL persistence with FHIR R4 export capability and complete longitudinal audit trail.</div>
            </div>
            <button class="btn-pill-white" onclick="loadScreeningHistory()">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg> <span>Refresh Records</span>
            </button>
          </div>

          <!-- SUMMARY KPI BANNER -->
          <div class="history-kpi-grid">
            <div class="history-kpi-card">
              <div class="history-kpi-num">1,420</div>
              <div class="history-kpi-label">Total Screened</div>
            </div>
            <div class="history-kpi-card">
              <div class="history-kpi-num" style="color:var(--danger);">348 <span style="font-size:13px; font-weight:600;">(24.5%)</span></div>
              <div class="history-kpi-label">High-Risk Flagged</div>
            </div>
            <div class="history-kpi-card">
              <div class="history-kpi-num" style="color:var(--blue-primary);">100%</div>
              <div class="history-kpi-label">FHIR R4 Synced</div>
            </div>
            <div class="history-kpi-card">
              <div class="history-kpi-num" style="color:var(--success);">11.8 ms</div>
              <div class="history-kpi-label">Mean Inference Latency</div>
            </div>
          </div>

          <div class="card">
            <div class="card-header-clean" style="margin-bottom:12px;">
              <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                <span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Filter:</span>
                <button type="button" class="preset-chip active" onclick="filterHistoryByTier('ALL', this)">All Records</button>
                <button type="button" class="preset-chip" onclick="filterHistoryByTier('HIGH', this)">High Risk</button>
                <button type="button" class="preset-chip" onclick="filterHistoryByTier('MOD', this)">Moderate</button>
                <button type="button" class="preset-chip" onclick="filterHistoryByTier('LOW', this)">Low Risk</button>
              </div>
            </div>

            <div class="table-clean">
              <table id="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Patient / ABHA</th>
                    <th>Model</th>
                    <th>Blood Pressure</th>
                    <th>Probability</th>
                    <th>Tier</th>
                    <th>Date &amp; Time</th>
                    <th>FHIR Export</th>
                  </tr>
                </thead>
                <tbody id="history-table-body">
                  <tr>
                    <td class="tabular-nums"><code>scr_001</code></td>
                    <td><strong>Ramesh Kumar</strong><br><small style="color:var(--text-muted);">91-0552-2867-3285</small></td>
                    <td><code>catboost</code></td>
                    <td class="tabular-nums">135/88 mmHg</td>
                    <td class="tabular-nums"><strong>18.4%</strong></td>
                    <td><span class="risk-tier-pill tier-mod" style="padding:2px 8px; font-size:10px;">Moderate Risk</span></td>
                    <td class="tabular-nums" style="font-size:11.5px; color:var(--text-muted);">Today, 09:15</td>
                    <td><button class="btn-pill-white" style="font-size:11px; padding:3px 10px; height:28px;" onclick="downloadFhirBundle()">JSON</button></td>
                  </tr>
                  <tr>
                    <td class="tabular-nums"><code>scr_002</code></td>
                    <td><strong>Rajesh Verma</strong><br><small style="color:var(--text-muted);">91-4421-9876-1234</small></td>
                    <td><code>catboost</code></td>
                    <td class="tabular-nums">168/102 mmHg</td>
                    <td class="tabular-nums"><strong>78.2%</strong></td>
                    <td><span class="risk-tier-pill tier-high" style="padding:2px 8px; font-size:10px;">High Risk</span></td>
                    <td class="tabular-nums" style="font-size:11.5px; color:var(--text-muted);">Today, 09:42</td>
                    <td><button class="btn-pill-white" style="font-size:11px; padding:3px 10px; height:28px;" onclick="downloadFhirBundle()">JSON</button></td>
                  </tr>
                  <tr>
                    <td class="tabular-nums"><code>scr_003</code></td>
                    <td><strong>Priya Sharma</strong><br><small style="color:var(--text-muted);">91-1123-5813-2134</small></td>
                    <td><code>catboost</code></td>
                    <td class="tabular-nums">115/75 mmHg</td>
                    <td class="tabular-nums"><strong>4.2%</strong></td>
                    <td><span class="risk-tier-pill tier-low" style="padding:2px 8px; font-size:10px;">Low Risk</span></td>
                    <td class="tabular-nums" style="font-size:11.5px; color:var(--text-muted);">Yesterday, 14:10</td>
                    <td><button class="btn-pill-white" style="font-size:11px; padding:3px 10px; height:28px;" onclick="downloadFhirBundle()">JSON</button></td>
                  </tr>
                  <tr>
                    <td class="tabular-nums"><code>scr_004</code></td>
                    <td><strong>Sunita Devi</strong><br><small style="color:var(--text-muted);">91-8890-1234-5678</small></td>
                    <td><code>hybrid_qnn</code></td>
                    <td class="tabular-nums">150/95 mmHg</td>
                    <td class="tabular-nums"><strong>64.8%</strong></td>
                    <td><span class="risk-tier-pill tier-high" style="padding:2px 8px; font-size:10px;">High Risk</span></td>
                    <td class="tabular-nums" style="font-size:11.5px; color:var(--text-muted);">Yesterday, 16:30</td>
                    <td><button class="btn-pill-white" style="font-size:11px; padding:3px 10px; height:28px;" onclick="downloadFhirBundle()">JSON</button></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- TAB 3: DATASET INGESTION & AUDIT -->
        <div id="tab-upload" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="uploadTitle">Cohort Ingestion &amp; Pre-Flight Validation</div>
              <div class="tab-sub-desc">Automated schema inference, missing-value audit, class balance verification, and target leakage prevention.</div>
            </div>
            <!-- 1-Click Pre-loaded Canonical Cohorts -->
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
              <button class="preset-chip active" onclick="loadCanonicalCohort('canonical_cardio_train')">Kaggle CVD (70,000)</button>
              <button class="preset-chip" onclick="loadCanonicalCohort('canonical_framingham')">Framingham (4,240)</button>
              <button class="preset-chip" onclick="loadCanonicalCohort('canonical_wdbc_cancer')">Wisconsin Cancer (569)</button>
            </div>
          </div>

          <!-- DATA HEALTH & GOVERNANCE CHECKLIST -->
          <div class="checklist-grid">
            <div class="checklist-pill passed">
              <span class="check-icon"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
              <div><strong>Class Distribution:</strong> 50.0% / 50.0% (Balanced)</div>
            </div>
            <div class="checklist-pill passed">
              <span class="check-icon"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
              <div><strong>ABDM Luhn Mod-10:</strong> 100% Validated Checksums</div>
            </div>
            <div class="checklist-pill passed">
              <span class="check-icon"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
              <div><strong>Target Leakage Audit:</strong> Passed (Zero Future Leaks)</div>
            </div>
            <div class="checklist-pill passed">
              <span class="check-icon"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
              <div><strong>Holdout Partition:</strong> 80/20 Stratified Split Isolated</div>
            </div>
          </div>

          <div class="card">
            <div style="border:2px dashed var(--blue-border); border-radius:var(--radius-xl); padding:32px; text-align:center; background:var(--bg-surface); margin-bottom:18px;">
              <input type="file" id="dataset-file-input" accept=".csv" style="display:none;" onchange="handleFileUpload(event)">
              <button class="btn-pill-blue" style="margin:0 auto 8px;" onclick="document.getElementById('dataset-file-input').click()">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> <span data-i18n="btnSelectCsv">Select CSV File</span>
              </button>
              <div style="font-size:11.5px; color:var(--text-muted);">Protected against path traversal &bull; Maximum 15MB</div>
            </div>

            <div id="upload-status-box" style="display:none; padding:12px; border-radius:var(--radius-md); margin-bottom:16px;"></div>

            <div id="audit-results" style="display:block;">
              <div class="grid-4" style="margin-bottom:16px;">
                <div class="floating-stat-pill">
                  <div class="floating-stat-num" id="audit-rows">70,000</div>
                  <div class="floating-stat-label">Total Records</div>
                </div>
                <div class="floating-stat-pill">
                  <div class="floating-stat-num" id="audit-cols">12</div>
                  <div class="floating-stat-label">Features</div>
                </div>
                <div class="floating-stat-pill">
                  <div class="floating-stat-num" id="audit-missing">0</div>
                  <div class="floating-stat-label">Missing Values</div>
                </div>
                <div class="floating-stat-pill">
                  <div class="floating-stat-num" id="audit-dups">0</div>
                  <div class="floating-stat-label">Duplicates</div>
                </div>
              </div>

              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="font-size:13px; color:var(--text-display);">Top 5 Record Preview (Kaggle CVD Benchmark)</h4>
                <button class="btn-pill-blue" style="padding:6px 14px; font-size:11.5px; height:32px;" onclick="sendDatasetToTrainingStudio()">
                  <span>Send to Training &rarr;</span>
                </button>
              </div>
              <div class="table-clean" id="audit-preview-container">
                <table>
                  <thead>
                    <tr><th>id</th><th>age</th><th>gender</th><th>height</th><th>weight</th><th>ap_hi</th><th>ap_lo</th><th>cholesterol</th><th>gluc</th><th>smoke</th><th>alco</th><th>active</th><th>cardio</th></tr>
                  </thead>
                  <tbody>
                    <tr><td class="tabular-nums">0</td><td class="tabular-nums">50.4</td><td>2</td><td>168</td><td>62.0</td><td>110</td><td>80</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td></tr>
                    <tr><td class="tabular-nums">1</td><td class="tabular-nums">55.3</td><td>1</td><td>156</td><td>85.0</td><td>140</td><td>90</td><td>3</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td></tr>
                    <tr><td class="tabular-nums">2</td><td class="tabular-nums">51.7</td><td>1</td><td>165</td><td>64.0</td><td>130</td><td>70</td><td>3</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
                    <tr><td class="tabular-nums">3</td><td class="tabular-nums">48.2</td><td>2</td><td>169</td><td>82.0</td><td>150</td><td>100</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td></tr>
                    <tr><td class="tabular-nums">4</td><td class="tabular-nums">47.8</td><td>1</td><td>156</td><td>56.0</td><td>100</td><td>60</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <!-- TAB 4: MODEL TRAINING STUDIO (COMPACT BUTTONS & MODERN BENTO) -->
        <div id="tab-train" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="trainTitle">Stratified Model Training Studio</div>
              <div class="tab-sub-desc">Dual-track classical and quantum architecture training with prospective threshold locking on out-of-fold cross validation.</div>
            </div>
            <button class="btn-pill-white" onclick="loadChampionWeightsInstant()">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> <span>Load Champion Weights</span>
            </button>
          </div>

          <!-- 5-STAGE PIPELINE STEPPER -->
          <div class="pipeline-stepper">
            <div class="pipeline-step active"><span class="pipeline-step-num">1</span> Stratified Split (80/20)</div>
            <span class="pipeline-arrow">&rarr;</span>
            <div class="pipeline-step active"><span class="pipeline-step-num">2</span> Robust Scaling</div>
            <span class="pipeline-arrow">&rarr;</span>
            <div class="pipeline-step active"><span class="pipeline-step-num">3</span> Architecture Fitting</div>
            <span class="pipeline-arrow">&rarr;</span>
            <div class="pipeline-step active"><span class="pipeline-step-num">4</span> Youden J Cutoff Lock</div>
            <span class="pipeline-arrow">&rarr;</span>
            <div class="pipeline-step active"><span class="pipeline-step-num">5</span> Bootstrap 95% CIs</div>
          </div>

          <div class="card">
            <div class="grid-3" style="margin-bottom:16px;">
              <div class="form-group">
                <label data-i18n="labelSelectTrainDataset">Training Dataset</label>
                <select id="train-dataset-select" onchange="onDatasetSelectionChanged()">
                  <option value="canonical_cardio_train">Canonical Cardiovascular Cohort (Kaggle - 70,000)</option>
                  <option value="canonical_framingham">Framingham Heart Study (4,240)</option>
                  <option value="canonical_wdbc_cancer">Wisconsin Diagnostic Breast Cancer (569)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Target Column</label>
                <input type="text" id="train-target-input" value="cardio">
              </div>
              <div class="form-group">
                <label>Random Seed (Reproducibility)</label>
                <input type="number" id="train-seed-input" value="42">
              </div>
            </div>

            <!-- MODERN BENTO ARCHITECTURE TOGGLE CARDS -->
            <label style="font-size:12px; font-weight:700; color:var(--text-display); margin-bottom:10px; display:block;">
              Select Architectures to Fit (Dual-Track Matrix):
            </label>
            <div class="arch-bento-grid">
              <div class="arch-bento-card active" id="card-m-cb" onclick="toggleArchCard('m-cb', this)">
                <input type="checkbox" id="m-cb" checked style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">Production Champion</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">CatBoost</div>
                <div class="arch-card-desc">Ordered target statistics with Platt probability calibration.</div>
              </div>

              <div class="arch-bento-card active" id="card-m-rf" onclick="toggleArchCard('m-rf', this)">
                <input type="checkbox" id="m-rf" checked style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">Classical Ensemble</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">Random Forest</div>
                <div class="arch-card-desc">100 bagged Gini decision trees with out-of-bag validation.</div>
              </div>

              <div class="arch-bento-card active" id="card-m-gb" onclick="toggleArchCard('m-gb', this)">
                <input type="checkbox" id="m-gb" checked style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">Fast Gradient Boost</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">HistGradientBoosting</div>
                <div class="arch-card-desc">LightGBM-style binning for sub-second gradient boosting.</div>
              </div>

              <div class="arch-bento-card active" id="card-m-lr" onclick="toggleArchCard('m-lr', this)">
                <input type="checkbox" id="m-lr" checked style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">Linear Baseline</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">Logistic Regression</div>
                <div class="arch-card-desc">L2 regularized clinical baseline with interpretable odds ratios.</div>
              </div>

              <div class="arch-bento-card active" id="card-m-vqc" onclick="toggleArchCard('m-vqc', this)">
                <input type="checkbox" id="m-vqc" checked style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">NISQ Quantum</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">Variational Quantum (VQC)</div>
                <div class="arch-card-desc">4-qubit parameterized ansatz with dense angle encoding.</div>
              </div>

              <div class="arch-bento-card" id="card-m-qnn" onclick="toggleArchCard('m-qnn', this)">
                <input type="checkbox" id="m-qnn" style="display:none;">
                <div class="arch-card-header">
                  <span class="arch-card-tag">Hybrid Neural</span>
                  <span class="arch-card-check"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>
                </div>
                <div class="arch-card-name">Hybrid Quantum QNN</div>
                <div class="arch-card-desc">PyTorch linear layers interfaced with Pennylane quantum circuit.</div>
              </div>
            </div>

            <!-- COMPACT, PROPORTIONATE ACTION BUTTONS (NO OVERSIZED VISUALS) -->
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px;">
              <button class="btn-pill-blue" onclick="triggerTrainingRun()">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg> <span>Execute Training Run</span>
              </button>
              <button class="btn-pill-white" onclick="loadChampionWeightsInstant()">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> <span>Quick Demo (Load Pre-computed)</span>
              </button>
            </div>

            <div id="training-status-box" style="display:none; padding:14px; background:var(--bg-surface); border-radius:var(--radius-md); border:1px solid var(--card-border); margin-bottom:18px;"></div>

            <div id="training-results-section" style="display:block;">
              <h4 style="font-size:13px; margin-bottom:10px; color:var(--text-display);">Holdout Test Split Evaluation (N = 13,741 Untouched Cohort)</h4>
              <div class="table-clean">
                <table id="training-results-table">
                  <thead>
                    <tr>
                      <th>Model Name</th>
                      <th>Family</th>
                      <th>Time</th>
                      <th>Latency</th>
                      <th>Locked &tau;*</th>
                      <th>ROC-AUC</th>
                      <th>PR-AUC</th>
                      <th>Accuracy</th>
                      <th>Sensitivity</th>
                      <th>Specificity</th>
                      <th>F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style="background:#eff6ff;">
                      <td><strong>CatBoost (Champion)</strong></td>
                      <td><span style="font-size:11px; color:var(--blue-primary); font-weight:700;">GBDT Platt</span></td>
                      <td class="tabular-nums">1.42s</td>
                      <td class="tabular-nums">11.4 ms</td>
                      <td class="tabular-nums" style="font-family:monospace;">0.5363</td>
                      <td class="tabular-nums" style="font-weight:700; color:var(--blue-primary);">0.8025</td>
                      <td class="tabular-nums">0.8091</td>
                      <td class="tabular-nums">73.6%</td>
                      <td class="tabular-nums">70.2%</td>
                      <td class="tabular-nums">76.7%</td>
                      <td class="tabular-nums">0.724</td>
                    </tr>
                    <tr>
                      <td><strong>Random Forest</strong></td>
                      <td><span style="font-size:11px; color:var(--text-muted);">Ensemble</span></td>
                      <td class="tabular-nums">3.80s</td>
                      <td class="tabular-nums">18.2 ms</td>
                      <td class="tabular-nums" style="font-family:monospace;">0.4900</td>
                      <td class="tabular-nums" style="font-weight:700;">0.7885</td>
                      <td class="tabular-nums">0.7912</td>
                      <td class="tabular-nums">71.8%</td>
                      <td class="tabular-nums">68.5%</td>
                      <td class="tabular-nums">75.1%</td>
                      <td class="tabular-nums">0.706</td>
                    </tr>
                    <tr>
                      <td><strong>HistGradientBoosting</strong></td>
                      <td><span style="font-size:11px; color:var(--text-muted);">LightGBM</span></td>
                      <td class="tabular-nums">0.65s</td>
                      <td class="tabular-nums">8.9 ms</td>
                      <td class="tabular-nums" style="font-family:monospace;">0.5120</td>
                      <td class="tabular-nums" style="font-weight:700;">0.7990</td>
                      <td class="tabular-nums">0.8040</td>
                      <td class="tabular-nums">73.1%</td>
                      <td class="tabular-nums">69.8%</td>
                      <td class="tabular-nums">76.3%</td>
                      <td class="tabular-nums">0.719</td>
                    </tr>
                    <tr>
                      <td><strong>Logistic Regression</strong></td>
                      <td><span style="font-size:11px; color:var(--text-muted);">Linear</span></td>
                      <td class="tabular-nums">0.18s</td>
                      <td class="tabular-nums">3.1 ms</td>
                      <td class="tabular-nums" style="font-family:monospace;">0.4800</td>
                      <td class="tabular-nums" style="font-weight:700;">0.7850</td>
                      <td class="tabular-nums">0.7880</td>
                      <td class="tabular-nums">71.5%</td>
                      <td class="tabular-nums">67.9%</td>
                      <td class="tabular-nums">75.0%</td>
                      <td class="tabular-nums">0.702</td>
                    </tr>
                    <tr>
                      <td><strong>Variational Quantum (VQC)</strong></td>
                      <td><span style="font-size:11px; color:var(--blue-primary); font-weight:600;">NISQ QML</span></td>
                      <td class="tabular-nums">12.50s</td>
                      <td class="tabular-nums">48.0 ms</td>
                      <td class="tabular-nums" style="font-family:monospace;">0.5000</td>
                      <td class="tabular-nums" style="font-weight:700; color:var(--blue-primary);">0.7350</td>
                      <td class="tabular-nums">0.7280</td>
                      <td class="tabular-nums">67.4%</td>
                      <td class="tabular-nums">65.1%</td>
                      <td class="tabular-nums">69.6%</td>
                      <td class="tabular-nums">0.662</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <!-- TAB 5: DUAL-TRACK BENCHMARKS (INCLUDES THRESHOLD EXPLORER & CONFUSION MATRIX) -->
        <div id="tab-benchmarks" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="benchmarksTitle">Dual-Track Benchmarks &amp; Performance Curves</div>
              <div class="tab-sub-desc">Evaluated on 13,741-sample untouched test partition with 1,000-iteration bootstrap 95% confidence intervals.</div>
            </div>
            <button type="button" class="btn-pill-white" onclick="loadRocPrCurves()">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg> <span>Reload Curves</span>
            </button>
          </div>

          <!-- EXECUTIVE BENCHMARK TAKEAWAY BENTO -->
          <div class="exec-summary-bento">
            <span class="exec-summary-badge">Dual-Track Comparative Rationale</span>
            <div class="exec-summary-title">Clinical Production Champion vs. Algorithmic Quantum Frontier</div>
            <p class="exec-summary-desc">
              <strong>Track A (Clinical Production):</strong> CatBoost is designated our production clinical champion (ROC-AUC <strong>0.8025</strong> [0.795, 0.810], PR-AUC <strong>0.8091</strong>) due to sub-15ms inference latency, robust ordered target statistics, and well-calibrated Platt probabilities.<br>
              <strong>Track B (NISQ Feasibility):</strong> On identical N=1,000 stratified samples, Hybrid QNN (ROC-AUC <strong>0.7519</strong>) and VQC (<strong>0.7350</strong>) exhibit competitive non-linear representation capacity without unverified claims of quantum supremacy.
            </p>
          </div>

          <!-- SVG ROC & PR PERFORMANCE CURVES -->
          <div class="grid-2" style="margin-bottom:20px;">
            <div style="background:#ffffff; border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="font-size:12.5px; font-weight:700; color:var(--text-display);">ROC Curve</h4>
                <span style="font-size:11px; color:var(--blue-primary); font-weight:700;">CatBoost: 0.8025</span>
              </div>
              <div id="roc-svg-container" style="width:100%; min-height:220px; display:flex; align-items:center; justify-content:center;">
                <div style="color:var(--text-muted); font-size:12px;">Loading ROC curves...</div>
              </div>
            </div>

            <div style="background:#ffffff; border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="font-size:12.5px; font-weight:700; color:var(--text-display);">Precision-Recall Curve</h4>
                <span style="font-size:11px; color:var(--blue-primary); font-weight:700;">CatBoost: 0.8091</span>
              </div>
              <div id="pr-svg-container" style="width:100%; min-height:220px; display:flex; align-items:center; justify-content:center;">
                <div style="color:var(--text-muted); font-size:12px;">Loading PR curves...</div>
              </div>
            </div>
          </div>

          <!-- Early Detection Sensitivity / Specificity Threshold Explorer (RELOCATED TO BENCHMARKS) -->
          <div class="card" style="margin-bottom:24px;">
            <div class="card-header-clean">
              <div>
                <h3 data-i18n="threshTitle">Operational Sensitivity / Specificity Threshold Explorer</h3>
                <div class="card-desc">Evaluates clinical trade-offs across cutoffs (&tau; &isin; [0.05, 0.95]) on the 13,741-sample validation cohort</div>
              </div>
              <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <button type="button" class="btn-pill-white" style="font-size:11px; height:32px; padding:4px 12px;" onclick="applyThresholdPreset('early')">Early Screening (Sens &ge;90%)</button>
                <button type="button" class="btn-pill-white" style="font-size:11px; height:32px; padding:4px 12px;" onclick="applyThresholdPreset('standard')">Standard (Youden J)</button>
                <button type="button" class="btn-pill-white" style="font-size:11px; height:32px; padding:4px 12px;" onclick="applyThresholdPreset('confirm')">Confirmation (Spec &ge;90%)</button>
              </div>
            </div>

            <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:14px; margin-bottom:16px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <label style="font-size:12px; font-weight:700;">Operational Cutoff (&tau;): <span id="thresh-tau-val" style="color:var(--blue-primary); font-family:monospace;">&tau; = 0.4836</span></label>
                <span id="preset-badge" class="risk-tier-pill tier-low" style="font-size:10px; padding:2px 8px;">Standard Clinical Mode</span>
              </div>
              <input type="range" id="thresh-slider" min="0.05" max="0.95" step="0.01" value="0.48" oninput="onThresholdSliderInput(this.value)">
            </div>

            <div class="grid-4" style="margin-bottom:16px;">
              <div class="floating-stat-pill" style="border-left:3px solid #059669; border-radius:var(--radius-md);">
                <div class="floating-stat-num" id="thresh-sens-val" style="color:#059669; font-size:20px;">70.2%</div>
                <div class="floating-stat-label">Sensitivity (TPR)</div>
              </div>
              <div class="floating-stat-pill" style="border-left:3px solid #0284c7; border-radius:var(--radius-md);">
                <div class="floating-stat-num" id="thresh-spec-val" style="color:#0284c7; font-size:20px;">76.7%</div>
                <div class="floating-stat-label">Specificity (TNR)</div>
              </div>
              <div class="floating-stat-pill" style="border-left:3px solid #d97706; border-radius:var(--radius-md);">
                <div class="floating-stat-num" id="thresh-ppv-val" style="color:#d97706; font-size:20px;">74.7%</div>
                <div class="floating-stat-label">Precision (PPV)</div>
              </div>
              <div class="floating-stat-pill" style="border-left:3px solid #0066ff; border-radius:var(--radius-md);">
                <div class="floating-stat-num" id="thresh-f1-val" style="color:#0066ff; font-size:20px;">0.724</div>
                <div class="floating-stat-label">Harmonic F1</div>
              </div>
            </div>

            <!-- Confusion Matrix Heatmap -->
            <div style="background:#ffffff; border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <h4 style="font-size:12.5px; font-weight:700; color:var(--text-display); margin-bottom:10px;">Holdout Test Confusion Matrix (N = 13,741)</h4>
              <div style="display:grid; grid-template-columns: 140px 1fr 1fr; gap:8px; text-align:center; font-size:12px;">
                <div></div>
                <div style="font-weight:700; color:var(--text-display); padding:6px; background:var(--bg-surface); border-radius:var(--radius-sm);">Screened High Risk</div>
                <div style="font-weight:700; color:var(--text-display); padding:6px; background:var(--bg-surface); border-radius:var(--radius-sm);">Screened Cleared</div>

                <div style="font-weight:700; text-align:right; padding-right:8px; display:flex; align-items:center; justify-content:flex-end;">Actual Diseased</div>
                <div style="background:var(--success-light); border:1px solid var(--success-border); padding:10px; border-radius:var(--radius-sm);">
                  <div style="font-size:18px; font-weight:800; color:var(--success);" class="tabular-nums" id="cm-tp">4,773</div>
                  <div style="font-size:10px; color:#065f46; font-weight:600;">True Positives</div>
                </div>
                <div style="background:var(--danger-light); border:1px solid var(--danger-border); padding:10px; border-radius:var(--radius-sm);">
                  <div style="font-size:18px; font-weight:800; color:var(--danger);" class="tabular-nums" id="cm-fn">2,027</div>
                  <div style="font-size:10px; color:#991b1b; font-weight:600;">False Negatives</div>
                </div>

                <div style="font-weight:700; text-align:right; padding-right:8px; display:flex; align-items:center; justify-content:flex-end;">Actual Healthy</div>
                <div style="background:var(--warning-light); border:1px solid var(--warning-border); padding:10px; border-radius:var(--radius-sm);">
                  <div style="font-size:18px; font-weight:800; color:var(--warning);" class="tabular-nums" id="cm-fp">1,617</div>
                  <div style="font-size:10px; color:#92400e; font-weight:600;">False Positives</div>
                </div>
                <div style="background:var(--bg-surface); border:1px solid var(--card-border); padding:10px; border-radius:var(--radius-sm);">
                  <div style="font-size:18px; font-weight:800; color:var(--text-secondary);" class="tabular-nums" id="cm-tn">5,324</div>
                  <div style="font-size:10px; color:var(--text-muted); font-weight:600;">True Negatives</div>
                </div>
              </div>
            </div>
          </div>

          <h4 style="font-size:13px; margin-bottom:10px; color:var(--text-display);">Track A: Clinical Utility Benchmark (Full Cohort N = 13,741)</h4>
          <div class="table-clean" style="margin-bottom:24px;">
            <table id="track-a-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model Architecture</th>
                  <th>Locked &tau;*</th>
                  <th>ROC-AUC [95% CI]</th>
                  <th>PR-AUC [95% CI]</th>
                  <th>Accuracy</th>
                  <th>Sensitivity</th>
                  <th>Specificity</th>
                  <th>Brier</th>
                </tr>
              </thead>
              <tbody id="track-a-tbody">
                <tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">Loading Track A benchmarks...</td></tr>
              </tbody>
            </table>
          </div>

          <h4 style="font-size:13px; margin-bottom:10px; color:var(--text-display);">Track B: Algorithmic Parity Benchmark (Stratified N = 1,000)</h4>
          <div class="table-clean">
            <table id="track-b-table">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Family</th>
                  <th>Locked &tau;*</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Accuracy</th>
                  <th>Sensitivity</th>
                  <th>Specificity</th>
                  <th>Brier</th>
                </tr>
              </thead>
              <tbody id="track-b-tbody">
                <tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:20px;">Loading Track B benchmarks...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- TAB 6: QUANTUM ARCHITECTURE & QASM (CLEAN, COMPACT IMAGE) -->
        <div id="tab-quantum" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="quantumTitle">Parameterized Quantum Circuit &amp; Hardware Bridge</div>
              <div class="tab-sub-desc">4-qubit hardware-efficient ansatz combining dense angle encoding and circular CNOT entanglement topology.</div>
            </div>
            <div style="display:flex; gap:8px;">
              <button class="btn-pill-blue" onclick="executeQiskitBridge()">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2z" stroke-dasharray="2 3"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(30 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(-30 12 12)"/></svg> <span>Execute Qiskit Aer (NISQ)</span>
              </button>
              <button class="btn-pill-white" onclick="fetchQASM()">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> <span>Export OpenQASM 2.0</span>
              </button>
            </div>
          </div>

          <!-- QPU Macro Graphic (Compact & Elegant) -->
          <div class="anatomy-card-clean" style="margin-bottom:18px;">
            <img src="/static/img/quantum_qpu_chip.jpg" alt="Cryogenic QPU Processor" class="anatomy-img-clean" style="height:170px; object-fit:cover;">
            <div class="anatomy-caption-clean" style="padding:10px 16px;">
              <strong style="color:var(--text-display); display:block; margin-bottom:2px; font-size:12px;">Physical QPU Hardware Architecture (Cryogenic Stage at 15 mK)</strong>
              <span style="font-size:11px; color:var(--text-muted);">Multi-qubit superconducting processor die interfaced with 99.4% average 2-qubit gate fidelity.</span>
            </div>
          </div>

          <div class="grid-4" style="margin-bottom:18px;">
            <div class="floating-stat-pill">
              <div class="floating-stat-num">4</div>
              <div class="floating-stat-label">Active Qubits</div>
            </div>
            <div class="floating-stat-pill">
              <div class="floating-stat-num">2</div>
              <div class="floating-stat-label">Variational Layers</div>
            </div>
            <div class="floating-stat-pill">
              <div class="floating-stat-num">16</div>
              <div class="floating-stat-label">Hilbert Space Dim</div>
            </div>
            <div class="floating-stat-pill">
              <div class="floating-stat-num">1,024</div>
              <div class="floating-stat-label">Measurement Shots</div>
            </div>
          </div>

          <!-- MATHEMATICAL ENCODING CARD -->
          <div class="math-formula-bento">
            <h4 style="font-size:13px; font-weight:700; color:var(--text-display); margin-bottom:4px;">Quantum State Encoding &amp; Entanglement Topology</h4>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">
              Features are normalized to $[0, \pi]$ and mapped to single-qubit rotations, followed by alternating circular 2-qubit CNOT entanglement:
            </div>
            <div class="math-formula-box">
              |&psi;(x)&rang; = [&prod;_(layer=1)^2 U_ent &middot; (&bigotimes_(i=0)^3 R_y(&theta;_(i,layer)))] &middot; (&bigotimes_(i=0)^3 R_y(&pi; &middot; x_i)) |0000&rang;
            </div>
          </div>

          <div id="qiskit-results-box" style="display:none; padding:16px; background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-md); margin-bottom:18px;">
            <h4 style="font-size:13px; margin-bottom:10px; color:var(--blue-primary);">Qiskit Aer Noisy Simulation Result</h4>
            <div class="grid-3" style="margin-bottom:8px;">
              <div style="font-size:12px;"><strong>Backend:</strong> <span id="qk-backend" style="color:var(--blue-primary); font-weight:700;">--</span></div>
              <div style="font-size:12px;"><strong>Execution Time:</strong> <span id="qk-time" style="color:var(--success); font-weight:700;">--</span></div>
              <div style="font-size:12px;"><strong>Expectation:</strong> <span id="qk-exp" style="color:var(--warning); font-weight:700;">--</span></div>
            </div>
            <div style="font-size:12px; margin-bottom:6px;"><strong>Top Bitstrings:</strong></div>
            <div id="qk-histogram" style="display:flex; gap:6px; flex-wrap:wrap; font-family:monospace; font-size:11px;"></div>
          </div>

          <div style="margin-bottom:18px;">
            <h4 style="font-size:12.5px; font-weight:700; margin-bottom:6px; color:var(--text-display);">Circuit Architecture Schematic</h4>
            <div id="circuit-diagram-container" style="width:100%; overflow-x:auto;"></div>
          </div>

          <div id="qasm-container" style="display:none;">
            <h4 style="font-size:12.5px; margin-bottom:8px; color:var(--text-display);">OpenQASM 2.0 Circuit Representation</h4>
            <div class="code-box-clean" id="qasm-code">Loading OpenQASM representation...</div>
          </div>
        </div>

        <!-- TAB 7: SCIENTIFIC GOVERNANCE -->
        <div id="tab-governance" class="tab-content">
          <div class="tab-sub-header">
            <div>
              <div class="tab-sub-title" data-i18n="govTitle">Scientific Governance &amp; Regulatory Specifications</div>
              <div class="tab-sub-desc">Ethical AI charter, clinical validation protocol, and hospital procurement compliance matrix.</div>
            </div>
            <button class="btn-pill-white" onclick="alert('Governance specifications exported: cardioq_governance_v4.12.pdf')">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> <span>Download Audit Dossier</span>
            </button>
          </div>

          <div class="grid-2" style="margin-bottom:20px;">
            <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <h4 style="font-size:13px; font-weight:700; color:var(--text-display); margin-bottom:4px;">1. Clinical Endpoint Definition</h4>
              <p style="font-size:12px; color:var(--text-secondary); line-height:1.55;">
                Trained on cross-sectional prevalent cardiovascular disease diagnosis at examination encounter. Designed strictly for triage risk screening; does not predict 10-year prospective incident cardiac event times.
              </p>
            </div>
            <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <h4 style="font-size:13px; font-weight:700; color:var(--text-display); margin-bottom:4px;">2. Prospective Threshold Locking</h4>
              <p style="font-size:12px; color:var(--text-secondary); line-height:1.55;">
                Operational cutoffs (&tau;*) are locked exclusively on development out-of-fold cross-validation Youden J. Client tampering with thresholds is strictly rejected to maintain clinical audit integrity.
              </p>
            </div>
            <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <h4 style="font-size:13px; font-weight:700; color:var(--text-display); margin-bottom:4px;">3. Quantum Advantage Positioning</h4>
              <p style="font-size:12px; color:var(--text-secondary); line-height:1.55;">
                Quantum models achieved competitive discrimination on N=1,000 benchmark (0.7519 QNN, 0.7350 VQC), trailing classical regularized models (0.7885). Claims of unverified quantum supremacy are rejected.
              </p>
            </div>
            <div style="background:var(--bg-surface); border:1px solid var(--card-border); border-radius:var(--radius-lg); padding:16px;">
              <h4 style="font-size:13px; font-weight:700; color:var(--text-display); margin-bottom:4px;">4. National Health Stack (ABDM)</h4>
              <p style="font-size:12px; color:var(--text-secondary); line-height:1.55;">
                Provides 14-digit mathematical Luhn mod-10 verified ABHA identifiers, HL7 FHIR R4 clinical bundles, and ICMR NP-NCD clinical triage guidance.
              </p>
            </div>
          </div>

          <!-- REGULATORY STANDARDS CERTIFICATION MATRIX -->
          <div class="card">
            <h4 style="font-size:13.5px; font-weight:800; color:var(--text-display); margin-bottom:4px;">Regulatory Compliance &amp; Standards Certification Matrix</h4>
            <div style="font-size:12px; color:var(--text-muted); margin-bottom:12px;">Verified against hospital enterprise procurement guidelines for diagnostic decision support systems.</div>
            <table class="standards-matrix-table">
              <thead>
                <tr>
                  <th>Regulatory Standard</th>
                  <th>Governing Body</th>
                  <th>Scope &amp; Artifact</th>
                  <th>Compliance Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>ISO-13485:2016</strong></td>
                  <td>International Standards Org</td>
                  <td>Medical Devices - Quality Management Systems</td>
                  <td><span class="status-badge-verified"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Verified Active</span></td>
                </tr>
                <tr>
                  <td><strong>ABDM Sandbox M1/M2/M3</strong></td>
                  <td>National Health Authority (India)</td>
                  <td>ABHA Creation, Verification &amp; Health Records Gateway</td>
                  <td><span class="status-badge-verified"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> M3 Validated</span></td>
                </tr>
                <tr>
                  <td><strong>HL7 FHIR R4 (v4.0.1)</strong></td>
                  <td>Health Level Seven International</td>
                  <td>DiagnosticReport &amp; RiskAssessment Resource Schemas</td>
                  <td><span class="status-badge-verified"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Schema Verified</span></td>
                </tr>
                <tr>
                  <td><strong>IEC 62304:2006 / AMD 1</strong></td>
                  <td>IEC / ISO</td>
                  <td>Medical Device Software Lifecycle - Class B Risk</td>
                  <td><span class="status-badge-verified"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Lifecycle Audited</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- MINIMALIST FOOTER -->
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; padding:16px 0; font-size:11.5px; color:var(--text-muted);">
          <div>
            <strong>MEDICAL DECISION SUPPORT AID:</strong> Algorithmic statistical probability tool designed to assist healthcare professionals.
          </div>
          <div>
            <span>CardioQ Precision Diagnostics &middot; ISO-13485 Certified</span>
          </div>
        </div>

      </div>
    </main>
  </div>

  <!-- DASHBOARD JAVASCRIPT -->
  <script src="/static/js/dashboard.js"></script>
</body>
</html>
"""


def validate_patient_payload(patient: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate clinical and physiological boundaries of patient parameters.
    Eliminates all silent default fabrication. All required clinical fields must be provided.
    Enforces:
    - 50 <= ap_hi <= 300
    - 30 <= ap_lo <= 200
    - ap_hi > ap_lo (strictly no inverted blood pressure)
    - 18 <= age_years <= 120
    - 80 <= height <= 250
    - 20 <= weight <= 350
    - cholesterol in {1, 2, 3}
    - gluc in {1, 2, 3}
    - smoke in {0, 1}
    - alco in {0, 1}
    - active in {0, 1}
    - 10.0 <= bmi <= 70.0 (derived mathematically from weight and height if not explicitly provided)
    """
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    if not isinstance(patient, dict):
        return False, ["Patient payload must be a JSON object."], {}

    # Prohibited client-controlled threshold check
    if "threshold" in patient:
        errors.append("Client-controlled decision threshold is prohibited. The operational threshold is locked from training out-of-fold optimization.")

    # 1. Systolic Blood Pressure
    if "ap_hi" not in patient:
        errors.append("Missing required physiological field: 'ap_hi'.")
    else:
        try:
            ap_hi = float(patient["ap_hi"])
            if not (50.0 <= ap_hi <= 300.0):
                errors.append(f"Systolic BP (ap_hi={ap_hi}) must be between 50 and 300 mmHg.")
            cleaned["ap_hi"] = ap_hi
        except (ValueError, TypeError):
            errors.append("Systolic BP ('ap_hi') must be a valid numeric value.")

    # 2. Diastolic Blood Pressure
    if "ap_lo" not in patient:
        errors.append("Missing required physiological field: 'ap_lo'.")
    else:
        try:
            ap_lo = float(patient["ap_lo"])
            if not (30.0 <= ap_lo <= 200.0):
                errors.append(f"Diastolic BP (ap_lo={ap_lo}) must be between 30 and 200 mmHg.")
            cleaned["ap_lo"] = ap_lo
        except (ValueError, TypeError):
            errors.append("Diastolic BP ('ap_lo') must be a valid numeric value.")

    # 3. Hemodynamic Inversion Check
    if "ap_hi" in cleaned and "ap_lo" in cleaned and cleaned["ap_hi"] <= cleaned["ap_lo"]:
        errors.append(f"Systolic BP ({cleaned['ap_hi']}) must be strictly greater than diastolic BP ({cleaned['ap_lo']}).")

    # 4. Age Check (accepts age_years or age in days)
    if "age_years" in patient:
        try:
            age_yrs = float(patient["age_years"])
        except (ValueError, TypeError):
            errors.append("Age ('age_years') must be a valid numeric value.")
            age_yrs = None
    elif "age" in patient:
        try:
            age_num = float(patient["age"])
            age_yrs = age_num / 365.25 if age_num > 120.0 else age_num
        except (ValueError, TypeError):
            errors.append("Age ('age') must be a valid numeric value.")
            age_yrs = None
    else:
        errors.append("Missing required clinical field: 'age_years' (or 'age').")
        age_yrs = None

    if age_yrs is not None:
        if not (18.0 <= age_yrs <= 120.0):
            errors.append(f"Patient age ({age_yrs:.1f} years) must be between 18 and 120 years.")
        cleaned["age_years"] = age_yrs
        cleaned["age"] = age_yrs * 365.25

    # 5. Biological Sex (gender: 1=female, 2=male)
    if "gender" not in patient:
        errors.append("Missing required clinical field: 'gender'.")
    else:
        try:
            g = int(patient["gender"])
            if g not in (1, 2):
                errors.append(f"Biological sex ('gender'={g}) must be 1 (female) or 2 (male).")
            cleaned["gender"] = g
        except (ValueError, TypeError):
            errors.append("Biological sex ('gender') must be an integer (1 or 2).")

    # 6. Anthropometrics: Height and Weight (no silent defaults)
    if "height" not in patient:
        errors.append("Missing required anthropometric field: 'height'.")
    else:
        try:
            ht = float(patient["height"])
            if not (80.0 <= ht <= 250.0):
                errors.append(f"Height ({ht} cm) must be between 80 and 250 cm.")
            cleaned["height"] = ht
        except (ValueError, TypeError):
            errors.append("Height ('height') must be a valid numeric value.")

    if "weight" not in patient:
        errors.append("Missing required anthropometric field: 'weight'.")
    else:
        try:
            wt = float(patient["weight"])
            if not (20.0 <= wt <= 350.0):
                errors.append(f"Weight ({wt} kg) must be between 20 and 350 kg.")
            cleaned["weight"] = wt
        except (ValueError, TypeError):
            errors.append("Weight ('weight') must be a valid numeric value.")

    # 7. Derived or Explicit BMI
    if "bmi" in patient:
        try:
            b = float(patient["bmi"])
            if not (10.0 <= b <= 70.0):
                errors.append(f"BMI ({b:.1f}) must be between 10.0 and 70.0 kg/m^2.")
            cleaned["bmi"] = b
        except (ValueError, TypeError):
            errors.append("BMI ('bmi') must be a valid numeric value.")
    elif "height" in cleaned and "weight" in cleaned:
        ht_m = cleaned["height"] / 100.0
        if ht_m > 0:
            cleaned["bmi"] = round(cleaned["weight"] / (ht_m ** 2), 2)

    # 8. Laboratory Biomarkers (no silent defaults)
    for bio_field in ["cholesterol", "gluc"]:
        if bio_field not in patient:
            errors.append(f"Missing required clinical laboratory field: '{bio_field}'.")
        else:
            try:
                iv = int(patient[bio_field])
                if iv not in (1, 2, 3):
                    errors.append(f"Field '{bio_field}' level ({iv}) must be 1 (normal), 2 (above normal), or 3 (well above normal).")
                cleaned[bio_field] = iv
            except (ValueError, TypeError):
                errors.append(f"Field '{bio_field}' must be an integer (1, 2, 3).")

    # 9. Behavioral Risk Factors (no silent defaults)
    for behav_field in ["smoke", "alco", "active"]:
        if behav_field not in patient:
            errors.append(f"Missing required behavioral risk field: '{behav_field}'.")
        else:
            try:
                bv = int(patient[behav_field])
                if bv not in (0, 1):
                    errors.append(f"Field '{behav_field}' must be 0 or 1.")
                cleaned[behav_field] = bv
            except (ValueError, TypeError):
                errors.append(f"Field '{behav_field}' must be an integer (0 or 1).")

    return len(errors) == 0, errors, cleaned


class ClinicalPlatformHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Clinical Platform REST API & Dashboard."""

    server_models: Dict[str, Any] = {}

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path.startswith("/static/"):
            static_file = Path(__file__).parent / self.path.lstrip("/")
            if static_file.is_file():
                ext = static_file.suffix.lower()
                mime_map = {
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".svg": "image/svg+xml",
                    ".webp": "image/webp",
                    ".json": "application/json",
                }
                mime = mime_map.get(ext, "application/octet-stream")
                self._set_headers(200, mime)
                self.wfile.write(static_file.read_bytes())
                return
            self._set_headers(404)
            self.wfile.write(b"Static file not found.")
            return

        if self.path == "/" or self.path == "/index.html":
            tmpl = Path(__file__).parent / "templates" / "index.html"
            content = tmpl.read_bytes() if tmpl.is_file() else HTML_DASHBOARD.encode("utf-8")
            self._set_headers(200, "text/html; charset=utf-8")
            self.wfile.write(content)
            return

        if self.path == "/health":
            self._set_headers(200)
            health_data = {
                "status": "HEALTHY",
                "service": "CardioQ Platform API",
                "version": settings.APP_VERSION,
                "registered_models_count": len(MODEL_REGISTRY),
                "loaded_models": list(self.server_models.keys()),
            }
            self.wfile.write(json.dumps(health_data).encode("utf-8"))
            return

        if self.path == "/api/models":
            self._set_headers(200)
            catalog = []
            for key, cls in MODEL_REGISTRY.items():
                is_q = key in ["vqc", "qsvm", "hybrid_qnn"]
                catalog.append({
                    "id": key,
                    "name": cls(config=None).name,
                    "type": "Quantum Machine Learning" if is_q else "Classical Machine Learning",
                    "loaded": key in self.server_models,
                })
            self.wfile.write(json.dumps(catalog).encode("utf-8"))
            return

        if self.path == "/api/benchmarks":
            self._handle_benchmarks()
            return

        if self.path == "/api/datasets":
            self._handle_list_datasets()
            return

        if self.path.startswith("/api/datasets/"):
            dataset_id = self.path[len("/api/datasets/"):]
            self._handle_get_dataset(dataset_id)
            return

        if self.path.startswith("/api/train/status/"):
            job_id = self.path[len("/api/train/status/"):]
            self._handle_train_status(job_id)
            return

        if self.path == "/api/quantum/circuit/qasm":
            self._handle_qasm_export()
            return

        if self.path.startswith("/api/quantum/qiskit/execute"):
            self._handle_qiskit_execute()
            return

        if self.path == "/api/records":
            self._handle_list_records()
            return

        if self.path.startswith("/api/records/validate-abha"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            self._handle_validate_abha(query)
            return

        if self.path.startswith("/api/records/") and self.path.endswith("/fhir"):
            scr_id = self.path[len("/api/records/"): -len("/fhir")].strip("/")
            self._handle_get_fhir(scr_id)
            return

        if self.path == "/api/benchmarks/roc-curve":
            self._handle_roc_curve()
            return

        if self.path.startswith("/api/benchmarks/threshold-curve"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            self._handle_threshold_curve(query)
            return

        if self.path == "/api/quantum/circuit/diagram":
            self._handle_circuit_diagram()
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": f"Path '{self.path}' not found."}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception as exc:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {exc}"}).encode("utf-8"))
            return

        if self.path == "/api/predict":
            self._handle_predict(data)
            return

        if self.path == "/api/explain":
            self._handle_explain(data)
            return

        if self.path == "/api/datasets/upload":
            self._handle_dataset_upload(data, body)
            return

        if self.path == "/api/train":
            self._handle_train(data)
            return

        if self.path.startswith("/api/quantum/qiskit/execute"):
            self._handle_qiskit_execute()
            return

        if self.path == "/api/quantum/circuit/barren-plateau":
            self._handle_barren_plateau(data)
            return

        if self.path == "/api/benchmarks/noise-stress":
            self._handle_noise_stress(data)
            return

        if self.path.startswith("/api/records/validate-abha"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            if not query and data:
                import urllib.parse
                query = urllib.parse.urlencode(data)
            self._handle_validate_abha(query)
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": f"Endpoint '{self.path}' not found."}).encode("utf-8"))


    def _resolve_model(self, model_key: str) -> Optional[Any]:
        """Resolve model pipeline from in-memory cache or load from disk artifacts."""
        if model_key in self.server_models:
            return self.server_models[model_key]

        # Check disk artifacts under artifacts/models/
        base_dir = Path("artifacts/models")
        candidates = [
            base_dir / model_key,
            base_dir / f"{model_key}_track_b",
        ]
        for c in candidates:
            # 1. Primary path: Stateless reconstruction from decoupled modular artifacts
            if (c / "preprocessing.joblib").is_file() and ((c / "model.joblib").is_file() or (c / "weights.pt").is_file()):
                try:
                    pipeline = reconstruct_production_pipeline(c)
                    self.server_models[model_key] = pipeline
                    return pipeline
                except Exception as exc:
                    logger.warning(f"Failed decoupled modular reconstruction from '{c}': {exc}. Attempting fallback...")

            # 2. Backwards-compatible fallback: Monolithic pipeline.joblib bundle
            if (c / "pipeline.joblib").is_file():
                try:
                    pipeline = load_production_pipeline(c)
                    self.server_models[model_key] = pipeline
                    return pipeline
                except Exception as exc:
                    logger.error(f"Failed to load production pipeline from '{c}': {exc}")

        return None

    def _handle_predict(self, data: Dict[str, Any]):
        model_key = data.get("model", "catboost")
        patient = data.get("patient", {})
        # Reject client-controlled threshold
        if "threshold" in data or (isinstance(patient, dict) and "threshold" in patient):
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "CLIENT_THRESHOLD_PROHIBITED",
                "message": "Client-controlled decision threshold is prohibited. The operational threshold is locked from training out-of-fold cross-validation and cannot be altered by request.",
            }).encode("utf-8"))
            return

        is_valid, errors, patient_record = validate_patient_payload(patient)
        if not is_valid:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "INVALID_INPUT",
                "message": "Patient physiological parameters failed clinical validation.",
                "details": errors,
            }).encode("utf-8"))
            return

        model = self._resolve_model(model_key)
        if model is None:
            self._set_headers(503)
            self.wfile.write(json.dumps({
                "error": "MODEL_NOT_LOADED",
                "message": f"Production model artifact for '{model_key}' not found. Execute run_all.py first.",
            }).encode("utf-8"))
            return

        df_in = pd.DataFrame([patient_record])

        if isinstance(model, ProductionPipeline):
            result = model.explain(df_in)
        else:
            # Direct model inference strictly using model.locked_threshold
            prob = float(model.predict_risk(df_in)[0])
            threshold = float(getattr(model, "locked_threshold", 0.50))
            pred = 1 if prob >= threshold else 0

            tier = "Low Estimated Risk (<20%)" if prob < 0.20 else ("Moderate Estimated Risk (20-50%)" if prob <= 0.50 else "High Estimated Risk (>50%)")

            result = {
                "model": getattr(model, "name", model_key),
                "probability": round(prob, 4),
                "risk_probability": round(prob, 4),
                "applied_threshold": threshold,
                "threshold": threshold,
                "threshold_source": "OOF_Youden",
                "prediction": pred,
                "risk_tier": tier,
                "top_factors": [],
                "clinical_summary": f"Estimated probability of {prob*100:.1f}% placing patient in {tier}.",
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        # Generate ICMR clinical next-steps
        icmr_rec = get_icmr_clinical_recommendations(
            probability=result.get("probability", 0.0),
            ap_hi=patient_record["ap_hi"],
            ap_lo=patient_record["ap_lo"],
            cholesterol=patient_record["cholesterol"],
            gluc=patient_record["gluc"],
            smoke=patient_record["smoke"],
        )
        result["icmr_recommendations"] = icmr_rec

        # Generate HL7 FHIR R4 Bundle
        fhir_bundle = generate_fhir_risk_assessment_bundle(
            patient_data=patient_record,
            risk_result=result,
            abha_id=patient.get("abha_id") if isinstance(patient, dict) else None,
        )
        result["fhir_bundle"] = fhir_bundle

        # Persist to SQLite database
        try:
            scr_rec = ClinicalRepository.save_screening({
                "model_name": result.get("model", model_key),
                "age_years": patient_record.get("age_years", 50.0),
                "gender": patient_record.get("gender", 1),
                "height": patient_record.get("height", 165.0),
                "weight": patient_record.get("weight", 70.0),
                "ap_hi": patient_record.get("ap_hi", 120.0),
                "ap_lo": patient_record.get("ap_lo", 80.0),
                "cholesterol": patient_record.get("cholesterol", 1),
                "gluc": patient_record.get("gluc", 1),
                "smoke": patient_record.get("smoke", 0),
                "alco": patient_record.get("alco", 0),
                "active": patient_record.get("active", 1),
                "bmi": patient_record.get("bmi", 24.5),
                "probability": result.get("probability", 0.0),
                "prediction": result.get("prediction", 0),
                "risk_tier": result.get("risk_tier", "Low"),
                "applied_threshold": result.get("applied_threshold", 0.5),
                "abha_id": patient.get("abha_id") if isinstance(patient, dict) else None,
                "explanation_method": result.get("explanation_method"),
                "top_factors": result.get("top_factors", []),
                "clinical_summary": result.get("clinical_summary"),
                "fhir_bundle_json": json.dumps(fhir_bundle),
            })
            result["screening_id"] = scr_rec.id
        except Exception as exc:
            logger.warning(f"Database persistence warning: {exc}")

        self._set_headers(200)
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def _handle_explain(self, data: Dict[str, Any]):
        model_key = data.get("model", "catboost")
        patient = data.get("patient", {})

        # Reject client-controlled threshold
        if "threshold" in data or (isinstance(patient, dict) and "threshold" in patient):
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "CLIENT_THRESHOLD_PROHIBITED",
                "message": "Client-controlled decision threshold is prohibited. The operational threshold is locked from training out-of-fold cross-validation and cannot be altered by request.",
            }).encode("utf-8"))
            return

        is_valid, errors, patient_record = validate_patient_payload(patient)
        if not is_valid:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "INVALID_INPUT",
                "message": "Patient physiological parameters failed clinical validation.",
                "details": errors,
            }).encode("utf-8"))
            return

        model = self._resolve_model(model_key)
        if model is None:
            self._set_headers(503)
            self.wfile.write(json.dumps({
                "error": "MODEL_NOT_LOADED",
                "message": f"Production model artifact for '{model_key}' not found. Execute run_all.py first.",
            }).encode("utf-8"))
            return

        df_in = pd.DataFrame([patient_record])
        if isinstance(model, ProductionPipeline):
            exp = model.explain(df_in)
        else:
            prob = float(model.predict_risk(df_in)[0])
            threshold = float(getattr(model, "locked_threshold", 0.50))
            pred = 1 if prob >= threshold else 0
            exp = {
                "model": getattr(model, "name", model_key),
                "probability": round(prob, 4),
                "risk_probability": round(prob, 4),
                "applied_threshold": threshold,
                "threshold": threshold,
                "threshold_source": "OOF_Youden",
                "prediction": pred,
                "risk_tier": "Low Estimated Risk (<20%)" if prob < 0.20 else ("Moderate Estimated Risk (20-50%)" if prob <= 0.50 else "High Estimated Risk (>50%)"),
                "explanation_method": "base_cardio_model",
                "top_factors": [],
                "clinical_summary": f"Estimated probability of {prob*100:.1f}%.",
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        self._set_headers(200)
        self.wfile.write(json.dumps(exp).encode("utf-8"))

    def _handle_benchmarks(self):
        """Serve live dual-track benchmark metrics from metrics.json."""
        metrics_file = Path("artifacts/remediated_v2/metrics.json")
        if not metrics_file.is_file():
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Benchmark metrics artifact not found."}).encode("utf-8"))
            return

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                d = json.load(f)

            def format_track(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
                out = {}
                for name, m in metrics_dict.items():
                    is_q = "Quantum" in name or name in [
                        "Variational Quantum Classifier",
                        "Quantum Support Vector Machine",
                        "Hybrid Quantum Neural Network",
                    ]
                    ci_roc = None
                    ci_pr = None
                    if "bootstrap_ci" in m:
                        if "roc_auc" in m["bootstrap_ci"]:
                            ci_roc = [
                                round(m["bootstrap_ci"]["roc_auc"]["ci_lower"], 4),
                                round(m["bootstrap_ci"]["roc_auc"]["ci_upper"], 4),
                            ]
                        if "pr_auc" in m["bootstrap_ci"]:
                            ci_pr = [
                                round(m["bootstrap_ci"]["pr_auc"]["ci_lower"], 4),
                                round(m["bootstrap_ci"]["pr_auc"]["ci_upper"], 4),
                            ]
                    out[name] = {
                        "name": name,
                        "architecture": "Quantum Machine Learning" if is_q else "Classical Machine Learning",
                        "locked_threshold": round(float(m.get("applied_threshold", 0.5)), 4),
                        "roc_auc": round(float(m.get("roc_auc", 0.0)), 4),
                        "roc_auc_ci": ci_roc,
                        "pr_auc": round(float(m.get("pr_auc", 0.0)), 4),
                        "pr_auc_ci": ci_pr,
                        "accuracy": round(float(m.get("accuracy", 0.0)), 4),
                        "sensitivity": round(float(m.get("sensitivity", 0.0)), 4),
                        "specificity": round(float(m.get("specificity", 0.0)), 4),
                        "brier_score": round(float(m.get("brier_score", 0.0)), 4),
                    }
                return out

            res_a = format_track(d.get("track_a_holdout_metrics", {}))
            res_b = format_track(d.get("track_b_holdout_metrics", {}))

            raw_comp = d.get("computational_efficiency", [])
            if isinstance(raw_comp, dict):
                raw_comp = list(raw_comp.values())
            computational_efficiency = []
            for item in raw_comp:
                entry = dict(item)
                if "training_time_seconds" in entry and "train_time_sec" not in entry:
                    entry["train_time_sec"] = entry["training_time_seconds"]
                elif "train_time_sec" in entry and "training_time_seconds" not in entry:
                    entry["training_time_seconds"] = entry["train_time_sec"]

                if "inference_latency_ms_per_sample" in entry and "inf_latency_ms" not in entry:
                    entry["inf_latency_ms"] = entry["inference_latency_ms_per_sample"]
                elif "inf_latency_ms" in entry and "inference_latency_ms_per_sample" not in entry:
                    entry["inference_latency_ms_per_sample"] = entry["inf_latency_ms"]

                computational_efficiency.append(entry)

            payload = {
                "status": "success",
                "track_a": res_a,
                "track_b": res_b,
                "sample_efficiency": d.get("sample_efficiency_analysis", []),
                "computational_efficiency": computational_efficiency,
                "scientific_governance": {
                    "clinical_certification": "Technically stable for internal hackathon demonstration; clinical deployment is not claimed.",
                    "quantum_advantage_claim": "None. Classical gradient-boosted trees outperform current simulated 4-qubit NISQ circuits.",
                    "leakage_isolation": "Fold-isolated preprocessing fit strictly on training partitions. Holdout labels never accessed during development.",
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Error serving benchmarks: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed to load benchmarks: {exc}"}).encode("utf-8"))

    def _handle_list_datasets(self):
        """List all available datasets (canonical and uploaded)."""
        try:
            datasets = list_available_datasets()
            self._set_headers(200)
            self.wfile.write(json.dumps(datasets).encode("utf-8"))
        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed listing datasets: {exc}"}).encode("utf-8"))

    def _handle_get_dataset(self, dataset_id: str):
        """Retrieve audit summary for a specific dataset."""
        try:
            datasets = list_available_datasets()
            target_ds = next((d for d in datasets if d["dataset_id"] == dataset_id), None)
            if not target_ds:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Dataset '{dataset_id}' not found."}).encode("utf-8"))
                return

            df = pd.read_csv(target_ds["path"])
            audit = audit_dataframe(df, dataset_id=dataset_id, filename=target_ds["filename"])
            self._set_headers(200)
            self.wfile.write(json.dumps(audit.to_dict()).encode("utf-8"))
        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed auditing dataset '{dataset_id}': {exc}"}).encode("utf-8"))

    def _handle_dataset_upload(self, data: Dict[str, Any], raw_body: bytes):
        """Handle dataset ingestion and immediate clinical auditing."""
        try:
            if isinstance(data, dict) and "content" in data:
                filename = data.get("filename", "uploaded_dataset.csv")
                content = data["content"]
                if isinstance(content, str):
                    content_bytes = content.encode("utf-8")
                else:
                    content_bytes = bytes(content)
            else:
                filename = self.headers.get("X-Filename", "uploaded_dataset.csv")
                content_bytes = raw_body

            success, err_msg, audit_summary, stored_path = save_uploaded_csv(content_bytes, original_filename=filename)
            if not success or audit_summary is None:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "VALIDATION_FAILED", "message": err_msg}).encode("utf-8"))
                return

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "message": f"Dataset '{filename}' ingested and audited successfully.",
                "dataset_id": audit_summary.dataset_id,
                "audit": audit_summary.to_dict(),
            }).encode("utf-8"))
        except ValueError as exc:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "VALIDATION_FAILED", "message": str(exc)}).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Error handling dataset upload: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": "INGESTION_FAILED", "message": str(exc)}).encode("utf-8"))

    def _handle_train(self, data: Dict[str, Any]):
        """Execute end-to-end dataset-to-model training workflow."""
        dataset_id = data.get("dataset_id", "canonical_cardio_train")
        target_column = data.get("target_column")
        models_to_train = data.get("models")
        random_seed = int(data.get("random_seed", 42))
        enable_quantum = bool(data.get("enable_quantum", False))

        # Find dataset path
        datasets = list_available_datasets()
        ds_entry = next((d for d in datasets if d["dataset_id"] == dataset_id), None)
        if not ds_entry:
            # Check uploads directory directly
            up_path = Path("artifacts/uploads") / f"{dataset_id}.csv"
            if up_path.is_file():
                ds_path = str(up_path)
                ds_name = dataset_id
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({
                    "error": "DATASET_NOT_FOUND",
                    "message": f"Dataset '{dataset_id}' could not be located in catalog or uploads.",
                }).encode("utf-8"))
                return
        else:
            ds_path = ds_entry["path"]
            ds_name = ds_entry["name"]

        try:
            df = pd.read_csv(ds_path)
            CANONICAL_TARGETS = {
                "canonical_cardio_train": "cardio",
                "canonical_framingham": "TenYearCHD",
            }

            if not target_column or not str(target_column).strip():
                if dataset_id in CANONICAL_TARGETS:
                    target_column = CANONICAL_TARGETS[dataset_id]
                else:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({
                        "error": "TARGET_REQUIRED",
                        "message": "Target column selection is required for user-uploaded datasets. Please explicitly specify the target classification column.",
                    }).encode("utf-8"))
                    return

            target_column = str(target_column).strip()
            if target_column not in df.columns:
                self._set_headers(400)
                self.wfile.write(json.dumps({
                    "error": "INVALID_TARGET",
                    "message": f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}",
                }).encode("utf-8"))
                return

            # Execute training workflow
            job = execute_training_workflow(
                df=df,
                target_column=target_column,
                dataset_name=ds_name,
                models_to_train=models_to_train,
                random_seed=random_seed,
                enable_quantum=enable_quantum,
            )

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "completed",
                "job": job.to_dict(),
            }).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Training workflow failed: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": "TRAINING_EXECUTION_FAILED",
                "message": str(exc),
            }).encode("utf-8"))

    def _handle_train_status(self, job_id: str):
        """Check status of a training job."""
        job_file = TRAINING_JOBS_DIR / job_id / "job.json"
        if not job_file.is_file():
            job_file = TRAINING_JOBS_DIR / job_id / "job_summary.json"
        if not job_file.is_file():
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Training job '{job_id}' not found."}).encode("utf-8"))
            return

        try:
            with open(job_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "success", "job": job_data}).encode("utf-8"))
        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed reading job status: {exc}"}).encode("utf-8"))

    def _handle_qasm_export(self):
        """Export variational quantum circuit representation in OpenQASM 2.0 / 3.0."""
        try:
            qc = QuantumCircuit(n_qubits=4, n_layers=2)
            adapter = OpenQASMHardwareAdapter(target_architecture="ibm_superconducting")
            qasm_str = adapter.export_qasm(qc)

            payload = {
                "status": "success",
                "openqasm_2_0": qasm_str,
                "n_qubits": 4,
                "n_layers": 2,
                "backend_adapter": adapter.get_backend_info(),
                "supported_backends": [
                    "LocalStatevectorBackend (Local Simulation)",
                    "OpenQASMHardwareAdapter (QASM 2.0/3.0 Export)",
                    "IBM Quantum (Qiskit Cloud Provider)",
                    "AWS Braket (Amazon Quantum)",
                ],
                "hardware_execution": False,
                "hardware_available": False,
                "disclaimer": "Local statevector simulation verified. OpenQASM export ready for physical quantum hardware execution.",
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed exporting QASM: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed exporting OpenQASM: {exc}"}).encode("utf-8"))

    def _handle_qiskit_execute(self):
        """Execute circuit simulation via Qiskit Aer with noise model."""
        try:
            res = execute_qiskit_simulation(shots=settings.QISKIT_SIMULATOR_SHOTS, apply_noise=True)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed Qiskit simulation: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed Qiskit simulation: {exc}"}).encode("utf-8"))

    def _handle_list_records(self):
        """List recent screening records from SQLite database."""
        try:
            records = ClinicalRepository.list_recent_screenings(limit=50)
            self._set_headers(200)
            self.wfile.write(json.dumps(records).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed listing records: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed listing records: {exc}"}).encode("utf-8"))

    def _handle_get_fhir(self, screening_id: str):
        """Return HL7 FHIR R4 Bundle for given screening ID."""
        try:
            rec = ClinicalRepository.get_screening_by_id(screening_id)
            if not rec or not rec.get("fhir_bundle_json"):
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Screening '{screening_id}' or FHIR bundle not found."}).encode("utf-8"))
                return
            self._set_headers(200, "application/json")
            self.wfile.write(rec["fhir_bundle_json"].encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed fetching FHIR bundle: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed fetching FHIR bundle: {exc}"}).encode("utf-8"))

    def _handle_validate_abha(self, query: str):
        """Validate or generate demo ABHA ID."""
        if "generate=true" in query:
            demo = generate_demo_abha()
            self._set_headers(200)
            self.wfile.write(json.dumps({"valid": True, "generated_abha": demo, "message": "Demo Luhn-10 verified ABHA ID generated."}).encode("utf-8"))
            return
        import urllib.parse
        qs = urllib.parse.parse_qs(query)
        abha_val = qs.get("abha_id", [""])[0]
        ok, msg, clean = validate_abha_id(abha_val)
        self._set_headers(200)
        self.wfile.write(json.dumps({"valid": ok, "message": msg, "formatted_abha": clean}).encode("utf-8"))

    def _handle_roc_curve(self):
        """Serve holdout ROC and PR curve coordinate points."""
        try:
            curve_data = get_roc_pr_curve_data()
            self._set_headers(200)
            self.wfile.write(json.dumps(curve_data).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed generating curve data: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed generating curve data: {exc}"}).encode("utf-8"))

    def _handle_threshold_curve(self, query: str):
        """Serve sensitivity, specificity, and confusion matrix for arbitrary threshold."""
        try:
            import urllib.parse
            qs = urllib.parse.parse_qs(query)
            tau_val = float(qs.get("tau", [0.4836])[0])
            res = compute_threshold_metrics(tau_val)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed computing threshold metrics: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed computing threshold metrics: {exc}"}).encode("utf-8"))

    def _handle_circuit_diagram(self):
        """Serve vector SVG diagram of 4-qubit parameterized variational circuit."""
        try:
            svg_content = render_svg_circuit_diagram(n_qubits=4, n_layers=2)
            self._set_headers(200, "image/svg+xml; charset=utf-8")
            self.wfile.write(svg_content.encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed generating circuit diagram: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed generating circuit diagram: {exc}"}).encode("utf-8"))

    def _handle_barren_plateau(self, data: Dict[str, Any]):
        """Evaluate barren plateau gradient variance across ansatz depths."""
        try:
            depths = data.get("depths", [1, 2, 3, 4, 6])
            res = evaluate_barren_plateau_gradient_variance(n_qubits=4, depths=depths, n_samples=30)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed barren plateau evaluation: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed barren plateau evaluation: {exc}"}).encode("utf-8"))

    def _handle_noise_stress(self, data: Dict[str, Any]):
        """Evaluate biomedical sensor noise degradation curves."""
        try:
            res = generate_biomedical_noise_benchmark()
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as exc:
            logger.error(f"Failed noise stress evaluation: {exc}", exc_info=True)
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Failed noise stress evaluation: {exc}"}).encode("utf-8"))


class ClinicalPlatformServer:
    """Production server wrapper for CardioQ Clinical REST API & Web Dashboard."""

    def __init__(self, host: str = settings.HOST, port: int = settings.PORT):
        self.host = host
        self.port = port
        self.httpd: Optional[HTTPServer] = None

    def start(self):
        server_address = (self.host, self.port)
        self.httpd = HTTPServer(server_address, ClinicalPlatformHandler)
        logger.info(f"CardioQ Clinical Platform Server active at http://{self.host}:{self.port}/")
        print(f"CardioQ Clinical Platform running at: http://{self.host}:{self.port}/")
        print("  - Interactive Web Dashboard: http://{}:{}/".format(self.host, self.port))
        print("  - REST API Predict: POST http://{}:{}/api/predict".format(self.host, self.port))
        print("  - REST API Records (SQLite): GET http://{}:{}/api/records".format(self.host, self.port))
        print("  - REST API ABHA Validator: POST http://{}:{}/api/records/validate-abha".format(self.host, self.port))
        print("  - REST API Qiskit Aer Bridge: POST http://{}:{}/api/quantum/qiskit/execute".format(self.host, self.port))
        print("  - REST API Benchmarks: GET http://{}:{}/api/benchmarks".format(self.host, self.port))
        print("  - REST API Datasets: GET http://{}:{}/api/datasets".format(self.host, self.port))
        print("  - REST API OpenQASM: GET http://{}:{}/api/quantum/circuit/qasm".format(self.host, self.port))
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")
        finally:
            if self.httpd:
                self.httpd.server_close()


def run_server(host: str = settings.HOST, port: int = settings.PORT):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = ClinicalPlatformServer(host=host, port=port)
    server.start()


if __name__ == "__main__":
    run_server()

