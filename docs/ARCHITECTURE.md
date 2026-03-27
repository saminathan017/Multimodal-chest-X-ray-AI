# ClinicalAI — Production Architecture & Safety Design

## System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        CLINICALAI PRODUCTION ARCHITECTURE                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   CLINICIAN WORKSTATION                    AWS CLOUD                          ║
║   ┌─────────────────┐         HTTPS        ┌─────────────────────────────┐   ║
║   │  Streamlit App  │ ──────────────────►  │   Application Load Balancer │   ║
║   │  (Demo UI)      │                      │   + AWS WAF + TLS 1.3        │   ║
║   └─────────────────┘                      └──────────────┬──────────────┘   ║
║                                                            │                   ║
║   ┌─────────────────┐                      ┌──────────────▼──────────────┐   ║
║   │  Hospital EHR   │ ──── REST API ─────► │   FastAPI Service (ECS)     │   ║
║   │  (DICOM export) │                      │   JWT Auth · Rate Limit      │   ║
║   └─────────────────┘                      └──────────────┬──────────────┘   ║
║                                                            │                   ║
║   SAFETY PIPELINE (runs in-process):       ┌──────────────▼──────────────┐   ║
║   ┌─────────────────────────────────┐       │   SAFETY LAYER               │   ║
║   │ 1. PHI Detector (18 HIPAA IDs) │◄─────│   PHI Strip → Validate →     │   ║
║   │ 2. Input Validator (img+text)  │       │   Adversarial Check          │   ║
║   │ 3. Adversarial Guard           │       └──────────────┬──────────────┘   ║
║   │ 4. HIPAA Audit Logger          │                       │                   ║
║   └─────────────────────────────────┘       ┌──────────────▼──────────────┐   ║
║                                              │   ML INFERENCE PIPELINE      │   ║
║   ML MODELS (SageMaker Endpoint):            │                              │   ║
║   ┌─────────────────────────────────┐        │   ┌──────────┐ ┌──────────┐ │   ║
║   │ BiomedCLIP ViT-B/16            │◄──────│   │Biomed    │ │Clinical  │ │   ║
║   │ Bio_ClinicalBERT               │        │   │CLIP      │ │BERT      │ │   ║
║   │ CrossModal Fusion (Attn+MLP)   │        │   └────┬─────┘ └─────┬────┘ │   ║
║   │ MC Dropout Uncertainty (N=30)  │        │        └──────┬───────┘      │   ║
║   │ Temperature Scaling Calibrator │        │         ┌─────▼──────┐       │   ║
║   └─────────────────────────────────┘        │         │  Fusion    │       │   ║
║                                              │         │  + MC Drop │       │   ║
║   DATA STORAGE:                              │         └─────┬──────┘       │   ║
║   ┌──────────────────────────────┐           └──────────────┼──────────────┘   ║
║   │ S3 (KMS encrypted)           │                           │                   ║
║   │  - Model artifacts            │           ┌──────────────▼──────────────┐   ║
║   │  - Inference logs (no PHI)   │           │   AWS BEDROCK (Claude Haiku) │   ║
║   │  - Drift metrics              │           │   Clinical Report Generation │   ║
║   │ CloudWatch (7yr retention)   │           └──────────────┬──────────────┘   ║
║   │  - HIPAA audit trail          │                           │                   ║
║   └──────────────────────────────┘           ┌──────────────▼──────────────┐   ║
║                                              │   MONITORING LAYER           │   ║
║   MONITORING:                                │   Drift Detector             │   ║
║   ┌──────────────────────────────┐           │   Fairness Monitor           │   ║
║   │ CloudWatch Alarms:           │◄─────────│   Performance Tracker        │   ║
║   │  - Latency P99 > 5s          │           └─────────────────────────────┘   ║
║   │  - Error rate > 1%           │                                               ║
║   │  - Data drift PSI > 0.2      │                                               ║
║   │  - Fairness disparity > 10%  │                                               ║
║   └──────────────────────────────┘                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Safety Design Principles

### 1. Privacy by Design (HIPAA)
- PHI is detected and redacted BEFORE any data leaves the local process
- 18 HIPAA Safe-Harbor identifiers are scrubbed from clinical notes
- DICOM files have all PHI tags stripped on load
- Audit logs contain only pseudonymized hashes — never actual PHI
- 7-year log retention via AWS CloudWatch (45 CFR 164.312(b))

### 2. Input Hardening
- Image quality validation rejects blank, overexposed, or adversarial images
- Adversarial perturbation detection via high-frequency noise analysis
- File size limits enforced at API and validation layers
- Clinical note length and medical plausibility checks
- SQL injection and prompt injection resistant (no string interpolation into queries)

### 3. Uncertainty-Aware Predictions
- Monte Carlo Dropout (N=30 passes) quantifies epistemic uncertainty
- Temperature scaling calibrates confidence scores post-training
- Conformal prediction sets provide guaranteed 95% coverage
- High-uncertainty predictions automatically trigger "CALL RADIOLOGIST" flag
- Confidence intervals shown alongside every prediction

### 4. Fairness Monitoring
- Demographic parity tracked across age groups and gender in real-time
- Population Stability Index (PSI) detects confidence disparities
- Disparate impact alerts trigger when gap exceeds 10%
- All fairness metrics exposed via `/api/v1/fairness/report`

### 5. Operational Safety
- Non-root Docker container (UID 1001)
- JWT authentication + RBAC (CLINICIAN / RADIOLOGIST / ADMIN)
- Rate limiting (60 req/min per token)
- Immutable audit log (append-only JSONL + CloudWatch)
- Health checks and CloudWatch alarms on all critical metrics
- Data drift detection with automatic CloudWatch alarm when PSI > 0.2
- Model retraining pipeline triggered on sustained drift

### 6. Clinical Guardrails
- Every response includes mandatory clinical disclaimer
- Urgency scores are always paired with "verify with radiologist" prompts
- No single finding is ever presented as a definitive diagnosis
- System explicitly states it is NOT FDA-cleared
- Feedback loop for radiologist corrections drives quarterly retraining

## File Structure

```
clinical-ai-demo/
├── src/
│   ├── safety/
│   │   ├── phi_detector.py         # 18 HIPAA PHI identifier detection
│   │   ├── input_validator.py      # Image + note quality validation
│   │   └── fairness_monitor.py     # Demographic parity tracking
│   ├── models/
│   │   ├── image_encoder.py        # BiomedCLIP + GradCAM
│   │   ├── text_encoder.py         # Bio_ClinicalBERT
│   │   ├── fusion_model.py         # Cross-modal attention + heads
│   │   └── uncertainty.py          # MC Dropout + Temperature Scaling
│   ├── pipeline/
│   │   ├── inference.py            # Base inference pipeline
│   │   └── training.py             # CheXpert/MIMIC fine-tuning
│   ├── api/
│   │   ├── main.py                 # FastAPI — full production endpoint
│   │   ├── auth.py                 # JWT + RBAC
│   │   └── audit.py                # HIPAA audit logger
│   └── utils/
│       └── dicom_handler.py        # DICOM load + PHI strip + windowing
├── monitoring/
│   └── drift_detector.py           # PSI + KS drift detection
├── tests/
│   └── test_safety.py              # 30/30 safety tests (all passing)
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile              # Dev container
│   │   └── Dockerfile.production  # Security-hardened (non-root, multi-stage)
│   ├── terraform/
│   │   └── main.tf                 # AWS: S3, ECR, CloudWatch, Alarms, IAM
│   └── sagemaker/
│       ├── deploy_endpoint.py      # SageMaker T4 GPU endpoint deploy
│       └── inference_handler.py   # SageMaker model_fn/predict_fn
├── app/
│   ├── streamlit_app.py            # Demo Streamlit UI (with safety indicators)
│   └── index.html                  # Executive-designed web preview
├── configs/config.yaml             # All hyperparameters + AWS config
├── requirements.txt
├── ARCHITECTURE.md                 # This file
├── README.md
└── LINKEDIN_POST_GUIDE.md
```

## Known Limitations & Responsible AI Disclosures

| Limitation | Mitigation |
|---|---|
| Training data (CheXpert) predominantly from one institution | Fairness monitor tracks demographic disparities; quarterly retraining |
| Model not validated on non-standard X-ray equipment | Image quality validator rejects low-quality inputs |
| LLM reports can hallucinate clinical details | Reports always labelled as AI-generated; disclaimer on every output |
| No FDA 510(k) clearance | Explicit disclaimer on every response; for research only |
| Model uncertainty increases at distribution boundaries | MC Dropout uncertainty flags automatically trigger human review |
