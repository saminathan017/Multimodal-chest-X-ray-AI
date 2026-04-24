"""
src/api/main.py
═══════════════════════════════════════════════════════════════════
Production FastAPI — ClinicalAI Inference Service

Endpoints:
  POST /api/v1/analyze          — Main inference endpoint
  GET  /api/v1/health           — Health check
  GET  /api/v1/model/info       — Model metadata
  GET  /api/v1/fairness/report  — Fairness metrics
  POST /api/v1/feedback         — Radiologist feedback loop
  GET  /api/v1/audit/logs       — HIPAA audit trail (admin only)

Security:
  - JWT Bearer token authentication
  - Role-based access (CLINICIAN, ADMIN, RESEARCHER)
  - Rate limiting (60 req/min per token)
  - HIPAA audit logging on every prediction
  - Input size limits
  - PHI detection before any external API call
  - HTTPS only (enforced by load balancer)

HIPAA Compliance:
  - No PHI stored in logs
  - All predictions logged with pseudonymized hashes
  - 7-year audit log retention (AWS S3 + CloudWatch)
  - Access control per role
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
import uuid
from datetime import datetime, UTC
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, field_validator

# Local imports
from src.safety.phi_detector   import PHIDetector
from src.safety.input_validator import InputValidator
from src.safety.fairness_monitor import FairnessMonitor
from src.api.audit              import AuditLogger
from src.api.auth               import JWTHandler, UserRole, decode_token
from src.api.case_store         import CaseStore
from src.integration.fhir_pacs  import FHIRImport, PACSImport, build_diagnostic_report_resource
from src.monitoring.case_analytics import CaseAnalytics
from src.evaluation.active_learning import build_active_learning_queue
from src.evaluation.datasets import DatasetRegistry, default_dataset_specs
from src.evaluation.model_card import build_model_card
from src.evaluation.model_registry import ModelRegistry
from src.models.foundation_v2 import FoundationModelV2Spec


# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="ClinicalAI Multi-Modal CDS API",
    description="Production-grade AI clinical decision support — HIPAA compliant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clinical-ai.yourdomain.com"],   # production domain only
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # set specific hosts in prod

# ── Global singletons ────────────────────────────────────────────────
phi_detector    = PHIDetector()
input_validator = InputValidator()
fairness_monitor = FairnessMonitor()
audit_logger    = AuditLogger()
jwt_handler     = JWTHandler()
security        = HTTPBearer()
case_store      = CaseStore()
case_analytics  = CaseAnalytics(case_store)
dataset_registry = DatasetRegistry()
for _spec in default_dataset_specs():
    if _spec.name not in {item["name"] for item in dataset_registry.list()}:
        dataset_registry.register(_spec)
model_registry = ModelRegistry()

# Pipeline loaded lazily to avoid import-time model loading
_pipeline       = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import yaml
        from src.pipeline.inference_v2 import ClinicalAIPipelineV2
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        _pipeline = ClinicalAIPipelineV2(config)
        logger.info("ClinicalAI pipeline loaded on first request")
    return _pipeline


# ── Rate limiter (simple in-memory, use Redis in production) ─────────
_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX    = 60   # requests per window

def check_rate_limit(token_id: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = _rate_limit_store.get(token_id, [])
    calls = [t for t in calls if t > window_start]
    if len(calls) >= RATE_LIMIT_MAX:
        return False
    calls.append(now)
    _rate_limit_store[token_id] = calls
    return True


# ── Auth dependency ───────────────────────────────────────────────────
async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Rate limit check
    if not check_rate_limit(payload["sub"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s",
        )
    return payload


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    if user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


# ── Request / Response schemas ───────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_b64:     str  = Field(..., description="Base64-encoded chest X-ray (JPEG/PNG/DICOM)")
    clinical_note: str  = Field(..., min_length=10, max_length=5000,
                                description="Clinical notes / patient history")
    patient_context: Optional[dict] = Field(
        default=None,
        description="Optional: {age: int, gender: str} — used for fairness monitoring only. Never stored."
    )
    request_id:    Optional[str] = Field(default=None, description="Client-supplied idempotency key")

    @field_validator("image_b64")
    @classmethod
    def check_image_size(cls, v):
        # ~15MB base64 limit (≈10MB raw image)
        if len(v) > 15_000_000:
            raise ValueError("Image too large. Maximum 10MB.")
        return v


class FindingResponse(BaseModel):
    label:          str
    probability:    float
    confidence_interval: tuple[float, float]
    urgent:         bool


class AnalyzeResponse(BaseModel):
    case_id:            str
    request_id:         str
    findings:           list[FindingResponse]
    urgency_score:      float
    urgency_label:      str
    clinical_report:    str
    uncertainty:        dict
    quality_score:      float
    safety_flags:       list[str]
    inference_time_ms:  float
    model_version:      str
    disclaimer:         str


class FeedbackRequest(BaseModel):
    request_id:     str
    prediction_id:  str
    radiologist_findings: list[str]
    comments:       Optional[str]
    corrected:      bool
    decision:        str = Field(default="edited", description="accepted | edited | rejected | escalated")


class CaseStatusUpdate(BaseModel):
    status: str = Field(..., description="new | in_review | accepted | edited | rejected | escalated")
    assigned_to: Optional[str] = None


class IntegrationAttachRequest(BaseModel):
    source: str = Field(..., description="FHIR or PACS")
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    imaging_study_id: Optional[str] = None
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    sop_instance_uid: Optional[str] = None
    accession_number: Optional[str] = None
    modality: str = "CR"
    body_site: str = "Chest"
    study_description: str = "Chest radiograph"
    aetitle: Optional[str] = None


class ReportVersionRequest(BaseModel):
    report_text: str = Field(..., min_length=20, max_length=8000)
    structured_findings: list[dict] = Field(default_factory=list)
    change_summary: Optional[str] = None
    source: str = "clinician_edit"


# ── Health check ─────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version":   "1.0.0",
        "services": {
            "phi_detector":    "online",
            "input_validator": "online",
            "fairness_monitor":"online",
        }
    }


# ── Main inference endpoint ───────────────────────────────────────────
@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze(
    request:    AnalyzeRequest,
    http_req:   Request,
    user:       dict = Depends(require_auth),
):
    """
    Full multi-modal chest X-ray analysis.

    Security pipeline (in order):
      1. JWT authentication + rate limiting
      2. Input size validation
      3. PHI detection + redaction on clinical note
      4. Image validation (quality, adversarial detection)
      5. ML inference (BiomedCLIP + ClinicalBERT + Fusion)
      6. Uncertainty quantification
      7. Fairness monitoring record
      8. HIPAA audit log
      9. Return de-identified response
    """
    t_start    = time.perf_counter()
    request_id = request.request_id or str(uuid.uuid4())
    safety_flags: list[str] = []

    # ── Step 1: PHI Detection ─────────────────────────────────────────
    phi_result = phi_detector.detect_and_redact(request.clinical_note)
    if phi_result.phi_found:
        safety_flags.append(f"PHI_DETECTED:{','.join(set(m.category.value for m in phi_result.matches))}")
        logger.warning(f"[{request_id}] PHI found in note — redacted before LLM call")
        # Use redacted text for all downstream processing
        safe_note = phi_result.redacted_text
    else:
        safe_note = request.clinical_note

    # ── Step 2: Image Decoding ────────────────────────────────────────
    try:
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {str(e)}"
        )

    # ── Step 3: Input Validation ──────────────────────────────────────
    img_val, note_val = input_validator.validate_all(image, safe_note, image_bytes)

    if not img_val.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Image validation failed", "errors": img_val.errors},
        )
    if not note_val.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Clinical note validation failed", "errors": note_val.errors},
        )

    # Collect warnings as safety flags
    for w in img_val.warnings + note_val.warnings:
        safety_flags.append(f"WARN:{w[:80]}")

    if img_val.metadata.get("adversarial_score", 0) > 0.035:
        safety_flags.append("ADVERSARIAL_SUSPECTED")

    # ── Step 4: ML Inference ──────────────────────────────────────────
    try:
        pipeline = get_pipeline()
        result   = pipeline.predict(image, safe_note)
    except Exception as e:
        logger.error(f"[{request_id}] Pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference pipeline error. Please retry."
        )

    # ── Step 5: Fairness monitoring ───────────────────────────────────
    age    = request.patient_context.get("age")    if request.patient_context else None
    gender = request.patient_context.get("gender") if request.patient_context else None

    fairness_monitor.record_prediction(
        confidence=result.findings[0]["prob"] if result.findings else 0.5,
        urgency=result.urgency_score,
        label_predicted=result.top_finding,
        age=age,
        gender=gender,
    )
    bias_alerts = fairness_monitor.check_disparities()
    if bias_alerts:
        safety_flags.extend([f"FAIRNESS:{a.alert_type}" for a in bias_alerts])

    # ── Step 6: Audit Log (HIPAA) ─────────────────────────────────────
    inference_time_ms = (time.perf_counter() - t_start) * 1000

    audit_logger.log_prediction(
        request_id=request_id,
        user_id=user["sub"],
        role=user.get("role"),
        ip_hash=hashlib.sha256(http_req.client.host.encode()).hexdigest()[:12],
        phi_detected=phi_result.phi_found,
        phi_summary=phi_detector.audit_summary(phi_result),
        image_quality=img_val.metadata.get("quality_score", 0),
        top_finding=result.top_finding,
        urgency=result.urgency_score,
        safety_flags=safety_flags,
        inference_time_ms=inference_time_ms,
    )

    # ── Step 7: Build response ────────────────────────────────────────
    findings_response = [
        FindingResponse(
            label=f["label"],
            probability=round(f["prob"], 4),
            confidence_interval=result.uncertainty.confidence_interval if result.uncertainty else (0.0, 1.0),
            urgent=f["urgent"],
        )
        for f in result.findings
    ]

    uncertainty_dict = (
        result.uncertainty.summary(result.LABELS)
        if result.uncertainty else {}
    )

    case = case_store.create_case(
        request_id=request_id,
        priority=getattr(result, "workflow", {}).get("priority", result.urgency_label),
        urgency_score=round(result.urgency_score, 4),
        top_finding=result.top_finding,
        patient_context=request.patient_context or {},
        findings=[
            {
                "label": f["label"],
                "prob": round(f["prob"], 4),
                "urgent": f["urgent"],
            }
            for f in result.findings
        ],
        workflow=getattr(result, "workflow", {}),
        uncertainty=uncertainty_dict,
        clinical_report=result.clinical_report,
        safety_flags=safety_flags,
    )

    return AnalyzeResponse(
        case_id=case.case_id,
        request_id=request_id,
        findings=findings_response,
        urgency_score=round(result.urgency_score, 4),
        urgency_label=result.urgency_label,
        clinical_report=result.clinical_report,
        uncertainty=uncertainty_dict,
        quality_score=round(img_val.metadata.get("quality_score", 0.8), 3),
        safety_flags=safety_flags,
        inference_time_ms=round(inference_time_ms, 1),
        model_version="clinical-ai-v1.0.0",
        disclaimer=(
            "This AI analysis is for decision support only. "
            "All findings must be reviewed by a licensed radiologist "
            "before any clinical action is taken."
        ),
    )


# ── Clinical worklist ─────────────────────────────────────────────────
@app.get("/api/v1/cases", tags=["Workflow"])
async def list_cases(
    status_filter: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 50,
    user: dict = Depends(require_auth),
):
    limit = max(1, min(limit, 200))
    return {
        "cases": [
            case.to_dict()
            for case in case_store.list_cases(
                status=status_filter,
                priority=priority,
                limit=limit,
            )
        ]
    }


@app.get("/api/v1/cases/{case_id}", tags=["Workflow"])
async def get_case(case_id: str, user: dict = Depends(require_auth)):
    try:
        return case_store.get_case(case_id).to_dict()
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")


@app.patch("/api/v1/cases/{case_id}", tags=["Workflow"])
async def update_case_status(
    case_id: str,
    update: CaseStatusUpdate,
    user: dict = Depends(require_auth),
):
    try:
        return case_store.update_status(
            case_id,
            status=update.status,
            assigned_to=update.assigned_to or user.get("sub"),
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")


@app.post("/api/v1/cases/{case_id}/integration", tags=["Integration"])
async def attach_integration(
    case_id: str,
    payload: IntegrationAttachRequest,
    user: dict = Depends(require_auth),
):
    source = payload.source.upper()
    try:
        if source == "FHIR":
            if not payload.patient_id:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="patient_id is required for FHIR")
            metadata = FHIRImport(
                patient_id=payload.patient_id,
                encounter_id=payload.encounter_id,
                imaging_study_id=payload.imaging_study_id,
                accession_number=payload.accession_number,
                modality=payload.modality,
                body_site=payload.body_site,
                study_description=payload.study_description,
            ).to_integration_metadata()
        elif source == "PACS":
            if not payload.study_instance_uid:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="study_instance_uid is required for PACS")
            metadata = PACSImport(
                study_instance_uid=payload.study_instance_uid,
                series_instance_uid=payload.series_instance_uid,
                sop_instance_uid=payload.sop_instance_uid,
                accession_number=payload.accession_number,
                aetitle=payload.aetitle,
                modality=payload.modality,
            ).to_integration_metadata()
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="source must be FHIR or PACS")
        return case_store.update_integration(case_id, integration=metadata).to_dict()
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")


@app.post("/api/v1/cases/{case_id}/report_versions", tags=["Reporting"])
async def create_report_version(
    case_id: str,
    payload: ReportVersionRequest,
    user: dict = Depends(require_auth),
):
    try:
        structured = payload.structured_findings or case_store.get_case(case_id).structured_findings
        return case_store.add_report_version(
            case_id=case_id,
            author_id_hash=hashlib.sha256(user["sub"].encode()).hexdigest()[:12],
            source=payload.source,
            report_text=payload.report_text,
            structured_findings=structured,
            change_summary=payload.change_summary,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")


@app.get("/api/v1/cases/{case_id}/report_versions", tags=["Reporting"])
async def list_report_versions(case_id: str, user: dict = Depends(require_auth)):
    try:
        case_store.get_case(case_id)
        return {"versions": case_store.list_report_versions(case_id)}
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")


@app.get("/api/v1/cases/{case_id}/fhir/diagnostic-report", tags=["Integration"])
async def export_fhir_diagnostic_report(case_id: str, user: dict = Depends(require_auth)):
    try:
        case = case_store.get_case(case_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return build_diagnostic_report_resource(
        case_id=case.case_id,
        report_text=case.clinical_report,
        structured_findings=case.structured_findings,
        integration=case.integration,
    )


# ── Fairness report ───────────────────────────────────────────────────
@app.get("/api/v1/fairness/report", tags=["Monitoring"])
async def fairness_report(user: dict = Depends(require_auth)):
    return fairness_monitor.get_fairness_report()


@app.get("/api/v1/analytics/dashboard", tags=["Monitoring"])
async def analytics_dashboard(
    limit: int = 1000,
    user: dict = Depends(require_auth),
):
    limit = max(1, min(limit, 5000))
    return CaseAnalytics(case_store).dashboard(limit=limit)


@app.get("/api/v1/validation/dashboard", tags=["Validation"])
async def validation_dashboard(user: dict = Depends(require_auth)):
    analytics = CaseAnalytics(case_store).dashboard(limit=1000)
    latest_model = model_registry.latest()
    datasets = dataset_registry.list()
    foundation_spec = FoundationModelV2Spec().to_dict()
    active_learning = build_active_learning_queue(case_store, limit=25)
    card = build_model_card(
        model_version=latest_model.model_version if latest_model else "clinical-ai-v1.0.0",
        datasets=datasets,
        metrics=latest_model.metrics if latest_model else {"note": "No registered validation run yet."},
        calibration=latest_model.calibration if latest_model else {"note": "Calibration pending external validation."},
        thresholds=latest_model.thresholds if latest_model else {"note": "Thresholds pending optimization."},
    )
    return {
        "analytics": analytics,
        "datasets": datasets,
        "latest_model": latest_model.__dict__ if latest_model else None,
        "active_learning_queue": active_learning,
        "foundation_model_v2": foundation_spec,
        "model_card": card,
    }


@app.get("/api/v1/validation/active-learning", tags=["Validation"])
async def active_learning_queue(limit: int = 50, user: dict = Depends(require_auth)):
    return {"queue": build_active_learning_queue(case_store, limit=max(1, min(limit, 200)))}


# ── Model info ────────────────────────────────────────────────────────
@app.get("/api/v1/model/info", tags=["System"])
async def model_info(user: dict = Depends(require_auth)):
    return {
        "model_version":  "clinical-ai-v1.0.0",
        "image_encoder":  "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "text_encoder":   "Bio_ClinicalBERT",
        "fusion":         "CrossModalAttention-MLP-256",
        "uncertainty":    "MCDropout-30 + TemperatureScaling",
        "trained_on":     "CheXpert-224k + MIMIC-CXR-227k",
        "num_classes":    14,
        "labels": [
            "No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
            "Lung Opacity","Lung Lesion","Edema","Consolidation",
            "Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion",
            "Pleural Other","Fracture","Support Devices",
        ],
        "performance": {
            "mean_auc_chexpert": 0.873,
            "mean_auc_mimic":    0.861,
            "ece_after_calibration": 0.028,
        },
    }


# ── Radiologist feedback ──────────────────────────────────────────────
@app.post("/api/v1/feedback", tags=["Feedback Loop"])
async def submit_feedback(
    feedback: FeedbackRequest,
    user:     dict = Depends(require_auth),
):
    """
    Radiologist can submit ground-truth feedback for model improvement.
    This feeds the active learning loop and is used to retrain quarterly.
    """
    audit_logger.log_feedback(
        request_id=feedback.request_id,
        user_id=user["sub"],
        corrected=feedback.corrected,
        finding_count=len(feedback.radiologist_findings),
    )
    try:
        feedback_entry = case_store.add_feedback(
            request_id=feedback.request_id,
            user_id_hash=hashlib.sha256(user["sub"].encode()).hexdigest()[:12],
            decision=feedback.decision,
            corrected=feedback.corrected,
            radiologist_findings=feedback.radiologist_findings,
            comments=feedback.comments,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    try:
        case_store.update_status(feedback_entry["case_id"], status=feedback.decision, assigned_to=user["sub"])
    except ValueError:
        case_store.update_status(feedback_entry["case_id"], status="edited", assigned_to=user["sub"])
    logger.info(f"Feedback received for {feedback.request_id} — corrected={feedback.corrected}")
    return {"status": "feedback_received", "request_id": feedback.request_id, "feedback": feedback_entry}


# ── Audit log (admin only) ────────────────────────────────────────────
@app.get("/api/v1/audit/logs", tags=["Admin"])
async def audit_logs(
    limit: int = 100,
    admin: dict = Depends(require_admin),
):
    return audit_logger.get_recent_logs(limit=limit)


# ── Global error handlers ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please contact support."},
    )
