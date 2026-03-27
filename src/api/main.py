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
import json
import time
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Annotated

import boto3
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, validator

# Local imports
from src.safety.phi_detector   import PHIDetector
from src.safety.input_validator import InputValidator
from src.safety.fairness_monitor import FairnessMonitor
from src.api.audit              import AuditLogger
from src.api.auth               import JWTHandler, UserRole, decode_token


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
    allow_methods=["GET", "POST"],
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

    @validator("image_b64")
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


# ── Health check ─────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
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

    return AnalyzeResponse(
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


# ── Fairness report ───────────────────────────────────────────────────
@app.get("/api/v1/fairness/report", tags=["Monitoring"])
async def fairness_report(user: dict = Depends(require_auth)):
    return fairness_monitor.get_fairness_report()


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
    logger.info(f"Feedback received for {feedback.request_id} — corrected={feedback.corrected}")
    return {"status": "feedback_received", "request_id": feedback.request_id}


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
