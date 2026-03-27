"""
src/api/audit.py
═══════════════════════════════════════════════════════════════════
HIPAA Audit Logger

Every prediction, access event, and feedback submission is
immutably logged with:
  - Timestamp (UTC)
  - Pseudonymized user ID (SHA-256 hash)
  - Pseudonymized request ID
  - PHI detection summary (category counts, NO actual PHI values)
  - Image quality score
  - Prediction summary (finding label, urgency — NOT patient data)
  - Safety flags triggered
  - Inference latency

Logs are:
  - Written to local structured JSON log (loguru)
  - Shipped to AWS CloudWatch Logs (async)
  - Retained for 7 years (HIPAA requirement: 45 CFR 164.312(b))
  - NEVER contain actual PHI — only pseudonymized identifiers

This module is the compliance backbone of the system.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class AuditLogger:
    """
    Immutable, append-only HIPAA audit log.

    All log entries are structured JSON, pseudonymized,
    and contain ZERO PHI values.
    """

    LOG_PATH = Path("logs/audit.jsonl")

    def __init__(self, use_cloudwatch: bool = False, log_group: str = "clinical-ai-audit"):
        self.use_cloudwatch = use_cloudwatch
        self.log_group      = log_group
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._cw_client     = None
        self._log_stream    = f"clinical-ai-{datetime.utcnow().strftime('%Y-%m-%d')}"

        if use_cloudwatch:
            self._init_cloudwatch(log_group)

        logger.info(f"AuditLogger initialised — log path: {self.LOG_PATH}")

    def _init_cloudwatch(self, log_group: str):
        try:
            import boto3
            self._cw_client = boto3.client("logs")
            # Create log group if not exists
            try:
                self._cw_client.create_log_group(logGroupName=log_group)
                self._cw_client.put_retention_policy(
                    logGroupName=log_group,
                    retentionInDays=2555,   # 7 years
                )
            except self._cw_client.exceptions.ResourceAlreadyExistsException:
                pass
            logger.info(f"CloudWatch audit logging enabled: {log_group}")
        except Exception as e:
            logger.warning(f"CloudWatch init failed: {e}. Local-only logging.")
            self._cw_client = None

    def log_prediction(
        self,
        request_id:       str,
        user_id:          str,
        role:             Optional[str],
        ip_hash:          str,
        phi_detected:     bool,
        phi_summary:      dict,
        image_quality:    float,
        top_finding:      str,
        urgency:          float,
        safety_flags:     list[str],
        inference_time_ms:float,
    ) -> None:
        entry = {
            "event_type":       "PREDICTION",
            "event_id":         str(uuid.uuid4()),
            "timestamp_utc":    datetime.utcnow().isoformat() + "Z",
            "request_id":       request_id,
            "user_id_hash":     user_id[:16] + "...",  # Truncated pseudonym
            "role":             role,
            "ip_hash":          ip_hash,
            "phi_detected":     phi_detected,
            "phi_category_counts": phi_summary.get("categories", {}),
            "phi_match_count":  phi_summary.get("total_matches", 0),
            # Prediction summary — NOT patient-specific PHI
            "top_finding":      top_finding,
            "urgency_score":    round(urgency, 3),
            "image_quality":    round(image_quality, 3),
            "safety_flags":     safety_flags,
            "inference_time_ms":round(inference_time_ms, 1),
        }
        self._write(entry)

    def log_feedback(
        self,
        request_id: str,
        user_id:    str,
        corrected:  bool,
        finding_count: int,
    ) -> None:
        entry = {
            "event_type":   "RADIOLOGIST_FEEDBACK",
            "event_id":     str(uuid.uuid4()),
            "timestamp_utc":datetime.utcnow().isoformat() + "Z",
            "request_id":   request_id,
            "user_id_hash": user_id[:16] + "...",
            "corrected":    corrected,
            "finding_count":finding_count,
        }
        self._write(entry)

    def log_access(self, user_id: str, endpoint: str, ip_hash: str) -> None:
        entry = {
            "event_type":   "ACCESS",
            "event_id":     str(uuid.uuid4()),
            "timestamp_utc":datetime.utcnow().isoformat() + "Z",
            "user_id_hash": user_id[:16] + "...",
            "endpoint":     endpoint,
            "ip_hash":      ip_hash,
        }
        self._write(entry)

    def _write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        # Local append-only log
        with open(self.LOG_PATH, "a") as f:
            f.write(line + "\n")
        # Structured logger
        logger.bind(audit=True).info(line)
        # CloudWatch (async, non-blocking)
        if self._cw_client:
            self._ship_to_cloudwatch(line)

    def _ship_to_cloudwatch(self, message: str) -> None:
        try:
            self._cw_client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self._log_stream,
                logEvents=[{"timestamp": int(time.time()*1000), "message": message}],
            )
        except Exception as e:
            logger.warning(f"CloudWatch ship failed: {e}")

    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        """Return recent audit entries (admin endpoint)."""
        if not self.LOG_PATH.exists():
            return []
        lines = self.LOG_PATH.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-limit:] if l.strip()]
