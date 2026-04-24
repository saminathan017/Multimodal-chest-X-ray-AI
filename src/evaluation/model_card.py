"""Model card generation for validation reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_model_card(
    *,
    model_version: str,
    datasets: list[dict[str, Any]],
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    thresholds: dict[str, Any],
    subgroups: dict[str, Any] | None = None,
    intended_use: str = "Clinical decision support for chest radiograph triage and reporting.",
) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "intended_use": intended_use,
        "not_for": [
            "Autonomous diagnosis",
            "Replacement of radiologist interpretation",
            "Use outside validated population or imaging protocol",
        ],
        "datasets": datasets,
        "metrics": metrics,
        "calibration": calibration,
        "thresholds": thresholds,
        "subgroups": subgroups or {},
        "safety": {
            "requires_human_review": True,
            "uses_abstention": True,
            "monitors_drift": True,
            "stores_feedback": True,
        },
    }

