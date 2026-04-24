"""External validation orchestration."""

from __future__ import annotations

from typing import Any

from .calibration import calibration_report
from .metrics import classification_report
from .thresholds import optimize_thresholds


def external_validation_report(
    *,
    dataset_name: str,
    y_true,
    y_prob,
    labels: list[str],
    thresholds=None,
) -> dict[str, Any]:
    metrics = classification_report(y_true, y_prob, labels=labels, thresholds=thresholds)
    calibration = calibration_report(y_true, y_prob, labels=labels)
    optimized = optimize_thresholds(y_true, y_prob, labels=labels)
    return {
        "dataset": dataset_name,
        "metrics": metrics,
        "calibration": calibration,
        "threshold_optimization": optimized,
    }

