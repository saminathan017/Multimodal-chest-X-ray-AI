"""Clinical threshold optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import threshold_metrics_binary


def optimize_thresholds(
    y_true,
    y_prob,
    labels: list[str] | None = None,
    *,
    min_sensitivity: float = 0.90,
    objective: str = "youden",
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        p = p.reshape(-1, 1)
    labels = labels or [f"class_{i}" for i in range(y.shape[1])]
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 91)

    result = {}
    for idx in range(y.shape[1]):
        candidates = []
        for threshold in grid:
            metrics = threshold_metrics_binary(y[:, idx], p[:, idx], float(threshold))
            if metrics["sensitivity"] < min_sensitivity:
                continue
            if objective == "f1":
                score = metrics["f1"]
            elif objective == "npv":
                score = metrics["npv"]
            else:
                score = metrics["sensitivity"] + metrics["specificity"] - 1
            candidates.append((score, metrics))
        if not candidates:
            fallback = threshold_metrics_binary(y[:, idx], p[:, idx], 0.5)
            fallback["selection_note"] = "No threshold met minimum sensitivity; using 0.50 fallback."
            selected = fallback
        else:
            selected = max(candidates, key=lambda item: item[0])[1]
            selected["selection_note"] = f"Optimized for {objective} with sensitivity >= {min_sensitivity:.2f}."
        result[labels[idx] if idx < len(labels) else f"class_{idx}"] = selected
    return {"thresholds": result, "min_sensitivity": min_sensitivity, "objective": objective}

