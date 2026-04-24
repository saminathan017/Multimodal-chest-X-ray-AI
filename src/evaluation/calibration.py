"""Calibration, conformal prediction, and abstention utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=float).reshape(-1), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (p >= start) & (p < end if end < 1 else p <= end)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / len(p)) * abs(acc - conf)
    return round(float(ece), 6)


def brier_score(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return round(float(np.mean((p - y) ** 2)), 6)


def temperature_scale_logits(logits, temperature: float):
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


def conformal_prediction_sets(
    probabilities,
    labels: list[str],
    coverage: float = 0.95,
) -> list[list[str]]:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    sets = []
    for row in probs:
        order = np.argsort(-row)
        cumulative = 0.0
        pred_set = []
        for idx in order:
            pred_set.append(labels[idx] if idx < len(labels) else f"class_{idx}")
            cumulative += float(row[idx])
            if cumulative >= coverage:
                break
        sets.append(pred_set)
    return sets


def abstention_mask(
    probabilities,
    *,
    max_confidence_threshold: float = 0.45,
    margin_threshold: float = 0.08,
    uncertainty_scores=None,
    uncertainty_threshold: float = 0.20,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    max_conf = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1] if probs.shape[1] > 1 else sorted_probs[:, 0]
    abstain = (max_conf < max_confidence_threshold) | (margin < margin_threshold)
    if uncertainty_scores is not None:
        abstain = abstain | (np.asarray(uncertainty_scores, dtype=float) >= uncertainty_threshold)
    return abstain


def calibration_report(y_true, y_prob, labels: list[str] | None = None, n_bins: int = 15) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        p = p.reshape(-1, 1)
    labels = labels or [f"class_{i}" for i in range(y.shape[1])]
    per_class = {}
    for idx in range(y.shape[1]):
        name = labels[idx] if idx < len(labels) else f"class_{idx}"
        per_class[name] = {
            "ece": expected_calibration_error(y[:, idx], p[:, idx], n_bins=n_bins),
            "brier": brier_score(y[:, idx], p[:, idx]),
            "mean_confidence": round(float(np.mean(p[:, idx])), 6),
            "prevalence": round(float(np.mean(y[:, idx])), 6),
        }
    return {
        "macro_ece": round(float(np.mean([v["ece"] for v in per_class.values()])), 6),
        "macro_brier": round(float(np.mean([v["brier"] for v in per_class.values()])), 6),
        "per_class": per_class,
    }

