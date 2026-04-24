"""
Clinical classification metrics.

Implements dependency-light AUROC/AUPRC and threshold metrics for multi-label
medical AI validation. These functions are intentionally explicit so validation
reports remain reproducible in constrained clinical environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_prob={prob.shape}")
    if true.ndim == 1:
        true = true.reshape(-1, 1)
        prob = prob.reshape(-1, 1)
    return true, np.clip(prob, 0.0, 1.0)


def roc_auc_binary(y_true, y_score) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    unique_scores = np.unique(s)
    for score in unique_scores:
        idx = np.where(s == score)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    auc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return round(float(auc), 6)


def average_precision_binary(y_true, y_score) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    ap = float(np.sum(precision[y_sorted == 1]) / positives)
    return round(ap, 6)


def threshold_metrics_binary(y_true, y_score, threshold: float = 0.5) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(tp / max(tp + fn, 1), 6),
        "specificity": round(tn / max(tn + fp, 1), 6),
        "ppv": round(tp / max(tp + fp, 1), 6),
        "npv": round(tn / max(tn + fn, 1), 6),
        "f1": round((2 * tp) / max(2 * tp + fp + fn, 1), 6),
        "false_negative_rate": round(fn / max(tp + fn, 1), 6),
    }


def classification_report(
    y_true,
    y_prob,
    labels: list[str] | None = None,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    true, prob = _as_arrays(y_true, y_prob)
    n_classes = true.shape[1]
    labels = labels or [f"class_{i}" for i in range(n_classes)]
    thresholds_arr = np.asarray(thresholds if thresholds is not None else [0.5] * n_classes)
    if thresholds_arr.size == 1:
        thresholds_arr = np.repeat(thresholds_arr, n_classes)

    per_class = {}
    aurocs = []
    auprcs = []
    fn_rates = []
    for idx in range(n_classes):
        auc = roc_auc_binary(true[:, idx], prob[:, idx])
        ap = average_precision_binary(true[:, idx], prob[:, idx])
        tm = threshold_metrics_binary(true[:, idx], prob[:, idx], float(thresholds_arr[idx]))
        if auc is not None:
            aurocs.append(auc)
        if ap is not None:
            auprcs.append(ap)
        fn_rates.append(float(tm["false_negative_rate"]))
        per_class[labels[idx] if idx < len(labels) else f"class_{idx}"] = {
            "auroc": auc,
            "auprc": ap,
            **tm,
            "support_positive": int(np.sum(true[:, idx] == 1)),
            "support_negative": int(np.sum(true[:, idx] == 0)),
        }

    return {
        "macro_auroc": round(float(np.mean(aurocs)), 6) if aurocs else None,
        "macro_auprc": round(float(np.mean(auprcs)), 6) if auprcs else None,
        "mean_false_negative_rate": round(float(np.mean(fn_rates)), 6) if fn_rates else None,
        "num_samples": int(true.shape[0]),
        "num_classes": int(n_classes),
        "per_class": per_class,
    }

