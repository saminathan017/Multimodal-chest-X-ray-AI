"""Subgroup and external validation analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import classification_report


def subgroup_report(
    y_true,
    y_prob,
    subgroup_values,
    labels: list[str] | None = None,
    thresholds=None,
) -> dict[str, Any]:
    groups = np.asarray(subgroup_values)
    true = np.asarray(y_true)
    prob = np.asarray(y_prob)
    reports = {}
    macro_aurocs = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        report = classification_report(true[mask], prob[mask], labels=labels, thresholds=thresholds)
        reports[str(group)] = report
        if report["macro_auroc"] is not None:
            macro_aurocs.append(report["macro_auroc"])
    disparity = round(float(max(macro_aurocs) - min(macro_aurocs)), 6) if len(macro_aurocs) >= 2 else 0.0
    return {
        "groups": reports,
        "macro_auroc_disparity": disparity,
        "num_groups": len(reports),
    }

