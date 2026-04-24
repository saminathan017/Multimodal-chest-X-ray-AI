"""
Clinical workflow and AI governance analytics.

Aggregates persisted case records into metrics that matter operationally:
review backlog, safety flags, correction rate, turnaround proxies, structured
reporting adoption, integration coverage, and model behavior by finding.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from src.api.case_store import CaseRecord, CaseStore


REVIEWED_STATUSES = {"accepted", "edited", "rejected", "escalated"}


def _pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


class CaseAnalytics:
    def __init__(self, store: CaseStore):
        self.store = store

    def dashboard(self, limit: int = 1000) -> dict[str, Any]:
        cases = self.store.list_cases(limit=limit)
        return build_dashboard(cases)


def build_dashboard(cases: list[CaseRecord]) -> dict[str, Any]:
    total = len(cases)
    reviewed = [case for case in cases if case.status in REVIEWED_STATUSES]
    feedback_cases = [case for case in cases if case.feedback]
    corrected_feedback = [
        feedback
        for case in cases
        for feedback in case.feedback
        if feedback.get("corrected")
    ]
    all_feedback = [feedback for case in cases for feedback in case.feedback]

    safety_flags = Counter(flag.split(":", 1)[0] for case in cases for flag in case.safety_flags)
    status_counts = Counter(case.status for case in cases)
    priority_counts = Counter(case.priority for case in cases)
    finding_counts = Counter(case.top_finding for case in cases)
    integration_counts = Counter(case.integration.get("source", "Unlinked") for case in cases)

    urgency_values = [case.urgency_score for case in cases]
    high_urgency = [case for case in cases if case.urgency_score >= 0.75]
    uncertainty_flags = [
        case
        for case in cases
        if bool(case.uncertainty.get("uncertainty_flag"))
    ]
    report_versions = [version for case in cases for version in case.report_versions]
    clinician_versions = [version for version in report_versions if version.get("source") != "ai_draft"]
    structured_cases = [case for case in cases if case.structured_findings]

    by_finding: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        grouped[case.top_finding].append(case)
    for finding, group in grouped.items():
        finding_feedback = [feedback for case in group for feedback in case.feedback]
        by_finding[finding] = {
            "case_count": len(group),
            "avg_urgency": round(mean([case.urgency_score for case in group]), 4),
            "correction_rate": _pct(
                sum(1 for feedback in finding_feedback if feedback.get("corrected")),
                len(finding_feedback),
            ),
            "review_rate": _pct(sum(1 for case in group if case.status in REVIEWED_STATUSES), len(group)),
        }

    alerts = []
    if total and _pct(len(high_urgency), total) >= 0.35:
        alerts.append("High-urgency workload is elevated.")
    if all_feedback and _pct(len(corrected_feedback), len(all_feedback)) >= 0.25:
        alerts.append("Clinician correction rate exceeds governance threshold.")
    if total and _pct(len(uncertainty_flags), total) >= 0.20:
        alerts.append("Uncertainty flags are elevated; review calibration and data drift.")
    if safety_flags:
        alerts.append("Safety flags present in recent case volume.")

    return {
        "summary": {
            "total_cases": total,
            "reviewed_cases": len(reviewed),
            "open_cases": total - len(reviewed),
            "review_rate": _pct(len(reviewed), total),
            "feedback_coverage": _pct(len(feedback_cases), total),
            "correction_rate": _pct(len(corrected_feedback), len(all_feedback)),
            "avg_urgency": round(mean(urgency_values), 4) if urgency_values else 0.0,
            "high_urgency_rate": _pct(len(high_urgency), total),
            "uncertainty_flag_rate": _pct(len(uncertainty_flags), total),
            "structured_reporting_rate": _pct(len(structured_cases), total),
            "clinician_edit_rate": _pct(len(clinician_versions), total),
            "integration_coverage": _pct(
                sum(1 for case in cases if case.integration.get("source")),
                total,
            ),
        },
        "status_counts": dict(status_counts),
        "priority_counts": dict(priority_counts),
        "top_findings": dict(finding_counts.most_common(10)),
        "safety_flags": dict(safety_flags),
        "integration_sources": dict(integration_counts),
        "by_finding": by_finding,
        "alerts": alerts,
    }

