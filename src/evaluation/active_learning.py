"""Active learning queue from clinician feedback and uncertainty signals."""

from __future__ import annotations

from typing import Any

from src.api.case_store import CaseRecord, CaseStore


def score_case_for_labeling(case: CaseRecord) -> float:
    score = 0.0
    if case.uncertainty.get("uncertainty_flag"):
        score += 0.35
    if case.status in {"edited", "rejected", "escalated"}:
        score += 0.25
    if any(feedback.get("corrected") for feedback in case.feedback):
        score += 0.25
    if case.urgency_score >= 0.75:
        score += 0.10
    if case.safety_flags:
        score += 0.05
    return round(min(score, 1.0), 4)


def build_active_learning_queue(store: CaseStore, limit: int = 50) -> list[dict[str, Any]]:
    candidates = []
    for case in store.list_cases(limit=1000):
        score = score_case_for_labeling(case)
        if score <= 0:
            continue
        reasons = []
        if case.uncertainty.get("uncertainty_flag"):
            reasons.append("uncertainty")
        if any(feedback.get("corrected") for feedback in case.feedback):
            reasons.append("clinician_correction")
        if case.status in {"edited", "rejected", "escalated"}:
            reasons.append(f"status_{case.status}")
        if case.urgency_score >= 0.75:
            reasons.append("high_urgency")
        if case.safety_flags:
            reasons.append("safety_flag")
        candidates.append(
            {
                "case_id": case.case_id,
                "request_id": case.request_id,
                "score": score,
                "top_finding": case.top_finding,
                "urgency_score": case.urgency_score,
                "status": case.status,
                "reasons": reasons,
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]

