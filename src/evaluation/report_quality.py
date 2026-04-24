"""Report quality checks for grounded clinical report generation."""

from __future__ import annotations

from typing import Any


def unsupported_claims(report_text: str, structured_findings: list[dict[str, Any]]) -> list[str]:
    report_l = report_text.lower()
    supported = {str(item.get("label", "")).lower() for item in structured_findings}
    supported.update(str(item.get("location", "")).lower() for item in structured_findings)
    high_risk_terms = ["pneumothorax", "pneumonia", "edema", "effusion", "fracture", "cardiomegaly"]
    unsupported = []
    for term in high_risk_terms:
        if term in report_l and not any(term in item for item in supported):
            unsupported.append(term)
    return unsupported


def report_quality_report(report_text: str, structured_findings: list[dict[str, Any]]) -> dict[str, Any]:
    missing_sections = [
        section
        for section in ["CLINICAL INDICATION", "FINDINGS", "IMPRESSION"]
        if section.lower() not in report_text.lower()
    ]
    claims = unsupported_claims(report_text, structured_findings)
    return {
        "missing_sections": missing_sections,
        "unsupported_claims": claims,
        "is_grounded": not claims,
        "section_complete": not missing_sections,
    }

