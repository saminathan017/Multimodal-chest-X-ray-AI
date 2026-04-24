"""
Clinician workflow utilities for case review.

These helpers keep product behavior testable outside Streamlit/FastAPI:
triage priority, evidence extraction, suggested actions, and feedback labels.
They are intentionally conservative. The system supports clinicians; it does
not diagnose or prescribe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


HIGH_RISK_FINDINGS = {
    "Pneumothorax",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Pleural Effusion",
}

SYMPTOM_TERMS = {
    "fever": ("fever", "pyrexia", "febrile"),
    "cough": ("cough", "productive cough"),
    "dyspnea": ("dyspnea", "shortness of breath", "sob"),
    "chest pain": ("chest pain", "pleuritic pain"),
    "hypoxia": ("hypoxia", "o2 sat", "oxygen saturation", "spo2"),
    "orthopnea": ("orthopnea",),
    "edema": ("leg edema", "swelling", "edema"),
}


@dataclass(frozen=True)
class WorkflowSummary:
    priority: str
    priority_reason: str
    suggested_actions: list[str]
    evidence: list[str]
    documentation_status: str
    human_review_required: bool


def extract_evidence(note: str, findings: Iterable[dict]) -> list[str]:
    """Extract short, non-PHI clinical evidence snippets from note and findings."""
    note_l = note.lower()
    evidence: list[str] = []

    for label, terms in SYMPTOM_TERMS.items():
        if any(term in note_l for term in terms):
            evidence.append(f"Clinical note mentions {label}.")

    for finding in findings:
        prob = float(finding.get("prob", 0.0))
        if prob >= 0.45:
            evidence.append(f"Model flags {finding.get('label', 'finding')} at {prob:.0%}.")

    return evidence[:8]


def compute_workflow_summary(
    *,
    note: str,
    findings: list[dict],
    urgency_score: float,
    quality_score: float = 0.8,
    uncertainty_flag: bool = False,
) -> WorkflowSummary:
    """Create a clinician-facing case summary from model and validation signals."""
    top_finding = findings[0]["label"] if findings else "No Finding"
    top_prob = float(findings[0].get("prob", 0.0)) if findings else 0.0

    if quality_score < 0.45:
        priority = "Image Quality Review"
        reason = "Input quality is low enough to limit AI reliability."
    elif uncertainty_flag:
        priority = "Radiologist Review"
        reason = "Prediction uncertainty is elevated."
    elif urgency_score >= 0.75 or (top_finding in HIGH_RISK_FINDINGS and top_prob >= 0.75):
        priority = "Critical Review"
        reason = f"{top_finding} is high-confidence or clinically urgent."
    elif urgency_score >= 0.45 or top_prob >= 0.45:
        priority = "Routine Review"
        reason = "Clinically relevant abnormality is possible."
    else:
        priority = "Low Priority"
        reason = "No high-confidence urgent abnormality is present."

    actions = ["Verify image quality and patient identity before interpretation."]
    if priority == "Critical Review":
        actions.append("Escalate for immediate clinician or radiologist review.")
    if uncertainty_flag or quality_score < 0.65:
        actions.append("Treat AI output as low-confidence and prioritize human review.")
    if top_finding != "No Finding":
        actions.append("Correlate with symptoms, vitals, prior imaging, and labs.")
    actions.append("Capture accept, edit, or reject feedback for model monitoring.")

    doc_status = "Draft ready for physician edit" if top_prob >= 0.45 else "Needs clinician completion"
    human_review = priority != "Low Priority" or uncertainty_flag

    return WorkflowSummary(
        priority=priority,
        priority_reason=reason,
        suggested_actions=actions,
        evidence=extract_evidence(note, findings),
        documentation_status=doc_status,
        human_review_required=human_review,
    )

