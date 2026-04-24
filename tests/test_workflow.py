import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clinical_workflow import compute_workflow_summary, extract_evidence


def test_critical_pneumonia_requires_human_review():
    findings = [{"label": "Pneumonia", "prob": 0.88, "urgent": True}]

    summary = compute_workflow_summary(
        note="72yo male with fever, productive cough, dyspnea, O2 sat 89%.",
        findings=findings,
        urgency_score=0.86,
    )

    assert summary.priority == "Critical Review"
    assert summary.human_review_required
    assert any("immediate" in action.lower() for action in summary.suggested_actions)


def test_low_quality_image_overrides_other_triage():
    findings = [{"label": "No Finding", "prob": 0.91, "urgent": False}]

    summary = compute_workflow_summary(
        note="Routine pre-op chest radiograph. No respiratory symptoms.",
        findings=findings,
        urgency_score=0.12,
        quality_score=0.30,
    )

    assert summary.priority == "Image Quality Review"
    assert summary.human_review_required


def test_extract_evidence_combines_note_and_model_signals():
    evidence = extract_evidence(
        "Patient has fever and shortness of breath.",
        [{"label": "Lung Opacity", "prob": 0.63}],
    )

    assert "Clinical note mentions fever." in evidence
    assert any("Lung Opacity" in item for item in evidence)

