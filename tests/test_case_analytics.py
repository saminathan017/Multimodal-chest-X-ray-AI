import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.case_store import CaseStore
from src.monitoring.case_analytics import CaseAnalytics


def test_case_analytics_dashboard_counts_governance_metrics(tmp_path):
    store = CaseStore(tmp_path / "analytics.db")
    case = store.create_case(
        request_id="req-analytics",
        priority="Critical Review",
        urgency_score=0.91,
        top_finding="Pneumonia",
        patient_context={"age": 70},
        findings=[{"label": "Pneumonia", "prob": 0.91, "urgent": True}],
        workflow={"priority": "Critical Review"},
        uncertainty={"uncertainty_flag": True},
        clinical_report="AI draft report.",
        safety_flags=["WARN:low quality"],
        integration={"source": "FHIR"},
    )
    store.add_feedback(
        request_id=case.request_id,
        user_id_hash="doctorhash",
        decision="edited",
        corrected=True,
        radiologist_findings=["Pneumonia"],
        comments="Corrected laterality.",
    )
    store.update_status(case.case_id, status="edited", assigned_to="doctor-1")

    dashboard = CaseAnalytics(store).dashboard()

    assert dashboard["summary"]["total_cases"] == 1
    assert dashboard["summary"]["review_rate"] == 1.0
    assert dashboard["summary"]["correction_rate"] == 1.0
    assert dashboard["summary"]["integration_coverage"] == 1.0
    assert dashboard["summary"]["uncertainty_flag_rate"] == 1.0
    assert dashboard["safety_flags"]["WARN"] == 1
    assert dashboard["by_finding"]["Pneumonia"]["correction_rate"] == 1.0

