import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.case_store import CaseStore


def test_case_store_persists_status_and_feedback(tmp_path):
    store = CaseStore(tmp_path / "cases.db")

    case = store.create_case(
        request_id="req-1",
        priority="Critical Review",
        urgency_score=0.88,
        top_finding="Pneumonia",
        patient_context={"age": 72, "gender": "male"},
        findings=[{"label": "Pneumonia", "prob": 0.88, "urgent": True}],
        workflow={"priority": "Critical Review"},
        uncertainty={"uncertainty_flag": False},
        clinical_report="AI report",
        safety_flags=["DEMO"],
    )

    updated = store.update_status(case.case_id, status="in_review", assigned_to="doctor-1")
    assert updated.status == "in_review"
    assert updated.assigned_to == "doctor-1"

    feedback = store.add_feedback(
        request_id="req-1",
        user_id_hash="doctorhash",
        decision="accepted",
        corrected=False,
        radiologist_findings=["Pneumonia"],
        comments="Agree with draft.",
    )
    assert feedback["case_id"] == case.case_id

    reloaded = store.get_case(case.case_id)
    assert len(reloaded.feedback) == 1
    assert reloaded.feedback[0]["decision"] == "accepted"
    assert len(reloaded.report_versions) == 1

    version = store.add_report_version(
        case_id=case.case_id,
        author_id_hash="doctorhash",
        source="clinician_edit",
        report_text="Edited report with structured impression.",
        structured_findings=[
            {
                "label": "Pneumonia",
                "status": "confirmed",
                "probability": 0.88,
                "laterality": "right",
                "location": "lower zone",
                "severity": "moderate",
                "clinician_note": "Correlates with fever.",
            }
        ],
        change_summary="Confirmed location and severity.",
    )
    assert version["source"] == "clinician_edit"

    final_case = store.update_integration(
        case.case_id,
        integration={"source": "FHIR", "patient_hash": "abc", "imaging_study_id": "img-1"},
    )
    assert final_case.integration["source"] == "FHIR"
    assert final_case.structured_findings[0]["status"] == "confirmed"
    assert len(final_case.report_versions) == 2
