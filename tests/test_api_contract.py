import base64
import io
import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient
from PIL import Image

from src.api.auth import JWTHandler, UserRole
from src.api.case_store import CaseStore


def _sample_image_b64() -> str:
    rng = np.random.default_rng(42)
    arr = rng.normal(85, 32, (512, 512)).clip(0, 255).astype("uint8")
    image = Image.fromarray(arr, mode="L").convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_analyze_endpoint_returns_clinical_contract(monkeypatch):
    from src.api import main

    class FakePipeline:
        def predict(self, image, clinical_note):
            from src.pipeline.inference_v2 import ClinicalAIPipelineV2

            return ClinicalAIPipelineV2({}).predict(image, clinical_note)

    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())

    token = JWTHandler().create_token("doctor-1", UserRole.RADIOLOGIST, "demo-hospital")
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "image_b64": _sample_image_b64(),
            "clinical_note": (
                "72yo male with fever, productive cough, dyspnea, and O2 sat 89%. "
                "Concern for pneumonia."
            ),
            "patient_context": {"age": 72, "gender": "male"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"]
    assert payload["request_id"]
    assert payload["findings"]
    assert 0.0 <= payload["urgency_score"] <= 1.0
    assert payload["clinical_report"]
    assert "decision support" in payload["disclaimer"].lower()
    assert "uncertainty" in payload
    assert "model_version" in payload


def test_worklist_endpoints_round_trip(monkeypatch, tmp_path):
    from src.api import main

    class FakePipeline:
        def predict(self, image, clinical_note):
            from src.pipeline.inference_v2 import ClinicalAIPipelineV2

            return ClinicalAIPipelineV2({}).predict(image, clinical_note)

    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(main, "case_store", CaseStore(tmp_path / "worklist.db"))

    token = JWTHandler().create_token("doctor-1", UserRole.RADIOLOGIST, "demo-hospital")
    client = TestClient(main.app)
    headers = {"Authorization": f"Bearer {token}"}

    analyze_response = client.post(
        "/api/v1/analyze",
        headers=headers,
        json={
            "image_b64": _sample_image_b64(),
            "clinical_note": "72yo male with fever, cough, dyspnea, and concern for pneumonia.",
            "patient_context": {"age": 72, "gender": "male"},
        },
    )
    assert analyze_response.status_code == 200
    payload = analyze_response.json()

    list_response = client.get("/api/v1/cases", headers=headers)
    assert list_response.status_code == 200
    assert list_response.json()["cases"][0]["case_id"] == payload["case_id"]

    patch_response = client.patch(
        f"/api/v1/cases/{payload['case_id']}",
        headers=headers,
        json={"status": "in_review"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "in_review"

    feedback_response = client.post(
        "/api/v1/feedback",
        headers=headers,
        json={
            "request_id": payload["request_id"],
            "prediction_id": payload["case_id"],
            "radiologist_findings": ["Pneumonia"],
            "comments": "Agree with AI draft.",
            "corrected": False,
            "decision": "accepted",
        },
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["feedback"]["decision"] == "accepted"

    integration_response = client.post(
        f"/api/v1/cases/{payload['case_id']}/integration",
        headers=headers,
        json={
            "source": "FHIR",
            "patient_id": "patient-123",
            "encounter_id": "enc-456",
            "imaging_study_id": "img-789",
            "accession_number": "ACC-1",
        },
    )
    assert integration_response.status_code == 200
    assert integration_response.json()["integration"]["source"] == "FHIR"
    assert "patient-123" not in str(integration_response.json())

    version_response = client.post(
        f"/api/v1/cases/{payload['case_id']}/report_versions",
        headers=headers,
        json={
            "report_text": "CLINICAL INDICATION\nCough.\n\nIMPRESSION\nRight lower zone pneumonia is suspected.",
            "structured_findings": [
                {
                    "label": "Pneumonia",
                    "status": "confirmed",
                    "probability": 0.82,
                    "laterality": "right",
                    "location": "lower zone",
                    "severity": "moderate",
                    "clinician_note": "Edited by radiologist.",
                }
            ],
            "change_summary": "Confirmed right lower zone location.",
        },
    )
    assert version_response.status_code == 200
    assert version_response.json()["source"] == "clinician_edit"

    fhir_response = client.get(
        f"/api/v1/cases/{payload['case_id']}/fhir/diagnostic-report",
        headers=headers,
    )
    assert fhir_response.status_code == 200
    assert fhir_response.json()["resourceType"] == "DiagnosticReport"
    assert fhir_response.json()["result"][0]["status"] == "confirmed"

    analytics_response = client.get("/api/v1/analytics/dashboard", headers=headers)
    assert analytics_response.status_code == 200
    assert analytics_response.json()["summary"]["total_cases"] == 1
    assert analytics_response.json()["summary"]["integration_coverage"] == 1.0

    validation_response = client.get("/api/v1/validation/dashboard", headers=headers)
    assert validation_response.status_code == 200
    assert validation_response.json()["datasets"]
    assert validation_response.json()["foundation_model_v2"]["fusion"] == "patch_token_cross_attention"

    active_response = client.get("/api/v1/validation/active-learning", headers=headers)
    assert active_response.status_code == 200
    assert "queue" in active_response.json()
