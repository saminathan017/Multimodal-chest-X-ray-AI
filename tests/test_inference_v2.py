import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.inference_v2 import ClinicalAIPipelineV2


def test_pipeline_v2_fallback_returns_structured_prediction():
    pipeline = ClinicalAIPipelineV2({}, allow_fallback=True)

    result = pipeline.predict(
        object(),
        "72yo male with fever, productive cough, dyspnea, and O2 sat 89%.",
    )

    assert result.findings
    assert result.top_finding in {finding["label"] for finding in result.findings}
    assert 0.0 <= result.urgency_score <= 1.0
    assert result.uncertainty is not None
    assert result.workflow["priority"] in {
        "Critical Review",
        "Routine Review",
        "Radiologist Review",
        "Low Priority",
        "Image Quality Review",
    }
    assert "licensed clinician" in result.clinical_report
