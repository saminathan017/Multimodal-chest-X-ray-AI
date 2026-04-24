import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.case_store import CaseStore
from src.evaluation.active_learning import build_active_learning_queue
from src.evaluation.calibration import (
    abstention_mask,
    calibration_report,
    conformal_prediction_sets,
)
from src.evaluation.datasets import DatasetRegistry, DatasetSpec, clean_uncertain_label
from src.evaluation.metrics import classification_report
from src.evaluation.model_registry import ModelRegistry
from src.evaluation.report_quality import report_quality_report
from src.evaluation.subgroup_analysis import subgroup_report
from src.evaluation.thresholds import optimize_thresholds
from src.models.foundation_v2 import FoundationModelV2Spec


def test_classification_calibration_thresholds_and_subgroups():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.91, 0.10], [0.82, 0.20], [0.30, 0.80], [0.20, 0.72], [0.77, 0.15], [0.10, 0.92]])
    labels = ["Pneumonia", "Effusion"]

    metrics = classification_report(y_true, y_prob, labels=labels)
    calibration = calibration_report(y_true, y_prob, labels=labels)
    thresholds = optimize_thresholds(y_true, y_prob, labels=labels, min_sensitivity=0.9)
    subgroups = subgroup_report(y_true, y_prob, ["A", "A", "B", "B", "A", "B"], labels=labels)

    assert metrics["macro_auroc"] == 1.0
    assert calibration["macro_ece"] >= 0
    assert thresholds["thresholds"]["Pneumonia"]["sensitivity"] >= 0.9
    assert subgroups["num_groups"] == 2


def test_abstention_conformal_and_report_grounding():
    probs = np.array([[0.40, 0.38, 0.22], [0.90, 0.05, 0.05]])
    assert abstention_mask(probs).tolist() == [True, False]
    sets = conformal_prediction_sets(probs, ["A", "B", "C"], coverage=0.75)
    assert sets[0] == ["A", "B"]

    quality = report_quality_report(
        "CLINICAL INDICATION cough. FINDINGS opacity. IMPRESSION pneumonia.",
        [{"label": "Pneumonia", "location": "lower lobe"}],
    )
    assert quality["is_grounded"]
    assert quality["section_complete"]


def test_dataset_and_model_registries(tmp_path):
    dataset_registry = DatasetRegistry(tmp_path / "datasets.json")
    dataset_registry.register(DatasetSpec("MIMIC-CXR", "/data/mimic", "CXR", has_reports=True))
    assert dataset_registry.get("MIMIC-CXR").has_reports
    assert clean_uncertain_label(-1, "u_one") == 1.0
    assert clean_uncertain_label(-1, "u_ignore") is None

    model_registry = ModelRegistry(tmp_path / "models.json")
    model_registry.register(
        model_version="v1",
        checkpoint_uri="s3://bucket/model.tar.gz",
        datasets=["MIMIC-CXR"],
        metrics={"macro_auroc": 0.91},
        calibration={"macro_ece": 0.04},
        thresholds={"Pneumonia": 0.42},
    )
    promoted = model_registry.promote("v1")
    assert promoted.status == "production"
    assert model_registry.latest("production").model_version == "v1"


def test_active_learning_and_foundation_spec(tmp_path):
    store = CaseStore(tmp_path / "active.db")
    case = store.create_case(
        request_id="req-active",
        priority="Radiologist Review",
        urgency_score=0.82,
        top_finding="Pneumonia",
        patient_context={},
        findings=[{"label": "Pneumonia", "prob": 0.82, "urgent": True}],
        workflow={},
        uncertainty={"uncertainty_flag": True},
        clinical_report="AI draft",
        safety_flags=["WARN:uncertain"],
    )
    store.add_feedback(
        request_id=case.request_id,
        user_id_hash="doctor",
        decision="edited",
        corrected=True,
        radiologist_findings=["Pneumonia"],
        comments=None,
    )
    queue = build_active_learning_queue(store)
    assert queue[0]["case_id"] == case.case_id
    assert "clinician_correction" in queue[0]["reasons"]

    spec = FoundationModelV2Spec().to_dict()
    assert spec["fusion"] == "patch_token_cross_attention"
    assert any(head["name"] == "urgency" for head in spec["heads"])

