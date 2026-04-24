"""
Production-compatible inference facade.

The FastAPI service imports ClinicalAIPipelineV2. This module keeps that path
stable and gives the application a robust fallback when real model artifacts or
cloud credentials are unavailable in a demo/development environment.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.clinical_workflow import compute_workflow_summary

if TYPE_CHECKING:  # pragma: no cover
    from src.pipeline.inference import ClinicalAIPipeline, PredictionResult
    from PIL import Image


LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class LightweightUncertainty:
    mean_probs: np.ndarray
    std_probs: np.ndarray
    confidence_interval: tuple[float, float]
    uncertainty_flag: bool
    prediction_set: list[str] = field(default_factory=list)

    def summary(self, labels: list[str]) -> dict:
        top_idx = int(np.argmax(self.mean_probs))
        return {
            "top_prediction": labels[top_idx] if top_idx < len(labels) else "Unknown",
            "mean_confidence": round(float(self.mean_probs[top_idx]), 4),
            "std_confidence": round(float(self.std_probs[top_idx]), 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 3),
                round(self.confidence_interval[1], 3),
            ),
            "uncertainty_flag": self.uncertainty_flag,
            "prediction_set_95pct": self.prediction_set,
            "method": "heuristic-fallback" if self.uncertainty_flag else "lightweight-estimate",
        }


@dataclass
class PredictionResultV2:
    findings: list[dict]
    urgency_score: float
    heatmap: np.ndarray | None
    clinical_report: str
    inference_time_ms: float
    clinical_entities: dict = field(default_factory=dict)
    raw_probs: list[float] = field(default_factory=list)
    uncertainty: LightweightUncertainty | object | None = None
    model_mode: str = "fallback"
    workflow: dict = field(default_factory=dict)
    LABELS: list[str] = field(default_factory=lambda: LABELS.copy())

    @property
    def top_finding(self) -> str:
        return self.findings[0]["label"] if self.findings else "No significant findings"

    @property
    def urgency_label(self) -> str:
        if self.urgency_score >= 0.75:
            return "HIGH"
        if self.urgency_score >= 0.45:
            return "MODERATE"
        return "LOW"


class ClinicalAIPipelineV2:
    """
    Stable API-facing pipeline.

    It attempts to use the real multimodal model. If checkpoints/dependencies
    are unavailable, it uses a deterministic clinical fallback so API demos,
    safety tests, and UI workflows remain end-to-end.
    """

    LABELS = LABELS

    def __init__(self, config: dict, device: str | None = None, allow_fallback: bool = True):
        self.config = config
        self.device = device
        self.allow_fallback = allow_fallback
        self._real_pipeline: ClinicalAIPipeline | None = None
        self._load_error: str | None = None

        if not self._should_load_real_pipeline(config):
            self._load_error = "Model checkpoints not configured or unavailable; using fallback mode."
            return

        try:
            from src.pipeline.inference import ClinicalAIPipeline

            self._real_pipeline = ClinicalAIPipeline(config, device=device)
        except Exception as exc:  # pragma: no cover - depends on local model artifacts
            self._load_error = str(exc)
            if not allow_fallback:
                raise

    def _should_load_real_pipeline(self, config: dict) -> bool:
        model_cfg = config.get("models", {}) if isinstance(config, dict) else {}
        paths = [
            model_cfg.get("image", {}).get("checkpoint_path"),
            model_cfg.get("text", {}).get("checkpoint_path"),
            model_cfg.get("fusion", {}).get("checkpoint_path"),
        ]
        return bool(paths) and all(path and Path(path).exists() for path in paths)

    def predict(self, image: "Image.Image", clinical_note: str) -> PredictionResultV2:
        if self._real_pipeline is not None:
            return self._from_real_result(self._real_pipeline.predict(image, clinical_note))
        return self._fallback_predict(image, clinical_note)

    def _from_real_result(self, result: "PredictionResult") -> PredictionResultV2:
        uncertainty = getattr(result, "uncertainty", None)
        workflow = compute_workflow_summary(
            note=result.clinical_report,
            findings=result.findings,
            urgency_score=result.urgency_score,
            uncertainty_flag=bool(getattr(uncertainty, "uncertainty_flag", False)),
        )
        return PredictionResultV2(
            findings=result.findings,
            urgency_score=result.urgency_score,
            heatmap=result.heatmap,
            clinical_report=result.clinical_report,
            inference_time_ms=result.inference_time_ms,
            clinical_entities=result.clinical_entities,
            raw_probs=result.raw_probs,
            uncertainty=uncertainty,
            model_mode="model",
            workflow=workflow.__dict__,
        )

    def _fallback_predict(self, image: "Image.Image", clinical_note: str) -> PredictionResultV2:
        t0 = time.perf_counter()
        note_l = clinical_note.lower()
        seed = int(hashlib.sha256(clinical_note.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        scores = {label: float(rng.uniform(0.02, 0.12)) for label in LABELS}
        scores["No Finding"] = 0.28

        rules = [
            (("fever", "cough", "sputum", "pneumonia"), {"Pneumonia": 0.82, "Consolidation": 0.66, "Lung Opacity": 0.72}),
            (("dyspnea", "orthopnea", "bnp", "edema"), {"Edema": 0.78, "Pleural Effusion": 0.68, "Cardiomegaly": 0.58}),
            (("effusion", "pleural"), {"Pleural Effusion": 0.86, "Atelectasis": 0.48}),
            (("pneumothorax", "trauma", "sudden chest pain"), {"Pneumothorax": 0.84, "Lung Opacity": 0.45}),
            (("routine", "pre-op", "normal", "no respiratory"), {"No Finding": 0.91}),
        ]
        for terms, updates in rules:
            if any(term in note_l for term in terms):
                for label, score in updates.items():
                    scores[label] = max(scores[label], min(0.98, score + float(rng.uniform(-0.03, 0.03))))

        if scores["No Finding"] > 0.8:
            for label in LABELS:
                if label != "No Finding":
                    scores[label] = min(scores[label], float(rng.uniform(0.01, 0.16)))

        probs = np.array([scores[label] for label in LABELS], dtype=float)
        findings = [
            {"label": label, "prob": round(float(prob), 3), "urgent": bool(prob >= 0.75), "class_idx": idx}
            for idx, (label, prob) in enumerate(zip(LABELS, probs))
            if prob >= 0.40
        ]
        findings.sort(key=lambda item: item["prob"], reverse=True)
        if not findings:
            findings = [{"label": "No Finding", "prob": round(float(probs[0]), 3), "urgent": False, "class_idx": 0}]

        urgency = self._compute_urgency(findings, note_l)
        uncertainty = self._estimate_uncertainty(probs, findings)
        entities = self._extract_entities(clinical_note)
        workflow = compute_workflow_summary(
            note=clinical_note,
            findings=findings,
            urgency_score=urgency,
            uncertainty_flag=uncertainty.uncertainty_flag,
        )
        report = self._build_report(clinical_note, findings, urgency, workflow)

        return PredictionResultV2(
            findings=findings,
            urgency_score=urgency,
            heatmap=None,
            clinical_report=report,
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 1),
            clinical_entities=entities,
            raw_probs=probs.tolist(),
            uncertainty=uncertainty,
            model_mode="fallback",
            workflow=workflow.__dict__,
        )

    def _compute_urgency(self, findings: list[dict], note_l: str) -> float:
        top = findings[0] if findings else {"label": "No Finding", "prob": 0.2}
        urgency = float(top["prob"]) * (0.95 if top["label"] != "No Finding" else 0.25)
        if any(term in note_l for term in ("o2 sat 8", "spo2 8", "hypoxia", "respiratory distress")):
            urgency += 0.18
        if top["label"] in {"Pneumothorax", "Edema", "Pneumonia", "Consolidation"}:
            urgency += 0.08
        return round(max(0.0, min(1.0, urgency)), 3)

    def _estimate_uncertainty(self, probs: np.ndarray, findings: list[dict]) -> LightweightUncertainty:
        top_prob = float(findings[0]["prob"]) if findings else float(np.max(probs))
        margin = 0.12
        if len(findings) > 1:
            margin = max(0.02, top_prob - float(findings[1]["prob"]))
        uncertainty_flag = top_prob < 0.50 or margin < 0.08
        std = np.full_like(probs, 0.04 if not uncertainty_flag else 0.11)
        ci = (max(0.0, top_prob - 1.96 * std.max()), min(1.0, top_prob + 1.96 * std.max()))
        pred_set = [f["label"] for f in findings[:3]] or ["No Finding"]
        return LightweightUncertainty(probs, std, ci, uncertainty_flag, pred_set)

    def _extract_entities(self, note: str) -> dict:
        tokens = note.replace(",", " ").replace(".", " ").split()
        age = None
        for token in tokens:
            cleaned = token.lower().replace("-year-old", "").replace("yo", "")
            if cleaned.isdigit() and 0 < int(cleaned) < 120:
                age = int(cleaned)
                break
        note_l = note.lower()
        symptoms = [
            term
            for term in ("fever", "cough", "dyspnea", "chest pain", "orthopnea", "hypoxia")
            if term in note_l
        ]
        gender = "Male" if "male" in note_l else ("Female" if "female" in note_l else None)
        return {"age": age, "gender": gender, "symptoms": symptoms}

    def _build_report(
        self,
        note: str,
        findings: list[dict],
        urgency: float,
        workflow,
    ) -> str:
        findings_text = "\n".join(
            f"- {f['label']}: {f['prob']:.0%} estimated probability" for f in findings[:5]
        )
        recommendation = "Immediate physician/radiologist review recommended." if urgency >= 0.75 else "Correlate with clinical findings and prior imaging."
        return f"""CLINICAL INDICATION
{note[:260]}

AI FINDINGS
{findings_text}

IMPRESSION
AI triage priority: {workflow.priority}. {workflow.priority_reason}

RECOMMENDATION
{recommendation}

SAFETY NOTE
This is a deterministic fallback analysis because production model artifacts were not available. It is suitable for workflow demonstration and API testing only, not diagnosis. All findings must be reviewed by a licensed clinician."""
