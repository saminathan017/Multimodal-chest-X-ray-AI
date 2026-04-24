"""
Pipeline package exports.

Imports are intentionally lazy so lightweight modules such as inference_v2 can
be used in CI/demo environments that do not have all ML or AWS dependencies.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"ClinicalAIPipeline", "PredictionResult"}:
        from .inference import ClinicalAIPipeline, PredictionResult

        return {"ClinicalAIPipeline": ClinicalAIPipeline, "PredictionResult": PredictionResult}[name]
    if name == "ClinicalAIPipelineV2":
        from .inference_v2 import ClinicalAIPipelineV2

        return ClinicalAIPipelineV2
    raise AttributeError(name)


__all__ = ["ClinicalAIPipeline", "PredictionResult", "ClinicalAIPipelineV2"]
