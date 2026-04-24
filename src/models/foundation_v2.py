"""
Advanced multimodal architecture blueprint.

This file defines configuration objects for the next-generation radiology
foundation model: image patch tokens + report/note tokens + cross-attention
fusion + multi-task heads. The heavy training implementation can plug into this
contract without changing evaluation/governance code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultiTaskHeadSpec:
    name: str
    task_type: str
    output_dim: int
    loss: str
    clinical_purpose: str


@dataclass(frozen=True)
class FoundationModelV2Spec:
    vision_encoder: str = "swin_or_eva_medical_clip"
    text_encoder: str = "long_context_clinical_encoder"
    fusion: str = "patch_token_cross_attention"
    pretraining: list[str] = field(default_factory=lambda: ["image_report_contrastive", "masked_report_modeling"])
    heads: list[MultiTaskHeadSpec] = field(
        default_factory=lambda: [
            MultiTaskHeadSpec("pathology", "multilabel_classification", 14, "bce", "CheXpert pathology prediction"),
            MultiTaskHeadSpec("urgency", "regression", 1, "mse_or_bce", "Clinical triage"),
            MultiTaskHeadSpec("image_quality", "classification", 3, "cross_entropy", "Input quality gating"),
            MultiTaskHeadSpec("view_position", "classification", 4, "cross_entropy", "PA/AP/lateral/portable detection"),
            MultiTaskHeadSpec("support_devices", "multilabel_classification", 6, "bce", "Tube and line detection"),
            MultiTaskHeadSpec("uncertainty", "regression", 1, "evidential_or_variance", "Abstention and human review"),
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["design_note"] = (
            "Use contrastive image-report pretraining before supervised fine-tuning. "
            "Fuse image patch tokens with report/note tokens, not pooled vectors only."
        )
        return data

