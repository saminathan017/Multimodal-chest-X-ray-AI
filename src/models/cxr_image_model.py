"""
Chest X-ray image backbones for high-signal CheXpert training.

The first cloud baseline used the multimodal stack with synthetic notes. For
CheXpert-only supervision, an image-first model is a stronger and cleaner
baseline: every trainable parameter receives real signal from the X-ray labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CXRImageClassifier(nn.Module):
    """Torchvision ImageNet backbone adapted for multilabel CXR prediction."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(in_features, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_small(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(in_features, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Use convnext_tiny, convnext_small, efficientnet_b3, or resnet50."
            )

        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        is_head = any(part in name for part in ("classifier", "fc"))
        param.requires_grad = trainable or is_head
