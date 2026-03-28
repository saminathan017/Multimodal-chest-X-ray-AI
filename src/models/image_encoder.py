"""
src/models/image_encoder.py
─────────────────────────────────────────────────────────────────────
BiomedCLIP-based chest X-ray encoder with:
  - Feature extraction (512-d embeddings)
  - 14-class CheXpert pathology classification head
  - GradCAM explainability hook
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from loguru import logger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip not available – image encoder will run in mock mode")


# ── Classification Head ──────────────────────────────────────────────
class PathologyClassifier(nn.Module):
    """Lightweight MLP head on top of frozen BiomedCLIP image features."""

    def __init__(self, in_features: int = 512, num_classes: int = 14, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Full Image Encoder ───────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """
    Wraps BiomedCLIP vision tower + classification head.

    Usage:
        encoder = ImageEncoder.from_pretrained(device="cuda")
        features, logits = encoder(image_tensor)
    """

    BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes

        if OPEN_CLIP_AVAILABLE:
            self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
                self.BIOMEDCLIP_MODEL
            )
            # Detect output dim robustly across open_clip versions
            try:
                feat_dim = self.backbone.visual.output_dim
            except AttributeError:
                try:
                    feat_dim = self.backbone.visual.head.in_features
                except AttributeError:
                    # Run a dummy forward pass to get the actual dim
                    with torch.no_grad():
                        _dummy = torch.zeros(1, 3, 224, 224)
                        feat_dim = self.backbone.encode_image(_dummy).shape[-1]
        else:
            # Mock backbone for CI / environments without GPU
            self.backbone = None
            self.preprocess = None
            feat_dim = 512

        if freeze_backbone and self.backbone is not None:
            for param in self.backbone.visual.parameters():
                param.requires_grad = False
            logger.info("BiomedCLIP backbone frozen — only classification head will train")

        self.classifier = PathologyClassifier(feat_dim, num_classes)
        self._feat_dim = feat_dim

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 3, 224, 224) preprocessed tensor
        Returns:
            features: (B, 512)
            logits:   (B, 14)
        """
        if self.backbone is not None:
            features = self.backbone.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)  # L2-norm
        else:
            features = torch.randn(images.shape[0], self._feat_dim, device=images.device)

        logits = self.classifier(features)
        return features, logits

    # ── Class factory ────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        num_classes: int = 14,
        device: str = "cuda",
        freeze_backbone: bool = True,
    ) -> "ImageEncoder":
        device = device if torch.cuda.is_available() else "cpu"
        model = cls(num_classes=num_classes, freeze_backbone=freeze_backbone)
        model = model.to(device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            logger.info(f"Loaded image encoder weights from {checkpoint_path}")

        model.eval()
        return model

    # ── Preprocessing helper ─────────────────────────────────────────
    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert a PIL image → preprocessed tensor (1, 3, 224, 224)."""
        if self.preprocess is not None:
            return self.preprocess(pil_image).unsqueeze(0)

        # Fallback: manual resize + normalize
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(pil_image).unsqueeze(0)


# ── GradCAM Explainability ───────────────────────────────────────────
class XRayExplainer:
    """
    Wraps pytorch-grad-cam to produce heatmaps for a given
    predicted class on the BiomedCLIP ViT backbone.
    """

    def __init__(self, model: ImageEncoder, device: str = "cpu"):
        self.model = model
        self.device = device

        # Target the last transformer block's norm layer inside the ViT
        if model.backbone is not None and hasattr(model.backbone, "visual"):
            try:
                target_layers = [model.backbone.visual.trunk.blocks[-1].norm1]
                self.cam = GradCAM(
                    model=self._wrapped_model(),
                    target_layers=target_layers,
                    use_cuda=(device == "cuda"),
                )
                self._available = True
            except Exception as e:
                logger.warning(f"GradCAM init failed: {e}. Heatmaps will be skipped.")
                self._available = False
        else:
            self._available = False

    def _wrapped_model(self):
        """GradCAM needs a model that returns logits directly."""
        class _Wrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, x):
                _, logits = self.encoder(x)
                return logits

        return _Wrapper(self.model)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        raw_image: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        Args:
            image_tensor: (1, 3, 224, 224)
            raw_image:    (224, 224, 3) float32 in [0, 1] — original for overlay
            target_class: class index to explain
        Returns:
            overlay: (224, 224, 3) uint8 heatmap overlaid on original image
        """
        if not self._available:
            return (raw_image * 255).astype(np.uint8)

        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        overlay = show_cam_on_image(raw_image, grayscale_cam[0], use_rgb=True)
        return overlay
