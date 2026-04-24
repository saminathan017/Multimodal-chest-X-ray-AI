"""
Foundation model feature extractors.

These wrappers expose a small, stable API around third-party foundation models
so the rest of ClinicalAI can use embeddings without caring about Hugging Face,
OpenCLIP, or model-specific tokenizer details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


BIOMEDCLIP_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CXR_BERT_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"
GOOGLE_CXR_FOUNDATION_ID = "google/cxr-foundation"


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return array / norm


@dataclass
class EmbeddingResult:
    model_id: str
    modality: str
    embedding: np.ndarray
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "modality": self.modality,
            "shape": list(self.embedding.shape),
            "mean": round(float(np.mean(self.embedding)), 6),
            "std": round(float(np.std(self.embedding)), 6),
            "metadata": self.metadata,
        }


class BiomedCLIPExtractor:
    def __init__(self, model_id: str = BIOMEDCLIP_ID, device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.open_clip = open_clip
        self.model_id = model_id
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:" + model_id,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer("hf-hub:" + model_id)
        self.model.eval()

    def encode_image(self, image: Image.Image) -> EmbeddingResult:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_image(tensor).detach().cpu().float().numpy()
        features = _l2_normalize(features)
        return EmbeddingResult(
            model_id=self.model_id,
            modality="image",
            embedding=features,
            metadata={"feature_dim": int(features.shape[-1]), "normalized": True},
        )

    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        tokens = self.tokenizer(texts).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_text(tokens).detach().cpu().float().numpy()
        features = _l2_normalize(features)
        return EmbeddingResult(
            model_id=self.model_id,
            modality="text",
            embedding=features,
            metadata={"feature_dim": int(features.shape[-1]), "normalized": True, "count": len(texts)},
        )


class CXRBertExtractor:
    def __init__(self, model_id: str = CXR_BERT_ID, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> EmbeddingResult:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            output = self.model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        features = pooled.detach().cpu().float().numpy()
        features = _l2_normalize(features)
        return EmbeddingResult(
            model_id=self.model_id,
            modality="text",
            embedding=features,
            metadata={"feature_dim": int(features.shape[-1]), "normalized": True, "count": len(texts)},
        )


class GoogleCXRFoundationExtractor:
    def __init__(self, model_path: str):
        try:
            import tensorflow as tf
        except Exception as exc:
            raise RuntimeError(
                "google/cxr-foundation uses TensorFlow SavedModel files. "
                "Install TensorFlow in a compatible Python environment and pass the downloaded model path."
            ) from exc
        self.tf = tf
        self.model_path = model_path
        self.model = tf.saved_model.load(model_path)

    def signatures(self) -> list[str]:
        return list(self.model.signatures.keys())

