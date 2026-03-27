"""
src/models/text_encoder.py
─────────────────────────────────────────────────────────────────────
Bio_ClinicalBERT-based clinical notes encoder.
Extracts a 768-d [CLS] embedding from free-text patient notes.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from loguru import logger


CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class TextEncoder(nn.Module):
    """
    Encodes free-text clinical notes into a dense 768-d vector using
    Bio_ClinicalBERT, with a projection head to match image feature dim.

    Usage:
        encoder = TextEncoder.from_pretrained(device="cuda")
        features = encoder(notes=["68yo male, smoker, shortness of breath..."])
    """

    def __init__(
        self,
        model_name: str = CLINICAL_BERT_MODEL,
        output_dim: int = 512,
        max_length: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.max_length = max_length

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            self._bert_dim = self.bert.config.hidden_size  # 768
            self._available = True
            logger.info(f"Loaded {model_name} (hidden_size={self._bert_dim})")
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}. Running in mock mode.")
            self.tokenizer = None
            self.bert = None
            self._bert_dim = 768
            self._available = False

        if freeze_backbone and self.bert is not None:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("ClinicalBERT backbone frozen")

        # Projection: 768 → output_dim (matches image feature dim)
        self.projection = nn.Sequential(
            nn.Linear(self._bert_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, notes: list[str], device: str = "cpu") -> torch.Tensor:
        """
        Args:
            notes:  list of clinical note strings (batch)
            device: target device
        Returns:
            features: (B, output_dim) projected embeddings
        """
        if self._available and self.tokenizer is not None:
            encoding = self.tokenizer(
                notes,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad() if not self.training else torch.enable_grad():
                output = self.bert(**encoding)

            # Mean-pool over token dimension (more stable than [CLS] alone)
            attention_mask = encoding["attention_mask"].unsqueeze(-1).float()
            token_embeddings = output.last_hidden_state  # (B, T, 768)
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1e-9)
            cls_embedding = sum_embeddings / count              # (B, 768)
        else:
            # Mock mode
            batch_size = len(notes)
            cls_embedding = torch.randn(batch_size, self._bert_dim, device=device)

        features = self.projection(cls_embedding.to(device))
        return features                                          # (B, 512)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        output_dim: int = 512,
        max_length: int = 512,
        device: str = "cuda",
        freeze_backbone: bool = True,
    ) -> "TextEncoder":
        device = device if torch.cuda.is_available() else "cpu"
        model = cls(
            output_dim=output_dim,
            max_length=max_length,
            freeze_backbone=freeze_backbone,
        )
        model = model.to(device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            logger.info(f"Loaded text encoder weights from {checkpoint_path}")

        model.eval()
        return model

    # ── Utility ──────────────────────────────────────────────────────
    def extract_clinical_entities(self, note: str) -> dict:
        """
        Simple regex-based extraction of age, gender, symptoms from raw notes.
        Used to populate the structured metadata panel in the UI.
        """
        import re

        entities: dict = {"age": None, "gender": None, "symptoms": []}

        # Age
        age_match = re.search(r"(\d{1,3})\s*(?:year[s]?[-\s]?old|yo|y\.o\.)", note, re.I)
        if age_match:
            entities["age"] = int(age_match.group(1))

        # Gender
        if re.search(r"\b(male|man|boy|he|his)\b", note, re.I):
            entities["gender"] = "Male"
        elif re.search(r"\b(female|woman|girl|she|her)\b", note, re.I):
            entities["gender"] = "Female"

        # Common symptom keywords
        symptom_keywords = [
            "cough", "fever", "dyspnea", "shortness of breath", "chest pain",
            "fatigue", "hemoptysis", "wheezing", "tachycardia", "hypoxia",
            "hypoxemia", "pleuritic pain", "night sweats", "weight loss",
        ]
        entities["symptoms"] = [kw for kw in symptom_keywords if kw.lower() in note.lower()]

        return entities
