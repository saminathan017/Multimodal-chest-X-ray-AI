"""
src/models/bart_text_encoder.py
─────────────────────────────────────────────────────────────────────
BioBART encoder-only clinical notes encoder.

Why BART over ClinicalBERT?
  • Denoising pre-training (text-infilling, sentence permutation) →
    richer contextual representations, especially for noisy clinical notes
  • 1024-token context window  (vs 512 for BERT) →
    handles long admission notes without truncation loss
  • BioBART pre-trained on PubMed + PMC full-text →
    strong biomedical vocabulary coverage
  • Encoder co-trained with a generation decoder →
    features capture both semantic understanding AND generative context

Architecture:
  BioBART encoder (768-d) → masked mean-pool → Linear → LayerNorm → GELU → 512-d

Only the encoder is loaded; the decoder is discarded immediately after
weight extraction, saving ~50% memory at inference time.

Default model : GanjinZero/biobart-v2-base  (PubMed + PMC, BART-base)
Fallback model: facebook/bart-base          (general English BART-base)
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BartModel
from loguru import logger


BIOBART_MODEL  = "GanjinZero/biobart-v2-base"
FALLBACK_MODEL = "facebook/bart-base"


class BartClinicalEncoder(nn.Module):
    """
    Encodes free-text clinical notes into a dense vector using the
    encoder half of BioBART, projected to match the image feature dim.

    Drop-in replacement for TextEncoder — identical forward signature
    and from_pretrained() interface.

    Usage:
        encoder = BartClinicalEncoder.from_pretrained(device="cuda")
        features = encoder(notes=["68yo male, O2 sat 88%, productive cough..."])
        # → (1, 512) tensor
    """

    def __init__(
        self,
        model_name: str = BIOBART_MODEL,
        output_dim: int = 512,
        max_length: int = 1024,
        freeze_backbone: bool = True,
        num_unfrozen_layers: int = 0,
    ):
        """
        Args:
            model_name:          HuggingFace model ID
            output_dim:          Projection output size — must match image feat dim (512)
            max_length:          Tokenizer truncation limit (BART supports up to 1024)
            freeze_backbone:     Freeze all encoder weights (recommended for fine-tuning fusion only)
            num_unfrozen_layers: If > 0, unfreeze the last N transformer layers for end-to-end fine-tuning
        """
        super().__init__()
        self.max_length = max_length

        self.encoder, self.tokenizer, self._encoder_dim, self._available = (
            self._load_encoder(model_name)
        )

        if freeze_backbone and self.encoder is not None:
            self._freeze(num_unfrozen_layers)

        # Projection head: encoder_dim → output_dim
        self.projection = nn.Sequential(
            nn.Linear(self._encoder_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _load_encoder(
        self, model_name: str
    ) -> tuple:
        """Try BioBART, fall back to facebook/bart-base, then mock mode."""
        for name in (model_name, FALLBACK_MODEL):
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                bart      = BartModel.from_pretrained(name)
                encoder   = bart.encoder         # keep encoder only
                dim       = encoder.config.d_model
                del bart                          # free decoder weights
                logger.info(
                    f"[BartClinicalEncoder] Loaded '{name}' "
                    f"(d_model={dim}, max_pos={encoder.config.max_position_embeddings})"
                )
                return encoder, tokenizer, dim, True
            except Exception as exc:
                logger.warning(f"[BartClinicalEncoder] Could not load '{name}': {exc}")

        logger.warning("[BartClinicalEncoder] All models failed — running in mock mode.")
        return None, None, 768, False

    def _freeze(self, num_unfrozen_layers: int) -> None:
        """Freeze backbone; optionally keep the last N transformer layers trainable."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        if num_unfrozen_layers > 0 and hasattr(self.encoder, "layers"):
            for layer in self.encoder.layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(
                f"[BartClinicalEncoder] Backbone frozen "
                f"(last {num_unfrozen_layers} encoder layers trainable)"
            )
        else:
            logger.info("[BartClinicalEncoder] Backbone fully frozen")

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,   # (B, T, D)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:                 # (B, D)
        """Masked mean-pool over the token dimension."""
        mask        = attention_mask.unsqueeze(-1).float()    # (B, T, 1)
        sum_hidden  = (hidden_states * mask).sum(dim=1)       # (B, D)
        token_count = mask.sum(dim=1).clamp(min=1e-9)        # (B, 1)
        return sum_hidden / token_count                        # (B, D)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, notes: list[str], device: str = "cpu") -> torch.Tensor:
        """
        Encode a batch of clinical notes.

        Args:
            notes:  list[str] — raw clinical note strings
            device: target torch device string

        Returns:
            Tensor of shape (B, output_dim) — projected encoder features
        """
        if self._available and self.tokenizer is not None:
            encoding = self.tokenizer(
                notes,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            ctx = torch.no_grad() if not self.training else torch.enable_grad()
            with ctx:
                encoder_out = self.encoder(
                    input_ids      = encoding["input_ids"],
                    attention_mask = encoding["attention_mask"],
                )

            # encoder_out.last_hidden_state: (B, T, d_model)
            pooled = self._mean_pool(
                encoder_out.last_hidden_state,
                encoding["attention_mask"],
            )                                                  # (B, d_model)
        else:
            # Mock mode — random features of correct shape
            pooled = torch.randn(len(notes), self._encoder_dim, device=device)

        return self.projection(pooled.to(device))             # (B, output_dim)

    # ── Constructors ─────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        output_dim: int = 512,
        max_length: int = 1024,
        device: str = "cuda",
        freeze_backbone: bool = True,
        num_unfrozen_layers: int = 0,
    ) -> "BartClinicalEncoder":
        """
        Build a BartClinicalEncoder, optionally loading fine-tuned
        projection weights from a checkpoint.

        Args:
            checkpoint_path:     Path to a saved state_dict (.pt / .pth).
                                 Only the projection head is expected if the
                                 backbone is frozen.
            output_dim:          Projection output dim (must match fusion model).
            max_length:          Tokenizer max length.
            device:              "cuda" or "cpu" (auto-falls back if no GPU).
            freeze_backbone:     Freeze the BioBART encoder weights.
            num_unfrozen_layers: Number of trailing encoder layers to keep trainable.
        """
        device = device if torch.cuda.is_available() else "cpu"

        model = cls(
            output_dim          = output_dim,
            max_length          = max_length,
            freeze_backbone     = freeze_backbone,
            num_unfrozen_layers = num_unfrozen_layers,
        ).to(device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info(f"[BartClinicalEncoder] Loaded weights from {checkpoint_path}")

        model.eval()
        return model

    # ── Utility ───────────────────────────────────────────────────────

    def extract_clinical_entities(self, note: str) -> dict:
        """
        Lightweight regex extraction of structured fields from raw notes.
        Returns age, gender, and symptom keywords detected in the text.

        Same interface as TextEncoder.extract_clinical_entities().
        """
        entities: dict = {"age": None, "gender": None, "symptoms": []}

        age_match = re.search(
            r"(\d{1,3})\s*(?:year[s]?[-\s]?old|yo|y\.o\.)", note, re.I
        )
        if age_match:
            entities["age"] = int(age_match.group(1))

        if re.search(r"\b(male|man|boy|he|his)\b", note, re.I):
            entities["gender"] = "Male"
        elif re.search(r"\b(female|woman|girl|she|her)\b", note, re.I):
            entities["gender"] = "Female"

        symptom_keywords = [
            "cough", "fever", "dyspnea", "shortness of breath", "chest pain",
            "fatigue", "hemoptysis", "wheezing", "tachycardia", "hypoxia",
            "hypoxemia", "pleuritic pain", "night sweats", "weight loss",
        ]
        entities["symptoms"] = [
            kw for kw in symptom_keywords if kw.lower() in note.lower()
        ]

        return entities

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "live" if self._available else "mock"
        return (
            f"BartClinicalEncoder("
            f"d_model={self._encoder_dim}, "
            f"max_length={self.max_length}, "
            f"status={status})"
        )
