"""
src/models/fusion_model.py
─────────────────────────────────────────────────────────────────────
Cross-modal fusion network.

Takes image features (512-d) + text features (512-d) and produces:
  - Fused pathology logits (14 classes)
  - Urgency score (0-1)
  - Confidence per class
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Scaled dot-product cross-attention: image queries attend to text keys/values.
    Helps the model learn which clinical note context is relevant to each
    visual region.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        img_feat: torch.Tensor,   # (B, 512)
        txt_feat: torch.Tensor,   # (B, 512)
    ) -> torch.Tensor:
        # Add sequence dimension for MultiheadAttention
        q  = self.norm_q(img_feat).unsqueeze(1)   # (B, 1, 512)
        kv = self.norm_kv(txt_feat).unsqueeze(1)  # (B, 1, 512)
        fused, _ = self.attn(q, kv, kv)
        return fused.squeeze(1)                   # (B, 512)


class FusionModel(nn.Module):
    """
    Multimodal fusion with cross-attention + MLP classification head.

    Architecture:
      img_feat (512) ─┐
                       ├─► CrossModalAttention ─► concat(img, txt, attn) ─► MLP ─► logits (14)
      txt_feat (512) ─┘                                                          └─► urgency (1)
    """

    def __init__(
        self,
        feat_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 14,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cross_attn = CrossModalAttention(feat_dim)

        # Input: concat of img_feat + txt_feat + cross_attn_out = 512*3 = 1536
        fused_dim = feat_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            {
                "logits":   (B, 14)  — raw logits for BCEWithLogitsLoss
                "probs":    (B, 14)  — sigmoid probabilities
                "urgency":  (B, 1)   — overall urgency score
                "fused":    (B, 1536) — fused representation (for downstream)
            }
        """
        attn_out = self.cross_attn(img_feat, txt_feat)
        fused = torch.cat([img_feat, txt_feat, attn_out], dim=-1)  # (B, 1536)

        logits  = self.classifier(fused)
        probs   = torch.sigmoid(logits)
        urgency = self.urgency_head(fused)

        return {"logits": logits, "probs": probs, "urgency": urgency, "fused": fused}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> "FusionModel":
        device = device if torch.cuda.is_available() else "cpu"
        model = cls(**kwargs).to(device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)

        model.eval()
        return model
