"""
src/pipeline/finetune_text_encoder.py
─────────────────────────────────────────────────────────────────────
Fine-tune the BioBART (or ClinicalBERT) text encoder on MIMIC-CXR
radiology reports using a contrastive image-text alignment objective
(CLIP-style) + optional MLM denoising loss.

Why this matters:
  BioBART was pre-trained on PubMed abstracts — formal academic text.
  MIMIC-CXR reports are terse, abbreviated, and domain-specific:
    "Bibasilar opacities. No PTX. Mild cardiomegaly. Stable."
  Fine-tuning adapts the encoder's representations to this style,
  improving the image-text fusion downstream.

Training objective:
  1. Contrastive loss  — pull image + matching report embeddings
                         together, push non-matching pairs apart.
  2. (Optional) MLM    — standard masked language modelling on reports
                         to keep the encoder from catastrophic forgetting.

Run:
    python src/pipeline/finetune_text_encoder.py \
        --mimic_dir  /data/mimic-cxr \
        --output_dir models/ \
        --encoder    bart \
        --epochs     5 \
        --batch_size 32 \
        --unfreeze_layers 2
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from src.models import get_text_encoder
from src.models.image_encoder import ImageEncoder
from src.utils.mimic_cxr_dataset import MimicCXRDataset


# ── Contrastive loss ─────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss.

    Pulls paired (image, text) embeddings close in the shared 512-d
    space, pushes all other pairs in the batch apart.

    Args:
        temperature: Softmax temperature — lower = sharper distribution.
                     Start at 0.07 (CLIP default).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        img_feat: torch.Tensor,   # (B, D) L2-normalised
        txt_feat: torch.Tensor,   # (B, D) L2-normalised
    ) -> torch.Tensor:
        logits   = (img_feat @ txt_feat.T) / self.temperature.clamp(min=0.01)
        targets  = torch.arange(len(logits), device=logits.device)
        loss_i2t = F.cross_entropy(logits,   targets)   # image → text
        loss_t2i = F.cross_entropy(logits.T, targets)   # text → image
        return (loss_i2t + loss_t2i) / 2


# ── Metrics ──────────────────────────────────────────────────────────

def recall_at_k(
    img_feats: np.ndarray,
    txt_feats: np.ndarray,
    k: int = 10,
) -> float:
    """
    Image-to-text Recall@K:
    For each image, check if its matching report is in the top-K
    most similar texts. Ground truth: diagonal (i-th image matches i-th report).

    Args:
        img_feats: (N, D) normalised image embeddings
        txt_feats: (N, D) normalised text embeddings
        k:         top-K to consider

    Returns:
        Recall@K as a float in [0, 1]
    """
    sims    = img_feats @ txt_feats.T                   # (N, N)
    top_k   = np.argsort(sims, axis=1)[:, -k:]          # (N, K)
    correct = sum(i in top_k[i] for i in range(len(img_feats)))
    return correct / len(img_feats)


# ── Trainer ──────────────────────────────────────────────────────────

class TextEncoderFinetuner:
    """
    Fine-tunes the text encoder projection + (optionally) the last N
    transformer layers using contrastive image-text alignment on MIMIC-CXR.

    The image encoder (BiomedCLIP) is kept fully frozen throughout —
    only the text encoder learns to align its space with the image space
    that was already established during CheXpert training.
    """

    def __init__(self, args: argparse.Namespace):
        self.args   = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Fine-tuning on {self.device}")

        # Image encoder — frozen (already trained on CheXpert)
        self.image_encoder = ImageEncoder.from_pretrained(
            checkpoint_path = args.image_checkpoint or None,
            num_classes     = 14,
            device          = self.device,
            freeze_backbone = True,
        )
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        # Text encoder — projection + last N layers trainable
        self.text_encoder = get_text_encoder(
            encoder_type        = args.encoder,
            output_dim          = 512,
            max_length          = 1024 if args.encoder == "bart" else 512,
            device              = self.device,
            freeze_backbone     = True,
            num_unfrozen_layers = args.unfreeze_layers,
        )

        # Loss
        self.contrastive = InfoNCELoss(temperature=0.07).to(self.device)

        # Optimizer: text encoder trainable params + temperature
        trainable = (
            list(self.text_encoder.projection.parameters()) +
            list(self.contrastive.parameters())
        )
        if args.unfreeze_layers > 0 and hasattr(self.text_encoder, "encoder"):
            trainable += [
                p for p in self.text_encoder.encoder.parameters()
                if p.requires_grad
            ]

        self.optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=1e-4
        )
        self.history: list[dict] = []

    # ── Train one epoch ──────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.text_encoder.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            notes  = batch["note"]                      # list[str]

            self.optimizer.zero_grad()

            # Image features (frozen BiomedCLIP)
            with torch.no_grad():
                img_feat, _ = self.image_encoder(images)
            img_feat = F.normalize(img_feat, dim=-1)    # (B, 512)

            # Text features (fine-tuning)
            txt_feat = self.text_encoder(notes, device=self.device)
            txt_feat = F.normalize(txt_feat, dim=-1)    # (B, 512)

            loss = self.contrastive(img_feat, txt_feat)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.text_encoder.projection.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Temp: {self.contrastive.temperature.item():.4f}"
                )

        return total_loss / len(loader)

    # ── Evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """
        Compute image-to-text retrieval metrics on the validation set.

        Metrics:
          - Recall@1  : matching report is the #1 result
          - Recall@5  : matching report is in top 5
          - Recall@10 : matching report is in top 10
          - Mean similarity: average cosine sim of matching pairs
        """
        self.text_encoder.eval()

        all_img, all_txt = [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            notes  = batch["note"]

            img_feat, _ = self.image_encoder(images)
            txt_feat    = self.text_encoder(notes, device=self.device)

            all_img.append(F.normalize(img_feat, dim=-1).cpu().numpy())
            all_txt.append(F.normalize(txt_feat, dim=-1).cpu().numpy())

        img_feats = np.concatenate(all_img)   # (N, 512)
        txt_feats = np.concatenate(all_txt)   # (N, 512)

        # Diagonal cosine similarity (matching pairs)
        mean_sim = float(np.mean(np.sum(img_feats * txt_feats, axis=1)))

        return {
            "recall@1":       recall_at_k(img_feats, txt_feats, k=1),
            "recall@5":       recall_at_k(img_feats, txt_feats, k=5),
            "recall@10":      recall_at_k(img_feats, txt_feats, k=10),
            "mean_similarity": mean_sim,
        }

    # ── Save ─────────────────────────────────────────────────────────

    def save(self, output_dir: str, epoch: int, metrics: dict) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Text encoder state dict (projection + unfrozen backbone layers)
        torch.save(
            self.text_encoder.state_dict(),
            out / "text_encoder_finetuned.pt",
        )

        # Training history
        self.history.append({"epoch": epoch, **metrics})
        with open(out / "text_encoder_finetune_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(
            f"Saved text encoder → {out}/text_encoder_finetuned.pt\n"
            f"  Recall@1={metrics['recall@1']:.3f}  "
            f"Recall@5={metrics['recall@5']:.3f}  "
            f"Recall@10={metrics['recall@10']:.3f}  "
            f"MeanSim={metrics['mean_similarity']:.4f}"
        )


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune text encoder on MIMIC-CXR radiology reports"
    )
    parser.add_argument("--mimic_dir",        required=True,
                        help="Root directory of mimic-cxr-jpg")
    parser.add_argument("--output_dir",       default="models/",
                        help="Where to save the fine-tuned weights")
    parser.add_argument("--encoder",          default="bart",
                        choices=["bart", "bert"],
                        help="Text encoder backbone")
    parser.add_argument("--image_checkpoint", default=None,
                        help="Path to trained image encoder checkpoint (.pt)")
    parser.add_argument("--epochs",           type=int,   default=5)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--unfreeze_layers",  type=int,   default=2,
                        help="Number of trailing encoder layers to unfreeze")
    parser.add_argument("--num_workers",      type=int,   default=4)
    args = parser.parse_args()

    # ── Data ────────────────────────────────────────────────────────
    train_ds = MimicCXRDataset(args.mimic_dir, split="train",    augment=True)
    val_ds   = MimicCXRDataset(args.mimic_dir, split="validate", augment=False)

    logger.info(f"Label distribution (train):\n{train_ds.label_stats().to_string(index=False)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Train ───────────────────────────────────────────────────────
    trainer  = TextEncoderFinetuner(args)
    best_r1  = 0.0

    for epoch in range(1, args.epochs + 1):
        t0         = time.perf_counter()
        train_loss = trainer.train_epoch(train_loader, epoch)
        metrics    = trainer.evaluate(val_loader)
        elapsed    = time.perf_counter() - t0

        logger.info(
            f"\nEpoch {epoch}/{args.epochs} — {elapsed:.0f}s\n"
            f"  Train loss : {train_loss:.4f}\n"
            f"  Recall@1   : {metrics['recall@1']:.3f}\n"
            f"  Recall@5   : {metrics['recall@5']:.3f}\n"
            f"  Recall@10  : {metrics['recall@10']:.3f}\n"
            f"  Mean sim   : {metrics['mean_similarity']:.4f}"
        )

        if metrics["recall@1"] > best_r1:
            best_r1 = metrics["recall@1"]
            trainer.save(args.output_dir, epoch, metrics)
            logger.info(f"New best Recall@1: {best_r1:.3f} — checkpoint saved")

    logger.info(f"\nFine-tuning complete. Best Recall@1: {best_r1:.3f}")
    logger.info(
        f"Load fine-tuned encoder:\n"
        f"  get_text_encoder('{args.encoder}', "
        f"checkpoint_path='{args.output_dir}/text_encoder_finetuned.pt')"
    )


if __name__ == "__main__":
    main()
