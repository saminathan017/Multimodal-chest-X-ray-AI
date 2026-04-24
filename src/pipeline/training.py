"""
src/pipeline/training.py
─────────────────────────────────────────────────────────────────────
Fine-tuning script for the fusion model on CheXpert / MIMIC-CXR.

Run on AWS SageMaker:
    python src/pipeline/training.py \
        --data_dir s3://clinical-ai-demo-bucket/data/chexpert \
        --output_dir s3://clinical-ai-demo-bucket/models \
        --epochs 10 \
        --batch_size 32

Run locally (subset for testing):
    python src/pipeline/training.py --data_dir data/ --epochs 2 --dry_run
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from loguru import logger

from src.models.image_encoder import ImageEncoder
from src.models.fusion_model import FusionModel
from src.models import get_text_encoder
from src.utils.mimic_cxr_dataset import MimicCXRDataset


# ── CheXpert Dataset (image-only labels, synthetic notes) ────────────
class CheXpertDataset(Dataset):
    """
    Loads CheXpert CSV + image files.
    Download: https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]

    def __init__(self, csv_path: str, root_dir: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.split = split

        # Fill NaN: treat uncertain (-1) as 0 (ignore strategy)
        for col in self.LABELS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).replace(-1, 0)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(5) if split == "train" else transforms.Lambda(lambda x: x),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info(f"Loaded CheXpert {split}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        labels = torch.tensor(
            [float(row.get(l, 0)) for l in self.LABELS],
            dtype=torch.float32,
        )

        # Synthetic note (in real setup, join with MIMIC clinical notes)
        note = f"Chest radiograph. {row.get('AP/PA', 'PA')} view."

        return {"image": image, "note": note, "labels": labels}


# ── Trainer ───────────────────────────────────────────────────────────
class MultiModalTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Models
        self.image_encoder = ImageEncoder.from_pretrained(
            num_classes=14, device=self.device, freeze_backbone=True
        )
        self.text_encoder  = get_text_encoder(
            encoder_type    = args.encoder,
            output_dim      = 512,
            device          = self.device,
            freeze_backbone = True,
        )
        self.fusion_model  = FusionModel(feat_dim=512, hidden_dim=256, num_classes=14).to(self.device)

        # Optimizer: only train fusion + classification heads
        params = (
            list(self.image_encoder.classifier.parameters()) +
            list(self.text_encoder.projection.parameters()) +
            list(self.fusion_model.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.image_encoder.train()
        self.text_encoder.train()
        self.fusion_model.train()

        total_loss = 0.0
        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            notes  = batch["note"]
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            img_feat, _ = self.image_encoder(images)
            txt_feat    = self.text_encoder(notes, device=self.device)
            out         = self.fusion_model(img_feat, txt_feat)

            loss = self.criterion(out["logits"], labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(self.fusion_model.parameters()) +
                list(self.image_encoder.classifier.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.fusion_model.eval()

        all_preds, all_labels = [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            notes  = batch["note"]
            labels = batch["labels"]

            img_feat, _ = self.image_encoder(images)
            txt_feat    = self.text_encoder(notes, device=self.device)
            out         = self.fusion_model(img_feat, txt_feat)
            probs       = out["probs"].cpu()

            all_preds.append(probs)
            all_labels.append(labels)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(all_labels.shape[1]):
            if all_labels[:, i].sum() > 0:
                aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))

        mean_auc = np.mean(aucs) if aucs else 0.0
        return {"mean_auc": mean_auc, "per_class_auc": aucs}

    def save_checkpoint(self, output_dir: str, epoch: int):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.image_encoder.state_dict(), f"{output_dir}/image_model.pt")
        torch.save(self.text_encoder.state_dict(),  f"{output_dir}/text_model.pt")
        torch.save(self.fusion_model.state_dict(),  f"{output_dir}/fusion_model.pt")
        logger.info(f"Checkpoint saved to {output_dir} (epoch {epoch})")


# ── Entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default="data/chexpert",
                        help="CheXpert root dir (or use --mimic_dir for MIMIC-CXR)")
    parser.add_argument("--mimic_dir",       default=None,
                        help="MIMIC-CXR root dir — uses real radiology reports when set")
    parser.add_argument("--output_dir",      default="models/")
    parser.add_argument("--encoder",         default="bart", choices=["bart", "bert"])
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--steps_per_epoch", type=int,   default=500)
    parser.add_argument("--dry_run",         action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run — skipping actual training")
        return

    # ── Dataset: prefer MIMIC-CXR (real reports) over CheXpert ──────
    if args.mimic_dir:
        logger.info(f"Using MIMIC-CXR with real radiology reports: {args.mimic_dir}")
        train_ds = MimicCXRDataset(args.mimic_dir, split="train",    augment=True)
        val_ds   = MimicCXRDataset(args.mimic_dir, split="validate", augment=False)
    else:
        logger.warning(
            "Using CheXpert with SYNTHETIC notes ('Chest radiograph. PA view.'). "
            "Text encoder will not learn from real clinical language. "
            "Pass --mimic_dir to use real MIMIC-CXR reports."
        )
        train_csv = f"{args.data_dir}/train.csv"
        val_csv   = f"{args.data_dir}/valid.csv"
        if not Path(val_csv).exists():
            logger.warning("valid.csv not found — splitting train.csv 90/10 for validation")
            full_df = pd.read_csv(train_csv)
            split_idx = int(len(full_df) * 0.9)
            full_df.iloc[:split_idx].to_csv("/tmp/chexpert_train_split.csv", index=False)
            full_df.iloc[split_idx:].to_csv("/tmp/chexpert_val_split.csv", index=False)
            train_csv = "/tmp/chexpert_train_split.csv"
            val_csv   = "/tmp/chexpert_val_split.csv"
        train_ds = CheXpertDataset(train_csv, args.data_dir, "train")
        val_ds   = CheXpertDataset(val_csv,   args.data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    args.steps_per_epoch = len(train_loader)
    trainer = MultiModalTrainer(args)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        metrics    = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val AUC: {metrics['mean_auc']:.4f}")

        if metrics["mean_auc"] > best_auc:
            best_auc = metrics["mean_auc"]
            trainer.save_checkpoint(args.output_dir, epoch)

    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
