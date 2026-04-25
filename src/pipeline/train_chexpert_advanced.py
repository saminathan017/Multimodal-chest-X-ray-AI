"""
Advanced CheXpert image trainer.

This trainer is designed for the first serious A100 run:
  - real image-only signal, no synthetic text bottleneck
  - patient-level split support
  - uncertainty-label policy
  - class imbalance weighting
  - mixed precision
  - progressive unfreezing
  - per-class AUROC checkpoint metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.models.cxr_image_model import CXRImageClassifier, set_backbone_trainable


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

U_ONES = {"Atelectasis", "Edema"}
CRITICAL_LABELS = {"Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Pneumothorax", "Pleural Effusion"}


class AsymmetricLoss(nn.Module):
    """
    Multilabel loss commonly used for long-tailed medical labels.

    It down-weights easy negatives more aggressively than positives, which helps
    rare findings without making every negative example dominate the gradient.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs
        if self.clip and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
            loss_pos = loss_pos * focal_weight
            loss_neg = loss_neg * focal_weight

        loss = loss_pos + loss_neg
        if self.pos_weight is not None:
            loss = loss * (1.0 + targets * (self.pos_weight - 1.0))
        return -loss.mean()


def resolve_csvs(data_dir: Path) -> tuple[Path, Path]:
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv in {data_dir}")
    if valid_csv.exists():
        return train_csv, valid_csv

    logger.warning("valid.csv not found; creating patient-level split files.")
    df = pd.read_csv(train_csv)
    patient_ids = df["Path"].astype(str).str.extract(r"(patient\d+)")[0]
    unique_patients = sorted(patient_ids.dropna().unique())
    if not unique_patients:
        raise ValueError("Could not extract patient IDs from Path column.")

    val_patients = set(unique_patients[::10])
    valid = df[patient_ids.isin(val_patients)]
    train = df[~patient_ids.isin(val_patients)]
    train_split = data_dir / "train_split_auto.csv"
    valid_split = data_dir / "valid_split_auto.csv"
    train.to_csv(train_split, index=False)
    valid.to_csv(valid_split, index=False)
    return train_split, valid_split


def clean_labels(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    cleaned = df.copy()
    for label in CHEXPERT_LABELS:
        if label not in cleaned.columns:
            cleaned[label] = 0.0
        series = cleaned[label].fillna(0.0)
        if policy == "zeros":
            series = series.replace(-1, 0)
        elif policy == "ones":
            series = series.replace(-1, 1)
        elif policy == "uones":
            series = series.replace(-1, 1 if label in U_ONES else 0)
        else:
            raise ValueError(f"Unknown uncertainty policy: {policy}")
        cleaned[label] = series.astype("float32")
    return cleaned


class CheXpertImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_dir: Path,
        image_size: int,
        split: str,
        uncertainty_policy: str,
        frontal_only: bool = True,
        fail_on_missing: bool = False,
    ) -> None:
        df = clean_labels(pd.read_csv(csv_path), uncertainty_policy)
        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"]
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.split = split
        self.labels = CHEXPERT_LABELS
        self.image_size = image_size
        self.fail_on_missing = fail_on_missing

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size + 24, image_size + 24)),
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=7),
                    transforms.ColorJitter(brightness=0.08, contrast=0.12),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        logger.info(f"Loaded CheXpert {split}: {len(self.df)} rows from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _image_path(self, raw_path: str) -> Path:
        path = self.data_dir / raw_path
        if path.exists():
            return path

        # Kaggle CheXpert small often stores CSV paths as
        # CheXpert-v1.0-small/train/... while users point data_dir directly at
        # that CheXpert-v1.0-small directory. Strip the duplicated root.
        parts = Path(raw_path).parts
        if parts and parts[0] == self.data_dir.name:
            stripped = self.data_dir.joinpath(*parts[1:])
            if stripped.exists():
                return stripped

        # Also support pointing data_dir at the parent folder, data/chexpert.
        nested = self.data_dir / self.data_dir.name / raw_path
        if nested.exists():
            return nested

        return path

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        path = self._image_path(str(row["Path"]))
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as exc:
            if self.fail_on_missing:
                raise FileNotFoundError(f"Could not read image {path}: {exc}") from exc
            logger.warning(f"Could not read image {path}: {exc}")
            image_tensor = torch.zeros(3, self.image_size, self.image_size)

        labels = torch.tensor([float(row[label]) for label in self.labels], dtype=torch.float32)
        return {"image": image_tensor, "labels": labels}


def positive_weights(dataset: CheXpertImageDataset, clip: float = 12.0) -> torch.Tensor:
    labels = dataset.df[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, clip)
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights(dataset: CheXpertImageDataset, cap: float = 8.0) -> torch.Tensor:
    """Give rare-positive rows a higher chance without oversampling too wildly."""
    labels = dataset.df[CHEXPERT_LABELS].to_numpy(dtype=np.float32)
    positives = labels.sum(axis=0)
    frequencies = positives / max(labels.shape[0], 1)
    inverse = 1.0 / np.maximum(frequencies, 1e-4)
    inverse = inverse / inverse.mean()
    row_weights = 1.0 + (labels * inverse).sum(axis=1)
    row_weights = np.clip(row_weights, 1.0, cap)
    return torch.tensor(row_weights, dtype=torch.double)


def mean_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for idx, label in enumerate(CHEXPERT_LABELS):
        positives = y_true[:, idx].sum()
        negatives = y_true.shape[0] - positives
        if positives == 0 or negatives == 0:
            continue
        per_class[label] = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
    return (float(np.mean(list(per_class.values()))) if per_class else 0.0, per_class)


def critical_auroc(per_class: dict[str, float]) -> float:
    values = [value for label, value in per_class.items() if label in CRITICAL_LABELS]
    return float(np.mean(values)) if values else 0.0


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv, valid_csv = resolve_csvs(data_dir)
    train_ds = CheXpertImageDataset(
        train_csv,
        data_dir,
        image_size=args.image_size,
        split="train",
        uncertainty_policy=args.uncertainty_policy,
        frontal_only=not args.include_lateral,
        fail_on_missing=args.fail_on_missing,
    )
    valid_ds = CheXpertImageDataset(
        valid_csv,
        data_dir,
        image_size=args.image_size,
        split="valid",
        uncertainty_policy=args.uncertainty_policy,
        frontal_only=not args.include_lateral,
        fail_on_missing=args.fail_on_missing,
    )

    sampler = None
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(sample_weights(train_ds), num_samples=len(train_ds), replacement=True)
        logger.info("Using weighted row sampler for rare-positive enrichment.")

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)
    valid_loader = make_loader(valid_ds, args.batch_size * 2, args.num_workers, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CXRImageClassifier(
        backbone=args.backbone,
        num_classes=len(CHEXPERT_LABELS),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    set_backbone_trainable(model, trainable=args.unfreeze_from_epoch <= 1)

    pos_weight = positive_weights(train_ds, clip=args.pos_weight_clip).to(device)
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == "asymmetric":
        criterion = AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    metadata = {
        "backbone": args.backbone,
        "labels": CHEXPERT_LABELS,
        "image_size": args.image_size,
        "uncertainty_policy": args.uncertainty_policy,
        "frontal_only": not args.include_lateral,
        "train_rows": len(train_ds),
        "valid_rows": len(valid_ds),
        "pos_weight": [float(x) for x in pos_weight.detach().cpu().tolist()],
        "loss": args.loss,
        "weighted_sampler": args.weighted_sampler,
        "tta": args.tta,
    }
    (output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2))

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_from_epoch:
            logger.info("Unfreezing backbone for fine-tuning.")
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * args.unfreeze_lr_scale, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - epoch + 1, 1))

        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                logger.info(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {np.mean(losses[-args.log_every:]):.4f}")
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

        scheduler.step()
        val_auc, per_class = evaluate(model, valid_loader, device, args.amp, args.tta)
        critical_auc = critical_auroc(per_class)
        train_loss = float(np.mean(losses)) if losses else 0.0
        logger.info(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Critical AUC: {critical_auc:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "critical_auc": critical_auc,
            "per_class_auc": per_class,
        }
        (output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(json.dumps(epoch_metrics, indent=2))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "critical_auc": critical_auc,
                    "per_class_auc": per_class,
                    "config": metadata,
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"Saved new best checkpoint: {output_dir / 'best_model.pt'}")

    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, amp: bool, tta: bool = False) -> tuple[float, dict[str, float]]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device == "cuda"):
            logits = model(images)
            if tta:
                flipped_logits = model(torch.flip(images, dims=[3]))
                logits = (logits + flipped_logits) / 2.0
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch["labels"].numpy())
    return mean_auroc(np.concatenate(labels), np.concatenate(probs))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/chexpert/CheXpert-v1.0-small")
    parser.add_argument("--output_dir", default="models/chexpert_advanced")
    parser.add_argument("--backbone", default="convnext_tiny", choices=["convnext_tiny", "convnext_small", "efficientnet_b3", "resnet50"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--loss", default="asymmetric", choices=["bce", "asymmetric"])
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="0 means full epoch.")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--uncertainty_policy", default="uones", choices=["zeros", "ones", "uones"])
    parser.add_argument("--pos_weight_clip", type=float, default=12.0)
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--unfreeze_from_epoch", type=int, default=2)
    parser.add_argument("--unfreeze_lr_scale", type=float, default=0.25)
    parser.add_argument("--include_lateral", action="store_true")
    parser.add_argument("--fail_on_missing", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
