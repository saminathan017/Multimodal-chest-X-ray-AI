"""
src/utils/mimic_cxr_dataset.py
─────────────────────────────────────────────────────────────────────
MIMIC-CXR dataset loader with real radiology report extraction.

MIMIC-CXR directory structure expected:
    mimic-cxr/
    ├── files/
    │   └── p10/
    │       └── p10000032/
    │           ├── s50414267.txt          ← radiology report
    │           └── s50414267/
    │               └── <dicom_id>.jpg     ← chest X-ray image
    ├── mimic-cxr-2.0.0-chexpert.csv      ← 14-label annotations
    ├── mimic-cxr-2.0.0-split.csv         ← train / validate / test
    └── mimic-cxr-2.0.0-metadata.csv      ← view position, etc.

Access: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
        (free, requires credentialed PhysioNet account)

Report sections extracted (in priority order):
    IMPRESSION  → radiologist's final conclusion   (most informative)
    FINDINGS    → detailed observations            (rich clinical detail)
    INDICATION  → reason for exam / clinical context
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from loguru import logger


# ── CheXpert 14-label set (matches MIMIC-CXR chexpert.csv columns) ──
LABELS = [
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

# Report section headers to search for
_SECTION_RE = re.compile(
    r"(IMPRESSION|FINDINGS|INDICATION|HISTORY|TECHNIQUE|COMPARISON)[:\s]*",
    re.IGNORECASE,
)


# ── Report parsing ────────────────────────────────────────────────────

def parse_report(report_path: Path) -> str:
    """
    Read a MIMIC-CXR .txt report and return the most clinically
    informative text: IMPRESSION first, then FINDINGS, then INDICATION.
    Falls back to the full report if no sections are found.

    Args:
        report_path: Path to the .txt file

    Returns:
        Cleaned report string (max ~512 words)
    """
    try:
        raw = report_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""

    sections = _split_sections(raw)

    # Priority: IMPRESSION → FINDINGS → INDICATION → full text
    text = (
        sections.get("IMPRESSION", "")
        or sections.get("FINDINGS", "")
        or sections.get("INDICATION", "")
        or raw
    )

    # If both IMPRESSION and FINDINGS exist, combine them
    if sections.get("IMPRESSION") and sections.get("FINDINGS"):
        text = sections["FINDINGS"] + " " + sections["IMPRESSION"]

    return _clean(text)


def _split_sections(text: str) -> dict[str, str]:
    """Split report text into {SECTION_NAME: content} dict."""
    parts   = _SECTION_RE.split(text)
    result  = {}
    i       = 1  # parts[0] is preamble before first header
    while i < len(parts) - 1:
        header  = parts[i].strip().upper().rstrip(":")
        content = parts[i + 1] if i + 1 < len(parts) else ""
        result[header] = _clean(content)
        i += 2
    return result


def _clean(text: str) -> str:
    """Normalise whitespace and remove artefacts."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text


def report_path_for(
    root: Path,
    subject_id: int,
    study_id: int,
) -> Path:
    """
    Reconstruct the .txt report path from subject_id + study_id.

    MIMIC-CXR stores reports at:
        files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt
    """
    prefix = f"p{str(subject_id)[:2]}"
    return root / "files" / prefix / f"p{subject_id}" / f"s{study_id}.txt"


def image_path_for(
    root: Path,
    subject_id: int,
    study_id: int,
    dicom_id: str,
) -> Path:
    """
    Reconstruct the .jpg image path from subject/study/dicom IDs.
    """
    prefix = f"p{str(subject_id)[:2]}"
    return (
        root / "files" / prefix
        / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    )


# ── Dataset ───────────────────────────────────────────────────────────

class MimicCXRDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR-JPG with real radiology reports.

    Replaces the synthetic "Chest radiograph. PA view." notes used
    during CheXpert training with actual FINDINGS + IMPRESSION text,
    enabling the text encoder to learn from genuine clinical language.

    Args:
        root_dir:  Path to the mimic-cxr root directory
        split:     "train" | "validate" | "test"
        max_length: Max report length in characters (soft truncation)
        augment:   Apply random augmentation (train split only)

    Usage:
        ds = MimicCXRDataset("/data/mimic-cxr", split="train")
        sample = ds[0]
        # sample["image"]  → (3, 224, 224) tensor
        # sample["note"]   → "Bilateral lower lobe consolidation..."
        # sample["labels"] → (14,) float tensor
        # sample["meta"]   → {"subject_id": ..., "study_id": ..., "report_length": ...}
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        max_length: int = 512,
        augment: bool = True,
    ):
        self.root      = Path(root_dir)
        self.split     = split
        self.max_length = max_length

        self.df = self._build_index()
        self.transform = self._build_transform(split, augment)

        logger.info(
            f"[MimicCXRDataset] {split}: {len(self.df)} samples | "
            f"report coverage: {self._report_coverage():.1%}"
        )

    # ── Index building ────────────────────────────────────────────────

    def _build_index(self) -> pd.DataFrame:
        """
        Join chexpert labels + split + metadata into one DataFrame.
        Each row = one study (one patient visit, potentially multiple images).
        We take the first frontal image per study.
        """
        chexpert_csv = self.root / "mimic-cxr-2.0.0-chexpert.csv"
        split_csv    = self.root / "mimic-cxr-2.0.0-split.csv"
        meta_csv     = self.root / "mimic-cxr-2.0.0-metadata.csv"

        for p in (chexpert_csv, split_csv, meta_csv):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required MIMIC-CXR file not found: {p}\n"
                    f"Download from: https://physionet.org/content/mimic-cxr-jpg/2.0.0/"
                )

        labels_df = pd.read_csv(chexpert_csv)
        split_df  = pd.read_csv(split_csv)
        meta_df   = pd.read_csv(meta_csv)

        # Keep only the target split
        split_ids = split_df[split_df["split"] == self.split][
            ["dicom_id", "subject_id", "study_id"]
        ]

        # Prefer frontal views (PA > AP) for each study
        frontal = meta_df[meta_df["ViewPosition"].isin(["PA", "AP"])].copy()
        frontal["view_rank"] = frontal["ViewPosition"].map({"PA": 0, "AP": 1})
        frontal = (
            frontal.sort_values("view_rank")
            .groupby("study_id", as_index=False)
            .first()
        )

        # Merge: split → frontal image → labels
        df = split_ids.merge(frontal[["dicom_id", "study_id"]], on="dicom_id", how="inner")
        df = df.merge(labels_df, on=["subject_id", "study_id"], how="inner")

        # Fill uncertain (-1) and NaN → 0 (U-zeroes strategy)
        for col in LABELS:
            if col in df.columns:
                df[col] = df[col].fillna(0).replace(-1, 0).astype(float)

        df = df.reset_index(drop=True)
        return df

    def _report_coverage(self) -> float:
        """Fraction of rows where a .txt report file actually exists."""
        exists = sum(
            1 for _, row in self.df.iterrows()
            if report_path_for(self.root, row["subject_id"], row["study_id"]).exists()
        )
        return exists / max(len(self.df), 1)

    # ── Transforms ───────────────────────────────────────────────────

    @staticmethod
    def _build_transform(split: str, augment: bool) -> transforms.Compose:
        base = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if split == "train" and augment:
            base = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        return transforms.Compose(base)

    # ── Dataset protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image ──────────────────────────────────────────────────
        img_path = image_path_for(
            self.root, row["subject_id"], row["study_id"], row["dicom_id"]
        )
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        # ── Radiology report ───────────────────────────────────────
        rpt_path = report_path_for(self.root, row["subject_id"], row["study_id"])
        note     = parse_report(rpt_path)

        if not note:
            # Fallback: minimal structured note from metadata
            note = f"Chest radiograph. {row.get('ViewPosition', 'PA')} view."

        # Soft-truncate to max_length characters
        note = note[: self.max_length]

        # ── Labels ─────────────────────────────────────────────────
        labels = torch.tensor(
            [float(row.get(lbl, 0.0)) for lbl in LABELS],
            dtype=torch.float32,
        )

        return {
            "image":  image,
            "note":   note,
            "labels": labels,
            "meta": {
                "subject_id":    int(row["subject_id"]),
                "study_id":      int(row["study_id"]),
                "report_length": len(note),
            },
        }

    # ── Utilities ────────────────────────────────────────────────────

    def label_stats(self) -> pd.DataFrame:
        """
        Return positive label frequency per class.
        Useful for computing class weights to handle label imbalance.
        """
        stats = []
        for lbl in LABELS:
            if lbl in self.df.columns:
                pos   = int(self.df[lbl].sum())
                total = len(self.df)
                stats.append({
                    "label":     lbl,
                    "positives": pos,
                    "frequency": pos / total,
                    "weight":    total / max(pos, 1),
                })
        return pd.DataFrame(stats)

    def sample_report(self, idx: int = 0) -> str:
        """Print a sample report for inspection."""
        row      = self.df.iloc[idx]
        rpt_path = report_path_for(self.root, row["subject_id"], row["study_id"])
        return parse_report(rpt_path)
