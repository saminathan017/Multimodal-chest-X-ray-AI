"""Audit CheXpert extraction, labels, and path coverage before cloud training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def resolve_path(data_dir: Path, raw_path: str) -> Path:
    path = data_dir / raw_path
    if path.exists():
        return path

    parts = Path(raw_path).parts
    if parts and parts[0] == data_dir.name:
        stripped = data_dir.joinpath(*parts[1:])
        if stripped.exists():
            return stripped
    return path


def audit_split(csv_path: Path, data_dir: Path, sample_limit: int = 0) -> dict:
    df = pd.read_csv(csv_path)
    checked = df if sample_limit <= 0 else df.head(sample_limit)
    exists = checked["Path"].astype(str).map(lambda p: resolve_path(data_dir, p).exists())

    prevalence = {}
    for label in LABELS:
        if label in df.columns:
            series = df[label].fillna(0)
            prevalence[label] = {
                "positive": int((series == 1).sum()),
                "uncertain": int((series == -1).sum()),
                "blank": int(df[label].isna().sum()),
                "positive_rate": round(float((series == 1).mean()), 5),
            }

    patient_ids = df["Path"].astype(str).str.extract(r"(patient\d+)")[0]
    return {
        "csv": str(csv_path),
        "rows": int(len(df)),
        "patients": int(patient_ids.nunique()),
        "frontal_rows": int((df.get("Frontal/Lateral", "") == "Frontal").sum()) if "Frontal/Lateral" in df else None,
        "lateral_rows": int((df.get("Frontal/Lateral", "") == "Lateral").sum()) if "Frontal/Lateral" in df else None,
        "path_check_rows": int(len(checked)),
        "path_exists": int(exists.sum()),
        "path_missing": int((~exists).sum()),
        "path_coverage": round(float(exists.mean()) if len(exists) else 0.0, 6),
        "label_prevalence": prevalence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/chexpert/CheXpert-v1.0-small")
    parser.add_argument("--sample_limit", type=int, default=5000, help="0 checks every path; 5000 is fast.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    result = {}
    for name in ("train.csv", "valid.csv", "train_full.csv"):
        csv_path = data_dir / name
        if csv_path.exists():
            result[name] = audit_split(csv_path, data_dir, args.sample_limit)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
