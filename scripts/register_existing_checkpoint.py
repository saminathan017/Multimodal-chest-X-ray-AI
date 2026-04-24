"""
Register an existing trained checkpoint into the ClinicalAI model registry.

This is useful when a checkpoint was trained outside the current environment
(for example Kaggle/SageMaker) and copied into models/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.evaluation.model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model_final.pth")
    parser.add_argument("--model-version", default="clinical-ai-chexpert-v1")
    parser.add_argument("--registry", default="models/model_registry.json")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    auc = float(payload.get("auc", 0.0)) if isinstance(payload, dict) else 0.0
    all_aucs = payload.get("all_aucs", {}) if isinstance(payload, dict) else {}
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    phase = payload.get("phase") if isinstance(payload, dict) else None

    metrics = {
        "macro_auroc": round(auc, 6),
        "per_class_auroc": {str(k): float(v) for k, v in all_aucs.items()},
        "epoch": epoch,
        "phase": phase,
        "source": "existing_checkpoint",
    }
    calibration = {"status": "pending", "note": "Run calibration on locked validation set before production use."}
    thresholds = {"status": "pending", "note": "Run threshold optimizer on clinical validation set."}

    entry = ModelRegistry(args.registry).register(
        model_version=args.model_version,
        checkpoint_uri=str(checkpoint),
        datasets=["CheXpert"],
        metrics=metrics,
        calibration=calibration,
        thresholds=thresholds,
        status="candidate",
        notes="Imported local checkpoint; requires external validation and calibration.",
    )
    print(json.dumps(entry.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()

