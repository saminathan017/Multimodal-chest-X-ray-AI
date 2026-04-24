"""
Tiny local smoke training for the fusion head.

This does not replace real CheXpert/MIMIC training. It verifies the training
loop, optimizer, loss, checkpoint save, and registry path on CPU without
requiring protected clinical datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.evaluation.metrics import classification_report
from src.evaluation.model_registry import ModelRegistry
from src.models.fusion_model import FusionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="models/smoke_fusion_model.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    n = args.samples
    img_feat = torch.randn(n, 512)
    txt_feat = torch.randn(n, 512)
    labels = torch.zeros(n, 14)
    signal = (img_feat[:, 0] + txt_feat[:, 0]) > 0
    labels[:, 7] = signal.float()  # Pneumonia toy label
    labels[:, 0] = (~signal).float()

    model = FusionModel(feat_dim=512, hidden_dim=64, num_classes=14)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(img_feat, txt_feat)
        loss = loss_fn(out["logits"], labels)
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 6)})

    model.eval()
    with torch.no_grad():
        probs = model(img_feat, txt_feat)["probs"].numpy()
    report = classification_report(labels.numpy(), probs, labels=[f"class_{i}" for i in range(14)])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    entry = ModelRegistry().register(
        model_version="smoke-fusion-local",
        checkpoint_uri=str(output),
        datasets=["synthetic-smoke"],
        metrics=report,
        calibration={"status": "not_applicable"},
        thresholds={"status": "not_applicable"},
        status="candidate",
        notes="Synthetic smoke training only; not clinically meaningful.",
    )
    print(json.dumps({"history": history, "metrics": report, "registry": entry.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()

