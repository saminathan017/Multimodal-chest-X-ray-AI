"""
Credential-aware clinical data pull helper.

Full CheXpert and MIMIC-CXR access requires dataset terms/credential approval.
This script checks credentials and gives exact commands/paths instead of hiding
access failures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["chexpert-kaggle", "mimic-physionet", "hf-mimic-sample"], required=True)
    parser.add_argument("--output", default="data")
    parser.add_argument("--limit", type=int, default=128, help="HF sample row limit.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {"dataset": args.dataset, "output": str(output), "actions": []}

    if args.dataset == "chexpert-kaggle":
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists() and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
            report["status"] = "blocked"
            report["message"] = "Kaggle credentials missing. Add ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."
            report["next_command"] = "python3 scripts/pull_clinical_data.py --dataset chexpert-kaggle --output data/chexpert"
        elif not shutil.which("kaggle"):
            report["status"] = "blocked"
            report["message"] = "Install Kaggle CLI: pip3 install kaggle"
        else:
            target = output / "chexpert"
            target.mkdir(parents=True, exist_ok=True)
            report["actions"].append(run(["kaggle", "datasets", "download", "-d", "ashery/chexpert", "-p", str(target), "--unzip"]))
            report["status"] = "attempted"

    elif args.dataset == "mimic-physionet":
        report["status"] = "manual_credential_required"
        report["message"] = (
            "MIMIC-CXR requires PhysioNet credentialing and data-use agreement. "
            "After approval, place MIMIC-CXR-JPG under data/mimic with metadata CSVs."
        )
        report["expected_files"] = [
            "data/mimic/mimic-cxr-2.0.0-chexpert.csv",
            "data/mimic/mimic-cxr-2.0.0-split.csv",
            "data/mimic/mimic-cxr-2.0.0-metadata.csv",
            "data/mimic/files/...",
        ]

    elif args.dataset == "hf-mimic-sample":
        try:
            from datasets import load_dataset
        except Exception as exc:
            report["status"] = "blocked"
            report["message"] = f"Install datasets first: pip3 install datasets. Import error: {exc}"
        else:
            ds = load_dataset("MLforHealthcare/mimic-cxr", split=f"train[:{args.limit}]")
            target = output / "hf_mimic_sample"
            target.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(target))
            report["status"] = "downloaded"
            report["rows"] = len(ds)
            report["path"] = str(target)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

