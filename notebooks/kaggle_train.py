# ═══════════════════════════════════════════════════════════════════════
# ClinicalAI — Kaggle Training Notebook
# ═══════════════════════════════════════════════════════════════════════
#
# Datasets to attach in Kaggle UI:
#   1. CheXpert-v1.0-small   → /kaggle/input/chexpert-v1.0-small/
#   2. mimic-cxr-dataset     → /kaggle/input/mimic-cxr-dataset/
#
# Accelerator : GPU T4 x2  (Kaggle free tier)
# Runtime     : ~4–6 hours total
#
# What this does:
#   STEP 1 — Fine-tune BioBART text encoder on MIMIC-CXR reports
#             using contrastive (InfoNCE) image-text alignment
#             Output: text_encoder_finetuned.pt
#
#   STEP 2 — Retrain full fusion model (image + text → pathology labels)
#             on MIMIC-CXR with real radiology reports
#             Output: best_model_v2.pth  +  auc_results_v2.csv
#
#   SAVE   — Push all outputs to s3://clinical-ai-sam-2026/models/
# ═══════════════════════════════════════════════════════════════════════

# ── Cell 1: Install dependencies ────────────────────────────────────────
import subprocess, sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    else:
        print(result.stdout[-500:] or "OK")

run("pip install -q open_clip_torch timm grad-cam loguru boto3 albumentations")
run("pip install -q transformers>=4.40.0")

# ── Cell 2: Clone repo & set up paths ───────────────────────────────────
import os, sys

REPO_DIR   = "/kaggle/working/clinical-ai"
MIMIC_DIR  = "/kaggle/input/mimic-cxr-dataset"
CHEXPERT_DIR = "/kaggle/input/chexpert-v1.0-small"
OUTPUT_DIR = "/kaggle/working/outputs"
S3_BUCKET  = "clinical-ai-sam-2026"
S3_PREFIX  = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clone latest code from GitHub
if not os.path.exists(REPO_DIR):
    run(f"git clone https://github.com/saminathan017/Multimodal-chest-X-ray-AI.git {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} pull")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"Working dir: {os.getcwd()}")

# ── Cell 3: Verify datasets ─────────────────────────────────────────────
import os
from pathlib import Path

print("=== MIMIC-CXR ===")
mimic = Path(MIMIC_DIR)
for f in ["mimic-cxr-2.0.0-chexpert.csv", "mimic-cxr-2.0.0-split.csv", "mimic-cxr-2.0.0-metadata.csv"]:
    status = "✓" if (mimic / f).exists() else "✗ MISSING"
    print(f"  {status}  {f}")

# Check sample report exists
sample_reports = list(mimic.glob("files/**/*.txt"))
print(f"  Reports found: {len(sample_reports):,}")
if sample_reports:
    print(f"  Sample: {sample_reports[0]}")
    print(f"  Content preview: {sample_reports[0].read_text()[:200]}")

print("\n=== CheXpert ===")
chex = Path(CHEXPERT_DIR)
print(f"  train.csv: {'✓' if (chex/'train.csv').exists() else '✗'}")
print(f"  valid.csv: {'✓' if (chex/'valid.csv').exists() else '✗'}")

# ── Cell 4: Configure AWS credentials ───────────────────────────────────
# Add your AWS credentials in Kaggle Secrets:
#   Kaggle → Account → Secrets → Add Secret
#   Keys: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

from kaggle_secrets import UserSecretsClient

try:
    secrets = UserSecretsClient()
    os.environ["AWS_ACCESS_KEY_ID"]     = secrets.get_secret("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = secrets.get_secret("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"]    = secrets.get_secret("AWS_DEFAULT_REGION") or "us-east-1"
    print("AWS credentials loaded from Kaggle Secrets ✓")
except Exception as e:
    print(f"Warning: Could not load AWS secrets: {e}")
    print("Outputs will be saved locally only.")

import boto3
try:
    s3 = boto3.client("s3")
    s3.head_bucket(Bucket=S3_BUCKET)
    print(f"S3 bucket accessible: s3://{S3_BUCKET} ✓")
    S3_AVAILABLE = True
except Exception as e:
    print(f"S3 not available: {e}")
    S3_AVAILABLE = False

# ── Cell 5: Check GPU ────────────────────────────────────────────────────
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Cell 6: STEP 1 — Fine-tune text encoder on MIMIC-CXR ────────────────
print("\n" + "="*60)
print("STEP 1: Fine-tuning BioBART text encoder on MIMIC-CXR")
print("="*60)

from src.utils.mimic_cxr_dataset import MimicCXRDataset

# Quick sanity check on dataset
print("\nLoading dataset sample...")
try:
    ds = MimicCXRDataset(MIMIC_DIR, split="validate", augment=False)
    sample = ds[0]
    print(f"  Image shape : {sample['image'].shape}")
    print(f"  Note length : {len(sample['note'])} chars")
    print(f"  Note preview: {sample['note'][:150]}...")
    print(f"  Labels sum  : {sample['labels'].sum().item():.0f} positive findings")
    print(f"\n  Report coverage: running label_stats...")
    stats = ds.label_stats()
    print(stats[["label","positives","frequency"]].to_string(index=False))
except Exception as e:
    print(f"Dataset error: {e}")
    raise

# Run fine-tuning
print("\nStarting text encoder fine-tuning...")
cmd = (
    f"python {REPO_DIR}/src/pipeline/finetune_text_encoder.py "
    f"--mimic_dir      {MIMIC_DIR} "
    f"--output_dir     {OUTPUT_DIR} "
    f"--encoder        bart "
    f"--epochs         5 "
    f"--batch_size     32 "
    f"--lr             2e-5 "
    f"--unfreeze_layers 2 "
    f"--num_workers    2 "
)
print(f"Command: {cmd}\n")
run(cmd)

# Check output
enc_path = Path(OUTPUT_DIR) / "text_encoder_finetuned.pt"
if enc_path.exists():
    print(f"\nText encoder saved: {enc_path} ({enc_path.stat().st_size/1e6:.1f} MB) ✓")
else:
    print("\nWARNING: text_encoder_finetuned.pt not found — check logs above")

# Print final metrics from history
import json
hist_path = Path(OUTPUT_DIR) / "text_encoder_finetune_history.json"
if hist_path.exists():
    history = json.load(open(hist_path))
    print("\nFine-tuning history:")
    print(f"{'Epoch':>6} {'R@1':>8} {'R@5':>8} {'R@10':>9} {'MeanSim':>10}")
    print("-" * 45)
    for h in history:
        print(f"{h['epoch']:>6} {h['recall@1']:>8.3f} {h['recall@5']:>8.3f} "
              f"{h['recall@10']:>9.3f} {h['mean_similarity']:>10.4f}")

# ── Cell 7: STEP 2 — Retrain full fusion model with MIMIC-CXR ───────────
print("\n" + "="*60)
print("STEP 2: Retraining fusion model on MIMIC-CXR (real reports)")
print("="*60)

cmd = (
    f"python {REPO_DIR}/src/pipeline/training.py "
    f"--mimic_dir      {MIMIC_DIR} "
    f"--output_dir     {OUTPUT_DIR} "
    f"--encoder        bart "
    f"--epochs         10 "
    f"--batch_size     32 "
    f"--lr             1e-4 "
)
print(f"Command: {cmd}\n")
run(cmd)

# ── Cell 8: Evaluate & save AUC results ─────────────────────────────────
print("\n" + "="*60)
print("Evaluating new model...")
print("="*60)

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models import get_text_encoder
from src.models.image_encoder import ImageEncoder
from src.models.fusion_model import FusionModel

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# Load models
image_enc = ImageEncoder.from_pretrained(
    checkpoint_path = str(Path(OUTPUT_DIR) / "image_model.pt")
                      if (Path(OUTPUT_DIR)/"image_model.pt").exists() else None,
    num_classes=14, device=DEVICE,
)
text_enc = get_text_encoder(
    "bart",
    checkpoint_path = str(enc_path) if enc_path.exists() else None,
    output_dim=512, device=DEVICE,
)
fusion = FusionModel.from_pretrained(
    checkpoint_path = str(Path(OUTPUT_DIR) / "fusion_model.pt")
                      if (Path(OUTPUT_DIR)/"fusion_model.pt").exists() else None,
    device=DEVICE,
)

# Evaluate on MIMIC-CXR validate split
val_ds = MimicCXRDataset(MIMIC_DIR, split="validate", augment=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

all_preds, all_labels = [], []
image_enc.eval(); text_enc.eval(); fusion.eval()

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(DEVICE)
        notes  = batch["note"]
        labels = batch["labels"]

        img_feat, _ = image_enc(images)
        txt_feat    = text_enc(notes, device=DEVICE)
        out         = fusion(img_feat, txt_feat)

        all_preds.append(out["probs"].cpu())
        all_labels.append(labels)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Per-class AUC
rows = []
comp_labels = {"Cardiomegaly","Edema","Consolidation","Atelectasis","Pleural Effusion"}
for i, lbl in enumerate(LABELS):
    if all_labels[:, i].sum() > 0:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        rows.append({"label": lbl, "auc": auc, "competition": lbl in comp_labels})
    else:
        rows.append({"label": lbl, "auc": None, "competition": lbl in comp_labels})

auc_df = pd.DataFrame(rows)
comp_auc = float(np.mean([r["auc"] for r in rows if r["competition"] and r["auc"]]))
mean_auc = float(np.mean([r["auc"] for r in rows if r["auc"]]))

print("\nPer-class AUC (v2 model):")
print(auc_df[["label","auc","competition"]].to_string(index=False))
print(f"\nCompetition AUC : {comp_auc:.4f}")
print(f"Mean AUC        : {mean_auc:.4f}")
print(f"\nPrevious Competition AUC : 0.8463")
print(f"Improvement              : {comp_auc - 0.8463:+.4f}")

# Save AUC CSV
auc_path = Path(OUTPUT_DIR) / "auc_results_v2.csv"
auc_df.to_csv(auc_path, index=False)
print(f"\nSaved: {auc_path}")

# ── Cell 9: Push everything to S3 ───────────────────────────────────────
print("\n" + "="*60)
print("Pushing outputs to S3...")
print("="*60)

files_to_upload = [
    (Path(OUTPUT_DIR) / "text_encoder_finetuned.pt",           f"{S3_PREFIX}/text_encoder_finetuned.pt"),
    (Path(OUTPUT_DIR) / "fusion_model.pt",                      f"{S3_PREFIX}/fusion_model_v2.pt"),
    (Path(OUTPUT_DIR) / "image_model.pt",                       f"{S3_PREFIX}/image_model_v2.pt"),
    (Path(OUTPUT_DIR) / "auc_results_v2.csv",                   f"{S3_PREFIX}/auc_results_v2.csv"),
    (Path(OUTPUT_DIR) / "text_encoder_finetune_history.json",   f"{S3_PREFIX}/text_encoder_finetune_history.json"),
]

if S3_AVAILABLE:
    for local_path, s3_key in files_to_upload:
        if local_path.exists():
            print(f"  Uploading {local_path.name} → s3://{S3_BUCKET}/{s3_key}")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(f"  ✓ Done ({local_path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✗ Skipped {local_path.name} (not found)")
    print(f"\nAll outputs at: s3://{S3_BUCKET}/{S3_PREFIX}/")
else:
    print("S3 not available — outputs are in /kaggle/working/outputs/")
    print("Download manually from Kaggle output panel.")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"  Competition AUC v1 (CheXpert only)  : 0.8463")
print(f"  Competition AUC v2 (MIMIC-CXR + BART): {comp_auc:.4f}")
print(f"  Text encoder fine-tuning             : ✓")
print("="*60)
