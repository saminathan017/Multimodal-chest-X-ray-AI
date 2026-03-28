# ═══════════════════════════════════════════════════════════════════════
# ClinicalAI — Kaggle Training Notebook (v2)
# ═══════════════════════════════════════════════════════════════════════
# Datasets attached:
#   1. mimic-cxr-dataset  (SimhadriSadaram) → /kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/
#   2. chexpert           (ashery)          → /kaggle/input/datasets/ashery/chexpert/
#
# STEP 1 — Fine-tune BioBART text encoder on MIMIC-CXR real reports
# STEP 2 — Retrain full fusion model on CheXpert + fine-tuned text encoder
# SAVE   — Push outputs to s3://clinical-ai-sam-2026/models/
# ═══════════════════════════════════════════════════════════════════════

import subprocess, sys, os, ast, json
from pathlib import Path

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stderr[-2000:] if result.returncode != 0 else result.stdout[-500:] or "OK")

# ── Install dependencies ─────────────────────────────────────────────
run("pip install -q open_clip_torch timm grad-cam loguru boto3 albumentations")
run("pip install -q 'transformers>=4.40.0'")

# ── Paths ────────────────────────────────────────────────────────────
MIMIC_BASE   = "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset"
MIMIC_IMG    = f"{MIMIC_BASE}/official_data_iccv_final"
MIMIC_TRAIN  = f"{MIMIC_BASE}/mimic_cxr_aug_train.csv"
MIMIC_VAL    = f"{MIMIC_BASE}/mimic_cxr_aug_validate.csv"
CHEXPERT_DIR = "/kaggle/input/datasets/ashery/chexpert"
OUTPUT_DIR   = "/kaggle/working/outputs"
S3_BUCKET    = "clinical-ai-sam-2026"
S3_PREFIX    = "models"
REPO_DIR     = "/kaggle/working/clinical-ai"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Clone repo ───────────────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    run(f"git clone https://github.com/saminathan017/Multimodal-chest-X-ray-AI.git {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} pull")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"Working dir: {os.getcwd()}")

# ── AWS credentials ──────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["AWS_ACCESS_KEY_ID"]     = secrets.get_secret("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = secrets.get_secret("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"]    = secrets.get_secret("AWS_DEFAULT_REGION") or "us-east-1"
    print("AWS credentials loaded ✓")
except Exception as e:
    print(f"AWS secrets not loaded: {e} — outputs will be local only")

import boto3
try:
    s3 = boto3.client("s3")
    s3.head_bucket(Bucket=S3_BUCKET)
    print(f"S3 accessible: s3://{S3_BUCKET} ✓")
    S3_AVAILABLE = True
except Exception as e:
    print(f"S3 not available: {e}")
    S3_AVAILABLE = False

# ── GPU check ────────────────────────────────────────────────────────
import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════
# MIMIC-CXR Dataset (uses this Kaggle dataset's actual structure)
# ═══════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class MimicAugDataset(Dataset):
    """
    Works with SimhadriSadaram/mimic-cxr-dataset on Kaggle.
    CSV columns: subject_id, image, view, AP, PA, Lateral, text, text_augment
    """
    def __init__(self, csv_path, img_root, split="train", augment=True):
        self.df       = pd.read_csv(csv_path).dropna(subset=["text"]).reset_index(drop=True)
        self.img_root = Path(img_root)
        self.augment  = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        print(f"MimicAugDataset {split}: {len(self.df)} samples")

    def _get_image_path(self, row):
        """Try PA first, then AP, then first image in image list."""
        for col in ["PA", "AP", "image"]:
            val = row.get(col, None)
            if pd.isna(val) or val == "[]" or not val:
                continue
            try:
                paths = ast.literal_eval(str(val))
                if paths:
                    p = self.img_root / paths[0]
                    if p.exists():
                        return p
            except Exception:
                pass
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = self._get_image_path(row)
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        # Text — use real radiology report
        note = str(row.get("text", "")).strip()
        if not note:
            note = "Chest radiograph."
        note = note[:512]

        return {"image": image, "note": note}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Fine-tune BioBART text encoder on MIMIC-CXR real reports
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Fine-tuning BioBART text encoder on MIMIC-CXR")
print("="*60)

# Sanity check
train_ds = MimicAugDataset(MIMIC_TRAIN, MIMIC_IMG, split="train",    augment=True)
val_ds   = MimicAugDataset(MIMIC_VAL,   MIMIC_IMG, split="validate", augment=False)

sample = val_ds[0]
print(f"  Image shape : {sample['image'].shape}")
print(f"  Note preview: {sample['note'][:150]}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Load models
from src.models import get_text_encoder
from src.models.image_encoder import ImageEncoder

image_encoder = ImageEncoder.from_pretrained(num_classes=14, device=DEVICE, freeze_backbone=True)
for p in image_encoder.parameters():
    p.requires_grad = False
image_encoder.eval()

text_encoder = get_text_encoder("bart", output_dim=512, device=DEVICE,
                                 freeze_backbone=True, num_unfrozen_layers=2)

# InfoNCE loss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, img_feat, txt_feat):
        logits  = (img_feat @ txt_feat.T) / self.temperature.clamp(min=0.01)
        targets = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2

contrastive = InfoNCELoss(0.07).to(DEVICE)
optimizer   = torch.optim.AdamW(
    list(text_encoder.projection.parameters()) + list(contrastive.parameters()),
    lr=2e-5, weight_decay=1e-4
)

best_loss = float("inf")
history   = []

for epoch in range(1, 6):
    text_encoder.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        images = batch["image"].to(DEVICE)
        notes  = batch["note"]

        optimizer.zero_grad()
        with torch.no_grad():
            img_feat, _ = image_encoder(images)
        img_feat = F.normalize(img_feat, dim=-1)

        txt_feat = text_encoder(notes, device=DEVICE)
        txt_feat = F.normalize(txt_feat, dim=-1)

        loss = contrastive(img_feat, txt_feat)
        loss.backward()
        nn.utils.clip_grad_norm_(text_encoder.projection.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    history.append({"epoch": epoch, "train_loss": avg_loss})
    print(f"\nEpoch {epoch}/5 — Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        enc_path = Path(OUTPUT_DIR) / "text_encoder_finetuned.pt"
        torch.save(text_encoder.state_dict(), enc_path)
        print(f"  Saved best encoder (loss={best_loss:.4f}) ✓")

with open(Path(OUTPUT_DIR) / "text_encoder_finetune_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nStep 1 complete. Best loss: {best_loss:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Retrain full fusion model on CheXpert + fine-tuned encoder
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Retraining fusion model on CheXpert")
print("="*60)

cmd = (
    f"python {REPO_DIR}/src/pipeline/training.py "
    f"--data_dir   {CHEXPERT_DIR} "
    f"--output_dir {OUTPUT_DIR} "
    f"--encoder    bart "
    f"--epochs     10 "
    f"--batch_size 32 "
    f"--lr         1e-4 "
)
print(f"Command: {cmd}\n")
run(cmd)

# ═══════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Evaluating...")
print("="*60)

from sklearn.metrics import roc_auc_score
from src.models.fusion_model import FusionModel

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# Load CheXpert val set for evaluation
from src.pipeline.training import CheXpertDataset
val_csv = f"{CHEXPERT_DIR}/valid.csv"
eval_ds = CheXpertDataset(val_csv, CHEXPERT_DIR, split="val")
eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=2)

enc_path = Path(OUTPUT_DIR) / "text_encoder_finetuned.pt"
image_enc = ImageEncoder.from_pretrained(
    checkpoint_path=str(Path(OUTPUT_DIR)/"image_model.pt") if (Path(OUTPUT_DIR)/"image_model.pt").exists() else None,
    num_classes=14, device=DEVICE,
)
text_enc = get_text_encoder("bart",
    checkpoint_path=str(enc_path) if enc_path.exists() else None,
    output_dim=512, device=DEVICE,
)
fusion = FusionModel.from_pretrained(
    checkpoint_path=str(Path(OUTPUT_DIR)/"fusion_model.pt") if (Path(OUTPUT_DIR)/"fusion_model.pt").exists() else None,
    device=DEVICE,
)

all_preds, all_labels = [], []
image_enc.eval(); text_enc.eval(); fusion.eval()

with torch.no_grad():
    for batch in eval_loader:
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

rows = []
comp_labels = {"Cardiomegaly","Edema","Consolidation","Atelectasis","Pleural Effusion"}
for i, lbl in enumerate(LABELS):
    if all_labels[:, i].sum() > 0:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        rows.append({"label": lbl, "auc": auc, "competition": lbl in comp_labels})
    else:
        rows.append({"label": lbl, "auc": None, "competition": lbl in comp_labels})

auc_df   = pd.DataFrame(rows)
comp_auc = float(np.mean([r["auc"] for r in rows if r["competition"] and r["auc"]]))
mean_auc = float(np.mean([r["auc"] for r in rows if r["auc"]]))

print(auc_df[["label","auc","competition"]].to_string(index=False))
print(f"\nCompetition AUC : {comp_auc:.4f}")
print(f"Mean AUC        : {mean_auc:.4f}")
print(f"Previous AUC    : 0.8463")
print(f"Improvement     : {comp_auc - 0.8463:+.4f}")

auc_path = Path(OUTPUT_DIR) / "auc_results_v2.csv"
auc_df.to_csv(auc_path, index=False)

# ═══════════════════════════════════════════════════════════════════════
# Push to S3
# ═══════════════════════════════════════════════════════════════════════
files_to_upload = [
    (Path(OUTPUT_DIR) / "text_encoder_finetuned.pt",         f"{S3_PREFIX}/text_encoder_finetuned.pt"),
    (Path(OUTPUT_DIR) / "fusion_model.pt",                    f"{S3_PREFIX}/fusion_model_v2.pt"),
    (Path(OUTPUT_DIR) / "image_model.pt",                     f"{S3_PREFIX}/image_model_v2.pt"),
    (Path(OUTPUT_DIR) / "auc_results_v2.csv",                 f"{S3_PREFIX}/auc_results_v2.csv"),
    (Path(OUTPUT_DIR) / "text_encoder_finetune_history.json", f"{S3_PREFIX}/text_encoder_finetune_history.json"),
]

if S3_AVAILABLE:
    for local_path, s3_key in files_to_upload:
        if local_path.exists():
            print(f"  Uploading {local_path.name} → s3://{S3_BUCKET}/{s3_key}")
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(f"  ✓ ({local_path.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  ✗ Skipped {local_path.name} (not found)")
    print(f"\nAll outputs: s3://{S3_BUCKET}/{S3_PREFIX}/")
else:
    print("S3 not available — download from Kaggle Output tab.")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"  Competition AUC v1 : 0.8463")
print(f"  Competition AUC v2 : {comp_auc:.4f}")
print(f"  Improvement        : {comp_auc - 0.8463:+.4f}")
print("="*60)
