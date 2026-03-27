# ClinicalAI — Multi-Modal Clinical Decision Support

> AI-powered chest X-ray analysis fused with clinical notes using **BiomedCLIP + ClinicalBERT + AWS Bedrock**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Bedrock-ff9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![CI](https://github.com/saminathan017/Multimodal-chest-X-ray-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/saminathan017/Multimodal-chest-X-ray-AI/actions)

---

## Problem

Radiologists review 50–100 chest X-rays per day, often without the patient's full clinical context. Adverse events from missed findings cost the US healthcare system **$17 billion annually**. Existing AI tools analyze images in isolation — ignoring the rich clinical narrative in the EHR.

**This system fuses both modalities** to deliver context-aware, explainable AI findings in under 2 seconds.

---

## Architecture

```
 Chest X-Ray (JPEG/PNG/DCM)        Clinical Notes (free text)
         │                                   │
         ▼                                   ▼
 ┌─────────────────┐              ┌──────────────────────┐
 │  BiomedCLIP     │              │  Bio_ClinicalBERT    │
 │  ViT-B/16       │              │  + Linear projection │
 │  → 512-d feat   │              │  → 512-d feat        │
 └────────┬────────┘              └──────────┬───────────┘
          └──────────────┬──────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Cross-Modal Fusion │
              │  Multi-Head Attn    │
              │  Concat → MLP 256-d │
              └──────────┬──────────┘
                         ▼
         ┌───────────────┬─────────────────┐
         ▼               ▼                 ▼
  14-class logits    Urgency score     GradCAM heatmap
         │
         ▼
  AWS Bedrock (Claude Haiku) → Clinical report
         │
         ▼
  Streamlit demo — findings + heatmap + download
```

---

## Quick Start

```bash
git clone https://github.com/saminathan017/Multimodal-chest-X-ray-AI.git
cd Multimodal-chest-X-ray-AI

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Fill in your AWS credentials and HuggingFace token
```

```bash
# Run the demo (no GPU or model download required — uses mock inference)
make run
# or: streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` — toggle **Demo Mode** in the sidebar to run without downloading models.

---

## Model Performance

Trained on **CheXpert** (224,316 frontal chest X-rays, 14 pathology labels) — see [Dataset Credits](#dataset-credits).

| Pathology          | AUC    | Competition label |
|--------------------|--------|:-----------------:|
| Pleural Other      | 0.9614 |                   |
| Pleural Effusion   | 0.8994 | ✓                 |
| Consolidation      | 0.8943 | ✓                 |
| Edema              | 0.8856 | ✓                 |
| Lung Opacity       | 0.8733 |                   |
| No Finding         | 0.8492 |                   |
| Pneumothorax       | 0.8208 |                   |
| Support Devices    | 0.7933 |                   |
| Atelectasis        | 0.7891 | ✓                 |
| Cardiomegaly       | 0.7630 | ✓                 |
| **Competition AUC**| **0.8463** | avg of 5 above |
| **Mean AUC**       | **0.772**  | all 13 labels  |

---

## Tech Stack

| Layer          | Technology                                |
|----------------|-------------------------------------------|
| Vision encoder | BiomedCLIP ViT-B/16 (Microsoft / HF)     |
| Text encoder   | Bio_ClinicalBERT (Emily Alsentzer / HF)  |
| Fusion         | Cross-modal attention + MLP head         |
| Explainability | GradCAM (pytorch-grad-cam)               |
| Report gen     | AWS Bedrock — Claude 3 Haiku             |
| Training       | PyTorch 2.2, HuggingFace Transformers    |
| Serving        | AWS SageMaker (ml.g4dn.xlarge), FastAPI  |
| Demo UI        | Streamlit 1.34, Plotly                   |
| Infrastructure | AWS S3, ECR, CloudWatch, Terraform       |

---

## Project Structure

```
clinical-ai-demo/
├── app/
│   └── streamlit_app.py          # Demo UI (Streamlit)
├── src/
│   ├── models/
│   │   ├── image_encoder.py      # BiomedCLIP + GradCAM
│   │   ├── text_encoder.py       # Bio_ClinicalBERT + projection
│   │   ├── fusion_model.py       # Cross-modal attention head
│   │   └── uncertainty.py        # MC-Dropout uncertainty
│   ├── pipeline/
│   │   ├── inference.py          # End-to-end inference pipeline
│   │   └── training.py           # CheXpert fine-tuning
│   ├── api/
│   │   ├── main.py               # FastAPI — production endpoints
│   │   ├── auth.py               # JWT + RBAC
│   │   └── audit.py              # HIPAA audit logging
│   ├── safety/
│   │   ├── phi_detector.py       # PHI scrubbing before Bedrock
│   │   ├── input_validator.py    # Image/text input validation
│   │   └── fairness_monitor.py   # Demographic fairness metrics
│   └── utils/
│       └── dicom_handler.py      # DICOM → PIL conversion
├── configs/
│   └── config.yaml               # Hyperparameters + AWS config
├── data/
│   ├── auc_results.csv           # Validation AUC scores
│   └── sample/                   # Sample X-rays (add your own)
├── deploy/
│   ├── docker/Dockerfile         # Container for EC2/ECS
│   ├── sagemaker/
│   │   ├── deploy_endpoint.py    # Deploy/delete SageMaker endpoint
│   │   └── inference_handler.py  # model_fn / predict_fn
│   └── terraform/main.tf         # IaC for AWS resources
├── docs/
│   └── ARCHITECTURE.md           # Detailed system design
├── models/                       # Model checkpoints (gitignored)
├── monitoring/
│   └── drift_detector.py         # Data drift detection
├── tests/
│   └── test_safety.py            # Safety layer test suite
├── .github/workflows/ci.yml      # Lint + test on push
├── .streamlit/config.toml        # Streamlit theme config
├── .env.example
├── Makefile                      # make run / test / lint / docker-build
├── pyproject.toml                # Tool config (black, ruff, pytest)
├── requirements.txt
└── LICENSE                       # MIT
```

---

## Training

```bash
# Fine-tune fusion head on CheXpert (BiomedCLIP weights frozen)
python src/pipeline/training.py \
    --data_dir data/chexpert \
    --output_dir models/ \
    --epochs 10 \
    --batch_size 32
```

**Data access:**
- CheXpert: [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (free)
- MIMIC-CXR: [physionet.org](https://physionet.org) (free, ~1-day approval)

**Training cost on AWS:** `ml.g4dn.xlarge` (T4, $0.74/hr) × ~5 hrs = **~$4**

---

## Deployment

```bash
# SageMaker real-time endpoint (T4 GPU)
make deploy-sagemaker

# Docker (local or EC2)
make docker-build && make docker-run

# Streamlit Cloud (free)
# Push to GitHub → share.streamlit.io → connect repo → add secrets
```

---

## Development

```bash
make install-dev   # Install dev tools (black, ruff, pytest)
make test          # Run test suite
make lint          # Ruff lint
make format        # Black format
make clean         # Remove __pycache__, .pyc, build artifacts
```

---

## Dataset Credits

**CheXpert** — Irvin et al., Stanford ML Group (2019)
> *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.*
> Used under the [CheXpert Dataset License](https://stanfordmlgroup.github.io/competitions/chexpert/) for non-commercial research purposes.

**Pre-trained models**
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Microsoft Research
- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Emily Alsentzer et al.

---

## Disclaimer

This system is for **research and educational purposes only**. It is not FDA-approved and must not be used for clinical decision-making without review by a licensed radiologist.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
