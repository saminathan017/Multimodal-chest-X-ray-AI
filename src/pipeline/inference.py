"""
src/pipeline/inference.py
─────────────────────────────────────────────────────────────────────
End-to-end inference pipeline:
  1. Preprocess X-ray image
  2. Encode image → features + GradCAM heatmap
  3. Encode clinical notes → features
  4. Fuse features → predictions
  5. Generate natural-language clinical report via AWS Bedrock
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import numpy as np
import torch
from PIL import Image
from loguru import logger

from src.models.image_encoder import ImageEncoder, XRayExplainer
from src.models.fusion_model import FusionModel
from src.models import get_text_encoder


# ── Output schema ────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    findings: list[dict]         # [{"label": str, "prob": float, "urgent": bool}]
    urgency_score: float         # 0–1
    heatmap: np.ndarray | None   # (224, 224, 3) uint8
    clinical_report: str         # LLM-generated narrative
    inference_time_ms: float
    clinical_entities: dict = field(default_factory=dict)
    raw_probs: list[float] = field(default_factory=list)

    @property
    def top_finding(self) -> str:
        if not self.findings:
            return "No significant findings"
        return self.findings[0]["label"]

    @property
    def urgency_label(self) -> str:
        if self.urgency_score >= 0.75:
            return "🔴 HIGH"
        elif self.urgency_score >= 0.45:
            return "🟡 MODERATE"
        return "🟢 LOW"


# ── Main pipeline class ──────────────────────────────────────────────
class ClinicalAIPipeline:
    """
    Orchestrates the full multi-modal inference pipeline.

    Args:
        config: dict loaded from configs/config.yaml
        device: "cuda" | "cpu"
    """

    LABELS: list[str] = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]

    def __init__(self, config: dict, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = config.get("inference", {}).get("confidence_threshold", 0.40)
        self.high_urgency_threshold = config.get("inference", {}).get("high_urgency_threshold", 0.75)

        logger.info(f"Initialising ClinicalAI pipeline on {self.device}")
        self._load_models()
        self._init_bedrock()

    # ── Model loading ────────────────────────────────────────────────
    def _load_models(self):
        model_cfg = self.config.get("models", {})

        self.image_encoder = ImageEncoder.from_pretrained(
            checkpoint_path=model_cfg.get("image", {}).get("checkpoint_path"),
            num_classes=len(self.LABELS),
            device=self.device,
        )

        text_cfg          = model_cfg.get("text", {})
        self.text_encoder = get_text_encoder(
            encoder_type    = text_cfg.get("encoder_type", "bert"),
            checkpoint_path = text_cfg.get("checkpoint_path"),
            output_dim      = 512,
            device          = self.device,
        )
        logger.info(f"Text encoder: {self.text_encoder}")

        self.fusion_model = FusionModel.from_pretrained(
            checkpoint_path=model_cfg.get("fusion", {}).get("checkpoint_path"),
            feat_dim=512,
            hidden_dim=256,
            num_classes=len(self.LABELS),
            device=self.device,
        )

        self.explainer = XRayExplainer(self.image_encoder, device=self.device)
        logger.info("All models loaded successfully")

    # ── AWS Bedrock for report generation ────────────────────────────
    def _init_bedrock(self):
        try:
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.config.get("aws", {}).get("region", "us-east-1"),
            )
            self._bedrock_available = True
            logger.info("AWS Bedrock client initialized")
        except Exception as e:
            logger.warning(f"Bedrock unavailable: {e}. Reports will use template fallback.")
            self.bedrock = None
            self._bedrock_available = False

    # ── Main inference entry point ───────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        clinical_note: str,
    ) -> PredictionResult:
        """
        Run full multi-modal inference.

        Args:
            image:         PIL Image (chest X-ray, any size/mode)
            clinical_note: free-text patient history & presentation

        Returns:
            PredictionResult with findings, heatmap, report
        """
        t0 = time.perf_counter()

        # 1. Preprocess image
        image_rgb = image.convert("RGB").resize((224, 224))
        image_tensor = self.image_encoder.preprocess_image(image_rgb).to(self.device)

        # Raw numpy for GradCAM overlay (float32, [0,1])
        raw_np = np.array(image_rgb).astype(np.float32) / 255.0

        # 2. Image encoding
        img_feat, img_logits = self.image_encoder(image_tensor)

        # 3. Text encoding
        txt_feat = self.text_encoder([clinical_note], device=self.device)

        # 4. Fusion
        fusion_out = self.fusion_model(img_feat, txt_feat)
        probs      = fusion_out["probs"].squeeze(0).cpu().numpy()   # (14,)
        urgency    = float(fusion_out["urgency"].squeeze().cpu())

        # 5. Build findings list (above threshold, sorted by probability)
        findings = []
        for idx, (label, prob) in enumerate(zip(self.LABELS, probs)):
            if prob >= self.conf_threshold:
                findings.append({
                    "label":  label,
                    "prob":   round(float(prob), 3),
                    "urgent": prob >= self.high_urgency_threshold,
                    "class_idx": idx,
                })
        findings.sort(key=lambda x: x["prob"], reverse=True)

        # If nothing above threshold, report "No significant findings"
        if not findings:
            findings = [{"label": "No Finding", "prob": float(probs[0]), "urgent": False, "class_idx": 0}]

        # 6. GradCAM heatmap for top finding
        heatmap = None
        if findings:
            top_class = findings[0]["class_idx"]
            heatmap = self.explainer.generate_heatmap(image_tensor, raw_np, top_class)

        # 7. Clinical entity extraction
        entities = self.text_encoder.extract_clinical_entities(clinical_note)

        # 8. Generate natural-language report
        report = self._generate_report(findings, urgency, clinical_note, entities)

        elapsed = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            findings=findings,
            urgency_score=urgency,
            heatmap=heatmap,
            clinical_report=report,
            inference_time_ms=round(elapsed, 1),
            clinical_entities=entities,
            raw_probs=probs.tolist(),
        )

    # ── Report generation via AWS Bedrock ────────────────────────────
    def _generate_report(
        self,
        findings: list[dict],
        urgency: float,
        clinical_note: str,
        entities: dict,
    ) -> str:
        findings_text = "\n".join(
            f"  - {f['label']}: {f['prob']*100:.1f}% confidence"
            for f in findings
        )
        urgency_label = "HIGH" if urgency >= 0.75 else ("MODERATE" if urgency >= 0.45 else "LOW")

        prompt = f"""You are an expert radiologist AI assistant. Based on the following chest X-ray analysis results and patient history, generate a concise, structured clinical radiology report.

Patient Clinical Notes:
{clinical_note}

AI-Detected Findings (above threshold):
{findings_text}

Overall Urgency Level: {urgency_label} ({urgency:.2f})

Instructions:
- Write in professional radiological language
- Structure as: CLINICAL INDICATION, FINDINGS, IMPRESSION, RECOMMENDATION
- Be concise (150-200 words max)
- Always include the disclaimer: "This report was generated by AI and must be reviewed by a licensed radiologist."
- Do NOT make definitive diagnoses — use qualifying language ("suggests", "consistent with", "cannot exclude")

Generate the report:"""

        if self._bedrock_available and self.bedrock:
            try:
                model_id = self.config.get("report", {}).get(
                    "bedrock_model_id", "anthropic.claude-3-haiku-20240307-v1:0"
                )
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                })
                response = self.bedrock.invoke_model(body=body, modelId=model_id)
                result = json.loads(response["body"].read())
                return result["content"][0]["text"]
            except Exception as e:
                logger.warning(f"Bedrock call failed: {e}. Using template report.")

        # ── Fallback template report ──────────────────────────────────
        top_finding = findings[0]["label"] if findings else "No significant abnormality"
        return f"""CLINICAL INDICATION:
{clinical_note[:200]}...

FINDINGS:
Chest X-ray reviewed with AI assistance. The analysis suggests {top_finding.lower()}
as the primary finding (confidence: {findings[0]['prob']*100:.1f}% if findings else 'N/A').
{'Additional findings noted: ' + ', '.join(f['label'] for f in findings[1:3]) + '.' if len(findings) > 1 else ''}

IMPRESSION:
AI analysis indicates {urgency_label} urgency. Findings are {top_finding.lower()} pattern.

RECOMMENDATION:
{"Immediate clinical review recommended." if urgency >= 0.75 else "Correlation with clinical findings recommended."}

⚠️ DISCLAIMER: This report was generated by AI and must be reviewed by a licensed radiologist."""
