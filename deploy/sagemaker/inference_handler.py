"""
deploy/sagemaker/inference_handler.py
─────────────────────────────────────────────────────────────────────
SageMaker inference handler (model_fn + predict_fn + output_fn).
This file is the entrypoint for the SageMaker endpoint container.
─────────────────────────────────────────────────────────────────────
"""

import io
import json
import base64
import yaml
from pathlib import Path
from PIL import Image

from src.pipeline.inference import ClinicalAIPipeline


# ── Called once when container starts ───────────────────────────────
def model_fn(model_dir: str):
    config_path = Path(model_dir) / "configs" / "config.yaml"
    config = yaml.safe_load(open(config_path)) if config_path.exists() else {}
    pipeline = ClinicalAIPipeline(config, device="cuda")
    return pipeline


# ── Called on every request ──────────────────────────────────────────
def input_fn(request_body, content_type: str = "application/json"):
    data = json.loads(request_body)

    # Image: base64-encoded PNG/JPEG
    image_bytes = base64.b64decode(data["image_b64"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    clinical_note = data.get("clinical_note", "No clinical history provided.")
    return image, clinical_note


def predict_fn(input_data, model):
    image, clinical_note = input_data
    result = model.predict(image, clinical_note)
    return result


def output_fn(prediction, accept: str = "application/json"):
    import numpy as np

    output = {
        "findings": prediction.findings,
        "urgency_score": prediction.urgency_score,
        "urgency_label": prediction.urgency_label,
        "clinical_report": prediction.clinical_report,
        "inference_time_ms": prediction.inference_time_ms,
        "clinical_entities": prediction.clinical_entities,
        # Heatmap as base64 PNG
        "heatmap_b64": _array_to_b64(prediction.heatmap) if prediction.heatmap is not None else None,
    }
    return json.dumps(output), accept


def _array_to_b64(arr: "np.ndarray") -> str:
    import numpy as np
    img = Image.fromarray(arr.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
