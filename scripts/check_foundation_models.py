"""
Check and optionally load/download foundation models used by ClinicalAI.

Usage:
    python3 scripts/check_foundation_models.py --metadata-only
    python3 scripts/check_foundation_models.py --download
    python3 scripts/check_foundation_models.py --load-biomedclip
    python3 scripts/check_foundation_models.py --load-cxr-bert
    python3 scripts/check_foundation_models.py --load-google-cxr

Notes:
    google/cxr-foundation may require accepting model terms and/or a Hugging Face
    token depending on the current model card settings. Its model files are
    TensorFlow SavedModels, not PyTorch checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_IDS = {
    "google_cxr_foundation": "google/cxr-foundation",
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "cxr_bert": "microsoft/BiomedVLP-CXR-BERT-specialized",
}


def model_metadata(model_id: str) -> dict:
    api = HfApi()
    info = api.model_info(model_id)
    siblings = [s.rfilename for s in info.siblings or []]
    return {
        "model_id": model_id,
        "sha": info.sha,
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "tags": info.tags,
        "files": siblings[:25],
        "file_count": len(siblings),
    }


def download_model(model_id: str, output_dir: Path) -> str:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir / model_id.replace("/", "__"),
        local_dir_use_symlinks=False,
    )
    return path


def load_biomedclip(model_id: str) -> dict:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:" + model_id,
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:" + model_id)
    prompts = tokenizer(["chest x-ray showing pneumonia", "normal chest x-ray"])
    image = preprocess(Image.new("RGB", (224, 224), (70, 70, 70))).unsqueeze(0)
    with torch.no_grad():
        text_features = model.encode_text(prompts)
        image_features = model.encode_image(image)
    return {
        "loaded": True,
        "model_id": model_id,
        "text_feature_shape": list(text_features.shape),
        "image_feature_shape": list(image_features.shape),
        "preprocess": str(preprocess),
    }


def load_cxr_bert(model_id: str) -> dict:
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    encoded = tokenizer("No acute cardiopulmonary abnormality.", return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    hidden = getattr(output, "last_hidden_state", None)
    return {
        "loaded": True,
        "model_id": model_id,
        "hidden_shape": list(hidden.shape) if hidden is not None else None,
    }


def load_google_cxr_foundation(model_id: str, output_dir: Path) -> dict:
    try:
        import tensorflow as tf
    except Exception as exc:
        return {
            "loaded": False,
            "model_id": model_id,
            "error": (
                "TensorFlow is required for google/cxr-foundation SavedModel loading. "
                f"Import failed: {exc}"
            ),
        }

    try:
        path = download_model(model_id, output_dir)
        model_path = Path(path) / "elixr-c-v2-pooled"
        model = tf.saved_model.load(str(model_path))
        return {
            "loaded": True,
            "model_id": model_id,
            "snapshot_path": path,
            "saved_model_path": str(model_path),
            "signatures": list(model.signatures.keys()),
        }
    except Exception as exc:
        return {
            "loaded": False,
            "model_id": model_id,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true", help="Only fetch HF metadata.")
    parser.add_argument("--download", action="store_true", help="Download snapshots into models/hf.")
    parser.add_argument("--load-biomedclip", action="store_true", help="Instantiate BiomedCLIP on CPU.")
    parser.add_argument("--load-cxr-bert", action="store_true", help="Instantiate CXR-BERT on CPU.")
    parser.add_argument("--load-google-cxr", action="store_true", help="Instantiate Google CXR Foundation if TensorFlow/access are available.")
    parser.add_argument("--output", default="models/hf", help="Snapshot output directory.")
    args = parser.parse_args()

    results = {"metadata": {}, "downloads": {}, "loads": {}}
    for key, model_id in MODEL_IDS.items():
        try:
            results["metadata"][key] = model_metadata(model_id)
        except Exception as exc:
            results["metadata"][key] = {"model_id": model_id, "error": str(exc)}

    if args.download:
        output_dir = Path(args.output)
        for key, model_id in MODEL_IDS.items():
            try:
                results["downloads"][key] = download_model(model_id, output_dir)
            except Exception as exc:
                results["downloads"][key] = {"error": str(exc)}

    if args.load_biomedclip:
        try:
            results["loads"]["biomedclip"] = load_biomedclip(MODEL_IDS["biomedclip"])
        except Exception as exc:
            results["loads"]["biomedclip"] = {"error": str(exc)}

    if args.load_cxr_bert:
        try:
            results["loads"]["cxr_bert"] = load_cxr_bert(MODEL_IDS["cxr_bert"])
        except Exception as exc:
            results["loads"]["cxr_bert"] = {"error": str(exc)}

    if args.load_google_cxr:
        results["loads"]["google_cxr_foundation"] = load_google_cxr_foundation(
            MODEL_IDS["google_cxr_foundation"],
            Path(args.output),
        )

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
