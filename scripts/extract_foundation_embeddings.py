"""
Extract smoke-test embeddings with the runnable foundation models.

Examples:
    python3 scripts/extract_foundation_embeddings.py --demo
    python3 scripts/extract_foundation_embeddings.py --image path/to/cxr.png --text "portable chest x-ray..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.models.foundation_extractors import BiomedCLIPExtractor, CXRBertExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Optional image path.")
    parser.add_argument("--text", default="No acute cardiopulmonary abnormality.", help="Clinical/radiology text.")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic grayscale image.")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), (72, 72, 72))

    biomed = BiomedCLIPExtractor(device="cpu")
    cxr_bert = CXRBertExtractor(device="cpu")

    results = {
        "biomedclip_image": biomed.encode_image(image).summary(),
        "biomedclip_text": biomed.encode_text([args.text]).summary(),
        "cxr_bert_text": cxr_bert.encode_text([args.text]).summary(),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

