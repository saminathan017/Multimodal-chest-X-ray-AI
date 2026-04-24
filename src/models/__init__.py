"""Model package exports with lazy heavyweight imports."""

from __future__ import annotations


def get_text_encoder(
    encoder_type: str = "bert",
    **kwargs,
) -> "TextEncoder | BartClinicalEncoder":
    """
    Factory — return the right text encoder by name.

    Args:
        encoder_type: "bart"  → BartClinicalEncoder (BioBART encoder-only)
                      "bert"  → TextEncoder         (Bio_ClinicalBERT)
        **kwargs:     Forwarded to the encoder's from_pretrained() method.

    Usage:
        enc = get_text_encoder("bart", output_dim=512, device="cuda")
        enc = get_text_encoder("bert", output_dim=512, device="cpu")
    """
    key = encoder_type.lower().strip()
    if key == "bart":
        from .bart_text_encoder import BartClinicalEncoder

        return BartClinicalEncoder.from_pretrained(**kwargs)
    if key == "bert":
        from .text_encoder import TextEncoder

        return TextEncoder.from_pretrained(**kwargs)
    if key not in {"bart", "bert"}:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            "Choose from: ['bart', 'bert']"
        )


def __getattr__(name: str):
    if name in {"ImageEncoder", "XRayExplainer"}:
        from .image_encoder import ImageEncoder, XRayExplainer

        return {"ImageEncoder": ImageEncoder, "XRayExplainer": XRayExplainer}[name]
    if name == "TextEncoder":
        from .text_encoder import TextEncoder

        return TextEncoder
    if name == "BartClinicalEncoder":
        from .bart_text_encoder import BartClinicalEncoder

        return BartClinicalEncoder
    if name == "FusionModel":
        from .fusion_model import FusionModel

        return FusionModel
    if name in {"FoundationModelV2Spec", "MultiTaskHeadSpec"}:
        from .foundation_v2 import FoundationModelV2Spec, MultiTaskHeadSpec

        return {
            "FoundationModelV2Spec": FoundationModelV2Spec,
            "MultiTaskHeadSpec": MultiTaskHeadSpec,
        }[name]
    if name in {"BiomedCLIPExtractor", "CXRBertExtractor", "GoogleCXRFoundationExtractor"}:
        from .foundation_extractors import (
            BiomedCLIPExtractor,
            CXRBertExtractor,
            GoogleCXRFoundationExtractor,
        )

        return {
            "BiomedCLIPExtractor": BiomedCLIPExtractor,
            "CXRBertExtractor": CXRBertExtractor,
            "GoogleCXRFoundationExtractor": GoogleCXRFoundationExtractor,
        }[name]
    raise AttributeError(name)


__all__ = [
    "ImageEncoder",
    "XRayExplainer",
    "TextEncoder",
    "BartClinicalEncoder",
    "FusionModel",
    "FoundationModelV2Spec",
    "MultiTaskHeadSpec",
    "BiomedCLIPExtractor",
    "CXRBertExtractor",
    "GoogleCXRFoundationExtractor",
    "get_text_encoder",
]
