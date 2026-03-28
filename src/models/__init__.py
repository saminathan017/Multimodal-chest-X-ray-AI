from .image_encoder      import ImageEncoder, XRayExplainer
from .text_encoder       import TextEncoder
from .bart_text_encoder  import BartClinicalEncoder
from .fusion_model       import FusionModel


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
    encoders = {
        "bart": BartClinicalEncoder,
        "bert": TextEncoder,
    }
    key = encoder_type.lower().strip()
    if key not in encoders:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: {list(encoders.keys())}"
        )
    return encoders[key].from_pretrained(**kwargs)


__all__ = [
    "ImageEncoder",
    "XRayExplainer",
    "TextEncoder",
    "BartClinicalEncoder",
    "FusionModel",
    "get_text_encoder",
]
