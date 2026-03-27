"""
src/safety/input_validator.py
═══════════════════════════════════════════════════════════════════
Clinical Input Validation & Adversarial Image Detection

Validates:
  1. Image quality (resolution, contrast, noise, exposure)
  2. Image authenticity (adversarial perturbation detection)
  3. Non-medical image detection (face photos, documents, etc.)
  4. Clinical note quality (length, language, completeness)
  5. Payload safety (size limits, format checks)

Any input that fails validation is REJECTED before reaching the
ML pipeline — protecting both the model and the patient.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image, ImageStat
from loguru import logger


class ValidationStatus(str, Enum):
    PASS    = "PASS"
    WARN    = "WARN"
    REJECT  = "REJECT"


@dataclass
class ValidationIssue:
    code:     str
    message:  str
    severity: ValidationStatus


@dataclass
class ValidationResult:
    status:     ValidationStatus
    issues:     list[ValidationIssue]
    quality_score: float              # 0.0 – 1.0, higher is better
    metadata:   dict

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.REJECT

    @property
    def warnings(self) -> list[str]:
        return [i.message for i in self.issues if i.severity == ValidationStatus.WARN]

    @property
    def errors(self) -> list[str]:
        return [i.message for i in self.issues if i.severity == ValidationStatus.REJECT]


# ── Image Validator ──────────────────────────────────────────────────
class ImageValidator:
    """
    Multi-layer image quality and safety validator for chest X-rays.
    Catches corrupt files, non-medical images, and adversarial inputs.
    """

    # Quality thresholds
    MIN_WIDTH           = 224
    MIN_HEIGHT          = 224
    MAX_FILE_SIZE_MB    = 50
    MIN_CONTRAST_STD    = 15.0     # pixel std dev — too low = blank/overexposed
    MAX_CONTRAST_STD    = 120.0    # too high = likely not an X-ray
    MAX_MEAN_BRIGHTNESS = 240.0    # too bright = overexposed
    MIN_MEAN_BRIGHTNESS = 10.0     # too dark = underexposed
    ADVERSARIAL_NOISE_THRESHOLD = 0.045   # high-freq noise energy ratio

    def validate(self, image: Image.Image, file_bytes: Optional[bytes] = None) -> ValidationResult:
        issues: list[ValidationIssue] = []
        metadata: dict = {}

        try:
            img_array = np.array(image.convert("L")).astype(np.float32)
            h, w = img_array.shape
            metadata["width"]  = w
            metadata["height"] = h
            metadata["mode"]   = image.mode

            # ── 1. Minimum resolution ──────────────────────────────
            if w < self.MIN_WIDTH or h < self.MIN_HEIGHT:
                issues.append(ValidationIssue(
                    code="IMG_LOW_RES",
                    message=f"Image too small ({w}x{h}). Minimum: {self.MIN_WIDTH}x{self.MIN_HEIGHT}px.",
                    severity=ValidationStatus.REJECT,
                ))

            # ── 2. File size ───────────────────────────────────────
            if file_bytes:
                size_mb = len(file_bytes) / (1024 * 1024)
                metadata["file_size_mb"] = round(size_mb, 2)
                if size_mb > self.MAX_FILE_SIZE_MB:
                    issues.append(ValidationIssue(
                        code="IMG_TOO_LARGE",
                        message=f"File size {size_mb:.1f}MB exceeds limit of {self.MAX_FILE_SIZE_MB}MB.",
                        severity=ValidationStatus.REJECT,
                    ))

            # ── 3. Brightness / exposure check ────────────────────
            mean_brightness = float(np.mean(img_array))
            std_brightness  = float(np.std(img_array))
            metadata["mean_brightness"] = round(mean_brightness, 2)
            metadata["std_brightness"]  = round(std_brightness, 2)

            if mean_brightness > self.MAX_MEAN_BRIGHTNESS:
                issues.append(ValidationIssue(
                    code="IMG_OVEREXPOSED",
                    message="Image appears overexposed. Analysis quality may be reduced.",
                    severity=ValidationStatus.WARN,
                ))
            elif mean_brightness < self.MIN_MEAN_BRIGHTNESS:
                issues.append(ValidationIssue(
                    code="IMG_UNDEREXPOSED",
                    message="Image appears underexposed. Analysis quality may be reduced.",
                    severity=ValidationStatus.WARN,
                ))

            # ── 4. Contrast check ──────────────────────────────────
            if std_brightness < self.MIN_CONTRAST_STD:
                issues.append(ValidationIssue(
                    code="IMG_LOW_CONTRAST",
                    message=f"Image has very low contrast (σ={std_brightness:.1f}). May be a blank or corrupted image.",
                    severity=ValidationStatus.REJECT,
                ))
            elif std_brightness > self.MAX_CONTRAST_STD:
                issues.append(ValidationIssue(
                    code="IMG_NOT_XRAY",
                    message=f"Image contrast pattern (σ={std_brightness:.1f}) is inconsistent with a chest X-ray.",
                    severity=ValidationStatus.WARN,
                ))

            # ── 5. Adversarial perturbation detection ──────────────
            # High-frequency noise analysis using Laplacian filter
            adversarial_score = self._compute_adversarial_score(img_array)
            metadata["adversarial_score"] = round(adversarial_score, 4)

            if adversarial_score > self.ADVERSARIAL_NOISE_THRESHOLD:
                issues.append(ValidationIssue(
                    code="IMG_ADVERSARIAL_SUSPECTED",
                    message=(
                        f"Unusually high-frequency noise detected (score={adversarial_score:.4f}). "
                        "Image may contain adversarial perturbations. Results should be treated with caution."
                    ),
                    severity=ValidationStatus.WARN,
                ))
                logger.warning(f"Adversarial input suspected — noise score: {adversarial_score:.4f}")

            # ── 6. Aspect ratio check ──────────────────────────────
            aspect = w / h
            metadata["aspect_ratio"] = round(aspect, 2)
            if aspect < 0.5 or aspect > 2.5:
                issues.append(ValidationIssue(
                    code="IMG_UNUSUAL_ASPECT",
                    message=f"Unusual aspect ratio ({aspect:.2f}). Standard X-rays are close to 1:1.",
                    severity=ValidationStatus.WARN,
                ))

            # ── 7. Compute overall quality score ───────────────────
            quality_score = self._compute_quality_score(
                std_brightness, adversarial_score, w, h
            )
            metadata["quality_score"] = round(quality_score, 3)

        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            issues.append(ValidationIssue(
                code="IMG_PARSE_ERROR",
                message=f"Could not parse image: {str(e)}",
                severity=ValidationStatus.REJECT,
            ))
            quality_score = 0.0

        # ── Determine overall status ───────────────────────────────
        if any(i.severity == ValidationStatus.REJECT for i in issues):
            status = ValidationStatus.REJECT
        elif any(i.severity == ValidationStatus.WARN for i in issues):
            status = ValidationStatus.WARN
        else:
            status = ValidationStatus.PASS

        return ValidationResult(
            status=status, issues=issues,
            quality_score=quality_score if issues or True else 1.0,
            metadata=metadata,
        )

    def _compute_adversarial_score(self, img: np.ndarray) -> float:
        """
        Computes ratio of high-frequency energy to total energy using
        a discrete Laplacian approximation. Adversarial perturbations
        typically inject imperceptible high-freq noise.
        """
        try:
            # Laplacian kernel
            laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
            from scipy.ndimage import convolve
            hf = convolve(img, laplacian)
            total_energy = np.sum(img**2) + 1e-9
            hf_energy    = np.sum(hf**2)
            return float(hf_energy / total_energy)
        except Exception:
            # scipy unavailable — use numpy fallback
            diff_h = np.diff(img, axis=0)
            diff_v = np.diff(img, axis=1)
            return float((np.sum(diff_h**2) + np.sum(diff_v**2)) / (np.sum(img**2) + 1e-9))

    def _compute_quality_score(
        self, std: float, adv_score: float, w: int, h: int
    ) -> float:
        # Resolution score (0–1, saturates at 512px)
        res_score = min(1.0, (w * h) / (512 * 512))
        # Contrast score (optimal range 25–80)
        contrast_score = max(0.0, 1.0 - abs(std - 52.5) / 52.5)
        # Adversarial penalty
        adv_penalty = min(1.0, adv_score / self.ADVERSARIAL_NOISE_THRESHOLD)
        # Combined (weighted)
        score = 0.4*res_score + 0.4*contrast_score + 0.2*(1.0 - adv_penalty)
        return max(0.0, min(1.0, score))


# ── Clinical Note Validator ──────────────────────────────────────────
class ClinicalNoteValidator:
    """
    Validates clinical note quality and completeness before processing.
    """
    MIN_LENGTH        = 20
    MAX_LENGTH        = 5000
    MIN_WORD_COUNT    = 5
    GIBBERISH_PATTERN = r"^[^a-zA-Z]*$"   # All non-letters = reject

    # Required context keywords (at least 1 must be present for medical plausibility)
    MEDICAL_KEYWORDS = [
        "pain", "fever", "cough", "dyspnea", "breath", "chest",
        "patient", "history", "year", "male", "female", "pmh",
        "medication", "vitals", "oxygen", "sat", "temperature",
        "history", "presenting", "complaint", "symptom", "diagnosis",
        "old", "yo", "pre-op", "post-op", "chronic", "acute",
    ]

    def validate(self, note: str) -> ValidationResult:
        issues:   list[ValidationIssue] = []
        metadata: dict = {}

        if not note:
            return ValidationResult(
                status=ValidationStatus.REJECT,
                issues=[ValidationIssue("NOTE_EMPTY","Clinical note is empty.", ValidationStatus.REJECT)],
                quality_score=0.0, metadata={},
            )

        note_stripped = note.strip()
        word_count    = len(note_stripped.split())
        char_count    = len(note_stripped)
        metadata.update({"word_count": word_count, "char_count": char_count})

        # Length checks
        if char_count < self.MIN_LENGTH:
            issues.append(ValidationIssue(
                "NOTE_TOO_SHORT",
                f"Note too short ({char_count} chars). Minimum {self.MIN_LENGTH} required.",
                ValidationStatus.REJECT,
            ))
        if char_count > self.MAX_LENGTH:
            issues.append(ValidationIssue(
                "NOTE_TOO_LONG",
                f"Note exceeds maximum length ({char_count} chars). Truncate to {self.MAX_LENGTH}.",
                ValidationStatus.WARN,
            ))

        # Word count
        if word_count < self.MIN_WORD_COUNT:
            issues.append(ValidationIssue(
                "NOTE_TOO_FEW_WORDS",
                f"Note has too few words ({word_count}). Please provide more clinical context.",
                ValidationStatus.WARN,
            ))

        # Medical plausibility (at least 1 keyword)
        note_lower = note_stripped.lower()
        found_kws  = [kw for kw in self.MEDICAL_KEYWORDS if kw in note_lower]
        metadata["medical_keywords_found"] = found_kws[:5]

        if not found_kws:
            issues.append(ValidationIssue(
                "NOTE_NOT_MEDICAL",
                "Note does not appear to contain medical content. Please enter patient clinical history.",
                ValidationStatus.WARN,
            ))

        # Gibberish / all-numeric check
        if re.match(self.GIBBERISH_PATTERN, note_stripped):
            issues.append(ValidationIssue(
                "NOTE_GIBBERISH",
                "Note contains no readable text.",
                ValidationStatus.REJECT,
            ))

        # Quality score
        kw_score   = min(1.0, len(found_kws) / 3)
        len_score  = min(1.0, char_count / 200)
        quality    = 0.5*kw_score + 0.5*len_score
        metadata["quality_score"] = round(quality, 3)

        status = (ValidationStatus.REJECT if any(i.severity==ValidationStatus.REJECT for i in issues)
                  else ValidationStatus.WARN if issues
                  else ValidationStatus.PASS)

        return ValidationResult(status=status, issues=issues,
                                quality_score=quality, metadata=metadata)


import re   # needed for ClinicalNoteValidator


# ── Combined validator ────────────────────────────────────────────────
class InputValidator:
    """Single entry point for all input validation."""

    def __init__(self):
        self.image_validator = ImageValidator()
        self.note_validator  = ClinicalNoteValidator()

    def validate_all(
        self,
        image:    Image.Image,
        note:     str,
        file_bytes: Optional[bytes] = None,
    ) -> tuple[ValidationResult, ValidationResult]:
        """
        Returns (image_result, note_result).
        Call .is_valid on each before proceeding to inference.
        """
        img_result  = self.image_validator.validate(image, file_bytes)
        note_result = self.note_validator.validate(note)

        if not img_result.is_valid:
            logger.warning(f"Image validation REJECTED: {img_result.errors}")
        if not note_result.is_valid:
            logger.warning(f"Note validation REJECTED: {note_result.errors}")

        return img_result, note_result
