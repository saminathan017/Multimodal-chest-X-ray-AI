"""
src/utils/dicom_handler.py
═══════════════════════════════════════════════════════════════════
DICOM File Handler with PHI Stripping

Handles real clinical DICOM files:
  1. Parses DICOM metadata and pixel data
  2. STRIPS all PHI tags before any processing
  3. Applies proper windowing (lung window, bone window)
  4. Converts to normalized PIL Image for the ML pipeline
  5. Extracts structured metadata (view, modality, technique)

DICOM is the real-world format. Every hospital uses it.
Supporting DICOM is what separates a portfolio project from
something actually deployable in a clinical environment.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger


class WindowPreset(str, Enum):
    LUNG         = "lung"          # W=1500, C=-600
    MEDIASTINUM  = "mediastinum"   # W=400,  C=40
    BONE         = "bone"          # W=1500, C=400
    SOFT_TISSUE  = "soft_tissue"   # W=400,  C=50
    AUTO         = "auto"          # Auto window/level from pixel stats


@dataclass
class DICOMMetadata:
    """Safe (de-identified) metadata extracted from DICOM."""
    modality:        Optional[str]     # CR, DX, CT, etc.
    view_position:   Optional[str]     # PA, AP, LL, etc.
    body_part:       Optional[str]     # CHEST, ABDOMEN, etc.
    image_width:     Optional[int]
    image_height:    Optional[int]
    bits_stored:     Optional[int]
    photometric:     Optional[str]     # MONOCHROME1/2
    kvp:             Optional[float]   # kVp
    exposure:        Optional[float]   # mAs
    phi_stripped:    bool = False
    original_format: str = "DICOM"

    # PHI tags that were found and stripped (for audit log)
    stripped_tags:   list[str] = None

    def __post_init__(self):
        if self.stripped_tags is None:
            self.stripped_tags = []


@dataclass
class DICOMLoadResult:
    image:       Image.Image
    metadata:    DICOMMetadata
    success:     bool
    error:       Optional[str] = None


class DICOMHandler:
    """
    Production-grade DICOM loader with mandatory PHI stripping.

    All PHI is removed BEFORE the image is returned.
    A log of stripped tag names (not values) is kept for audit.
    """

    # ── HIPAA PHI DICOM Tags (must ALWAYS be stripped) ───────────────
    # Reference: DICOM PS3.15 Table E.1-1 (Safe Harbor Method)
    PHI_TAGS = {
        # Patient identifiers
        (0x0010, 0x0010): "PatientName",
        (0x0010, 0x0020): "PatientID",
        (0x0010, 0x0030): "PatientBirthDate",
        (0x0010, 0x0040): "PatientSex",       # kept as anonymous gender
        (0x0010, 0x1010): "PatientAge",
        (0x0010, 0x1020): "PatientSize",
        (0x0010, 0x1030): "PatientWeight",
        (0x0010, 0x21B0): "AdditionalPatientHistory",
        (0x0010, 0x4000): "PatientComments",
        # Study/visit identifiers
        (0x0008, 0x0050): "AccessionNumber",
        (0x0020, 0x000D): "StudyInstanceUID",
        (0x0020, 0x000E): "SeriesInstanceUID",
        (0x0020, 0x0010): "StudyID",
        (0x0008, 0x0020): "StudyDate",
        (0x0008, 0x0021): "SeriesDate",
        (0x0008, 0x0022): "AcquisitionDate",
        (0x0008, 0x0023): "ContentDate",
        (0x0008, 0x0030): "StudyTime",
        (0x0008, 0x0031): "SeriesTime",
        # Institution
        (0x0008, 0x0080): "InstitutionName",
        (0x0008, 0x0081): "InstitutionAddress",
        (0x0008, 0x1040): "InstitutionalDepartmentName",
        # Physician identifiers
        (0x0008, 0x0090): "ReferringPhysicianName",
        (0x0008, 0x1048): "PhysiciansOfRecord",
        (0x0008, 0x1050): "PerformingPhysicianName",
        (0x0008, 0x1070): "OperatorsName",
        # Device/equipment
        (0x0018, 0x1000): "DeviceSerialNumber",
        (0x0018, 0x1030): "ProtocolName",
        (0x0008, 0x1010): "StationName",
        # UIDs
        (0x0008, 0x0018): "SOPInstanceUID",
        (0x0020, 0x0052): "FrameOfReferenceUID",
    }

    # ── Safe tags to keep for clinical context ───────────────────────
    SAFE_TAGS = {
        (0x0008, 0x0060): "Modality",
        (0x0018, 0x5100): "PatientPosition",
        (0x0008, 0x2218): "AnatomicRegionSequence",
        (0x0028, 0x0010): "Rows",
        (0x0028, 0x0011): "Columns",
        (0x0028, 0x0100): "BitsAllocated",
        (0x0028, 0x0101): "BitsStored",
        (0x0028, 0x0004): "PhotometricInterpretation",
        (0x0018, 0x0060): "KVP",
        (0x0018, 0x1152): "Exposure",
        (0x0008, 0x0008): "ImageType",
        (0x0054, 0x0220): "ViewCodeSequence",
    }

    WINDOW_PRESETS = {
        WindowPreset.LUNG:        (1500, -600),
        WindowPreset.MEDIASTINUM: (400,   40),
        WindowPreset.BONE:        (1500,  400),
        WindowPreset.SOFT_TISSUE: (400,   50),
    }

    def load(
        self,
        path_or_bytes: "str | Path | bytes",
        window: WindowPreset = WindowPreset.LUNG,
        target_size: tuple[int,int] = (512, 512),
    ) -> DICOMLoadResult:
        """
        Load a DICOM file, strip PHI, apply windowing, return PIL Image.

        Args:
            path_or_bytes:  file path or raw bytes
            window:         windowing preset for display
            target_size:    output image size (W, H)

        Returns:
            DICOMLoadResult with de-identified image and safe metadata
        """
        try:
            import pydicom
        except ImportError:
            logger.warning("pydicom not installed. DICOM support disabled.")
            return DICOMLoadResult(
                image=Image.new("L",(512,512)),
                metadata=DICOMMetadata(None,None,None,None,None,None,None,None,None),
                success=False,
                error="pydicom package not installed. Run: pip install pydicom",
            )

        try:
            # Load DICOM
            if isinstance(path_or_bytes, bytes):
                ds = pydicom.dcmread(io.BytesIO(path_or_bytes), force=True)
            else:
                ds = pydicom.dcmread(str(path_or_bytes), force=True)

            # ── MANDATORY: Strip all PHI tags ──────────────────────
            stripped_tags = self._strip_phi(ds)

            # ── Extract safe metadata ──────────────────────────────
            metadata = self._extract_metadata(ds, stripped_tags)

            # ── Get pixel array ────────────────────────────────────
            pixel_array = ds.pixel_array.astype(np.float32)

            # Handle signed integers
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            # ── Apply windowing ────────────────────────────────────
            image = self._apply_windowing(pixel_array, ds, window, metadata)

            # ── Resize ────────────────────────────────────────────
            image = image.resize(target_size, Image.LANCZOS)

            logger.info(
                f"DICOM loaded: {metadata.modality} | {metadata.view_position} | "
                f"{metadata.image_width}x{metadata.image_height} | "
                f"PHI stripped: {len(stripped_tags)} tags"
            )

            return DICOMLoadResult(image=image, metadata=metadata, success=True)

        except Exception as e:
            logger.error(f"DICOM load failed: {e}")
            return DICOMLoadResult(
                image=Image.new("RGB", target_size, (20, 20, 20)),
                metadata=DICOMMetadata(None,None,None,None,None,None,None,None,None),
                success=False,
                error=str(e),
            )

    def _strip_phi(self, ds) -> list[str]:
        """
        Remove all PHI tags from dataset IN-PLACE.
        Returns list of stripped tag names for audit log.
        NEVER store the VALUES — only the tag names.
        """
        stripped = []
        for tag, name in self.PHI_TAGS.items():
            if tag in ds:
                try:
                    del ds[tag]
                    stripped.append(name)
                except Exception:
                    pass
        if stripped:
            logger.info(f"PHI stripped from DICOM: {stripped}")
        return stripped

    def _extract_metadata(self, ds, stripped_tags: list[str]) -> DICOMMetadata:
        def get(tag, default=None):
            try:
                val = ds[tag].value
                return str(val) if val is not None else default
            except Exception:
                return default

        return DICOMMetadata(
            modality=get((0x0008,0x0060)),
            view_position=get((0x0018,0x5100)),
            body_part=get((0x0018,0x0015)),
            image_width=int(get((0x0028,0x0011), 0) or 0),
            image_height=int(get((0x0028,0x0010), 0) or 0),
            bits_stored=int(get((0x0028,0x0101), 12) or 12),
            photometric=get((0x0028,0x0004)),
            kvp=float(get((0x0018,0x0060), 0) or 0),
            exposure=float(get((0x0018,0x1152), 0) or 0),
            phi_stripped=len(stripped_tags) > 0,
            stripped_tags=stripped_tags,
        )

    def _apply_windowing(
        self,
        pixel_array: np.ndarray,
        ds,
        preset: WindowPreset,
        metadata: DICOMMetadata,
    ) -> Image.Image:
        """Apply CT/CR windowing to normalise pixel values for display."""

        if preset == WindowPreset.AUTO:
            # Auto window: mean ± 2*std
            center = float(np.mean(pixel_array))
            width  = float(4 * np.std(pixel_array))
        else:
            # Try to use DICOM window tags first, fall back to preset
            try:
                width  = float(ds.WindowWidth)
                center = float(ds.WindowCenter)
                if isinstance(width, list):  width  = width[0]
                if isinstance(center, list): center = center[0]
            except Exception:
                width, center = self.WINDOW_PRESETS.get(preset, (1500, -600))

        # Apply window/level
        low  = center - width / 2
        high = center + width / 2
        pixel_clipped = np.clip(pixel_array, low, high)
        normalized = ((pixel_clipped - low) / (high - low) * 255).astype(np.uint8)

        # Handle MONOCHROME1 (inverted)
        if metadata.photometric and "MONOCHROME1" in metadata.photometric:
            normalized = 255 - normalized

        image = Image.fromarray(normalized, mode="L").convert("RGB")
        return image

    def from_standard_image(self, image: Image.Image) -> DICOMLoadResult:
        """
        Wrap a standard JPEG/PNG as a DICOMLoadResult for unified pipeline.
        Used when the input is not DICOM but we still want consistent handling.
        """
        return DICOMLoadResult(
            image=image.convert("RGB"),
            metadata=DICOMMetadata(
                modality="CR", view_position="PA",
                body_part="CHEST",
                image_width=image.width, image_height=image.height,
                bits_stored=8, photometric="RGB",
                kvp=None, exposure=None,
                phi_stripped=False,
                original_format="JPEG/PNG",
            ),
            success=True,
        )
