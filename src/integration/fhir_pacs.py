"""
FHIR/PACS integration scaffolding.

These helpers produce de-identified, standards-shaped metadata for workflow
handoff. They do not connect to a hospital network yet; they define the stable
contract that future SMART-on-FHIR, HL7, DICOMweb, and PACS connectors can use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FHIRImport:
    patient_id: str
    encounter_id: str | None = None
    imaging_study_id: str | None = None
    accession_number: str | None = None
    modality: str = "CR"
    body_site: str = "Chest"
    study_description: str = "Chest radiograph"

    def to_integration_metadata(self) -> dict[str, Any]:
        return {
            "source": "FHIR",
            "patient_hash": stable_hash(self.patient_id),
            "encounter_hash": stable_hash(self.encounter_id) if self.encounter_id else None,
            "imaging_study_id": self.imaging_study_id,
            "accession_number": self.accession_number,
            "modality": self.modality,
            "body_site": self.body_site,
            "study_description": self.study_description,
        }


@dataclass(frozen=True)
class PACSImport:
    study_instance_uid: str
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    accession_number: str | None = None
    aetitle: str | None = None
    modality: str = "DX"

    def to_integration_metadata(self) -> dict[str, Any]:
        return {
            "source": "PACS",
            "study_uid_hash": stable_hash(self.study_instance_uid),
            "series_uid_hash": stable_hash(self.series_instance_uid) if self.series_instance_uid else None,
            "sop_uid_hash": stable_hash(self.sop_instance_uid) if self.sop_instance_uid else None,
            "accession_number": self.accession_number,
            "aetitle": self.aetitle,
            "modality": self.modality,
        }


def build_diagnostic_report_resource(
    *,
    case_id: str,
    report_text: str,
    structured_findings: list[dict[str, Any]],
    integration: dict[str, Any],
) -> dict[str, Any]:
    """Build a minimal FHIR DiagnosticReport-shaped export payload."""
    return {
        "resourceType": "DiagnosticReport",
        "id": case_id,
        "status": "preliminary",
        "category": [{"text": "Radiology"}],
        "code": {"text": integration.get("study_description", "Chest radiograph AI-assisted report")},
        "subject": {"identifier": {"value": integration.get("patient_hash", "deidentified")}},
        "imagingStudy": [{"identifier": {"value": integration.get("imaging_study_id") or integration.get("study_uid_hash")}}],
        "conclusion": report_text,
        "result": [
            {
                "display": finding.get("label"),
                "status": finding.get("status"),
                "probability": finding.get("probability"),
                "severity": finding.get("severity"),
                "location": finding.get("location"),
                "laterality": finding.get("laterality"),
            }
            for finding in structured_findings
        ],
    }

