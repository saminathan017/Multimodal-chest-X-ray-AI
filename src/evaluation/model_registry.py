"""Lightweight model registry for checkpoints, validation metrics, and thresholds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_version: str
    checkpoint_uri: str
    created_at: str
    datasets: list[str]
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Any]
    status: str = "candidate"
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, path: str | Path = "models/model_registry.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, ModelRegistryEntry] = {}
        if self.path.exists():
            self._load()

    def register(
        self,
        *,
        model_version: str,
        checkpoint_uri: str,
        datasets: list[str],
        metrics: dict[str, Any],
        calibration: dict[str, Any],
        thresholds: dict[str, Any],
        status: str = "candidate",
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRegistryEntry:
        entry = ModelRegistryEntry(
            model_version=model_version,
            checkpoint_uri=checkpoint_uri,
            created_at=datetime.now(UTC).isoformat(),
            datasets=datasets,
            metrics=metrics,
            calibration=calibration,
            thresholds=thresholds,
            status=status,
            notes=notes,
            metadata=metadata or {},
        )
        self._entries[model_version] = entry
        self.save()
        return entry

    def promote(self, model_version: str) -> ModelRegistryEntry:
        entry = self._entries[model_version]
        promoted = ModelRegistryEntry(**{**asdict(entry), "status": "production"})
        for version, current in list(self._entries.items()):
            if current.status == "production":
                self._entries[version] = ModelRegistryEntry(**{**asdict(current), "status": "archived"})
        self._entries[model_version] = promoted
        self.save()
        return promoted

    def latest(self, status: str | None = None) -> ModelRegistryEntry | None:
        entries = list(self._entries.values())
        if status:
            entries = [entry for entry in entries if entry.status == status]
        if not entries:
            return None
        return sorted(entries, key=lambda item: item.created_at, reverse=True)[0]

    def list(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in sorted(self._entries.values(), key=lambda item: item.created_at, reverse=True)]

    def save(self) -> None:
        self.path.write_text(json.dumps(self.list(), indent=2))

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        self._entries = {item["model_version"]: ModelRegistryEntry(**item) for item in raw}

