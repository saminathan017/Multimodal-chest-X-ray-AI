"""Dataset registry and label cleaning policies for clinical validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    modality: str
    split_strategy: str = "patient"
    label_policy: str = "u_zero"
    has_reports: bool = False
    supports_lateral: bool = False
    source_site: str | None = None


class DatasetRegistry:
    def __init__(self, path: str | Path = "data/dataset_registry.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, DatasetSpec] = {}
        if self.path.exists():
            self._load()

    def register(self, spec: DatasetSpec) -> None:
        self._items[spec.name] = spec
        self.save()

    def get(self, name: str) -> DatasetSpec:
        return self._items[name]

    def list(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._items.values()]

    def save(self) -> None:
        self.path.write_text(json.dumps(self.list(), indent=2))

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        self._items = {item["name"]: DatasetSpec(**item) for item in raw}


def clean_uncertain_label(value, policy: str = "u_zero") -> float | None:
    if value is None or value == "":
        return 0.0
    val = float(value)
    if val == -1:
        if policy == "u_one":
            return 1.0
        if policy == "u_ignore":
            return None
        return 0.0
    return 1.0 if val > 0 else 0.0


def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec("CheXpert", "data/chexpert", "CXR", has_reports=False, supports_lateral=True, source_site="Stanford"),
        DatasetSpec("MIMIC-CXR", "data/mimic", "CXR", has_reports=True, supports_lateral=True, source_site="BIDMC"),
        DatasetSpec("NIH ChestXray14", "data/nih", "CXR", has_reports=False, supports_lateral=False, source_site="NIH"),
        DatasetSpec("PadChest", "data/padchest", "CXR", has_reports=True, supports_lateral=True, source_site="BIMCV"),
    ]

