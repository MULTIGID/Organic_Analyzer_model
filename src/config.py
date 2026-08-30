from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    project_root: Path

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    def path(self, section: str, key: str) -> Path:
        value = Path(self.raw[section][key])
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required = {"project", "data", "model", "training", "paths"}
    missing = required.difference(raw or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return AppConfig(raw=raw, project_root=config_path.parent)
