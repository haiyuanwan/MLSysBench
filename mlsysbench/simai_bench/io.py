"""Input/output helpers for SimAI benchmark task files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a benchmark config file is malformed."""


def load_structured(path: str | Path) -> Any:
    """Load a JSON or YAML file.

    YAML support is optional and requires PyYAML. JSON is the dependency-free
    format used by tests and bundled example tasks.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)

        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ConfigError(
                    f"Cannot load {path}: PyYAML is not installed. Use JSON or install PyYAML."
                ) from exc
            return yaml.safe_load(handle)

    raise ConfigError(f"Unsupported file type for {path}; expected .json, .yaml, or .yml")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_task_path(task_dir: Path, relative_path: str) -> Path:
    candidate = (task_dir / relative_path).resolve()
    task_root = task_dir.resolve()
    if task_root not in candidate.parents and candidate != task_root:
        raise ConfigError(f"Task path escapes task directory: {relative_path}")
    return candidate

