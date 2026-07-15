"""Submission validation and config merging."""

from __future__ import annotations

from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured
from mlsysbench.simai_bench.schema import ActionSpec


FORBIDDEN_ACTIONS = {
    "replica_config_expert_model_parallel_size",
    "random_forrest_execution_time_predictor_config_simai_simulation_topo",
    "model_config_path",
}


def load_submission(path: str) -> dict[str, Any]:
    data = load_structured(path)
    if not isinstance(data, dict):
        raise ConfigError("submission must contain an object")
    changes = data.get("changes")
    if not isinstance(changes, dict):
        raise ConfigError("submission must include a changes object")
    files = data.get("files")
    if files is not None and (
        not isinstance(files, dict)
        or not all(isinstance(name, str) and isinstance(content, str) for name, content in files.items())
    ):
        raise ConfigError("submission.files must map relative file paths to UTF-8 text")
    return data


def validate_changes(
    changes: dict[str, Any],
    allowed_actions: dict[str, ActionSpec],
) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key, value in changes.items():
        if key in FORBIDDEN_ACTIONS:
            raise ConfigError(f"Action {key} is forbidden for the first SimAI benchmark version")
        if key not in allowed_actions:
            raise ConfigError(f"Action {key} is not allowed by this task")

        spec = allowed_actions[key]
        coerced = _coerce_value(key, value, spec.type)
        if spec.choices is not None and coerced not in spec.choices:
            raise ConfigError(f"Action {key}={coerced!r} not in choices {list(spec.choices)!r}")
        if spec.minimum is not None and isinstance(coerced, (int, float)) and coerced < spec.minimum:
            raise ConfigError(f"Action {key}={coerced!r} is below min {spec.minimum}")
        if spec.maximum is not None and isinstance(coerced, (int, float)) and coerced > spec.maximum:
            raise ConfigError(f"Action {key}={coerced!r} is above max {spec.maximum}")

        validated[key] = coerced
    return validated


def merge_config(baseline_config: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    merged = dict(baseline_config)
    merged.update(changes)
    return merged


def to_cli_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, bool):
            args.append(f"--{key}" if value else f"--no-{key}")
        elif value is None:
            continue
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _coerce_value(key: str, value: Any, action_type: str) -> Any:
    try:
        if action_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            raise ValueError
        if action_type == "int":
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if action_type == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if action_type == "str":
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Action {key} must be {action_type}, got {value!r}") from exc
    raise ConfigError(f"Unsupported action type {action_type!r}")
