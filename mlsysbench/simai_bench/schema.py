"""Dataclasses and schema validation for SimAI benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured, resolve_task_path


SUPPORTED_TRACKS = {"simai_gym", "infra_agent", "scale_up"}
SUPPORTED_DIRECTIONS = {"maximize", "minimize"}
SUPPORTED_RUNNERS = {"mock", "vidur"}


@dataclass(frozen=True)
class Objective:
    primary_metric: str
    direction: str
    score_cap: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Objective":
        direction = data.get("direction")
        if direction not in SUPPORTED_DIRECTIONS:
            raise ConfigError(f"objective.direction must be one of {sorted(SUPPORTED_DIRECTIONS)}")
        primary_metric = data.get("primary_metric")
        if not isinstance(primary_metric, str) or not primary_metric:
            raise ConfigError("objective.primary_metric must be a non-empty string")
        return cls(
            primary_metric=primary_metric,
            direction=direction,
            score_cap=float(data.get("score_cap", 2.0)),
        )


@dataclass(frozen=True)
class SLO:
    p99_ttft_ms: float | None = None
    p99_tbt_ms: float | None = None
    p99_e2e_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SLO":
        data = data or {}
        return cls(
            p99_ttft_ms=_optional_float(data.get("p99_ttft_ms")),
            p99_tbt_ms=_optional_float(data.get("p99_tbt_ms")),
            p99_e2e_ms=_optional_float(data.get("p99_e2e_ms")),
        )

    def check(self, metrics: dict[str, float]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        checks = [
            ("p99_ttft_ms", self.p99_ttft_ms),
            ("p99_tbt_ms", self.p99_tbt_ms),
            ("p99_e2e_ms", self.p99_e2e_ms),
        ]
        for metric_name, limit in checks:
            if limit is None:
                continue
            value = metrics.get(metric_name)
            if value is None:
                failures.append(f"missing {metric_name}")
            elif value > limit:
                failures.append(f"{metric_name}={value:.6g} exceeds {limit:.6g}")
        return not failures, failures


@dataclass(frozen=True)
class ActionSpec:
    name: str
    type: str
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ActionSpec":
        action_type = data.get("type")
        if action_type not in {"int", "float", "str", "bool"}:
            raise ConfigError(f"Action {name} has unsupported type {action_type!r}")
        choices = data.get("choices")
        if choices is not None and not isinstance(choices, list):
            raise ConfigError(f"Action {name}.choices must be a list")
        return cls(
            name=name,
            type=action_type,
            choices=tuple(choices) if choices is not None else None,
            minimum=_optional_float(data.get("min")),
            maximum=_optional_float(data.get("max")),
        )


@dataclass(frozen=True)
class RunnerSpec:
    type: str
    config: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunnerSpec":
        runner_type = data.get("type")
        if runner_type not in SUPPORTED_RUNNERS:
            raise ConfigError(f"runner.type must be one of {sorted(SUPPORTED_RUNNERS)}")
        return cls(type=runner_type, config={k: v for k, v in data.items() if k != "type"})


@dataclass(frozen=True)
class HiddenSpec:
    baseline_metrics: Path
    eval_workload: Path | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_dir: Path
    task_id: str
    track: str
    description: str
    baseline_config: Path
    allowed_actions: Path
    objective: Objective
    slo: SLO
    hidden: HiddenSpec
    runner: RunnerSpec

    @classmethod
    def load(cls, task_dir: str | Path) -> "TaskSpec":
        task_dir = Path(task_dir)
        task_file = task_dir / "task.json"
        if not task_file.exists():
            task_file = task_dir / "task.yaml"
        data = load_structured(task_file)

        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ConfigError("task_id must be a non-empty string")
        track = data.get("track")
        if track not in SUPPORTED_TRACKS:
            raise ConfigError(f"track must be one of {sorted(SUPPORTED_TRACKS)}")

        hidden_data = data.get("hidden") or {}
        if "baseline_metrics" not in hidden_data:
            raise ConfigError("hidden.baseline_metrics is required")

        return cls(
            task_dir=task_dir,
            task_id=task_id,
            track=track,
            description=str(data.get("description", "")),
            baseline_config=resolve_task_path(task_dir, data["baseline_config"]),
            allowed_actions=resolve_task_path(task_dir, data["allowed_actions"]),
            objective=Objective.from_dict(data["objective"]),
            slo=SLO.from_dict(data.get("slo")),
            hidden=HiddenSpec(
                baseline_metrics=resolve_task_path(task_dir, hidden_data["baseline_metrics"]),
                eval_workload=(
                    resolve_task_path(task_dir, hidden_data["eval_workload"])
                    if hidden_data.get("eval_workload")
                    else None
                ),
            ),
            runner=RunnerSpec.from_dict(data["runner"]),
        )

    def load_baseline_config(self) -> dict[str, Any]:
        data = load_structured(self.baseline_config)
        if not isinstance(data, dict):
            raise ConfigError("baseline_config must contain an object")
        return data

    def load_allowed_actions(self) -> dict[str, ActionSpec]:
        data = load_structured(self.allowed_actions)
        if not isinstance(data, dict):
            raise ConfigError("allowed_actions must contain an object")
        return {name: ActionSpec.from_dict(name, spec) for name, spec in data.items()}

    def load_baseline_metrics(self) -> dict[str, float]:
        data = load_structured(self.hidden.baseline_metrics)
        metrics = data.get("metrics", data) if isinstance(data, dict) else data
        if not isinstance(metrics, dict):
            raise ConfigError("baseline_metrics must contain an object")
        return {key: float(value) for key, value in metrics.items() if _is_number(value)}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

