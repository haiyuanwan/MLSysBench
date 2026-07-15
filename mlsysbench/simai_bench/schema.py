"""Dataclasses and schema validation for SimAI benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured, resolve_task_path


SUPPORTED_TRACKS = {
    "simai_gym",
    "infra_agent",
    "scale_up",
    "patch_transfer",
    "policy_transfer",
    "multifidelity",
}
SUPPORTED_TASK_SCHEMA_VERSIONS = {1, 2, 3}
SUPPORTED_DIRECTIONS = {"maximize", "minimize"}
SUPPORTED_RUNNERS = {"mock", "python_code", "vidur"}
SUPPORTED_SCENARIO_FAMILIES = {
    "prefill_heavy",
    "decode_heavy",
    "high_load",
    "balanced",
}
SUPPORTED_TRANSFERS = {
    "none",
    "scale_up",
    "workload_shift",
    "load_profile_shift",
    "network_shift",
    "hardware_shift",
}
SUPPORTED_STARTING_POINTS = {
    "from_scratch",
    "framework_default",
    "expert_template",
}
SUPPORTED_SOURCE_TYPES = {
    "hand_authored_fixture",
    "upstream_pr",
    "paper_artifact",
    "real_trace",
    "maintainer_blind",
}
SUPPORTED_PUBLICATION_STATUSES = {"fixture", "pilot", "candidate"}
SUPPORTED_CALIBRATION_STATUSES = {
    "uncalibrated",
    "proxy_only",
    "partially_calibrated",
    "calibrated",
    "real_hardware",
}
SUPPORTED_FIDELITY_KINDS = {"simulator", "hardware_proxy", "real_hardware"}
SCENARIO_PROFILES = {
    "prefill_heavy": {"short_prompt", "long_prompt"},
    "decode_heavy": {"short_output", "long_output"},
    "high_load": {"burst", "poisson", "constant"},
    "balanced": {"mixed_prompt_output", "mixed_concurrency"},
}
SCENARIO_OBJECTIVES = {
    "prefill_heavy": {
        ("p99_ttft_ms", "minimize"),
        ("goodput_rps", "maximize"),
    },
    "decode_heavy": {
        ("p99_tbt_ms", "minimize"),
        ("goodput_rps", "maximize"),
    },
    "high_load": {
        ("throughput_rps", "maximize"),
        ("goodput_rps", "maximize"),
        ("robust_goodput_rps", "maximize"),
    },
    "balanced": {
        ("goodput_rps", "maximize"),
        ("robust_goodput_rps", "maximize"),
    },
}


@dataclass(frozen=True)
class ScenarioSpec:
    family: str
    transfer: str
    starting_point: str
    profiles: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScenarioSpec":
        if not isinstance(data, dict):
            raise ConfigError("scenario must contain an object")

        family = data.get("family")
        if family not in SUPPORTED_SCENARIO_FAMILIES:
            raise ConfigError(
                "scenario.family must be one of "
                f"{sorted(SUPPORTED_SCENARIO_FAMILIES)}"
            )
        transfer = data.get("transfer")
        if transfer not in SUPPORTED_TRANSFERS:
            raise ConfigError(
                f"scenario.transfer must be one of {sorted(SUPPORTED_TRANSFERS)}"
            )
        starting_point = data.get("starting_point")
        if starting_point not in SUPPORTED_STARTING_POINTS:
            raise ConfigError(
                "scenario.starting_point must be one of "
                f"{sorted(SUPPORTED_STARTING_POINTS)}"
            )

        profiles = data.get("profiles")
        if not isinstance(profiles, list) or not profiles or not all(
            isinstance(profile, str) and profile for profile in profiles
        ):
            raise ConfigError("scenario.profiles must be a non-empty list of strings")
        if len(profiles) != len(set(profiles)):
            raise ConfigError("scenario.profiles must not contain duplicates")
        unknown_profiles = sorted(set(profiles) - SCENARIO_PROFILES[family])
        if unknown_profiles:
            raise ConfigError(
                f"scenario family {family!r} does not support profiles: "
                + ", ".join(unknown_profiles)
            )

        return cls(
            family=family,
            transfer=transfer,
            starting_point=starting_point,
            profiles=tuple(profiles),
        )


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
class MetricGate:
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def mapping_from_dict(cls, data: Any) -> dict[str, "MetricGate"]:
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError("metric_gates must contain an object")
        result: dict[str, MetricGate] = {}
        for name, spec in data.items():
            if not isinstance(name, str) or not name or not isinstance(spec, dict):
                raise ConfigError("metric_gates must map metric names to objects")
            minimum = _optional_float(spec.get("min"))
            maximum = _optional_float(spec.get("max"))
            if minimum is None and maximum is None:
                raise ConfigError(f"metric_gates.{name} requires min or max")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ConfigError(f"metric_gates.{name}.min must not exceed max")
            result[name] = cls(minimum=minimum, maximum=maximum)
        return result

    def check(self, name: str, metrics: dict[str, float]) -> list[str]:
        value = metrics.get(name)
        if value is None:
            return [f"missing gated metric {name}"]
        failures: list[str] = []
        if self.minimum is not None and value < self.minimum:
            failures.append(f"{name}={value:.6g} is below {self.minimum:.6g}")
        if self.maximum is not None and value > self.maximum:
            failures.append(f"{name}={value:.6g} exceeds {self.maximum:.6g}")
        return failures


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
class SubmissionSpec:
    type: str = "config"
    starter_dir: Path | None = None
    editable_files: tuple[str, ...] = ()
    max_file_bytes: int = 65_536

    @classmethod
    def from_dict(
        cls,
        task_dir: Path,
        data: dict[str, Any] | None,
        schema_version: int,
    ) -> "SubmissionSpec":
        if data is None:
            if schema_version >= 2:
                raise ConfigError("schema_version 2 tasks must declare submission")
            return cls()
        if not isinstance(data, dict):
            raise ConfigError("submission must contain an object")
        submission_type = data.get("type")
        if submission_type not in {"config", "code"}:
            raise ConfigError("submission.type must be one of ['code', 'config']")
        if submission_type == "config":
            return cls(type="config")

        starter_value = data.get("starter_dir")
        if not isinstance(starter_value, str) or not starter_value:
            raise ConfigError("code submission requires submission.starter_dir")
        editable_files = data.get("editable_files")
        if not isinstance(editable_files, list) or not editable_files:
            raise ConfigError("code submission requires non-empty submission.editable_files")
        if not all(isinstance(path, str) and _is_safe_relative_path(path) for path in editable_files):
            raise ConfigError(
                "submission.editable_files entries must be normalized relative paths"
            )
        if len(editable_files) != len(set(editable_files)):
            raise ConfigError("submission.editable_files must not contain duplicates")
        max_file_bytes = int(data.get("max_file_bytes", 65_536))
        if max_file_bytes <= 0:
            raise ConfigError("submission.max_file_bytes must be positive")
        starter_dir = resolve_task_path(task_dir, starter_value)
        if not starter_dir.is_dir():
            raise ConfigError(f"submission.starter_dir is not a directory: {starter_dir}")
        missing = [path for path in editable_files if not (starter_dir / path).is_file()]
        if missing:
            raise ConfigError(
                "submission.editable_files are missing from starter_dir: "
                + ", ".join(sorted(missing))
            )
        return cls(
            type="code",
            starter_dir=starter_dir,
            editable_files=tuple(editable_files),
            max_file_bytes=max_file_bytes,
        )


@dataclass(frozen=True)
class HiddenSpec:
    baseline_metrics: Path
    eval_workload: Path | None = None


@dataclass(frozen=True)
class DevelopmentSpec:
    baseline_metrics: Path
    eval_workload: Path | None = None
    fidelities: dict[str, "FidelitySpec"] | None = None
    default_fidelity: str | None = None


@dataclass(frozen=True)
class FidelitySpec:
    name: str
    kind: str
    cost_units: int
    baseline_metrics: Path
    eval_workload: Path | None = None
    max_queries: int | None = None

    @classmethod
    def from_dict(cls, task_dir: Path, name: str, data: Any) -> "FidelitySpec":
        if not isinstance(data, dict):
            raise ConfigError(f"development.fidelities.{name} must contain an object")
        if not _is_safe_identifier(name):
            raise ConfigError("development fidelity names must be safe identifiers")
        kind = data.get("kind")
        if kind not in SUPPORTED_FIDELITY_KINDS:
            raise ConfigError(
                f"development.fidelities.{name}.kind must be one of "
                f"{sorted(SUPPORTED_FIDELITY_KINDS)}"
            )
        cost_units = int(data.get("cost_units", 1))
        if cost_units <= 0:
            raise ConfigError(f"development.fidelities.{name}.cost_units must be positive")
        max_queries_value = data.get("max_queries")
        max_queries = int(max_queries_value) if max_queries_value is not None else None
        if max_queries is not None and max_queries <= 0:
            raise ConfigError(f"development.fidelities.{name}.max_queries must be positive")
        baseline_value = data.get("baseline_metrics")
        if not isinstance(baseline_value, str) or not baseline_value:
            raise ConfigError(
                f"development.fidelities.{name}.baseline_metrics is required"
            )
        workload_value = data.get("eval_workload")
        if workload_value is not None and not isinstance(workload_value, str):
            raise ConfigError(
                f"development.fidelities.{name}.eval_workload must be a path"
            )
        return cls(
            name=name,
            kind=kind,
            cost_units=cost_units,
            baseline_metrics=resolve_task_path(task_dir, baseline_value),
            eval_workload=(
                resolve_task_path(task_dir, workload_value) if workload_value else None
            ),
            max_queries=max_queries,
        )


@dataclass(frozen=True)
class ProvenanceSpec:
    source_type: str
    source_url: str | None
    source_revision: str | None
    license: str | None
    task_authors: tuple[str, ...]
    validators: tuple[str, ...]
    contamination_cutoff: str | None
    publication_status: str
    calibration_status: str
    notes: str | None

    @classmethod
    def from_dict(cls, data: Any, schema_version: int) -> "ProvenanceSpec | None":
        if data is None:
            if schema_version >= 3:
                raise ConfigError("schema_version 3 tasks must declare provenance")
            return None
        if not isinstance(data, dict):
            raise ConfigError("provenance must contain an object")
        source_type = data.get("source_type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ConfigError(
                f"provenance.source_type must be one of {sorted(SUPPORTED_SOURCE_TYPES)}"
            )
        publication_status = data.get("publication_status")
        if publication_status not in SUPPORTED_PUBLICATION_STATUSES:
            raise ConfigError(
                "provenance.publication_status must be one of "
                f"{sorted(SUPPORTED_PUBLICATION_STATUSES)}"
            )
        calibration_status = data.get("calibration_status")
        if calibration_status not in SUPPORTED_CALIBRATION_STATUSES:
            raise ConfigError(
                "provenance.calibration_status must be one of "
                f"{sorted(SUPPORTED_CALIBRATION_STATUSES)}"
            )
        authors = _string_tuple(data.get("task_authors", []), "provenance.task_authors")
        validators = _string_tuple(data.get("validators", []), "provenance.validators")
        return cls(
            source_type=source_type,
            source_url=_optional_string(data.get("source_url")),
            source_revision=_optional_string(data.get("source_revision")),
            license=_optional_string(data.get("license")),
            task_authors=authors,
            validators=validators,
            contamination_cutoff=_optional_string(data.get("contamination_cutoff")),
            publication_status=publication_status,
            calibration_status=calibration_status,
            notes=_optional_string(data.get("notes")),
        )


@dataclass(frozen=True)
class Constraints:
    max_gpu_units: int | None = None
    development_max_gpu_units: int | None = None
    max_steps: int = 8
    max_development_cost_units: int | None = None
    immutable_fields: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Constraints":
        data = data or {}
        max_gpu_units = data.get("max_gpu_units")
        development_max_gpu_units = data.get("development_max_gpu_units")
        max_steps = int(data.get("max_steps", 8))
        max_development_cost_units = data.get("max_development_cost_units")
        immutable_fields = data.get("immutable_fields", [])
        if max_gpu_units is not None and int(max_gpu_units) <= 0:
            raise ConfigError("constraints.max_gpu_units must be positive")
        if development_max_gpu_units is not None and int(development_max_gpu_units) <= 0:
            raise ConfigError("constraints.development_max_gpu_units must be positive")
        if max_steps <= 0:
            raise ConfigError("constraints.max_steps must be positive")
        if (
            max_development_cost_units is not None
            and int(max_development_cost_units) <= 0
        ):
            raise ConfigError("constraints.max_development_cost_units must be positive")
        if not isinstance(immutable_fields, list) or not all(
            isinstance(field, str) for field in immutable_fields
        ):
            raise ConfigError("constraints.immutable_fields must be a list of strings")
        return cls(
            max_gpu_units=int(max_gpu_units) if max_gpu_units is not None else None,
            development_max_gpu_units=(
                int(development_max_gpu_units)
                if development_max_gpu_units is not None
                else None
            ),
            max_steps=max_steps,
            max_development_cost_units=(
                int(max_development_cost_units)
                if max_development_cost_units is not None
                else None
            ),
            immutable_fields=tuple(immutable_fields),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_dir: Path
    schema_version: int
    task_id: str
    track: str
    description: str
    scenario: ScenarioSpec
    baseline_config: Path
    allowed_actions: Path
    objective: Objective
    slo: SLO
    metric_gates: dict[str, MetricGate]
    hidden: HiddenSpec
    development: DevelopmentSpec | None
    submission: SubmissionSpec
    runner: RunnerSpec
    constraints: Constraints
    provenance: ProvenanceSpec | None

    @classmethod
    def load(cls, task_dir: str | Path) -> "TaskSpec":
        task_dir = Path(task_dir)
        task_file = task_dir / "task.json"
        if not task_file.exists():
            task_file = task_dir / "task.yaml"
        data = load_structured(task_file)

        schema_version = data.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version not in SUPPORTED_TASK_SCHEMA_VERSIONS
        ):
            raise ConfigError(
                "schema_version must be one of "
                f"{sorted(SUPPORTED_TASK_SCHEMA_VERSIONS)}"
            )

        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ConfigError("task_id must be a non-empty string")
        track = data.get("track")
        if track not in SUPPORTED_TRACKS:
            raise ConfigError(f"track must be one of {sorted(SUPPORTED_TRACKS)}")

        hidden_data = data.get("hidden") or {}
        if "baseline_metrics" not in hidden_data:
            raise ConfigError("hidden.baseline_metrics is required")
        development_data = data.get("development")
        if development_data is not None and not isinstance(development_data, dict):
            raise ConfigError("development must contain an object")
        if development_data is not None and "baseline_metrics" not in development_data:
            raise ConfigError("development.baseline_metrics is required")

        fidelities_data = development_data.get("fidelities") if development_data else None
        if fidelities_data is not None and not isinstance(fidelities_data, dict):
            raise ConfigError("development.fidelities must contain an object")
        fidelities = (
            {
                name: FidelitySpec.from_dict(task_dir, name, spec)
                for name, spec in fidelities_data.items()
            }
            if fidelities_data
            else None
        )
        default_fidelity = development_data.get("default_fidelity") if development_data else None
        if default_fidelity is not None and (
            not isinstance(default_fidelity, str)
            or not fidelities
            or default_fidelity not in fidelities
        ):
            raise ConfigError("development.default_fidelity must name a declared fidelity")

        task = cls(
            task_dir=task_dir,
            schema_version=schema_version,
            task_id=task_id,
            track=track,
            description=str(data.get("description", "")),
            scenario=ScenarioSpec.from_dict(data.get("scenario")),
            baseline_config=resolve_task_path(task_dir, data["baseline_config"]),
            allowed_actions=resolve_task_path(task_dir, data["allowed_actions"]),
            objective=Objective.from_dict(data["objective"]),
            slo=SLO.from_dict(data.get("slo")),
            metric_gates=MetricGate.mapping_from_dict(data.get("metric_gates")),
            hidden=HiddenSpec(
                baseline_metrics=resolve_task_path(task_dir, hidden_data["baseline_metrics"]),
                eval_workload=(
                    resolve_task_path(task_dir, hidden_data["eval_workload"])
                    if hidden_data.get("eval_workload")
                    else None
                ),
            ),
            development=(
                DevelopmentSpec(
                    baseline_metrics=resolve_task_path(
                        task_dir, development_data["baseline_metrics"]
                    ),
                    eval_workload=(
                        resolve_task_path(task_dir, development_data["eval_workload"])
                        if development_data.get("eval_workload")
                        else None
                    ),
                    fidelities=fidelities,
                    default_fidelity=default_fidelity,
                )
                if development_data is not None
                else None
            ),
            submission=SubmissionSpec.from_dict(
                task_dir, data.get("submission"), schema_version
            ),
            runner=RunnerSpec.from_dict(data["runner"]),
            constraints=Constraints.from_dict(data.get("constraints")),
            provenance=ProvenanceSpec.from_dict(data.get("provenance"), schema_version),
        )
        return task

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

    def development_fidelity(self, fidelity: str | None = None) -> FidelitySpec | None:
        if self.development is None or not self.development.fidelities:
            if fidelity is not None:
                raise ConfigError("task does not define development fidelities")
            return None
        selected = fidelity or self.development.default_fidelity
        if selected is None:
            selected = next(iter(self.development.fidelities))
        try:
            return self.development.fidelities[selected]
        except KeyError as exc:
            raise ConfigError(f"unknown development fidelity {selected!r}") from exc

    def load_baseline_metrics(
        self,
        phase: str = "final",
        fidelity: str | None = None,
    ) -> dict[str, float]:
        fidelity_spec = self.development_fidelity(fidelity) if phase == "development" else None
        spec = fidelity_spec or (
            self.development if phase == "development" and self.development else self.hidden
        )
        data = load_structured(spec.baseline_metrics)
        if isinstance(data, dict) and data.get("valid") is False:
            reason = data.get("reason", "baseline metrics were invalidated")
            raise ConfigError(f"Cannot score task: {reason}")
        metrics = data.get("metrics", data) if isinstance(data, dict) else data
        if not isinstance(metrics, dict):
            raise ConfigError("baseline_metrics must contain an object")
        return {key: float(value) for key, value in metrics.items() if _is_number(value)}

    def eval_workload_path(
        self,
        phase: str = "final",
        fidelity: str | None = None,
    ) -> Path | None:
        fidelity_spec = self.development_fidelity(fidelity) if phase == "development" else None
        spec = fidelity_spec or (
            self.development if phase == "development" and self.development else self.hidden
        )
        return spec.eval_workload

    def load_eval_config_overrides(
        self,
        phase: str = "final",
        fidelity: str | None = None,
    ) -> dict[str, Any]:
        workload_path = self.eval_workload_path(phase, fidelity)
        if workload_path is None:
            return {}
        data = load_structured(workload_path)
        if not isinstance(data, dict):
            raise ConfigError("hidden eval_workload must contain an object")
        overrides = data.get("config_overrides", {})
        if not isinstance(overrides, dict):
            raise ConfigError("hidden eval_workload.config_overrides must contain an object")
        return overrides

    def validate_constraints(
        self,
        changes: dict[str, Any],
        config: dict[str, Any],
        phase: str = "final",
    ) -> int:
        immutable_changes = sorted(set(changes) & set(self.constraints.immutable_fields))
        if immutable_changes:
            raise ConfigError(
                "Submission changes immutable evaluation fields: "
                + ", ".join(immutable_changes)
            )

        replicas = int(config.get("cluster_config_num_replicas", 1))
        tensor_parallel = int(config.get("replica_config_tensor_parallel_size", 1))
        pipeline_parallel = int(config.get("replica_config_num_pipeline_stages", 1))
        gpu_units = replicas * tensor_parallel * pipeline_parallel
        gpu_budget = self.constraints.max_gpu_units
        if phase == "development" and self.constraints.development_max_gpu_units is not None:
            gpu_budget = self.constraints.development_max_gpu_units
        if gpu_budget is not None and gpu_units > gpu_budget:
            raise ConfigError(
                f"Configuration uses {gpu_units} GPU units, exceeding budget "
                f"{gpu_budget}"
            )
        return gpu_units


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        value == path.as_posix()
        and not path.is_absolute()
        and value not in {"", "."}
        and ".." not in path.parts
    )


def _is_safe_identifier(value: str) -> bool:
    compact = value.replace("-", "").replace("_", "")
    return bool(value) and compact.isalnum() and not value[0].isdigit()


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError("optional provenance strings must be non-empty when present")
    return value
