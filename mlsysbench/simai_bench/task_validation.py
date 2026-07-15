"""Pre-publication validation for benchmark task fixtures."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.evaluator import evaluate_changes
from mlsysbench.simai_bench.io import load_structured, resolve_task_path
from mlsysbench.simai_bench.runner import change_signature
from mlsysbench.simai_bench.schema import (
    SCENARIO_OBJECTIVES,
    SCENARIO_PROFILES,
    TaskSpec,
)
from mlsysbench.simai_bench.search import _candidate_configs


@dataclass(frozen=True)
class TaskValidationResult:
    task_id: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_task(
    task_dir: str | Path,
    *,
    run_real_baseline: bool = False,
    max_mock_configurations: int = 100_000,
) -> TaskValidationResult:
    task = TaskSpec.load(task_dir)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"schema_version": task.schema_version}

    _validate_scenario(task, errors, warnings, details)

    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    immutable_actions = sorted(set(allowed_actions) & set(task.constraints.immutable_fields))
    if immutable_actions:
        errors.append(
            "allowed actions are also immutable: " + ", ".join(immutable_actions)
        )

    if task.development is None:
        warnings.append("task has no separate development specification")
    else:
        development_hash = _sha256_file(task.development.eval_workload)
        final_hash = _sha256_file(task.hidden.eval_workload)
        details["development_workload_sha256"] = development_hash
        details["final_workload_sha256"] = final_hash
        if development_hash is not None and development_hash == final_hash:
            errors.append("development and final workload files have the same hash")
        development_seeds = _seed_values(task.development.eval_workload)
        final_seeds = _seed_values(task.hidden.eval_workload)
        details["development_seeds"] = development_seeds
        details["final_seeds"] = final_seeds
        if development_seeds and development_seeds == final_seeds:
            errors.append("development and final workloads use the same recorded seeds")

    if task.runner.type == "mock":
        _validate_mock_surfaces(
            task,
            baseline_config,
            allowed_actions,
            max_mock_configurations,
            errors,
            warnings,
            details,
        )
    if task.runner.type == "mock" or run_real_baseline:
        _validate_baseline_replay(
            task,
            baseline_config,
            allowed_actions,
            errors,
            details,
        )
    else:
        warnings.append("real baseline replay skipped; pass --run-real-baseline to execute it")

    return TaskValidationResult(
        task_id=task.task_id,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def _validate_scenario(
    task: TaskSpec,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> None:
    declared_profiles = set(task.scenario.profiles)
    canonical_profiles = SCENARIO_PROFILES[task.scenario.family]
    missing_profiles = sorted(canonical_profiles - declared_profiles)
    details["scenario"] = {
        "family": task.scenario.family,
        "transfer": task.scenario.transfer,
        "starting_point": task.scenario.starting_point,
        "profiles": list(task.scenario.profiles),
        "missing_canonical_profiles": missing_profiles,
    }
    objective = (task.objective.primary_metric, task.objective.direction)
    details["scenario"]["objective"] = {
        "primary_metric": objective[0],
        "direction": objective[1],
    }
    if objective not in SCENARIO_OBJECTIVES[task.scenario.family]:
        supported = ", ".join(
            f"{metric}:{direction}"
            for metric, direction in sorted(SCENARIO_OBJECTIVES[task.scenario.family])
        )
        errors.append(
            f"scenario {task.scenario.family!r} does not support objective "
            f"{objective[0]}:{objective[1]}; expected one of {supported}"
        )
    if missing_profiles:
        warnings.append(
            f"scenario {task.scenario.family} does not cover canonical profiles: "
            + ", ".join(missing_profiles)
        )

    phase_paths = {"final": task.hidden.eval_workload}
    if task.development is not None:
        phase_paths["development"] = task.development.eval_workload
    for phase, path in phase_paths.items():
        workload = _workload_scenario(path)
        details[f"{phase}_workload_scenario"] = workload
        workload_family = workload.get("scenario_family")
        if workload_family is None:
            errors.append(f"{phase} workload must declare scenario_family")
        elif workload_family != task.scenario.family:
            errors.append(
                f"{phase} workload scenario_family {workload_family!r} does not match "
                f"task scenario {task.scenario.family!r}"
            )

        workload_profiles = workload.get("profiles")
        if workload_profiles is None:
            errors.append(f"{phase} workload must declare profiles")
            continue
        unexpected_profiles = sorted(set(workload_profiles) - declared_profiles)
        if unexpected_profiles:
            errors.append(
                f"{phase} workload uses undeclared scenario profiles: "
                + ", ".join(unexpected_profiles)
            )
        if phase == "final":
            missing_from_final = sorted(declared_profiles - set(workload_profiles))
            if missing_from_final:
                errors.append(
                    "final workload does not exercise declared scenario profiles: "
                    + ", ".join(missing_from_final)
                )

    if task.scenario.transfer == "none" and task.development is not None:
        warnings.append("scenario transfer is 'none' but task defines a development phase")
    if task.scenario.transfer != "none" and task.development is None:
        warnings.append(
            f"scenario transfer {task.scenario.transfer!r} has no development phase"
        )


def _workload_scenario(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = load_structured(path)
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    if isinstance(payload.get("scenario_family"), str):
        result["scenario_family"] = payload["scenario_family"]
    profiles = payload.get("profiles")
    if isinstance(profiles, list) and all(isinstance(item, str) for item in profiles):
        result["profiles"] = profiles
    return result


def _validate_mock_surfaces(
    task: TaskSpec,
    baseline_config: dict[str, Any],
    allowed_actions: dict[str, Any],
    max_configurations: int,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> None:
    iterator = _candidate_configs(allowed_actions, baseline_config)
    candidates = list(itertools.islice(iterator, max_configurations + 1))
    if len(candidates) > max_configurations:
        errors.append(
            f"mock search space exceeds validation cap of {max_configurations} configurations"
        )
        return
    expected_signatures = {change_signature(candidate) for candidate in candidates}
    details["mock_legal_configurations"] = len(expected_signatures)

    for phase in ("development", "final"):
        path = _mock_metrics_path(task, phase)
        payload = load_structured(path)
        if not isinstance(payload, dict):
            errors.append(f"{phase} mock metrics must contain an object")
            continue
        mapped_signatures = {key for key in payload if key != "default"}
        missing = sorted(expected_signatures - mapped_signatures)
        extra = sorted(mapped_signatures - expected_signatures)
        phase_details = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "mapped_configurations": len(mapped_signatures),
            "missing_configurations": len(missing),
            "extra_configurations": len(extra),
            "has_default": "default" in payload,
        }
        details[f"{phase}_mock_surface"] = phase_details
        if missing:
            errors.append(
                f"{phase} mock surface is missing {len(missing)} of "
                f"{len(expected_signatures)} legal configurations"
            )
        if "default" in payload:
            warnings.append(
                f"{phase} mock surface has a default entry; unknown signatures should fail closed"
            )
        if extra:
            warnings.append(f"{phase} mock surface has {len(extra)} unknown signatures")

        baseline_actions = {
            name: baseline_config[name]
            for name in allowed_actions
            if name in baseline_config
        }
        baseline_entry = payload.get(change_signature(baseline_actions))
        if baseline_entry is None:
            errors.append(f"{phase} mock surface has no explicit baseline signature")
            continue
        mapped_metrics = _metrics_dict(baseline_entry)
        declared_metrics = task.load_baseline_metrics(phase)
        mismatches = _metric_mismatches(mapped_metrics, declared_metrics)
        if mismatches:
            errors.append(f"{phase} baseline metrics mismatch: " + "; ".join(mismatches))


def _validate_baseline_replay(
    task: TaskSpec,
    baseline_config: dict[str, Any],
    allowed_actions: dict[str, Any],
    errors: list[str],
    details: dict[str, Any],
) -> None:
    for phase in ("development", "final"):
        try:
            result = evaluate_changes(
                task,
                baseline_config,
                allowed_actions,
                {},
                phase=phase,
            )
        except Exception as exc:  # noqa: BLE001 - validation must aggregate failures.
            errors.append(f"{phase} baseline replay failed: {exc}")
            continue
        details[f"{phase}_baseline_replay"] = {
            "valid": result.valid,
            "ratio": result.ratio,
            "failures": result.failures,
        }
        if not result.valid:
            errors.append(f"{phase} baseline replay is invalid: {result.failures}")
        if not math.isclose(result.ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{phase} baseline replay ratio is {result.ratio}, expected 1.0")
        mismatches = _metric_mismatches(result.agent_metrics, result.baseline_metrics)
        if mismatches:
            errors.append(f"{phase} baseline replay mismatch: " + "; ".join(mismatches))


def _mock_metrics_path(task: TaskSpec, phase: str) -> Path:
    key = "mock_metrics_development" if phase == "development" else "mock_metrics"
    path_value = task.runner.config.get(key)
    if path_value is None and phase == "development":
        path_value = task.runner.config.get("mock_metrics")
    if path_value is None:
        raise ValueError(f"mock runner has no metrics file for {phase}")
    return resolve_task_path(task.task_dir, str(path_value))


def _metrics_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics = value.get("metrics", value)
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in metrics.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _metric_mismatches(
    actual: dict[str, float],
    expected: dict[str, float],
) -> list[str]:
    mismatches: list[str] = []
    for name in sorted(set(actual) | set(expected)):
        if name not in actual or name not in expected:
            mismatches.append(f"{name} missing")
        elif not math.isclose(actual[name], expected[name], rel_tol=1e-9, abs_tol=1e-9):
            mismatches.append(f"{name}={actual[name]} expected {expected[name]}")
    return mismatches


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_structured(path)
    seeds: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                if "seed" in str(key).lower() and isinstance(item, (str, int, float)):
                    seeds[child] = item
                else:
                    visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload, "")
    return seeds
