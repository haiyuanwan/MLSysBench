"""Validation and replay for publication baseline ladders.

The ladder is deliberately separate from the task's score denominator.  It
records every comparison needed for a paper result without silently changing
how an existing task is scored.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.actions import load_submission
from mlsysbench.simai_bench.evaluator import evaluate_submission
from mlsysbench.simai_bench.io import ConfigError, load_structured, write_json
from mlsysbench.simai_bench.schema import TaskSpec
from mlsysbench.simai_bench.search import run_search


STATIC_TIERS = ("naive", "framework_default", "expert_recipe")
REQUIRED_SEARCH_METHODS = {"random", "tpe", "smac"}
SUPPORTED_SEARCH_METHODS = REQUIRED_SEARCH_METHODS | {"grid"}


@dataclass(frozen=True)
class BaselineLadderValidationResult:
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


def validate_baseline_ladder(
    task_dir: str | Path,
    *,
    replay_static: bool = False,
) -> BaselineLadderValidationResult:
    """Validate one task's four-tier comparison contract.

    Static replay is opt-in because a publication task may launch an expensive
    real runner.  Structural validation still parses every submission and
    fingerprints every declared artifact.
    """

    task = TaskSpec.load(task_dir)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    if task.baseline_ladder is None:
        return BaselineLadderValidationResult(
            task_id=task.task_id,
            valid=False,
            errors=["task does not declare baseline_ladder"],
            warnings=[],
            details={},
        )
    if not task.baseline_ladder.is_file():
        return BaselineLadderValidationResult(
            task_id=task.task_id,
            valid=False,
            errors=[f"baseline ladder file does not exist: {task.baseline_ladder}"],
            warnings=[],
            details={},
        )

    payload = load_structured(task.baseline_ladder)
    if not isinstance(payload, dict):
        raise ConfigError("baseline ladder must contain an object")
    if payload.get("schema_version") != 1:
        errors.append("baseline ladder schema_version must be 1")

    tiers = payload.get("tiers")
    if not isinstance(tiers, dict):
        errors.append("baseline ladder tiers must contain an object")
        tiers = {}
    expected_tiers = set(STATIC_TIERS) | {"matched_search"}
    missing_tiers = sorted(expected_tiers - set(tiers))
    extra_tiers = sorted(set(tiers) - expected_tiers)
    if missing_tiers:
        errors.append("baseline ladder is missing tiers: " + ", ".join(missing_tiers))
    if extra_tiers:
        errors.append("baseline ladder has unknown tiers: " + ", ".join(extra_tiers))

    denominator = payload.get("score_denominator")
    if denominator not in expected_tiers:
        errors.append("score_denominator must name one of the four baseline tiers")
    details["score_denominator"] = denominator
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        errors.append("baseline ladder measurement must contain an object")
        measurement = {}
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 3:
        errors.append("baseline ladder measurement.repeats must be at least 3")
    details["measurement"] = {"repeats": repeats}

    static_details: dict[str, Any] = {}
    for name in STATIC_TIERS:
        spec = tiers.get(name)
        tier_details = _validate_static_tier(task, name, spec, errors)
        if replay_static and tier_details.get("submission_path"):
            try:
                result = evaluate_submission(task.task_dir, tier_details["submission_path"])
                tier_details["replay"] = result.to_dict()
                if not result.valid:
                    errors.append(f"baseline tier {name} replay is invalid: {result.failures}")
            except Exception as exc:  # noqa: BLE001 - validation reports all failures.
                errors.append(f"baseline tier {name} replay failed: {exc}")
        static_details[name] = tier_details
    details["static_tiers"] = static_details

    details["matched_search"] = _validate_matched_search(
        tiers.get("matched_search"), errors
    )
    details["human_expert"] = _validate_human_expert(
        task,
        payload.get("human_expert"),
        errors,
        warnings,
    )
    details["manifest_sha256"] = _sha256_file(task.baseline_ladder)
    details["result_bundle"] = _validate_result_bundle(
        task,
        payload.get("result_bundle"),
        repeats,
        details["matched_search"],
        errors,
        warnings,
    )
    return BaselineLadderValidationResult(
        task_id=task.task_id,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def run_baseline_ladder(
    task_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay static tiers and every declared matched-search seed.

    Optimizer import/runtime failures are retained as results so a missing TPE
    or SMAC run cannot disappear from a paper aggregate.
    """

    task = TaskSpec.load(task_dir)
    validation = validate_baseline_ladder(task_dir)
    if not validation.valid:
        raise ConfigError("invalid baseline ladder: " + "; ".join(validation.errors))
    assert task.baseline_ladder is not None
    payload = load_structured(task.baseline_ladder)
    tiers = payload["tiers"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records: list[dict[str, Any]] = []

    repeats = int(payload["measurement"]["repeats"])
    for name in STATIC_TIERS:
        submission = _task_artifact(task, tiers[name]["submission"])
        for repeat in range(repeats):
            record: dict[str, Any] = {
                "tier": name,
                "repeat": repeat,
                "submission_sha256": _sha256_file(submission),
            }
            try:
                evaluation = evaluate_submission(task.task_dir, submission)
                record["result"] = evaluation.to_dict()
                record["status"] = "completed"
            except Exception as exc:  # noqa: BLE001 - failures are benchmark results.
                record.update(status="failed", error=str(exc))
            records.append(record)
            write_json(output / "baseline_records.json", records)

    search = tiers["matched_search"]
    for method in search["methods"]:
        for seed in search["seeds"]:
            run_dir = output / "matched_search" / method / f"seed_{seed}"
            record = {"tier": "matched_search", "method": method, "seed": seed}
            try:
                result = run_search(
                    task.task_dir,
                    run_dir,
                    method=method,
                    budget=int(search["query_budget"]),
                    seed=int(seed),
                    wall_time_seconds=float(search["wall_time_seconds"]),
                )
                record.update(status="completed", result=result.to_dict())
            except Exception as exc:  # noqa: BLE001 - preserve unavailable optimizers.
                record.update(status="failed", error=str(exc))
            records.append(record)
            write_json(output / "baseline_records.json", records)

    result = {
        "schema_version": 1,
        "task_id": task.task_id,
        "baseline_ladder_sha256": _sha256_file(task.baseline_ladder),
        "started_unix_seconds": started,
        "completed_unix_seconds": time.time(),
        "records": records,
        "complete": all(record["status"] == "completed" for record in records),
    }
    write_json(output / "baseline_ladder_result.json", result)
    return result


def _validate_static_tier(
    task: TaskSpec,
    name: str,
    value: Any,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"baseline tier {name} must contain an object")
        return {}
    submission_value = value.get("submission")
    if not isinstance(submission_value, str) or not submission_value:
        errors.append(f"baseline tier {name} requires submission")
        return {}
    try:
        submission = _task_artifact(task, submission_value)
    except ConfigError as exc:
        errors.append(str(exc))
        return {}
    if not submission.is_file():
        errors.append(f"baseline tier {name} submission does not exist: {submission_value}")
        return {"submission_path": str(submission)}
    try:
        normalized_submission = load_submission(str(submission))
    except Exception as exc:  # noqa: BLE001 - report malformed artifact.
        errors.append(f"baseline tier {name} submission is invalid: {exc}")
        normalized_submission = None
    if (
        name == "framework_default"
        and normalized_submission is not None
        and normalized_submission.get("changes") != {}
    ):
        errors.append("framework_default submission must have empty changes")

    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("description"):
        errors.append(f"baseline tier {name} requires provenance.description")
        provenance = provenance if isinstance(provenance, dict) else {}
    if name == "framework_default" and not provenance.get("source_revision"):
        errors.append("framework_default requires provenance.source_revision")
    if name == "expert_recipe":
        missing = [
            field
            for field in ("source_url", "source_revision", "author")
            if not provenance.get(field)
        ]
        if missing:
            errors.append("expert_recipe provenance is missing: " + ", ".join(missing))
    return {
        "submission_path": str(submission),
        "submission_sha256": _sha256_file(submission),
        "provenance": provenance,
    }


def _validate_matched_search(value: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append("baseline tier matched_search must contain an object")
        return {}
    methods = value.get("methods")
    if not isinstance(methods, list) or not methods or not all(
        isinstance(method, str) for method in methods
    ):
        errors.append("matched_search.methods must be a non-empty string list")
        methods = []
    method_set = set(methods)
    unknown = sorted(method_set - SUPPORTED_SEARCH_METHODS)
    missing = sorted(REQUIRED_SEARCH_METHODS - method_set)
    if unknown:
        errors.append("matched_search has unknown methods: " + ", ".join(unknown))
    if missing:
        errors.append("matched_search is missing methods: " + ", ".join(missing))
    grid_applicable = value.get("grid_applicable")
    if not isinstance(grid_applicable, bool):
        errors.append("matched_search.grid_applicable must be boolean")
    elif grid_applicable and "grid" not in method_set:
        errors.append("matched_search must include grid when grid_applicable is true")
    if value.get("scope") not in {"restricted", "full"}:
        errors.append("matched_search.scope must be restricted or full")
    query_budget = value.get("query_budget")
    wall_time = value.get("wall_time_seconds")
    if not isinstance(query_budget, int) or isinstance(query_budget, bool) or query_budget <= 0:
        errors.append("matched_search.query_budget must be a positive integer")
    if not isinstance(wall_time, (int, float)) or isinstance(wall_time, bool) or wall_time <= 0:
        errors.append("matched_search.wall_time_seconds must be positive")
    seeds = value.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    ):
        errors.append("matched_search.seeds must contain at least three integer seeds")
        seeds = []
    if len(seeds) != len(set(seeds)):
        errors.append("matched_search.seeds must not contain duplicates")
    return {
        "methods": methods,
        "grid_applicable": grid_applicable,
        "scope": value.get("scope"),
        "query_budget": query_budget,
        "wall_time_seconds": wall_time,
        "seeds": seeds,
    }


def _validate_human_expert(
    task: TaskSpec,
    value: Any,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    is_candidate = bool(
        task.provenance and task.provenance.publication_status == "candidate"
    )
    if value is None:
        if is_candidate:
            errors.append("publication candidates require a human_expert baseline record")
        else:
            warnings.append("baseline ladder has no human_expert record")
        return {}
    if not isinstance(value, dict):
        errors.append("human_expert must contain an object")
        return {}
    participants = value.get("participants")
    if not isinstance(participants, int) or isinstance(participants, bool) or participants <= 0:
        errors.append("human_expert.participants must be a positive integer")
    if value.get("matched_budget") is not True:
        errors.append("human_expert.matched_budget must be true")
    if not isinstance(value.get("protocol"), str) or not value["protocol"]:
        errors.append("human_expert.protocol is required")
    results_value = value.get("results")
    results_path: Path | None = None
    if not isinstance(results_value, str) or not results_value:
        errors.append("human_expert.results is required")
    else:
        try:
            results_path = _task_artifact(task, results_value)
            if not results_path.is_file():
                errors.append(f"human_expert results do not exist: {results_value}")
            else:
                _validate_human_results(
                    task,
                    results_path,
                    int(participants) if isinstance(participants, int) else 0,
                    errors,
                )
        except ConfigError as exc:
            errors.append(str(exc))
    return {
        "participants": participants,
        "matched_budget": value.get("matched_budget"),
        "protocol": value.get("protocol"),
        "results_path": str(results_path) if results_path else None,
        "results_sha256": _sha256_file(results_path) if results_path else None,
    }


def _validate_human_results(
    task: TaskSpec,
    path: Path,
    participants: int,
    errors: list[str],
) -> None:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        errors.append("human_expert results must contain an object")
        return
    if payload.get("schema_version") != 1:
        errors.append("human_expert results schema_version must be 1")
    if payload.get("task_id") != task.task_id:
        errors.append("human_expert results task_id does not match task")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append("human_expert results must contain non-empty runs")
        return
    participant_ids: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"human_expert run {index} must contain an object")
            continue
        participant = run.get("participant_id")
        if not isinstance(participant, str) or not participant:
            errors.append(f"human_expert run {index} requires participant_id")
        else:
            participant_ids.add(participant)
        if run.get("status") != "completed":
            errors.append(f"human_expert run {index} is not completed")
        result = run.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("score"), (int, float)):
            errors.append(f"human_expert run {index} requires numeric result.score")
        budget = run.get("budget")
        if not isinstance(budget, dict):
            errors.append(f"human_expert run {index} requires budget")
        elif not isinstance(budget.get("wall_time_seconds"), (int, float)):
            errors.append(
                f"human_expert run {index} requires budget.wall_time_seconds"
            )
    if len(participant_ids) < participants:
        errors.append(
            "human_expert results contain fewer distinct participants than declared"
        )


def _validate_result_bundle(
    task: TaskSpec,
    value: Any,
    repeats: Any,
    search: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    is_candidate = bool(
        task.provenance and task.provenance.publication_status == "candidate"
    )
    if value is None:
        if is_candidate:
            errors.append("publication candidates require baseline result_bundle")
        else:
            warnings.append("baseline ladder has no replay result_bundle")
        return {}
    if not isinstance(value, str) or not value:
        errors.append("baseline result_bundle must be a path string")
        return {}
    try:
        path = _task_artifact(task, value)
    except ConfigError as exc:
        errors.append(str(exc))
        return {}
    if not path.is_file():
        errors.append(f"baseline result bundle does not exist: {value}")
        return {"path": str(path)}
    payload = load_structured(path)
    if not isinstance(payload, dict):
        errors.append("baseline result bundle must contain an object")
        return {"path": str(path)}
    if payload.get("schema_version") != 1:
        errors.append("baseline result bundle schema_version must be 1")
    if payload.get("task_id") != task.task_id:
        errors.append("baseline result bundle task_id does not match task")
    if payload.get("baseline_ladder_sha256") != _sha256_file(task.baseline_ladder):
        errors.append("baseline result bundle does not match the current ladder manifest")
    if payload.get("complete") is not True:
        errors.append("baseline result bundle is not complete")
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("baseline result bundle records must be a list")
        records = []
    completed = [
        record
        for record in records
        if isinstance(record, dict) and record.get("status") == "completed"
    ]
    if isinstance(repeats, int):
        for tier in STATIC_TIERS:
            count = sum(record.get("tier") == tier for record in completed)
            if count < repeats:
                errors.append(
                    f"baseline result bundle has {count}/{repeats} completed {tier} repeats"
                )
    methods = search.get("methods", [])
    seeds = search.get("seeds", [])
    for method in methods:
        for seed in seeds:
            if not any(
                record.get("tier") == "matched_search"
                and record.get("method") == method
                and record.get("seed") == seed
                for record in completed
            ):
                errors.append(
                    f"baseline result bundle is missing matched_search {method} seed {seed}"
                )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "records": len(records),
        "complete": payload.get("complete") is True,
    }


def _task_artifact(task: TaskSpec, value: str) -> Path:
    path = (task.task_dir / value).resolve()
    root = task.task_dir.resolve()
    if path != root and root not in path.parents:
        raise ConfigError(f"baseline ladder path escapes task directory: {value}")
    return path


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
