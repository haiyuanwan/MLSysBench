"""Aggregate immutable run artifacts without dropping matrix failures."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_PRACTICAL_DELTA = 0.03
_FAMILYWISE_CONFIDENCE = 0.95
_MATRIX_DIMENSIONS = (
    "task",
    "model",
    "scaffold",
    "starting_point",
    "budget",
    "seed",
    "repeat",
)
_PAIRING_DIMENSIONS = ("task", "scaffold", "budget", "seed", "repeat")
_FAILED_RUN_STATUSES = {"completed_with_agent_error", "failed", "timed_out"}


def aggregate_runs(
    runs_dir: str | Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    practical_delta: float = DEFAULT_PRACTICAL_DELTA,
) -> dict[str, Any]:
    """Aggregate standalone runs and matrix cells rooted at ``runs_dir``.

    Matrix cells are authoritative: their immutable ``cell_manifest.json`` supplies
    every experimental dimension, and terminal execution failures remain scored as
    zero even when no run-level manifest was produced.
    """

    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples <= 0
    ):
        raise ConfigError("bootstrap_samples must be a positive integer")
    if (
        not isinstance(practical_delta, (int, float))
        or isinstance(practical_delta, bool)
        or not math.isfinite(float(practical_delta))
        or practical_delta < 0
    ):
        raise ConfigError("practical_delta must be a finite non-negative number")

    root = Path(runs_dir).resolve()
    records, cell_manifest_paths = _discover_records(root)
    if not records:
        raise ConfigError(f"no run_manifest.json or cell_manifest.json files found under {root}")

    planned_cells = _load_planned_cells(root)
    by_model = _group_summary(records, "model", bootstrap_samples)
    _add_model_completeness(by_model, records, planned_cells)
    paired_comparisons, pairing_integrity = _paired_model_comparisons(
        records,
        planned_cells,
        bootstrap_samples=bootstrap_samples,
        practical_delta=float(practical_delta),
    )
    return {
        "schema_version": 1,
        "runs_root": str(root),
        "bootstrap_samples": bootstrap_samples,
        "practical_difference_delta": float(practical_delta),
        "failed_run_score": 0.0,
        "familywise_confidence_level": _FAMILYWISE_CONFIDENCE,
        "summary": _summarize(records, bootstrap_samples=bootstrap_samples),
        "by_task": _group_summary(records, "task_id", bootstrap_samples),
        "by_model": by_model,
        "paired_comparisons": paired_comparisons,
        "pairing_integrity": {
            **pairing_integrity,
            "block_dimensions": list(_PAIRING_DIMENSIONS),
            "cell_manifests": len(cell_manifest_paths),
            "planned_cells": len(planned_cells),
        },
        "runs": records,
    }


def _discover_records(root: Path) -> tuple[list[dict[str, Any]], set[Path]]:
    cell_manifest_paths = {path.resolve() for path in root.rglob("cell_manifest.json")}
    root_ancestor = _find_ancestor_file(root, "cell_manifest.json")
    if root_ancestor is not None:
        cell_manifest_paths.add(root_ancestor)
    artifact_manifests = [
        *root.rglob("run_manifest.json"),
        *root.rglob("search_manifest.json"),
    ]
    standalone_run_manifests: list[Path] = []
    for manifest_path in artifact_manifests:
        ancestor = _find_ancestor_file(manifest_path.parent, "cell_manifest.json")
        if ancestor is not None:
            cell_manifest_paths.add(ancestor)
        elif manifest_path.name == "run_manifest.json":
            standalone_run_manifests.append(manifest_path)

    records = [_load_cell(path) for path in sorted(cell_manifest_paths)]
    records.extend(_load_run(path) for path in sorted(standalone_run_manifests))
    records.sort(key=_record_sort_key)
    return records, cell_manifest_paths


def _find_ancestor_file(start: Path, name: str) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_cell(cell_manifest_path: Path) -> dict[str, Any]:
    cell = _load_object(cell_manifest_path, "cell manifest")
    dimensions = _normalize_dimensions(cell.get("dimensions"), cell_manifest_path)
    cell_dir = cell_manifest_path.parent
    status_path = cell_dir / "cell_status.json"
    status = _load_optional_object(status_path)
    state = str(status.get("state", "unknown"))
    artifact_dir = _cell_artifact_dir(cell_dir, status)

    run_manifest_path = artifact_dir / "run_manifest.json"
    search_manifest_path = artifact_dir / "search_manifest.json"
    if run_manifest_path.is_file():
        record = _load_run(run_manifest_path)
        artifact_kind = "cli_agent"
    elif search_manifest_path.is_file():
        record = _load_search_run(search_manifest_path)
        artifact_kind = "search"
    else:
        record = _load_artifact_directory(artifact_dir)
        artifact_kind = "missing" if not artifact_dir.exists() else "generic"

    run_status = str(record.get("status", "unknown"))
    reported_task = record.get("task_id")
    reported_model = record.get("model")
    reported_scaffold = record.get("scaffold")
    record.update(dimensions)
    record.update(
        {
            "dimensions": dimensions,
            "task_id": dimensions["task"],
            "matrix_id": str(cell.get("matrix_id", "unknown")),
            "cell_id": str(cell.get("cell_id", cell_dir.name)),
            "cell_manifest": str(cell_manifest_path.resolve()),
            "cell_state": state,
            "cell_attempt": status.get("attempt"),
            "cell_exit_code": status.get("exit_code"),
            "run_status": run_status,
            "record_source": f"matrix_{artifact_kind}",
            "reported_task_id": reported_task,
            "reported_model": reported_model,
            "reported_scaffold": reported_scaffold,
            "matrix_dimensions_complete": True,
            "pairing_eligible": state in {"completed", "failed"},
        }
    )
    effective_status = (
        run_status if state == "completed" and run_status != "unknown" else state
    )
    execution_failed = (
        state == "failed"
        or run_status in _FAILED_RUN_STATUSES
        or (state == "completed" and run_status == "unknown")
    )
    record["status"] = effective_status
    if state != "completed" or execution_failed:
        record["valid"] = False
        record["score"] = 0.0
        record["ratio"] = 0.0
        record["all_gates_passed"] = False
        if execution_failed:
            failures = list(record.get("final_failures", []))
            detail = status.get("execution_error")
            if detail:
                failures.append(str(detail))
            elif state == "failed":
                failures.append("matrix cell execution failed")
            elif run_status == "unknown":
                failures.append("matrix cell produced no recognizable run result")
            record["final_failures"] = failures
    record["execution_failed"] = execution_failed
    return record


def _cell_artifact_dir(cell_dir: Path, status: dict[str, Any]) -> Path:
    attempt = status.get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0:
        local = cell_dir / "attempts" / str(attempt) / "artifacts"
        if local.exists():
            return local
    declared = status.get("artifact_dir")
    if isinstance(declared, str) and declared:
        return Path(declared)
    return cell_dir / "attempts" / str(attempt or 1) / "artifacts"


def _load_run(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_object(manifest_path, "manifest")
    final = _load_optional_object(manifest_path.parent / "final_result.json")
    trajectory = _load_optional_list(manifest_path.parent / "development_trajectory.json")
    record = _record_from_artifacts(
        manifest_path.parent,
        manifest,
        final,
        trajectory,
        status=str(manifest.get("status", "unknown")),
    )
    if record["status"] in _FAILED_RUN_STATUSES:
        record["valid"] = False
        record["score"] = 0.0
        record["ratio"] = 0.0
        record["all_gates_passed"] = False
    return record


def _load_search_run(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_object(manifest_path, "search manifest")
    final = _load_optional_object(manifest_path.parent / "final_result.json")
    trajectory = _load_optional_list(manifest_path.parent / "trajectory.json")
    return _record_from_artifacts(
        manifest_path.parent,
        manifest,
        final,
        trajectory,
        status="completed" if final else "failed",
    )


def _load_artifact_directory(artifact_dir: Path) -> dict[str, Any]:
    final = _load_optional_object(artifact_dir / "final_result.json")
    trajectory = _load_optional_list(artifact_dir / "development_trajectory.json")
    if not trajectory:
        trajectory = _load_optional_list(artifact_dir / "trajectory.json")
    return _record_from_artifacts(
        artifact_dir,
        {},
        final,
        trajectory,
        status="completed" if final else "unknown",
    )


def _record_from_artifacts(
    run_dir: Path,
    manifest: dict[str, Any],
    final: dict[str, Any],
    trajectory: list[Any],
    *,
    status: str,
) -> dict[str, Any]:
    valid_steps = [
        item
        for item in trajectory
        if isinstance(item, dict)
        and isinstance(item.get("evaluation"), dict)
        and item["evaluation"].get("valid") is True
    ]
    development_ratios = [float(item["evaluation"].get("ratio", 0.0)) for item in valid_steps]
    signatures = [_experiment_signature(item) for item in trajectory if isinstance(item, dict)]
    duplicate_count = len(signatures) - len(set(signatures))
    first_improvement = next(
        (
            int(item.get("query", index + 1))
            for index, item in enumerate(trajectory)
            if isinstance(item, dict)
            and isinstance(item.get("evaluation"), dict)
            and item["evaluation"].get("valid") is True
            and float(item["evaluation"].get("ratio", 0.0)) > 1.0
        ),
        None,
    )
    final_ratio = float(final.get("ratio", 0.0)) if final else 0.0
    best_development_ratio = max(development_ratios) if development_ratios else None
    budgets = manifest.get("budgets", {}) if isinstance(manifest.get("budgets"), dict) else {}
    agent = manifest.get("agent", {}) if isinstance(manifest.get("agent"), dict) else {}
    gates = manifest.get("gates", {}) if isinstance(manifest.get("gates"), dict) else {}
    return {
        "run_dir": str(run_dir.resolve()),
        "task_id": str(manifest.get("task_id", "unknown")),
        "model": str(agent.get("model") or "unknown"),
        "scaffold": str(agent.get("scaffold") or "unknown"),
        "starting_point": None,
        "budget": None,
        "seed": manifest.get("seed"),
        "repeat": manifest.get("repeat"),
        "status": status,
        "valid": final.get("valid") is True,
        "score": float(final.get("score", 0.0)) if final else 0.0,
        "ratio": final_ratio,
        "final_failures": final.get("failures", []),
        "all_gates_passed": (
            gates.get("overall") is True if gates else final.get("valid") is True
        ),
        "queries_used": int(
            budgets.get("development_queries_used", manifest.get("evaluations", len(trajectory)))
        ),
        "cost_units_used": int(
            budgets.get(
                "development_cost_units_used",
                sum(
                    int(item.get("cost_units", 1))
                    for item in trajectory
                    if isinstance(item, dict)
                ),
            )
        ),
        "unique_experiments": len(set(signatures)),
        "duplicate_queries": duplicate_count,
        "duplicate_query_rate": duplicate_count / len(signatures) if signatures else 0.0,
        "invalid_development_queries": len(trajectory) - len(valid_steps),
        "first_improvement_query": first_improvement,
        "best_development_ratio": best_development_ratio,
        "generalization_gap": (
            final_ratio - best_development_ratio
            if best_development_ratio is not None and final.get("valid") is True
            else None
        ),
        "fidelity_queries": _fidelity_counts(trajectory),
        "matrix_dimensions_complete": False,
        "pairing_eligible": False,
        "execution_failed": status in _FAILED_RUN_STATUSES,
        "record_source": "standalone_run",
    }


def _normalize_dimensions(value: Any, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"matrix dimensions must contain an object: {source}")
    dimensions: dict[str, Any] = {}
    for name in _MATRIX_DIMENSIONS:
        item = value.get(name)
        if name in {"seed", "repeat"}:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ConfigError(f"matrix dimension {name} must be an integer: {source}")
        elif not isinstance(item, str) or not item:
            raise ConfigError(f"matrix dimension {name} must be a non-empty string: {source}")
        dimensions[name] = item
    return dimensions


def _load_planned_cells(root: Path) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for plan_path in sorted(root.rglob("matrix_plan.json")):
        plan = _load_object(plan_path, "matrix plan")
        cells = plan.get("cells")
        if not isinstance(cells, list):
            raise ConfigError(f"matrix plan cells must contain a list: {plan_path}")
        seen_cell_ids: set[str] = set()
        for item in cells:
            if not isinstance(item, dict):
                raise ConfigError(f"matrix plan cell must contain an object: {plan_path}")
            cell_id = str(item.get("cell_id", ""))
            if not cell_id or cell_id in seen_cell_ids:
                raise ConfigError(
                    f"matrix plan contains missing or duplicate cell ids: {plan_path}"
                )
            seen_cell_ids.add(cell_id)
            planned.append(
                {
                    "matrix_id": str(plan.get("matrix_id", "unknown")),
                    "cell_id": cell_id,
                    "dimensions": _normalize_dimensions(item.get("dimensions"), plan_path),
                }
            )
    return planned


def _add_model_completeness(
    by_model: dict[str, Any],
    records: list[dict[str, Any]],
    planned_cells: list[dict[str, Any]],
) -> None:
    planned_counts: dict[str, int] = {}
    for cell in planned_cells:
        model = str(cell["dimensions"]["model"])
        planned_counts[model] = planned_counts.get(model, 0) + 1
    recorded_cell_ids: dict[str, set[tuple[str, str]]] = {}
    for record in records:
        if not record.get("matrix_dimensions_complete"):
            continue
        model = str(record["model"])
        recorded_cell_ids.setdefault(model, set()).add(
            (str(record.get("matrix_id", "unknown")), str(record.get("cell_id", "unknown")))
        )
    for model in sorted(set(by_model) | set(planned_counts)):
        if model not in by_model:
            by_model[model] = _empty_summary()
        recorded = len(recorded_cell_ids.get(model, set()))
        planned = planned_counts.get(model, recorded)
        by_model[model].update(
            {
                "planned_runs": planned,
                "recorded_matrix_runs": recorded,
                "missing_runs": max(0, planned - recorded),
                "completion_rate": recorded / planned if planned else None,
            }
        )


def _paired_model_comparisons(
    records: list[dict[str, Any]],
    planned_cells: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    practical_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_records = [record for record in records if record.get("matrix_dimensions_complete")]
    observations: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = {}
    for record in matrix_records:
        key = _pairing_key(record)
        observations.setdefault((str(record["model"]), key), []).append(record)

    expected: dict[tuple[str, tuple[Any, ...]], list[str]] = {}
    for cell in planned_cells:
        dimensions = cell["dimensions"]
        key = _pairing_key(dimensions)
        expected.setdefault((str(dimensions["model"]), key), []).append(str(cell["cell_id"]))

    models = sorted(
        {str(record["model"]) for record in matrix_records}
        | {str(cell["dimensions"]["model"]) for cell in planned_cells}
    )
    model_pairs = list(itertools.combinations(models, 2))
    comparison_count = len(model_pairs)
    confidence_level = (
        1.0 - (1.0 - _FAMILYWISE_CONFIDENCE) / comparison_count
        if comparison_count
        else _FAMILYWISE_CONFIDENCE
    )

    comparisons: list[dict[str, Any]] = []
    total_missing = 0
    total_duplicates = 0
    for comparison_index, (model_a, model_b) in enumerate(model_pairs):
        if planned_cells:
            keys_a = {key for model, key in expected if model == model_a}
            keys_b = {key for model, key in expected if model == model_b}
            block_keys = sorted(keys_a & keys_b, key=_pairing_key_sort_key)
        else:
            keys_a = {key for model, key in observations if model == model_a}
            keys_b = {key for model, key in observations if model == model_b}
            block_keys = sorted(keys_a | keys_b, key=_pairing_key_sort_key)

        paired_differences: list[float] = []
        missing_blocks: list[dict[str, Any]] = []
        duplicate_blocks: list[dict[str, Any]] = []
        for key in block_keys:
            records_a = observations.get((model_a, key), [])
            records_b = observations.get((model_b, key), [])
            expected_a = expected.get((model_a, key), [])
            expected_b = expected.get((model_b, key), [])
            duplicates = []
            if len(records_a) > 1 or len(expected_a) > 1:
                duplicates.append(model_a)
            if len(records_b) > 1 or len(expected_b) > 1:
                duplicates.append(model_b)
            if duplicates:
                duplicate_blocks.append(
                    {"block": _pairing_block(key), "duplicate_models": duplicates}
                )
                continue

            missing = []
            if len(records_a) != 1 or not records_a[0].get("pairing_eligible"):
                missing.append(model_a)
            if len(records_b) != 1 or not records_b[0].get("pairing_eligible"):
                missing.append(model_b)
            if missing:
                missing_blocks.append({"block": _pairing_block(key), "missing_models": missing})
                continue
            paired_differences.append(float(records_a[0]["score"]) - float(records_b[0]["score"]))

        integrity_complete = bool(block_keys) and not missing_blocks and not duplicate_blocks
        mean_difference: float | None = None
        interval: list[float] | None = None
        decision = "inconclusive"
        better_model: str | None = None
        reason = "incomplete_pairing"
        if integrity_complete and paired_differences:
            mean_difference = statistics.fmean(paired_differences)
            lower, upper = _bootstrap_mean_interval(
                paired_differences,
                bootstrap_samples,
                confidence_level=confidence_level,
                seed=comparison_index,
            )
            interval = [lower, upper]
            decision, better_model = _classify_difference(
                lower,
                upper,
                model_a=model_a,
                model_b=model_b,
                practical_delta=practical_delta,
            )
            reason = "confidence_interval_vs_practical_delta"
        elif not block_keys:
            reason = "no_common_pairing_blocks"

        total_missing += len(missing_blocks)
        total_duplicates += len(duplicate_blocks)
        comparisons.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "difference_definition": "mean(score[model_a] - score[model_b])",
                "block_dimensions": list(_PAIRING_DIMENSIONS),
                "expected_blocks": len(block_keys),
                "paired_blocks": len(paired_differences),
                "missing_blocks": missing_blocks,
                "duplicate_blocks": duplicate_blocks,
                "bootstrap_samples": bootstrap_samples,
                "familywise_confidence_level": _FAMILYWISE_CONFIDENCE,
                "bonferroni_comparisons": comparison_count,
                "confidence_level": confidence_level,
                "confidence_level_percent": 100.0 * confidence_level,
                "mean_score_difference": mean_difference,
                "paired_bootstrap_ci": interval,
                "practical_difference_delta": practical_delta,
                "decision": decision,
                "better_model": better_model,
                "decision_reason": reason,
                "pairing_complete": integrity_complete,
            }
        )

    duplicate_observations = [
        {
            "model": model,
            "block": _pairing_block(key),
            "records": len(items),
        }
        for (model, key), items in sorted(
            observations.items(), key=lambda item: (item[0][0], _pairing_key_sort_key(item[0][1]))
        )
        if len(items) > 1
    ]
    planned_cell_ids = {
        (str(cell["matrix_id"]), str(cell["cell_id"])) for cell in planned_cells
    }
    recorded_cell_ids = {
        (str(record.get("matrix_id", "unknown")), str(record.get("cell_id", "unknown")))
        for record in matrix_records
    }
    missing_planned_cells = [
        {"matrix_id": matrix_id, "cell_id": cell_id}
        for matrix_id, cell_id in sorted(planned_cell_ids - recorded_cell_ids)
    ]
    unexpected_cell_records = [
        {"matrix_id": matrix_id, "cell_id": cell_id}
        for matrix_id, cell_id in sorted(recorded_cell_ids - planned_cell_ids)
    ] if planned_cells else []
    unpairable_records = [
        record["run_dir"] for record in records if not record.get("matrix_dimensions_complete")
    ]
    integrity = {
        "complete": (
            bool(comparisons)
            and all(comparison["pairing_complete"] for comparison in comparisons)
            and not missing_planned_cells
            and not unexpected_cell_records
            and not duplicate_observations
            and not unpairable_records
        ),
        "models": models,
        "model_comparisons": comparison_count,
        "missing_comparison_blocks": total_missing,
        "duplicate_comparison_blocks": total_duplicates,
        "duplicate_observations": duplicate_observations,
        "missing_planned_cells": missing_planned_cells,
        "unexpected_cell_records": unexpected_cell_records,
        "unpairable_standalone_runs": unpairable_records,
    }
    return comparisons, integrity


def _pairing_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[name] for name in _PAIRING_DIMENSIONS)


def _pairing_key_sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(value) for value in key)


def _pairing_block(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(_PAIRING_DIMENSIONS, key, strict=True))


def _classify_difference(
    lower: float,
    upper: float,
    *,
    model_a: str,
    model_b: str,
    practical_delta: float,
) -> tuple[str, str | None]:
    if lower > practical_delta:
        return "better", model_a
    if upper < -practical_delta:
        return "better", model_b
    if lower >= -practical_delta and upper <= practical_delta:
        return "equivalent", None
    return "inconclusive", None


def _experiment_signature(record: dict[str, Any]) -> str:
    payload = {
        "changes": record.get("changes"),
        "files": record.get("files"),
        "fidelity": record.get("fidelity"),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _fidelity_counts(trajectory: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        name = str(item.get("fidelity") or "default")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _group_summary(
    records: list[dict[str, Any]],
    key: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return {
        name: _summarize(group, bootstrap_samples=bootstrap_samples)
        for name, group in sorted(grouped.items())
    }


def _summarize(records: list[dict[str, Any]], *, bootstrap_samples: int) -> dict[str, Any]:
    if not records:
        return _empty_summary()
    scores = [float(record["score"]) for record in records]
    valid_ratios = [float(record["ratio"]) for record in records if record["valid"]]
    gaps = [
        float(record["generalization_gap"])
        for record in records
        if record["generalization_gap"] is not None
    ]
    mean_score = statistics.fmean(scores)
    lower, upper = _bootstrap_mean_interval(scores, bootstrap_samples)
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "runs": len(records),
        "valid_runs": sum(bool(record["valid"]) for record in records),
        "valid_rate": sum(bool(record["valid"]) for record in records) / len(records),
        "execution_failures": sum(bool(record.get("execution_failed")) for record in records),
        "zero_score_runs": sum(float(record["score"]) == 0.0 for record in records),
        "status_counts": dict(sorted(status_counts.items())),
        "all_gates_pass_rate": sum(bool(record["all_gates_passed"]) for record in records)
        / len(records),
        "mean_score_failure_as_zero": mean_score,
        "median_score_failure_as_zero": statistics.median(scores),
        "score_standard_deviation": statistics.pstdev(scores),
        "min_score_failure_as_zero": min(scores),
        "max_score_failure_as_zero": max(scores),
        "mean_score_95pct_bootstrap_ci": [lower, upper],
        "geomean_valid_ratio": (
            math.exp(statistics.fmean(math.log(value) for value in valid_ratios))
            if valid_ratios and all(value > 0 for value in valid_ratios)
            else None
        ),
        "worst_valid_ratio": min(valid_ratios) if valid_ratios else None,
        "mean_generalization_gap": statistics.fmean(gaps) if gaps else None,
        "mean_queries_used": statistics.fmean(record["queries_used"] for record in records),
        "mean_cost_units_used": statistics.fmean(record["cost_units_used"] for record in records),
        "mean_duplicate_query_rate": statistics.fmean(
            record["duplicate_query_rate"] for record in records
        ),
        "invalid_development_query_rate": (
            sum(record["invalid_development_queries"] for record in records)
            / max(1, sum(record["queries_used"] for record in records))
        ),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "runs": 0,
        "valid_runs": 0,
        "valid_rate": 0.0,
        "execution_failures": 0,
        "zero_score_runs": 0,
        "status_counts": {},
        "all_gates_pass_rate": 0.0,
        "mean_score_failure_as_zero": None,
        "median_score_failure_as_zero": None,
        "score_standard_deviation": None,
        "min_score_failure_as_zero": None,
        "max_score_failure_as_zero": None,
        "mean_score_95pct_bootstrap_ci": None,
        "geomean_valid_ratio": None,
        "worst_valid_ratio": None,
        "mean_generalization_gap": None,
        "mean_queries_used": None,
        "mean_cost_units_used": None,
        "mean_duplicate_query_rate": None,
        "invalid_development_query_rate": None,
    }


def _bootstrap_mean_interval(
    values: list[float],
    samples: int,
    *,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence_level) / 2.0
    return _quantile(means, tail), _quantile(means, 1.0 - tail)


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must contain an object: {path}")
    return payload


def _load_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = load_structured(path)
    return payload if isinstance(payload, dict) else {}


def _load_optional_list(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    payload = load_structured(path)
    return payload if isinstance(payload, list) else []


def _record_sort_key(record: dict[str, Any]) -> tuple[str, ...]:
    if record.get("matrix_dimensions_complete"):
        return tuple(str(record.get(name)) for name in _MATRIX_DIMENSIONS) + (
            str(record.get("cell_id")),
        )
    return ("~standalone", str(record.get("run_dir")))
