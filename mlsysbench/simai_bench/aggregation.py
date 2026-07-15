"""Aggregate immutable CLI-agent run artifacts without dropping failures."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured


def aggregate_runs(runs_dir: str | Path, *, bootstrap_samples: int = 2000) -> dict[str, Any]:
    root = Path(runs_dir)
    manifest_paths = sorted(root.rglob("run_manifest.json"))
    if not manifest_paths:
        raise ConfigError(f"no run_manifest.json files found under {root}")
    records = [_load_run(path) for path in manifest_paths]
    return {
        "schema_version": 1,
        "runs_root": str(root.resolve()),
        "summary": _summarize(records, bootstrap_samples=bootstrap_samples),
        "by_task": _group_summary(records, "task_id", bootstrap_samples),
        "by_model": _group_summary(records, "model", bootstrap_samples),
        "runs": records,
    }


def _load_run(manifest_path: Path) -> dict[str, Any]:
    manifest = load_structured(manifest_path)
    if not isinstance(manifest, dict):
        raise ConfigError(f"manifest must contain an object: {manifest_path}")
    final_path = manifest_path.parent / "final_result.json"
    final = load_structured(final_path) if final_path.is_file() else None
    if not isinstance(final, dict):
        final = {}
    trajectory_path = manifest_path.parent / "development_trajectory.json"
    trajectory = load_structured(trajectory_path) if trajectory_path.is_file() else []
    if not isinstance(trajectory, list):
        trajectory = []

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
        "run_dir": str(manifest_path.parent.resolve()),
        "task_id": str(manifest.get("task_id", "unknown")),
        "model": str(agent.get("model") or "unknown"),
        "scaffold": str(agent.get("scaffold") or "unknown"),
        "status": str(manifest.get("status", "unknown")),
        "valid": final.get("valid") is True,
        "score": float(final.get("score", 0.0)) if final else 0.0,
        "ratio": final_ratio,
        "final_failures": final.get("failures", []),
        "all_gates_passed": gates.get("overall") is True,
        "queries_used": int(budgets.get("development_queries_used", len(trajectory))),
        "cost_units_used": int(
            budgets.get(
                "development_cost_units_used",
                sum(int(item.get("cost_units", 1)) for item in trajectory if isinstance(item, dict)),
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
    }


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
    scores = [float(record["score"]) for record in records]
    valid_ratios = [float(record["ratio"]) for record in records if record["valid"]]
    gaps = [float(record["generalization_gap"]) for record in records if record["generalization_gap"] is not None]
    mean_score = statistics.fmean(scores)
    lower, upper = _bootstrap_mean_interval(scores, bootstrap_samples)
    return {
        "runs": len(records),
        "valid_runs": sum(bool(record["valid"]) for record in records),
        "valid_rate": sum(bool(record["valid"]) for record in records) / len(records),
        "all_gates_pass_rate": sum(bool(record["all_gates_passed"]) for record in records) / len(records),
        "mean_score_failure_as_zero": mean_score,
        "median_score_failure_as_zero": statistics.median(scores),
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


def _bootstrap_mean_interval(values: list[float], samples: int) -> tuple[float, float]:
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(0)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]
