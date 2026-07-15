"""End-to-end evaluator for SimAI benchmark submissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.actions import load_submission, merge_config, validate_changes
from mlsysbench.simai_bench.io import ConfigError, write_json
from mlsysbench.simai_bench.runner import make_runner
from mlsysbench.simai_bench.schema import TaskSpec
from mlsysbench.simai_bench.scoring import score_metrics


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    track: str
    valid: bool
    score: float
    ratio: float
    primary_metric: str
    baseline_metrics: dict[str, float]
    agent_metrics: dict[str, float]
    failures: list[str]
    merged_config: dict[str, Any]
    gpu_units: int
    runner_error: str | None = None
    fidelity: str | None = None
    fidelity_kind: str | None = None
    cost_units: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_submission(task_dir: str | Path, submission_path: str | Path) -> EvaluationResult:
    task = TaskSpec.load(task_dir)
    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    submission = load_submission(str(submission_path))
    _validate_submission_kind(task, submission)
    changes = validate_changes(submission["changes"], allowed_actions)
    return evaluate_changes(
        task,
        baseline_config,
        allowed_actions,
        changes,
        files=submission.get("files"),
    )


def evaluate_changes(
    task: TaskSpec,
    baseline_config: dict[str, Any],
    allowed_actions: dict[str, Any],
    changes: dict[str, Any],
    phase: str = "final",
    files: dict[str, str] | None = None,
    fidelity: str | None = None,
) -> EvaluationResult:
    if phase not in {"development", "final"}:
        raise ValueError("phase must be development or final")
    if task.submission.type != "code" and files:
        raise ConfigError("configuration tasks do not accept submitted source files")
    changes = validate_changes(changes, allowed_actions)
    merged_config = merge_config(baseline_config, changes)
    if phase == "final" and fidelity is not None:
        raise ConfigError("final evaluation does not accept a development fidelity")
    fidelity_spec = task.development_fidelity(fidelity) if phase == "development" else None
    merged_config.update(task.load_eval_config_overrides(phase, fidelity))
    gpu_units = task.validate_constraints(changes, merged_config, phase)

    baseline_metrics = task.load_baseline_metrics(phase, fidelity)
    runner = make_runner(task)
    run_result = runner.run(
        task,
        merged_config,
        changes,
        phase,
        files=files,
        fidelity=fidelity,
    )
    score = score_metrics(
        baseline_metrics=baseline_metrics,
        agent_metrics=run_result.metrics,
        objective=task.objective,
        slo=task.slo,
        runner_success=run_result.success,
        metric_gates=task.metric_gates,
    )

    return EvaluationResult(
        task_id=task.task_id,
        track=task.track,
        valid=score.valid,
        score=score.score,
        ratio=score.ratio,
        primary_metric=task.objective.primary_metric,
        baseline_metrics=baseline_metrics,
        agent_metrics=run_result.metrics,
        failures=score.failures,
        merged_config=merged_config,
        gpu_units=gpu_units,
        runner_error=run_result.error,
        fidelity=fidelity_spec.name if fidelity_spec is not None else None,
        fidelity_kind=fidelity_spec.kind if fidelity_spec is not None else None,
        cost_units=fidelity_spec.cost_units if fidelity_spec is not None else 1,
    )


def evaluate_and_write(
    task_dir: str | Path,
    submission_path: str | Path,
    output_path: str | Path,
    phase: str = "final",
    fidelity: str | None = None,
) -> EvaluationResult:
    task = TaskSpec.load(task_dir)
    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    submission = load_submission(str(submission_path))
    _validate_submission_kind(task, submission)
    result = evaluate_changes(
        task,
        baseline_config,
        allowed_actions,
        submission["changes"],
        phase=phase,
        files=submission.get("files"),
        fidelity=fidelity,
    )
    write_json(output_path, result.to_dict())
    return result


def _validate_submission_kind(task: TaskSpec, submission: dict[str, Any]) -> None:
    if task.submission.type != "code" and submission.get("files"):
        raise ConfigError("configuration tasks do not accept submitted source files")
