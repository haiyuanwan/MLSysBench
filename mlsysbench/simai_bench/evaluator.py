"""End-to-end evaluator for SimAI benchmark submissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.actions import load_submission, merge_config, validate_changes
from mlsysbench.simai_bench.io import write_json
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_submission(task_dir: str | Path, submission_path: str | Path) -> EvaluationResult:
    task = TaskSpec.load(task_dir)
    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    submission = load_submission(str(submission_path))
    changes = validate_changes(submission["changes"], allowed_actions)
    return evaluate_changes(task, baseline_config, allowed_actions, changes)


def evaluate_changes(
    task: TaskSpec,
    baseline_config: dict[str, Any],
    allowed_actions: dict[str, Any],
    changes: dict[str, Any],
    phase: str = "final",
) -> EvaluationResult:
    if phase not in {"development", "final"}:
        raise ValueError("phase must be development or final")
    changes = validate_changes(changes, allowed_actions)
    merged_config = merge_config(baseline_config, changes)
    merged_config.update(task.load_eval_config_overrides(phase))
    gpu_units = task.validate_constraints(changes, merged_config, phase)

    baseline_metrics = task.load_baseline_metrics(phase)
    runner = make_runner(task)
    run_result = runner.run(task, merged_config, changes, phase)
    score = score_metrics(
        baseline_metrics=baseline_metrics,
        agent_metrics=run_result.metrics,
        objective=task.objective,
        slo=task.slo,
        runner_success=run_result.success,
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
    )


def evaluate_and_write(
    task_dir: str | Path,
    submission_path: str | Path,
    output_path: str | Path,
    phase: str = "final",
) -> EvaluationResult:
    task = TaskSpec.load(task_dir)
    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    submission = load_submission(str(submission_path))
    result = evaluate_changes(
        task,
        baseline_config,
        allowed_actions,
        submission["changes"],
        phase=phase,
    )
    write_json(output_path, result.to_dict())
    return result
