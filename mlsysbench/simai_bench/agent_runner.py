"""Orchestrate model API generation and evaluator execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.agent_context import build_agent_context
from mlsysbench.simai_bench.evaluator import EvaluationResult, evaluate_and_write
from mlsysbench.simai_bench.io import write_json
from mlsysbench.simai_bench.model_client import MODEL_METADATA_KEY, ModelClient
from mlsysbench.simai_bench.schema import TaskSpec


@dataclass(frozen=True)
class AgentRunResult:
    submission_path: Path
    result_path: Path
    evaluation: EvaluationResult


@dataclass(frozen=True)
class AgentLoopResult:
    task_id: str
    steps_used: int
    trajectory_path: Path
    final_evaluation: EvaluationResult | None
    best_evaluation: EvaluationResult | None
    model_stats: dict[str, Any]
    development_cost_units_used: int
    fidelity_queries: dict[str, int]
    final_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps_used": self.steps_used,
            "trajectory_path": str(self.trajectory_path),
            "final_evaluation": (
                self.final_evaluation.to_dict() if self.final_evaluation is not None else None
            ),
            "best_evaluation": (
                self.best_evaluation.to_dict() if self.best_evaluation is not None else None
            ),
            "model_stats": self.model_stats,
            "development_cost_units_used": self.development_cost_units_used,
            "fidelity_queries": self.fidelity_queries,
            "final_error": self.final_error,
        }


def run_agent_once(
    task_dir: str | Path,
    output_dir: str | Path,
    client: ModelClient,
) -> AgentRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_agent_context(task_dir)
    context_path = output_dir / "prompt_context.json"
    write_json(context_path, context.to_dict())

    submission = client.generate_submission(context.to_dict())
    submission_path = output_dir / "submission.json"
    write_json(submission_path, submission)

    result_path = output_dir / "result.json"
    evaluation = evaluate_and_write(task_dir, submission_path, result_path)
    return AgentRunResult(
        submission_path=submission_path,
        result_path=result_path,
        evaluation=evaluation,
    )


def run_agent_loop(
    task_dir: str | Path,
    output_dir: str | Path,
    client: ModelClient,
    max_steps: int | None = None,
) -> AgentLoopResult:
    task = TaskSpec.load(task_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_budget = task.constraints.max_steps if max_steps is None else max_steps
    if step_budget <= 0:
        raise ValueError("max_steps must be positive")

    public_context = build_agent_context(task_dir).to_dict()
    write_json(output_dir / "prompt_context.json", public_context)

    trajectory: list[dict[str, Any]] = []
    best_evaluation: EvaluationResult | None = None
    final_evaluation: EvaluationResult | None = None
    final_error: str | None = None
    cost_units_used = 0
    fidelity_queries: dict[str, int] = {}
    last_files: dict[str, str] | None = None
    last_final_files: dict[str, str] | None = None

    for step_index in range(step_budget):
        step_context = dict(public_context)
        step_context["experiment_history"] = [
            {key: value for key, value in item.items() if key != "model_call"}
            for item in trajectory
        ]
        step_context["remaining_steps"] = step_budget - step_index
        step_context["development_cost_units_used"] = cost_units_used
        step_context["development_cost_units_remaining"] = (
            max(0, task.constraints.max_development_cost_units - cost_units_used)
            if task.constraints.max_development_cost_units is not None
            else None
        )
        step_context["instructions"] = (
            "Propose the next configuration experiment. Use prior measured results, "
            "change only allowed fields, and preserve the best-known configuration. "
            "Development results may use a smaller public scale. Set stop=true only "
            "when no further experiment is worthwhile. On the stopping step, optional "
            "final_changes may specify the configuration to submit to hidden evaluation."
        )
        submission = client.generate_submission(step_context)
        if isinstance(submission.get("files"), dict):
            last_files = submission["files"]
        if isinstance(submission.get("final_files"), dict):
            last_final_files = submission["final_files"]

        step_dir = output_dir / f"step_{step_index + 1:03d}"
        submission_path = step_dir / "submission.json"
        result_path = step_dir / "result.json"
        write_json(submission_path, submission)

        error: str | None = None
        fidelity = submission.get("fidelity")
        fidelity_spec = None
        query_cost = 1
        charge_query = False
        try:
            if fidelity is not None and not isinstance(fidelity, str):
                raise ValueError("fidelity must be a string")
            fidelity_spec = task.development_fidelity(fidelity)
            query_cost = fidelity_spec.cost_units if fidelity_spec is not None else 1
            if (
                fidelity_spec is not None
                and fidelity_spec.max_queries is not None
                and fidelity_queries.get(fidelity_spec.name, 0) >= fidelity_spec.max_queries
            ):
                raise ValueError(f"fidelity {fidelity_spec.name!r} query budget exhausted")
            if (
                task.constraints.max_development_cost_units is not None
                and cost_units_used + query_cost
                > task.constraints.max_development_cost_units
            ):
                raise ValueError("development cost-unit budget exhausted")
            charge_query = True
            evaluation = evaluate_and_write(
                task_dir,
                submission_path,
                result_path,
                phase="development",
                fidelity=fidelity,
            )
        except Exception as exc:  # noqa: BLE001 - invalid experiments belong in the trajectory.
            evaluation = None
            error = str(exc)

        if charge_query:
            cost_units_used += query_cost
        if charge_query and fidelity_spec is not None:
            fidelity_queries[fidelity_spec.name] = fidelity_queries.get(fidelity_spec.name, 0) + 1

        final_evaluation = evaluation
        is_best = False
        if evaluation is not None and evaluation.valid:
            if best_evaluation is None or evaluation.score > best_evaluation.score:
                best_evaluation = evaluation
                is_best = True

        trajectory.append(
            {
                "step": step_index + 1,
                "changes": submission.get("changes", {}),
                "files": (
                    {
                        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                        for name, content in submission.get("files", {}).items()
                    }
                    if isinstance(submission.get("files"), dict)
                    and all(
                        isinstance(name, str) and isinstance(content, str)
                        for name, content in submission["files"].items()
                    )
                    else None
                ),
                "notes": submission.get("notes", ""),
                "fidelity": fidelity_spec.name if fidelity_spec is not None else None,
                "fidelity_kind": fidelity_spec.kind if fidelity_spec is not None else None,
                "cost_units": query_cost,
                "cost_units_charged": charge_query,
                "stop": bool(submission.get("stop", False)),
                "final_changes": submission.get("final_changes"),
                "error": error,
                "evaluation": evaluation.to_dict() if evaluation is not None else None,
                "is_best_so_far": is_best,
                "model_call": submission.get(MODEL_METADATA_KEY),
            }
        )
        write_json(output_dir / "trajectory.json", trajectory)

        if submission.get("stop", False):
            break

    if trajectory:
        final_changes = trajectory[-1]["final_changes"] or trajectory[-1]["changes"]
        final_submission_path = output_dir / "final_submission.json"
        final_payload: dict[str, Any] = {"changes": final_changes}
        if task.submission.type == "code":
            final_files = last_final_files or last_files
            if final_files is not None:
                final_payload["files"] = final_files
        write_json(final_submission_path, final_payload)
        try:
            final_evaluation = evaluate_and_write(
                task_dir,
                final_submission_path,
                output_dir / "final_result.json",
                phase="final",
            )
        except Exception as exc:  # noqa: BLE001 - report final-evaluation failure in artifacts.
            final_evaluation = None
            final_error = str(exc)
            write_json(output_dir / "final_error.json", {"error": final_error})
    if best_evaluation is not None:
        write_json(output_dir / "best_result.json", best_evaluation.to_dict())

    return AgentLoopResult(
        task_id=task.task_id,
        steps_used=len(trajectory),
        trajectory_path=output_dir / "trajectory.json",
        final_evaluation=final_evaluation,
        best_evaluation=best_evaluation,
        model_stats=_summarize_model_calls(trajectory),
        development_cost_units_used=cost_units_used,
        fidelity_queries=dict(sorted(fidelity_queries.items())),
        final_error=final_error,
    )


def _summarize_model_calls(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [item.get("model_call") for item in trajectory]
    calls = [call for call in calls if isinstance(call, dict)]
    models = sorted(
        {
            str(call.get("response_model") or call.get("requested_model"))
            for call in calls
            if call.get("response_model") or call.get("requested_model")
        }
    )
    usage_totals: dict[str, float | int] = {}
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[key] = usage_totals.get(key, 0) + value
    return {
        "api_calls": len(calls),
        "models": models,
        "latency_seconds": round(
            sum(
                float(call.get("latency_seconds", 0.0))
                for call in calls
                if isinstance(call.get("latency_seconds", 0.0), (int, float))
            ),
            6,
        ),
        "usage": usage_totals,
    }
