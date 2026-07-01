"""Orchestrate model API generation and evaluator execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.agent_context import build_agent_context
from mlsysbench.simai_bench.evaluator import EvaluationResult, evaluate_and_write
from mlsysbench.simai_bench.io import write_json
from mlsysbench.simai_bench.model_client import ModelClient


@dataclass(frozen=True)
class AgentRunResult:
    submission_path: Path
    result_path: Path
    evaluation: EvaluationResult


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

