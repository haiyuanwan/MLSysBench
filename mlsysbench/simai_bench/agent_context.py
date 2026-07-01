"""Build public task context for model agents.

The agent context intentionally excludes hidden files. Hidden workload and
baseline metrics are only read by the evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import load_structured
from mlsysbench.simai_bench.schema import TaskSpec


PUBLIC_OPTIONAL_FILES = ("README.md", "symptoms.txt", "public_report.json")


@dataclass(frozen=True)
class AgentTaskContext:
    task_id: str
    track: str
    description: str
    readme: str | None
    symptoms: str | None
    public_report: dict[str, Any] | None
    baseline_config: dict[str, Any]
    allowed_actions: dict[str, dict[str, Any]]
    objective: dict[str, Any]
    slo: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_agent_context(task_dir: str | Path) -> AgentTaskContext:
    task = TaskSpec.load(task_dir)
    task_dir = Path(task_dir)

    readme = _read_text_if_exists(task_dir / "README.md")
    symptoms = _read_text_if_exists(task_dir / "symptoms.txt")
    public_report_path = task_dir / "public_report.json"
    public_report = load_structured(public_report_path) if public_report_path.exists() else None

    allowed_actions = {}
    for name, spec in task.load_allowed_actions().items():
        allowed_actions[name] = {
            "type": spec.type,
            "choices": list(spec.choices) if spec.choices is not None else None,
            "min": spec.minimum,
            "max": spec.maximum,
        }

    return AgentTaskContext(
        task_id=task.task_id,
        track=task.track,
        description=task.description,
        readme=readme,
        symptoms=symptoms,
        public_report=public_report,
        baseline_config=task.load_baseline_config(),
        allowed_actions=allowed_actions,
        objective={
            "primary_metric": task.objective.primary_metric,
            "direction": task.objective.direction,
            "score_cap": task.objective.score_cap,
        },
        slo={
            "p99_ttft_ms": task.slo.p99_ttft_ms,
            "p99_tbt_ms": task.slo.p99_tbt_ms,
            "p99_e2e_ms": task.slo.p99_e2e_ms,
        },
    )


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")

