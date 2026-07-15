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
    schema_version: int
    task_id: str
    track: str
    description: str
    scenario: dict[str, Any]
    readme: str | None
    symptoms: str | None
    public_report: dict[str, Any] | None
    baseline_config: dict[str, Any]
    allowed_actions: dict[str, dict[str, Any]]
    objective: dict[str, Any]
    slo: dict[str, Any]
    metric_gates: dict[str, dict[str, float | None]]
    constraints: dict[str, Any]
    submission: dict[str, Any]
    development_fidelities: dict[str, Any]
    provenance: dict[str, Any] | None

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
        schema_version=task.schema_version,
        task_id=task.task_id,
        track=task.track,
        description=task.description,
        scenario={
            "family": task.scenario.family,
            "transfer": task.scenario.transfer,
            "starting_point": task.scenario.starting_point,
            "profiles": list(task.scenario.profiles),
        },
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
        metric_gates={
            name: {"min": gate.minimum, "max": gate.maximum}
            for name, gate in task.metric_gates.items()
        },
        constraints={
            "max_gpu_units": task.constraints.max_gpu_units,
            "development_max_gpu_units": task.constraints.development_max_gpu_units,
            "max_steps": task.constraints.max_steps,
            "max_development_cost_units": task.constraints.max_development_cost_units,
            "immutable_fields": list(task.constraints.immutable_fields),
        },
        submission={
            "type": task.submission.type,
            "workspace_dir": "solution" if task.submission.type == "code" else None,
            "editable_files": list(task.submission.editable_files),
            "max_file_bytes": task.submission.max_file_bytes,
        },
        development_fidelities=(
            {
                name: {
                    "kind": spec.kind,
                    "cost_units": spec.cost_units,
                    "max_queries": spec.max_queries,
                    "is_default": name == task.development.default_fidelity,
                }
                for name, spec in task.development.fidelities.items()
            }
            if task.development is not None and task.development.fidelities
            else {}
        ),
        provenance=(
            {
                "source_type": task.provenance.source_type,
                "license": task.provenance.license,
                "publication_status": task.provenance.publication_status,
                "calibration_status": task.provenance.calibration_status,
                "contamination_cutoff": task.provenance.contamination_cutoff,
            }
            if task.provenance is not None
            else None
        ),
    )


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
