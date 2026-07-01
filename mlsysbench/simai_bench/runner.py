"""Runner backends for SimAI benchmark tasks."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlsysbench.simai_bench.actions import to_cli_args
from mlsysbench.simai_bench.io import ConfigError, load_structured, resolve_task_path
from mlsysbench.simai_bench.metrics import load_metrics_json, parse_vidur_output
from mlsysbench.simai_bench.schema import TaskSpec


@dataclass(frozen=True)
class RunResult:
    success: bool
    metrics: dict[str, float]
    output_dir: str | None = None
    error: str | None = None


class Runner(Protocol):
    def run(self, task: TaskSpec, config: dict[str, Any], changes: dict[str, Any]) -> RunResult:
        ...


def make_runner(task: TaskSpec) -> Runner:
    if task.runner.type == "mock":
        return MockRunner()
    if task.runner.type == "vidur":
        return VidurRunner()
    raise ConfigError(f"Unsupported runner type {task.runner.type}")


class MockRunner:
    """Dependency-free runner for evaluator tests and example tasks.

    The mock metrics file maps canonical change signatures to metric objects.
    This lets task authors unit-test scoring without a full SimAI build.
    """

    def run(self, task: TaskSpec, config: dict[str, Any], changes: dict[str, Any]) -> RunResult:
        metrics_path = task.runner.config.get("mock_metrics")
        if not metrics_path:
            raise ConfigError("mock runner requires runner.mock_metrics")
        metrics_file = resolve_task_path(task.task_dir, metrics_path)
        data = load_structured(metrics_file)
        signature = change_signature(changes)
        metrics_data = data.get(signature) or data.get("default")
        if metrics_data is None:
            return RunResult(False, {}, error=f"No mock metrics for signature {signature}")
        return RunResult(True, load_metrics_json(metrics_data))


class VidurRunner:
    def run(self, task: TaskSpec, config: dict[str, Any], changes: dict[str, Any]) -> RunResult:
        vidur_root = Path(task.runner.config.get("vidur_root", "third_party/SimAI/vidur-alibabacloud"))
        output_dir = Path(task.runner.config.get("output_dir", "runs/simai_bench")) / task.task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        run_config = dict(config)
        run_config.setdefault("metrics_config_output_dir", str(output_dir))
        python_bin = task.runner.config.get("python_bin", sys.executable)
        args = [python_bin, "-m", "vidur.main", *to_cli_args(run_config)]
        timeout = int(task.runner.config.get("timeout_seconds", 600))

        try:
            completed = subprocess.run(
                args,
                cwd=vidur_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(False, {}, output_dir=str(output_dir), error=f"timeout: {exc}")

        if completed.returncode != 0:
            return RunResult(
                False,
                {},
                output_dir=str(output_dir),
                error=completed.stderr[-4000:] or completed.stdout[-4000:],
            )

        try:
            metrics = parse_vidur_output(output_dir, task.slo)
        except Exception as exc:  # noqa: BLE001 - surface parser failure in result JSON.
            return RunResult(False, {}, output_dir=str(output_dir), error=str(exc))
        return RunResult(True, metrics, output_dir=str(output_dir))


def change_signature(changes: dict[str, Any]) -> str:
    return json.dumps(changes, sort_keys=True, separators=(",", ":"))
