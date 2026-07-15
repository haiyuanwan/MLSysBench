"""Runner backends for SimAI benchmark tasks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
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
    def run(
        self,
        task: TaskSpec,
        config: dict[str, Any],
        changes: dict[str, Any],
        phase: str = "final",
        files: dict[str, str] | None = None,
        fidelity: str | None = None,
    ) -> RunResult:
        ...


def make_runner(task: TaskSpec) -> Runner:
    if task.runner.type == "mock":
        return MockRunner()
    if task.runner.type == "python_code":
        return PythonCodeRunner()
    if task.runner.type == "vidur":
        return VidurRunner()
    raise ConfigError(f"Unsupported runner type {task.runner.type}")


class MockRunner:
    """Dependency-free runner for evaluator tests and example tasks.

    The mock metrics file maps canonical change signatures to metric objects.
    This lets task authors unit-test scoring without a full SimAI build.
    """

    def run(
        self,
        task: TaskSpec,
        config: dict[str, Any],
        changes: dict[str, Any],
        phase: str = "final",
        files: dict[str, str] | None = None,
        fidelity: str | None = None,
    ) -> RunResult:
        metrics_key = "mock_metrics_development" if phase == "development" else "mock_metrics"
        metrics_path = task.runner.config.get(metrics_key)
        if metrics_path is None and phase == "development":
            metrics_path = task.runner.config.get("mock_metrics")
        if not metrics_path:
            raise ConfigError("mock runner requires runner.mock_metrics")
        metrics_file = resolve_task_path(task.task_dir, metrics_path)
        data = load_structured(metrics_file)
        signature_candidates = [change_signature(changes)]
        allowed_config = {
            key: config[key]
            for key in task.load_allowed_actions()
            if key in config
        }
        signature_candidates.append(change_signature(allowed_config))

        metrics_data = None
        for signature in signature_candidates:
            metrics_data = data.get(signature)
            if metrics_data is not None:
                break
        metrics_data = metrics_data or data.get("default")
        if metrics_data is None:
            return RunResult(
                False,
                {},
                error=f"No mock metrics for signatures {signature_candidates}",
            )
        return RunResult(True, load_metrics_json(metrics_data))


class VidurRunner:
    def run(
        self,
        task: TaskSpec,
        config: dict[str, Any],
        changes: dict[str, Any],
        phase: str = "final",
        files: dict[str, str] | None = None,
        fidelity: str | None = None,
    ) -> RunResult:
        vidur_root = _resolve_runner_path(
            task,
            task.runner.config.get("vidur_root", "third_party/SimAI/vidur-alibabacloud"),
        )
        output_root = (
            _resolve_runner_path(task, task.runner.config.get("output_dir", "runs/simai_bench"))
            / task.task_id
            / phase
        )
        output_dir = output_root / uuid.uuid4().hex[:12]
        output_dir.mkdir(parents=True, exist_ok=True)

        run_config = dict(config)
        run_config.setdefault("metrics_config_output_dir", str(output_dir))
        python_bin = task.runner.config.get("python_bin", sys.executable)
        args = [python_bin, "-m", "vidur.main", *to_cli_args(run_config)]
        timeout = int(task.runner.config.get("timeout_seconds", 600))
        env = os.environ.copy()
        env_config = task.runner.config.get("env", {})
        if not isinstance(env_config, dict):
            raise ConfigError("vidur runner env must be an object")
        env.update({str(key): str(value) for key, value in env_config.items()})

        try:
            completed = subprocess.run(
                args,
                cwd=vidur_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
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
        fallback_error = detect_aicb_failure_or_default(completed.stdout, completed.stderr)
        if fallback_error is not None:
            return RunResult(
                False,
                {},
                output_dir=str(output_dir),
                error=fallback_error,
            )

        try:
            metrics = parse_vidur_output(output_dir, task.slo)
        except Exception as exc:  # noqa: BLE001 - surface parser failure in result JSON.
            return RunResult(False, {}, output_dir=str(output_dir), error=str(exc))
        return RunResult(True, metrics, output_dir=str(output_dir))


class PythonCodeRunner:
    """Run a trusted task evaluator against a bounded submitted source bundle.

    The task evaluator is responsible for executing untrusted candidate code in
    its own sandbox. The bundled scheduler fixture uses bubblewrap and exposes
    only JSON observations to the candidate process.
    """

    def run(
        self,
        task: TaskSpec,
        config: dict[str, Any],
        changes: dict[str, Any],
        phase: str = "final",
        files: dict[str, str] | None = None,
        fidelity: str | None = None,
    ) -> RunResult:
        if task.submission.type != "code" or task.submission.starter_dir is None:
            raise ConfigError("python_code runner requires a code submission")
        try:
            submitted_files = _validate_code_files(task, files or {})
        except ConfigError as exc:
            return RunResult(False, {}, error=str(exc))

        evaluator_value = task.runner.config.get("evaluator")
        if not isinstance(evaluator_value, str) or not evaluator_value:
            raise ConfigError("python_code runner requires runner.evaluator")
        if evaluator_value == "builtin:scheduler_policy_v1":
            evaluator = Path(__file__).with_name("scheduler_policy_evaluator.py")
        else:
            evaluator = resolve_task_path(task.task_dir, evaluator_value)
        if not evaluator.is_file():
            raise ConfigError(f"python_code evaluator does not exist: {evaluator}")
        eval_workload = task.eval_workload_path(phase, fidelity)
        if eval_workload is None:
            raise ConfigError(f"python_code runner requires a {phase} workload")

        timeout = int(task.runner.config.get("timeout_seconds", 60))
        if timeout <= 0:
            raise ConfigError("python_code runner timeout_seconds must be positive")

        with tempfile.TemporaryDirectory(prefix=f"mlsysbench-{task.task_id}-") as root_value:
            root = Path(root_value)
            solution_dir = root / "solution"
            shutil.copytree(task.submission.starter_dir, solution_dir)
            for relative_path, content in submitted_files.items():
                destination = solution_dir / relative_path
                destination.write_text(content, encoding="utf-8")
            config_path = root / "config.json"
            result_path = root / "result.json"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = [
                sys.executable,
                "-I",
                str(evaluator),
                "--solution-dir",
                str(solution_dir),
                "--workload",
                str(eval_workload),
                "--config",
                str(config_path),
                "--output",
                str(result_path),
            ]
            env = {
                "HOME": str(root / "home"),
                "LANG": "C.UTF-8",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONHASHSEED": "0",
                "TMPDIR": str(root),
            }
            try:
                completed = subprocess.run(
                    args,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                return RunResult(False, {}, error=f"code evaluator timeout: {exc}")
            if completed.returncode != 0:
                detail = completed.stderr[-4000:] or completed.stdout[-4000:]
                return RunResult(False, {}, error=f"code evaluator failed: {detail}")
            if not result_path.is_file():
                return RunResult(False, {}, error="code evaluator did not create result JSON")
            try:
                payload = load_structured(result_path)
            except Exception as exc:  # noqa: BLE001 - convert task output to a failed run.
                return RunResult(False, {}, error=f"invalid code evaluator result: {exc}")
            if not isinstance(payload, dict):
                return RunResult(False, {}, error="code evaluator result must be an object")
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                return RunResult(False, {}, error="code evaluator result requires metrics")
            numeric_metrics = load_metrics_json({"metrics": metrics})
            valid = payload.get("valid", True)
            failures = payload.get("failures", [])
            if valid is not True:
                if not isinstance(failures, list):
                    failures = [str(failures)]
                return RunResult(
                    False,
                    numeric_metrics,
                    error="; ".join(str(item) for item in failures) or "code correctness gate failed",
                )
            return RunResult(True, numeric_metrics)


def change_signature(changes: dict[str, Any]) -> str:
    return json.dumps(changes, sort_keys=True, separators=(",", ":"))


def _validate_code_files(task: TaskSpec, files: dict[str, str]) -> dict[str, str]:
    if not isinstance(files, dict) or not all(
        isinstance(name, str) and isinstance(content, str)
        for name, content in files.items()
    ):
        raise ConfigError("code submission files must map paths to UTF-8 text")
    allowed = set(task.submission.editable_files)
    unknown = sorted(set(files) - allowed)
    if unknown:
        raise ConfigError("submission includes non-editable files: " + ", ".join(unknown))
    validated: dict[str, str] = {}
    for relative_path, content in files.items():
        size = len(content.encode("utf-8"))
        if size > task.submission.max_file_bytes:
            raise ConfigError(
                f"submitted file {relative_path} is {size} bytes; maximum is "
                f"{task.submission.max_file_bytes}"
            )
        validated[relative_path] = content
    return validated


def _resolve_runner_path(task: TaskSpec, path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or str(path).startswith("runs/"):
        return cwd_candidate
    return (task.task_dir / path).resolve()


def detect_aicb_failure_or_default(stdout: str, stderr: str) -> str | None:
    combined = f"{stdout}\n{stderr}"
    markers = (
        "AICB data is empty",
        "using default attention execution time",
        "using default MLP execution time",
        "using default MoE execution time",
        "Expected CSV file was NOT created",
        "AICB command failed",
        "无法找到任何AICB CSV",
    )
    for marker in markers:
        if marker in combined:
            return f"AICB failure/default fallback detected: {marker}"
    return None
