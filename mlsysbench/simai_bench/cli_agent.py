"""Run filesystem-capable CLI agents against an isolated benchmark interface."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from mlsysbench.simai_bench.actions import load_submission
from mlsysbench.simai_bench.agent_context import build_agent_context
from mlsysbench.simai_bench.evaluator import EvaluationResult, evaluate_and_write, evaluate_changes
from mlsysbench.simai_bench.io import ConfigError, load_structured, resolve_task_path, write_json
from mlsysbench.simai_bench.landlock import (
    default_system_read_paths,
    landlock_abi_version,
    restrict_current_process,
)
from mlsysbench.simai_bench.model_client import load_dotenv
from mlsysbench.simai_bench.schema import TaskSpec


PUBLIC_EVALUATOR_HELPER = r'''#!/usr/bin/env python3
"""Submit one development configuration to the MLSysBench evaluator."""

import argparse
import json
import os
import pathlib
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="JSON file containing a changes object")
    parser.add_argument("--output", help="Optional path for the JSON response")
    args = parser.parse_args()

    with open(args.submission, "r", encoding="utf-8") as handle:
        submission = json.load(handle)
    changes = submission.get("changes") if isinstance(submission, dict) else None
    if not isinstance(changes, dict):
        raise SystemExit("submission must contain a changes object")

    request_dir_value = os.environ.get("MLSYSBENCH_EVAL_REQUEST_DIR")
    response_dir_value = os.environ.get("MLSYSBENCH_EVAL_RESPONSE_DIR")
    token = os.environ.get("MLSYSBENCH_EVAL_TOKEN")
    if not request_dir_value or not response_dir_value or not token:
        raise SystemExit("development evaluator environment is unavailable")
    request_dir = pathlib.Path(request_dir_value)
    response_dir = pathlib.Path(response_dir_value)
    request_id = uuid.uuid4().hex
    request_path = request_dir / (request_id + ".json")
    response_path = response_dir / (request_id + ".json")
    temporary_path = request_dir / (request_id + ".tmp")
    request_payload = {"token": token, "operation": "evaluate", "changes": changes}
    temporary_path.write_text(json.dumps(request_payload), encoding="utf-8")
    os.replace(temporary_path, request_path)

    deadline = time.monotonic() + 3600
    while not response_path.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("development evaluator response timed out")
        time.sleep(0.025)
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit("development evaluator returned invalid JSON") from error
    response_path.unlink(missing_ok=True)
    exit_code = 0 if payload.get("ok") else 1
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
'''


FINAL_SUBMISSION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["changes"],
    "properties": {
        "changes": {"type": "object"},
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class CLIAgentRunResult:
    task_id: str
    status: str
    output_dir: Path
    workspace: Path
    trajectory_path: Path
    manifest_path: Path
    queries_used: int
    agent_exit_code: int | None
    timed_out: bool
    final_evaluation: EvaluationResult | None
    final_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "workspace": str(self.workspace),
            "trajectory_path": str(self.trajectory_path),
            "manifest_path": str(self.manifest_path),
            "queries_used": self.queries_used,
            "agent_exit_code": self.agent_exit_code,
            "timed_out": self.timed_out,
            "final_evaluation": (
                self.final_evaluation.to_dict() if self.final_evaluation is not None else None
            ),
            "final_error": self.final_error,
        }


class PreparedAgentRuntime(Protocol):
    agent_command: Sequence[str]
    agent_environment: dict[str, str]
    environment_remove: Sequence[str]
    agent_read_paths: Sequence[str | Path]
    agent_read_write_paths: Sequence[str | Path]
    metadata: dict[str, Any]

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class AgentRuntime(Protocol):
    def prepare(
        self,
        *,
        output_dir: Path,
        workspace: Path,
        mission_path: Path,
    ) -> PreparedAgentRuntime:
        ...


def prepare_public_workspace(
    task_dir: str | Path,
    workspace: str | Path,
    *,
    max_queries: int,
    wall_time_seconds: int,
) -> Path:
    """Create the complete filesystem view intended for a CLI agent."""

    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise ConfigError(f"Agent workspace must be empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".tmp").mkdir()

    context = build_agent_context(task_dir).to_dict()
    write_json(workspace / "task_context.json", context)
    write_json(
        workspace / "budget.json",
        {
            "development_queries": max_queries,
            "wall_time_seconds": wall_time_seconds,
            "final_submission": "final_submission.json",
        },
    )
    write_json(workspace / "final_submission.schema.json", FINAL_SUBMISSION_SCHEMA)

    helper_path = workspace / "evaluate_dev.py"
    helper_path.write_text(PUBLIC_EVALUATOR_HELPER, encoding="utf-8")
    helper_path.chmod(0o755)

    mission_path = workspace / "MISSION.md"
    mission_path.write_text(
        _build_mission(context, max_queries=max_queries, wall_time_seconds=wall_time_seconds),
        encoding="utf-8",
    )
    return mission_path


def run_cli_agent(
    task_dir: str | Path,
    output_dir: str | Path,
    agent_command: str | Sequence[str] | None,
    *,
    wall_time_seconds: int = 3600,
    max_queries: int | None = None,
    isolation: str = "landlock",
    agent_read_paths: Sequence[str | Path] = (),
    agent_read_write_paths: Sequence[str | Path] = (),
    agent_environment: dict[str, str] | None = None,
    agent_runtime: AgentRuntime | None = None,
    agent_scaffold: str = "custom",
    benchmark_mode: bool = False,
) -> CLIAgentRunResult:
    """Run a CLI agent, then replay its final submission in a fresh evaluator."""

    if wall_time_seconds <= 0:
        raise ConfigError("wall_time_seconds must be positive")
    if benchmark_mode and agent_scaffold != "codex-cli+cc-switch":
        raise ConfigError(
            "benchmark mode requires the canonical codex-cli+cc-switch scaffold; "
            "use debug mode for custom or chat-completions agents"
        )
    if benchmark_mode and isolation != "bwrap":
        raise ConfigError(
            "benchmark mode requires bwrap process isolation; "
            "Landlock-only and unisolated runs are debug runs"
        )
    if isolation not in {"landlock", "bwrap", "none"}:
        raise ConfigError("isolation must be one of: landlock, bwrap, none")
    if isolation == "landlock" and landlock_abi_version() is None:
        raise ConfigError("Landlock isolation is unavailable on this host")
    if isolation == "bwrap" and not _bwrap_supported():
        raise ConfigError(
            "bwrap isolation is unavailable; use Landlock or explicitly choose none"
        )

    load_dotenv()
    task_dir = Path(task_dir).resolve()
    task = TaskSpec.load(task_dir)
    query_budget = task.constraints.max_steps if max_queries is None else max_queries
    if query_budget <= 0:
        raise ConfigError("max_queries must be positive")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConfigError(f"CLI agent output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    mission_path = prepare_public_workspace(
        task_dir,
        workspace,
        max_queries=query_budget,
        wall_time_seconds=wall_time_seconds,
    )

    prepared_runtime: PreparedAgentRuntime | None = None
    runtime_metadata: dict[str, Any] | None = None
    environment_remove: set[str] = set()
    effective_read_paths = list(agent_read_paths)
    effective_read_write_paths = list(agent_read_write_paths)
    effective_agent_environment = dict(agent_environment or {})
    if agent_runtime is not None:
        prepared_runtime = agent_runtime.prepare(
            output_dir=output_dir,
            workspace=workspace,
            mission_path=mission_path,
        )
        agent_command = prepared_runtime.agent_command
        effective_read_paths.extend(prepared_runtime.agent_read_paths)
        effective_read_write_paths.extend(prepared_runtime.agent_read_write_paths)
        effective_agent_environment.update(prepared_runtime.agent_environment)
        environment_remove.update(prepared_runtime.environment_remove)
        runtime_metadata = prepared_runtime.metadata
    if agent_command is None:
        raise ConfigError("agent_command or agent_runtime is required")

    trajectory_path = output_dir / "development_trajectory.json"
    server_error_path = output_dir / "development_server.log"
    stdout_path = output_dir / "agent.stdout.log"
    stderr_path = output_dir / "agent.stderr.log"
    manifest_path = output_dir / "run_manifest.json"

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + wall_time_seconds
    evaluator = DevelopmentEvaluatorProcess(
        task_dir=task_dir,
        workspace=workspace,
        max_queries=query_budget,
        deadline_monotonic=deadline_monotonic,
        trajectory_path=trajectory_path,
        error_log_path=server_error_path,
    )

    argv = _expand_agent_command(agent_command, mission_path, workspace)
    effective_argv = list(argv)
    preexec_fn = None
    executable_paths = _executable_read_paths(argv[0])
    if isolation == "landlock":
        read_only_paths = [
            *default_system_read_paths(),
            *executable_paths,
            *(Path(path) for path in effective_read_paths),
        ]
        _validate_landlock_read_paths(task_dir, read_only_paths)
        _validate_landlock_write_paths(task_dir, effective_read_write_paths)

        def install_landlock() -> None:
            restrict_current_process(
                read_only_paths=read_only_paths,
                read_write_paths=[
                    workspace,
                    *(Path(path) for path in effective_read_write_paths),
                ],
            )

        preexec_fn = install_landlock
    elif isolation == "bwrap":
        effective_argv = _wrap_with_bwrap(
            argv,
            workspace,
            task_dir,
            effective_read_write_paths,
        )

    environment = os.environ.copy()
    for key in environment_remove:
        environment.pop(key, None)
    environment.update(
        {
            "MLSYSBENCH_EVAL_REQUEST_DIR": str(workspace / ".eval_requests"),
            "MLSYSBENCH_EVAL_RESPONSE_DIR": str(workspace / ".eval_responses"),
            "MLSYSBENCH_EVAL_TOKEN": "",
            "MLSYSBENCH_PROMPT_FILE": str(mission_path),
            "MLSYSBENCH_CONTEXT_FILE": str(workspace / "task_context.json"),
            "MLSYSBENCH_FINAL_SUBMISSION": str(workspace / "final_submission.json"),
            "TMPDIR": str(workspace / ".tmp"),
            "PWD": str(workspace),
        }
    )
    environment.pop("OLDPWD", None)
    if effective_agent_environment:
        reserved = {
            "MLSYSBENCH_EVAL_REQUEST_DIR",
            "MLSYSBENCH_EVAL_RESPONSE_DIR",
            "MLSYSBENCH_EVAL_TOKEN",
            "MLSYSBENCH_PROMPT_FILE",
            "MLSYSBENCH_CONTEXT_FILE",
            "MLSYSBENCH_FINAL_SUBMISSION",
        }
        conflicts = sorted(reserved & set(effective_agent_environment))
        if conflicts:
            raise ConfigError(
                "agent_environment cannot override evaluator variables: " + ", ".join(conflicts)
            )
        environment.update(
            {str(key): str(value) for key, value in effective_agent_environment.items()}
        )

    agent_exit_code: int | None = None
    agent_launch_error: str | None = None
    timed_out = False
    evaluator_started = False
    runtime_stop_error: str | None = None
    try:
        if prepared_runtime is not None:
            prepared_runtime.start()
            environment.update(prepared_runtime.agent_environment)
        evaluator.start()
        evaluator_started = True
        environment["MLSYSBENCH_EVAL_TOKEN"] = evaluator.token
        remaining = max(0.001, deadline_monotonic - time.monotonic())
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            try:
                process = subprocess.Popen(
                    effective_argv,
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                    preexec_fn=preexec_fn,
                )
                try:
                    agent_exit_code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
                    agent_exit_code = process.returncode
            except (OSError, subprocess.SubprocessError) as exc:
                agent_launch_error = f"Agent process could not start: {exc}"
    except ConfigError as exc:
        agent_launch_error = f"Agent runtime could not start: {exc}"
    finally:
        if evaluator_started:
            evaluator.stop()
        if prepared_runtime is not None:
            try:
                prepared_runtime.stop()
            except ConfigError as exc:
                runtime_stop_error = str(exc)
            runtime_metadata = prepared_runtime.metadata

    trajectory = _load_json_list(trajectory_path)
    queries_used = len(trajectory)
    final_evaluation: EvaluationResult | None = None
    final_error = agent_launch_error or runtime_stop_error
    normalized_submission_path = output_dir / "final_submission.json"
    workspace_submission = workspace / "final_submission.json"

    if final_error is None:
        if timed_out:
            final_error = f"Agent exceeded the {wall_time_seconds}-second wall-time budget"
        elif not workspace_submission.exists():
            final_error = "Agent did not create final_submission.json"
        elif workspace_submission.is_symlink():
            final_error = "final_submission.json must be a regular file, not a symlink"
        else:
            try:
                submission = load_submission(str(workspace_submission))
                normalized_submission: dict[str, Any] = {"changes": submission["changes"]}
                if isinstance(submission.get("notes"), str):
                    normalized_submission["notes"] = submission["notes"]
                write_json(normalized_submission_path, normalized_submission)
                # The development service has stopped. This call loads a new task and runner.
                final_evaluation = evaluate_and_write(
                    task_dir,
                    normalized_submission_path,
                    output_dir / "final_result.json",
                    phase="final",
                )
            except Exception as exc:  # noqa: BLE001 - preserve failed agent artifacts.
                final_error = f"Final evaluation failed: {exc}"

    if final_error is not None:
        write_json(output_dir / "final_error.json", {"error": final_error})

    completed_at = _utc_now()
    elapsed_seconds = round(time.monotonic() - started_monotonic, 6)
    status = _run_status(final_evaluation, final_error, timed_out, agent_exit_code)
    manifest = _build_manifest(
        task=task,
        output_dir=output_dir,
        workspace=workspace,
        argv=argv,
        environment=environment,
        isolation=isolation,
        agent_read_paths=effective_read_paths,
        agent_read_write_paths=effective_read_write_paths,
        agent_runtime=runtime_metadata,
        agent_scaffold=agent_scaffold,
        benchmark_mode=benchmark_mode,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
        wall_time_seconds=wall_time_seconds,
        query_budget=query_budget,
        queries_used=queries_used,
        agent_exit_code=agent_exit_code,
        timed_out=timed_out,
        status=status,
        final_evaluation=final_evaluation,
        final_error=final_error,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    write_json(manifest_path, manifest)

    return CLIAgentRunResult(
        task_id=task.task_id,
        status=status,
        output_dir=output_dir,
        workspace=workspace,
        trajectory_path=trajectory_path,
        manifest_path=manifest_path,
        queries_used=queries_used,
        agent_exit_code=agent_exit_code,
        timed_out=timed_out,
        final_evaluation=final_evaluation,
        final_error=final_error,
    )


class _DevelopmentState:
    def __init__(
        self,
        *,
        task_dir: Path,
        max_queries: int,
        deadline_monotonic: float,
        trajectory_path: Path,
    ) -> None:
        self.task_dir = task_dir
        self.max_queries = max_queries
        self.deadline_monotonic = deadline_monotonic
        self.trajectory_path = trajectory_path
        self.trajectory: list[dict[str, Any]] = []
        self.task = TaskSpec.load(task_dir)
        self.baseline_config = self.task.load_baseline_config()
        self.allowed_actions = self.task.load_allowed_actions()
        _atomic_write_json(self.trajectory_path, self.trajectory)

    def status(self) -> dict[str, Any]:
        used = len(self.trajectory)
        return {
            "queries_used": used,
            "queries_remaining": max(0, self.max_queries - used),
            "seconds_remaining": round(
                max(0.0, self.deadline_monotonic - time.monotonic()), 3
            ),
        }

    def evaluate(self, payload: Any) -> tuple[int, dict[str, Any]]:
        if time.monotonic() >= self.deadline_monotonic:
            return 429, {"ok": False, "error": "wall-time budget exhausted", **self.status()}
        if len(self.trajectory) >= self.max_queries:
            return 429, {"ok": False, "error": "development query budget exhausted", **self.status()}

        query_index = len(self.trajectory) + 1
        changes = payload.get("changes") if isinstance(payload, dict) else None
        record: dict[str, Any] = {
            "query": query_index,
            "submitted_at": _utc_now(),
            "elapsed_seconds": None,
            "changes": changes if isinstance(changes, dict) else None,
            "evaluation": None,
            "error": None,
        }
        started = time.monotonic()
        status_code = 200
        try:
            if not isinstance(changes, dict):
                raise ConfigError("request must contain a changes object")
            evaluation = evaluate_changes(
                self.task,
                self.baseline_config,
                self.allowed_actions,
                changes,
                phase="development",
            )
            record["evaluation"] = evaluation.to_dict()
            response: dict[str, Any] = {
                "ok": True,
                "query": query_index,
                "evaluation": evaluation.to_dict(),
            }
        except ConfigError as exc:
            status_code = 400
            safe_error = _sanitize_task_error(str(exc), self.task_dir)
            record["error"] = safe_error
            response = {"ok": False, "query": query_index, "error": safe_error}
        except Exception:  # noqa: BLE001 - do not expose private paths or internals to the agent.
            status_code = 500
            record["error"] = "development evaluator failed"
            response = {
                "ok": False,
                "query": query_index,
                "error": "development evaluator failed",
            }
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
        self.trajectory.append(record)
        _atomic_write_json(self.trajectory_path, self.trajectory)
        response.update(self.status())
        return status_code, response


class DevelopmentEvaluatorProcess:
    """Own a private development evaluator in a killable child process."""

    def __init__(
        self,
        *,
        task_dir: Path,
        workspace: Path,
        max_queries: int,
        deadline_monotonic: float,
        trajectory_path: Path,
        error_log_path: Path,
    ) -> None:
        self.task_dir = task_dir
        self.workspace = workspace
        self.max_queries = max_queries
        self.deadline_monotonic = deadline_monotonic
        self.trajectory_path = trajectory_path
        self.error_log_path = error_log_path
        self.request_dir = workspace / ".eval_requests"
        self.response_dir = workspace / ".eval_responses"
        self.stop_path = trajectory_path.parent / ".development_evaluator_stop"
        self.process: multiprocessing.Process | None = None
        self.token = secrets.token_urlsafe(32)

    def start(self) -> None:
        self.request_dir.mkdir()
        self.response_dir.mkdir()
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_development_server_main,
            args=(
                str(self.task_dir),
                str(self.request_dir),
                str(self.response_dir),
                str(self.stop_path),
                self.max_queries,
                self.deadline_monotonic,
                str(self.trajectory_path),
                str(self.error_log_path),
                self.token,
                child_connection,
            ),
            name="mlsysbench-development-evaluator",
        )
        self.process.start()
        child_connection.close()
        try:
            if not parent_connection.poll(15):
                raise ConfigError("Development evaluator did not start within 15 seconds")
            message = parent_connection.recv()
        finally:
            parent_connection.close()
        if not isinstance(message, dict) or message.get("ready") is not True:
            error = message.get("error", "unknown startup error") if isinstance(message, dict) else message
            self.stop()
            raise ConfigError(f"Development evaluator failed to start: {error}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.stop_path.write_text("stop\n", encoding="utf-8")
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=3)
        self.stop_path.unlink(missing_ok=True)


def _development_server_main(
    task_dir: str,
    request_dir: str,
    response_dir: str,
    stop_path: str,
    max_queries: int,
    deadline_monotonic: float,
    trajectory_path: str,
    error_log_path: str,
    token: str,
    ready_connection: Any,
) -> None:
    try:
        state = _DevelopmentState(
            task_dir=Path(task_dir),
            max_queries=max_queries,
            deadline_monotonic=deadline_monotonic,
            trajectory_path=Path(trajectory_path),
        )
        request_root = Path(request_dir)
        response_root = Path(response_dir)
        stop_file = Path(stop_path)
        ready_connection.send({"ready": True})
        ready_connection.close()
        while not stop_file.exists():
            processed_any = False
            for request_path in sorted(request_root.glob("*.json")):
                processed_any = True
                response_path = response_root / request_path.name
                response = _process_development_request(state, request_path, token)
                _atomic_write_json(response_path, response)
                request_path.unlink(missing_ok=True)
            if not processed_any:
                time.sleep(0.02)
    except Exception:  # noqa: BLE001 - report full diagnostics only outside the agent workspace.
        detail = traceback.format_exc()
        Path(error_log_path).write_text(detail, encoding="utf-8")
        try:
            ready_connection.send({"error": detail.splitlines()[-1]})
            ready_connection.close()
        except (BrokenPipeError, EOFError, OSError):
            pass


def _process_development_request(
    state: _DevelopmentState,
    request_path: Path,
    token: str,
) -> dict[str, Any]:
    try:
        payload = _load_rpc_request(request_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"ok": False, "status_code": 400, "error": "request must be valid JSON"}
    provided_token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(provided_token, str) or not secrets.compare_digest(provided_token, token):
        return {"ok": False, "status_code": 401, "error": "unauthorized"}
    operation = payload.get("operation")
    if operation == "status":
        return {"ok": True, "status_code": 200, **state.status()}
    if operation != "evaluate":
        return {"ok": False, "status_code": 404, "error": "unknown operation"}
    status_code, response = state.evaluate({"changes": payload.get("changes")})
    response["status_code"] = status_code
    return response


def _load_rpc_request(path: Path) -> Any:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        size = os.fstat(descriptor).st_size
        if size > 1_048_576:
            raise ValueError("request is too large")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            return json.load(handle)
    finally:
        os.close(descriptor)


def _build_mission(
    context: dict[str, Any],
    *,
    max_queries: int,
    wall_time_seconds: int,
) -> str:
    objective = context["objective"]
    return f"""# MLSysBench CLI Agent Mission

Optimize task `{context['task_id']}` as an inference-systems engineer.

Primary objective: `{objective['direction']}` `{objective['primary_metric']}` while satisfying all validity, resource, and SLO gates.

You have {max_queries} development evaluator queries and {wall_time_seconds} seconds. Inspect `task_context.json` for the public task, baseline configuration, allowed actions, constraints, and metrics. The evaluator owns all non-public evaluation inputs.

For each useful experiment:

1. Form one explicit systems hypothesis from prior measurements.
2. Put only legal changed fields in a JSON file, for example `candidate.json` with `{{"changes": {{...}}}}`.
3. Run `python3 evaluate_dev.py candidate.json --output candidate_result.json`.
4. Compare against the baseline and best result, then retain or roll back the change.
5. Avoid duplicate configurations and reserve enough time to validate the final choice.

Before exiting, write exactly one `final_submission.json` matching `final_submission.schema.json`. It must contain a `changes` object and may contain `notes`. The final configuration may differ from the best development configuration when scale or workload transfer justifies it. Do not attempt to inspect files outside this workspace.
"""


def _expand_agent_command(
    command: str | Sequence[str],
    mission_path: Path,
    workspace: Path,
) -> list[str]:
    argv = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not argv:
        raise ConfigError("agent_command cannot be empty")
    prompt = mission_path.read_text(encoding="utf-8")
    replacements = {
        "{prompt}": prompt,
        "{prompt_file}": str(mission_path),
        "{workspace}": str(workspace),
    }
    expanded: list[str] = []
    for argument in argv:
        for marker, value in replacements.items():
            argument = argument.replace(marker, value)
        expanded.append(argument)
    executable = shutil.which(expanded[0])
    if executable is None:
        raise ConfigError(f"Agent executable was not found: {expanded[0]}")
    expanded[0] = str(Path(executable).resolve())
    return expanded


def _executable_read_paths(executable: str) -> list[Path]:
    path = Path(executable)
    paths = [path]
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return paths
    if resolved != path:
        paths.append(resolved)
    return paths


def _validate_landlock_read_paths(
    task_dir: Path,
    read_only_paths: Sequence[Path],
) -> None:
    task_dir = task_dir.resolve()
    for value in read_only_paths:
        path = Path(value).expanduser().resolve()
        overlaps = (
            path == task_dir
            or path in task_dir.parents
            or task_dir in path.parents
        )
        if overlaps:
            raise ConfigError(
                f"Landlock read-only path overlaps the private task tree: {path}"
            )


def _validate_landlock_write_paths(
    task_dir: Path,
    read_write_paths: Sequence[str | Path],
) -> None:
    task_dir = task_dir.resolve()
    for value in read_write_paths:
        path = Path(value).expanduser().resolve()
        overlaps = path == task_dir or path in task_dir.parents or task_dir in path.parents
        if overlaps:
            raise ConfigError(
                f"Landlock read-write path overlaps the private task tree: {path}"
            )


def _wrap_with_bwrap(
    argv: Sequence[str],
    workspace: Path,
    task_dir: Path,
    read_write_paths: Sequence[str | Path] = (),
) -> list[str]:
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        str(task_dir),
        "--bind",
        str(workspace),
        "/workspace",
        "--chdir",
        "/workspace",
    ]
    for value in read_write_paths:
        path = str(Path(value).resolve())
        command.extend(["--bind", path, path])
    command.extend(["--", *argv])
    return command


def _bwrap_supported() -> bool:
    executable = shutil.which("bwrap")
    if executable is None:
        return False
    completed = subprocess.run(
        [executable, "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    return completed.returncode == 0


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=3)


def _run_status(
    final_evaluation: EvaluationResult | None,
    final_error: str | None,
    timed_out: bool,
    agent_exit_code: int | None,
) -> str:
    if timed_out:
        return "timed_out"
    if final_error is not None or final_evaluation is None:
        return "failed"
    if agent_exit_code not in {0, None}:
        return "completed_with_agent_error"
    return "completed"


def _build_manifest(
    *,
    task: TaskSpec,
    output_dir: Path,
    workspace: Path,
    argv: Sequence[str],
    environment: dict[str, str],
    isolation: str,
    agent_read_paths: Sequence[str | Path],
    agent_read_write_paths: Sequence[str | Path],
    agent_runtime: dict[str, Any] | None,
    agent_scaffold: str,
    benchmark_mode: bool,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    wall_time_seconds: int,
    query_budget: int,
    queries_used: int,
    agent_exit_code: int | None,
    timed_out: bool,
    status: str,
    final_evaluation: EvaluationResult | None,
    final_error: str | None,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    git_state = _git_state(task.task_dir)
    reported_stats = _load_json_object(workspace / "chat_agent_stats.json")
    task_file = task.task_dir / "task.json"
    if not task_file.exists():
        task_file = task.task_dir / "task.yaml"
    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "track": task.track,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "budgets": {
            "wall_time_seconds": wall_time_seconds,
            "development_queries": query_budget,
            "development_queries_used": queries_used,
        },
        "agent": {
            "scaffold": agent_scaffold,
            "benchmark_mode": benchmark_mode,
            "command": _redact_argv(argv, environment),
            "exit_code": agent_exit_code,
            "timed_out": timed_out,
            "model": environment.get("MODEL_NAME"),
            "base_url": environment.get("MODEL_BASE_URL"),
            "max_output_tokens": _optional_int(environment.get("MODEL_MAX_TOKENS")),
            "context_window": _optional_int(environment.get("MODEL_CONTEXT_WINDOW")),
            "reported_stats": _compact_agent_stats(reported_stats),
        },
        "isolation": {
            "backend": isolation,
            "hard_filesystem_boundary": isolation in {"landlock", "bwrap"},
            "landlock_abi": landlock_abi_version() if isolation == "landlock" else None,
            "process_isolated": isolation == "bwrap",
            "network_isolated": False,
            "extra_read_only_paths": [str(Path(path).resolve()) for path in agent_read_paths],
            "extra_read_write_paths": [
                str(Path(path).resolve()) for path in agent_read_write_paths
            ],
        },
        "inputs": {
            "task_definition_sha256": _sha256_file(task_file),
            "baseline_config_sha256": _sha256_file(task.baseline_config),
            "allowed_actions_sha256": _sha256_file(task.allowed_actions),
            "development": _phase_fingerprint(task, "development"),
            "final": _phase_fingerprint(task, "final"),
            "public_context_sha256": _sha256_file(workspace / "task_context.json"),
            "mission_sha256": _sha256_file(workspace / "MISSION.md"),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "agent_stdout_sha256": _sha256_file(stdout_path),
            "agent_stderr_sha256": _sha256_file(stderr_path),
            "agent_stats_sha256": _sha256_file(workspace / "chat_agent_stats.json"),
            "agent_transcript_sha256": _sha256_file(
                workspace / "chat_agent_transcript.json"
            ),
        },
        "source": git_state,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "agent_runtime": agent_runtime,
        },
        "final": {
            "valid": final_evaluation.valid if final_evaluation is not None else False,
            "score": final_evaluation.score if final_evaluation is not None else 0.0,
            "error": final_error,
        },
        "gates": _gate_results(final_evaluation, final_error, timed_out, agent_exit_code),
    }


def _phase_fingerprint(task: TaskSpec, phase: str) -> dict[str, Any]:
    spec = task.development if phase == "development" and task.development else task.hidden
    workload = spec.eval_workload
    fingerprint: dict[str, Any] = {
        "workload_sha256": _sha256_file(workload) if workload is not None else None,
        "baseline_metrics_sha256": _sha256_file(spec.baseline_metrics),
        "seeds": _seed_values(workload),
    }
    if task.runner.type == "mock":
        key = "mock_metrics_development" if phase == "development" else "mock_metrics"
        path_value = task.runner.config.get(key)
        if path_value is None and phase == "development":
            path_value = task.runner.config.get("mock_metrics")
        if path_value is not None:
            metrics_path = resolve_task_path(task.task_dir, str(path_value))
            fingerprint["runner_input_sha256"] = _sha256_file(metrics_path)
    return fingerprint


def _seed_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = load_structured(path)
    except Exception:  # noqa: BLE001 - the file hash remains available for malformed fixtures.
        return {}
    seeds: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                if "seed" in str(key).lower() and isinstance(item, (str, int, float)):
                    seeds[child] = item
                else:
                    visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload, "")
    return seeds


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        root = root_result.stdout.strip()
        commit_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        dirty_result = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return {
            "git_commit": commit_result.stdout.strip(),
            "git_dirty": bool(dirty_result.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _redact_argv(argv: Sequence[str], environment: dict[str, str]) -> list[str]:
    secrets_to_redact = [
        value
        for key, value in environment.items()
        if value and len(value) >= 6 and any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    ]
    redacted: list[str] = []
    for argument in argv:
        safe = argument
        for secret_value in secrets_to_redact:
            safe = safe.replace(secret_value, "<redacted>")
        redacted.append(safe)
    return redacted


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_task_error(message: str, task_dir: Path) -> str:
    safe = message.replace(str(task_dir), "<task>")
    safe = safe.replace(str(task_dir.resolve()), "<task>")
    return safe


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _compact_agent_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    return {
        key: stats[key]
        for key in ("model", "api_calls", "latency_seconds", "usage")
        if key in stats
    }


def _gate_results(
    evaluation: EvaluationResult | None,
    final_error: str | None,
    timed_out: bool,
    agent_exit_code: int | None,
) -> dict[str, Any]:
    return {
        "agent_process": not timed_out and agent_exit_code == 0,
        "submission_schema": evaluation is not None and final_error is None,
        "resource": evaluation is not None,
        "simulator": evaluation is not None and evaluation.runner_error is None,
        "slo": evaluation is not None and not evaluation.failures,
        "overall": evaluation.valid if evaluation is not None else False,
        "failures": evaluation.failures if evaluation is not None else [],
    }


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
