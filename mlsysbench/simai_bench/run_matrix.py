"""Declarative, resumable execution of benchmark run matrices."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured, write_json
from mlsysbench.simai_bench.schema import TaskSpec


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CLI_PROFILES = {"custom", "chat-completions", "longcat", "codex"}
_AGENT_MODES = {"benchmark", "debug"}
_ISOLATION = {"landlock", "bwrap", "none"}
_SEARCH_METHODS = {"grid", "random", "tpe", "smac"}


@dataclass(frozen=True)
class MatrixCell:
    cell_id: str
    task_id: str
    task_path: Path
    model: dict[str, Any]
    scaffold: dict[str, Any]
    budget: dict[str, Any]
    seed: int
    repeat: int
    starting_point: str

    def dimensions(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "model": self.model["id"],
            "scaffold": self.scaffold["id"],
            "starting_point": self.starting_point,
            "budget": self.budget["id"],
            "seed": self.seed,
            "repeat": self.repeat,
        }


@dataclass(frozen=True)
class MatrixPlan:
    matrix_id: str
    manifest_path: Path
    manifest_sha256: str
    cells: tuple[MatrixCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "matrix_id": self.matrix_id,
            "source_manifest": str(self.manifest_path),
            "source_manifest_sha256": self.manifest_sha256,
            "cell_count": len(self.cells),
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "dimensions": cell.dimensions(),
                    "task_path": str(cell.task_path),
                    "executor_kind": cell.scaffold["kind"],
                }
                for cell in self.cells
            ],
        }


def plan_matrix(manifest_path: str | Path) -> MatrixPlan:
    path = Path(manifest_path).resolve()
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ConfigError("run matrix manifest must contain an object")
    if payload.get("schema_version") != 1:
        raise ConfigError("run matrix schema_version must be 1")
    matrix_id = _identifier(payload.get("matrix_id"), "matrix_id")
    _reject_secret_fields(payload)

    tasks = _task_axis(path.parent, payload.get("tasks"))
    models = _model_axis(payload.get("models"))
    scaffolds = _scaffold_axis(payload.get("scaffolds"), models)
    budgets = _budget_axis(payload.get("budgets"))
    seeds = _integer_axis(payload.get("seeds"), "seeds", minimum=None)
    repeats = payload.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ConfigError("repeats must be a positive integer")

    cells: list[MatrixCell] = []
    for task_entry, model, scaffold, budget, seed, repeat in itertools.product(
        tasks, models, scaffolds, budgets, seeds, range(repeats)
    ):
        allowed_models = scaffold.get("models")
        if allowed_models is not None and model["id"] not in allowed_models:
            continue
        if scaffold["kind"] == "search" and model.get("name") is not None:
            continue
        if scaffold["kind"] == "cli_agent" and not model.get("name"):
            continue
        semantic = {
            "matrix_id": matrix_id,
            "task": task_entry["id"],
            "task_sha256": task_entry["task_sha256"],
            "model": model,
            "scaffold": scaffold,
            "budget": budget,
            "seed": seed,
            "repeat": repeat,
            "starting_point": task_entry["starting_point"],
        }
        digest = _sha256_json(semantic)[:12]
        prefix = "__".join(
            _slug(str(value))
            for value in (
                task_entry["id"],
                model["id"],
                scaffold["id"],
                budget["id"],
                f"s{seed}",
                f"r{repeat}",
            )
        )
        cells.append(
            MatrixCell(
                cell_id=f"{prefix}__{digest}",
                task_id=task_entry["id"],
                task_path=task_entry["path"],
                model=model,
                scaffold=scaffold,
                budget=budget,
                seed=seed,
                repeat=repeat,
                starting_point=task_entry["starting_point"],
            )
        )
    if not cells:
        raise ConfigError("run matrix expands to zero applicable cells")
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ConfigError("run matrix contains duplicate cells")
    manifest_hash = _sha256_file(path)
    assert manifest_hash is not None
    return MatrixPlan(
        matrix_id=matrix_id,
        manifest_path=path,
        manifest_sha256=manifest_hash,
        cells=tuple(cells),
    )


def run_matrix(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    max_cells: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Execute applicable cells sequentially and preserve every outcome."""

    if max_cells is not None and max_cells <= 0:
        raise ConfigError("max_cells must be positive")
    plan = plan_matrix(manifest_path)
    if dry_run:
        return {**plan.to_dict(), "dry_run": True}

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "matrix_plan.json", plan.to_dict())
    executed = 0
    records: list[dict[str, Any]] = []
    for cell in plan.cells:
        cell_dir = root / "cells" / cell.cell_id
        status_path = cell_dir / "cell_status.json"
        previous = _load_optional_object(status_path)
        if previous.get("state") == "completed":
            records.append(_status_record(cell, previous, "resumed_completed"))
            continue
        if previous.get("state") == "failed" and not retry_failed:
            records.append(_status_record(cell, previous, "resumed_failed"))
            continue
        if max_cells is not None and executed >= max_cells:
            records.append({"cell_id": cell.cell_id, "state": "deferred"})
            continue

        cell_dir.mkdir(parents=True, exist_ok=True)
        attempt = int(previous.get("attempt", 0)) + 1
        attempt_dir = cell_dir / "attempts" / str(attempt)
        artifact_dir = attempt_dir / "artifacts"
        cell_spec = _cell_spec(plan, cell, cell_dir / "attempts")
        _write_immutable_json(cell_dir / "cell_manifest.json", cell_spec)
        command = _cell_command(cell, artifact_dir)
        started_at = _timestamp()
        write_json(
            status_path,
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "state": "running",
                "started_at": started_at,
                "attempt": attempt,
                "artifact_dir": str(artifact_dir),
            },
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        environment = os.environ.copy()
        environment["MLSYSBENCH_RUN_SEED"] = str(cell.seed)
        execution_error: str | None = None
        returncode: int | None = None
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    text=True,
                )
            returncode = completed.returncode
        except Exception as exc:  # noqa: BLE001 - every cell failure must be retained.
            execution_error = str(exc)
        state = "completed" if returncode == 0 else "failed"
        status = {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "state": state,
            "started_at": started_at,
            "completed_at": _timestamp(),
            "attempt": attempt,
            "artifact_dir": str(artifact_dir),
            "exit_code": returncode,
            "execution_error": execution_error,
            "stdout_sha256": _sha256_file(stdout_path),
            "stderr_sha256": _sha256_file(stderr_path),
        }
        write_json(status_path, status)
        records.append(_status_record(cell, status, "executed"))
        executed += 1
        _write_matrix_result(root, plan, records)

    return _write_matrix_result(root, plan, records)


def _cell_spec(
    plan: MatrixPlan,
    cell: MatrixCell,
    artifact_root: Path,
) -> dict[str, Any]:
    task_file = cell.task_path / "task.json"
    if not task_file.is_file():
        task_file = cell.task_path / "task.yaml"
    return {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "matrix_id": plan.matrix_id,
        "matrix_manifest_sha256": plan.manifest_sha256,
        "dimensions": cell.dimensions(),
        "executor": {
            key: value
            for key, value in cell.scaffold.items()
            if key not in {"agent_command"}
        },
        "task": {
            "path": str(cell.task_path),
            "definition_sha256": _sha256_file(task_file),
        },
        "model": cell.model,
        "budget": cell.budget,
        "artifact_root": str(artifact_root),
    }


def _cell_command(cell: MatrixCell, artifact_dir: Path) -> list[str]:
    scaffold = cell.scaffold
    budget = cell.budget
    base = [sys.executable, "-m", "mlsysbench.simai_bench"]
    if scaffold["kind"] == "search":
        return base + [
            "search",
            "--task",
            str(cell.task_path),
            "--output-dir",
            str(artifact_dir),
            "--method",
            scaffold["method"],
            "--budget",
            str(budget["max_queries"]),
            "--seed",
            str(cell.seed),
            "--wall-time-seconds",
            str(budget["wall_time_seconds"]),
        ]

    command = base + [
        "run-cli-agent",
        "--task",
        str(cell.task_path),
        "--output-dir",
        str(artifact_dir),
        "--agent-profile",
        scaffold["agent_profile"],
        "--agent-mode",
        scaffold["agent_mode"],
        "--isolation",
        scaffold["isolation"],
        "--model",
        str(cell.model["name"]),
        "--max-queries",
        str(budget["max_queries"]),
        "--wall-time-seconds",
        str(budget["wall_time_seconds"]),
    ]
    option_map = {
        "base_url": "--base-url",
        "max_output_tokens": "--max-output-tokens",
        "context_window": "--context-window",
        "thinking_budget": "--thinking-budget",
        "model_timeout_seconds": "--model-timeout-seconds",
    }
    for key, option in option_map.items():
        if cell.model.get(key) is not None:
            command.extend([option, str(cell.model[key])])
    if scaffold.get("agent_command") is not None:
        command.extend(["--agent-command", scaffold["agent_command"]])
    if scaffold.get("codex_asset_dir") is not None:
        command.extend(["--codex-asset-dir", scaffold["codex_asset_dir"]])
    for read_path in scaffold.get("agent_read_paths", []):
        command.extend(["--agent-read-path", read_path])
    return command


def _task_axis(base: Path, value: Any) -> list[dict[str, Any]]:
    entries = _object_axis(value, "tasks")
    result: list[dict[str, Any]] = []
    for entry in entries:
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ConfigError("every tasks entry requires path")
        path = (base / path_value).resolve()
        task = TaskSpec.load(path)
        declared_start = entry.get("starting_point")
        if declared_start is not None and declared_start != task.scenario.starting_point:
            raise ConfigError(
                f"task {entry['id']} starting_point does not match task definition"
            )
        task_file = path / "task.json"
        if not task_file.is_file():
            task_file = path / "task.yaml"
        result.append(
            {
                **entry,
                "path": path,
                "starting_point": task.scenario.starting_point,
                "task_sha256": _sha256_file(task_file),
            }
        )
    return result


def _object_axis(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a non-empty list")
    result: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise ConfigError(f"every {label} entry must contain an object")
        identifier = _identifier(item.get("id"), f"{label}.id")
        identifiers.append(identifier)
        result.append({**item, "id": identifier})
    if len(identifiers) != len(set(identifiers)):
        raise ConfigError(f"{label} ids must be unique")
    return result


def _scaffold_axis(value: Any, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = _object_axis(value, "scaffolds")
    model_ids = {model["id"] for model in models}
    for entry in entries:
        kind = entry.get("kind")
        if kind not in {"cli_agent", "search"}:
            raise ConfigError("scaffolds.kind must be cli_agent or search")
        selected_models = entry.get("models")
        if selected_models is not None:
            if not isinstance(selected_models, list) or not selected_models or not all(
                isinstance(item, str) for item in selected_models
            ):
                raise ConfigError("scaffolds.models must be a non-empty string list")
            unknown = sorted(set(selected_models) - model_ids)
            if unknown:
                raise ConfigError("scaffold names unknown models: " + ", ".join(unknown))
        if kind == "search":
            if entry.get("method") not in _SEARCH_METHODS:
                raise ConfigError("search scaffold method must be grid, random, tpe, or smac")
            continue
        if entry.get("agent_profile") not in _CLI_PROFILES:
            raise ConfigError(
                f"cli_agent scaffold requires agent_profile in {sorted(_CLI_PROFILES)}"
            )
        if entry.get("agent_mode") not in _AGENT_MODES:
            raise ConfigError(f"cli_agent scaffold requires agent_mode in {sorted(_AGENT_MODES)}")
        if entry.get("isolation") not in _ISOLATION:
            raise ConfigError(f"cli_agent scaffold requires isolation in {sorted(_ISOLATION)}")
        if entry["agent_profile"] == "custom" and not isinstance(
            entry.get("agent_command"), str
        ):
            raise ConfigError("custom cli_agent scaffold requires agent_command")
        read_paths = entry.get("agent_read_paths", [])
        if not isinstance(read_paths, list) or not all(
            isinstance(path, str) for path in read_paths
        ):
            raise ConfigError("scaffolds.agent_read_paths must be a string list")
    return entries


def _model_axis(value: Any) -> list[dict[str, Any]]:
    entries = _object_axis(value, "models")
    for entry in entries:
        name = entry.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ConfigError("models.name must be null or a non-empty string")
        base_url = entry.get("base_url")
        if base_url is not None and (not isinstance(base_url, str) or not base_url):
            raise ConfigError("models.base_url must be a non-empty string")
        for field in (
            "max_output_tokens",
            "context_window",
            "thinking_budget",
            "model_timeout_seconds",
        ):
            number = entry.get(field)
            if number is not None and (
                not isinstance(number, int) or isinstance(number, bool) or number <= 0
            ):
                raise ConfigError(f"models.{field} must be a positive integer")
    return entries


def _budget_axis(value: Any) -> list[dict[str, Any]]:
    entries = _object_axis(value, "budgets")
    for entry in entries:
        queries = entry.get("max_queries")
        wall_time = entry.get("wall_time_seconds")
        if not isinstance(queries, int) or isinstance(queries, bool) or queries <= 0:
            raise ConfigError("budgets.max_queries must be a positive integer")
        if not isinstance(wall_time, int) or isinstance(wall_time, bool) or wall_time <= 0:
            raise ConfigError("budgets.wall_time_seconds must be a positive integer")
    return entries


def _integer_axis(value: Any, label: str, minimum: int | None) -> list[int]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ConfigError(f"{label} must be a non-empty integer list")
    if minimum is not None and any(item < minimum for item in value):
        raise ConfigError(f"{label} values must be at least {minimum}")
    if len(value) != len(set(value)):
        raise ConfigError(f"{label} must not contain duplicates")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ConfigError(f"{label} must be a safe identifier")
    return value


def _reject_secret_fields(payload: Any, prefix: str = "") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_key", "token", "secret", "password"}:
                raise ConfigError(f"run matrix must not contain secret field {field}")
            _reject_secret_fields(value, field)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _reject_secret_fields(item, f"{prefix}[{index}]")


def _status_record(cell: MatrixCell, status: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "dimensions": cell.dimensions(),
        "state": status.get("state", "unknown"),
        "exit_code": status.get("exit_code"),
        "attempt": status.get("attempt"),
        "source": source,
    }


def _write_matrix_result(
    root: Path,
    plan: MatrixPlan,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        state = str(record.get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    result = {
        "schema_version": 1,
        "matrix_id": plan.matrix_id,
        "matrix_manifest_sha256": plan.manifest_sha256,
        "planned_cells": len(plan.cells),
        "recorded_cells": len(records),
        "status_counts": dict(sorted(counts.items())),
        "cells": records,
    }
    write_json(root / "matrix_result.json", result)
    return result


def _write_immutable_json(path: Path, payload: Any) -> None:
    if path.exists():
        existing = load_structured(path)
        if existing != payload:
            raise ConfigError(f"immutable manifest already exists with different content: {path}")
        return
    write_json(path, payload)


def _load_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = load_structured(path)
    return payload if isinstance(payload, dict) else {}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "cell"


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
