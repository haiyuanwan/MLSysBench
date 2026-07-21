#!/usr/bin/env python3
"""Replay inline shadow workloads on one real GPU with vLLM 0.11.0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_VLLM_VERSION = "0.11.0"
REPORT_SCHEMA_VERSION = 1
MAX_WORKLOAD_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


class ReplayError(RuntimeError):
    """Raised when a replay cannot produce a valid measurement."""


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int
    priority: int


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    requests: tuple[RequestSpec, ...]
    slo: Mapping[str, float]


@dataclass(frozen=True)
class Workload:
    path: Path
    sha256: str
    schema_version: int
    scenario_family: str | None
    profiles: tuple[str, ...]
    cases: tuple[WorkloadCase, ...]


@dataclass
class RequestState:
    spec: RequestSpec
    internal_id: str
    enqueue_started_ms: float | None = None
    enqueue_completed_ms: float | None = None
    first_token_time_ms: float | None = None
    completion_time_ms: float | None = None
    generated_tokens: int = 0
    finish_reason: str | None = None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay inline shadow-workload cases through the synchronous vLLM "
            "V1 engine on one physical GPU."
        )
    )
    parser.add_argument("--model", required=True, help="Local HF model/config directory")
    parser.add_argument("--workload", required=True, help="Inline shadow workload JSON")
    parser.add_argument("--output", required=True, help="Hardware replay report JSON")
    parser.add_argument(
        "--candidate-scheduler",
        help="Optional candidate vllm/v1/core/sched/scheduler.py",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--scheduling-policy",
        choices=("fcfs", "priority"),
        default="fcfs",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=(1, 8, 16, 32, 64, 128),
        default=16,
    )
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dummy-token-id", type=int, default=1)
    parser.add_argument("--max-progress-steps", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    positive = (
        "repeats",
        "warmup_requests",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_num_partial_prefills",
        "max_long_partial_prefills",
        "max_progress_steps",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_model_len < 16:
        parser.error("--max-model-len must be at least 16")
    if args.max_long_partial_prefills > args.max_num_partial_prefills:
        parser.error(
            "--max-long-partial-prefills cannot exceed --max-num-partial-prefills"
        )
    if args.long_prefill_token_threshold < 0:
        parser.error("--long-prefill-token-threshold cannot be negative")
    if args.num_gpu_blocks_override is not None and args.num_gpu_blocks_override < 1:
        parser.error("--num-gpu-blocks-override must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index cannot be negative")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.dummy_token_id < 0:
        parser.error("--dummy-token-id cannot be negative")
    if not math.isfinite(args.gpu_memory_utilization) or not (
        0.0 < args.gpu_memory_utilization < 1.0
    ):
        parser.error("--gpu-memory-utilization must be finite and between 0 and 1")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_json_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReplayError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"{label} must be a JSON object")
    return value


def _plain_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ReplayError(f"{name} must be at least {minimum}")
    return value


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ReplayError(f"{name} must be at least {minimum}")
    return result


def _positive_slo(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ReplayError(f"{name} must be positive")
    return result


def _load_workload(path: Path, max_model_len: int) -> Workload:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"workload is not a regular file: {path}")
    if path.stat().st_size > MAX_WORKLOAD_BYTES:
        raise ReplayError("workload exceeds the 64 MiB size limit")
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    payload = _load_json_bytes(raw, "workload")
    if "private_bundle" in payload:
        raise ReplayError(
            "hardware replay requires the resolved private workload with inline "
            "cases, not a private_bundle commitment"
        )
    schema_version = _plain_int(payload.get("schema_version"), "schema_version", minimum=1)
    if schema_version != 1:
        raise ReplayError("only shadow workload schema_version 1 is supported")
    scenario_value = payload.get("scenario_family")
    if scenario_value is not None and (
        not isinstance(scenario_value, str) or not scenario_value
    ):
        raise ReplayError("scenario_family must be a non-empty string")
    profiles_value = payload.get("profiles", [])
    if not isinstance(profiles_value, list) or any(
        not isinstance(item, str) or not item for item in profiles_value
    ):
        raise ReplayError("profiles must be an array of non-empty strings")
    cases_value = payload.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ReplayError("workload.cases must be a non-empty array")

    cases: list[WorkloadCase] = []
    case_names: set[str] = set()
    for case_index, case_value in enumerate(cases_value):
        case_label = f"cases[{case_index}]"
        if not isinstance(case_value, dict):
            raise ReplayError(f"{case_label} must be an object")
        name = case_value.get("name")
        if not isinstance(name, str) or not name:
            raise ReplayError(f"{case_label}.name must be a non-empty string")
        if name in case_names:
            raise ReplayError(f"case names must be unique: {name}")
        case_names.add(name)
        if "trace_file" in case_value:
            raise ReplayError(f"case {name} must contain inline requests, not trace_file")
        requests_value = case_value.get("requests")
        if not isinstance(requests_value, list) or not requests_value:
            raise ReplayError(f"case {name} requires a non-empty requests array")
        expected = case_value.get("expected_requests")
        if expected is not None and _plain_int(
            expected, f"case {name}.expected_requests", minimum=1
        ) != len(requests_value):
            raise ReplayError(f"case {name} request count does not match expected_requests")
        slo_value = case_value.get("slo")
        if not isinstance(slo_value, dict):
            raise ReplayError(f"case {name} requires an slo object")
        slo = {
            "ttft_ms": _positive_slo(slo_value.get("ttft_ms"), f"case {name}.slo.ttft_ms"),
            "tpot_ms": _positive_slo(slo_value.get("tpot_ms"), f"case {name}.slo.tpot_ms"),
            "e2e_ms": _positive_slo(slo_value.get("e2e_ms"), f"case {name}.slo.e2e_ms"),
        }

        requests: list[RequestSpec] = []
        request_ids: set[str] = set()
        for request_index, request_value in enumerate(requests_value):
            label = f"case {name}.requests[{request_index}]"
            if not isinstance(request_value, dict):
                raise ReplayError(f"{label} must be an object")
            request_id = request_value.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ReplayError(f"{label}.request_id must be a non-empty string")
            if request_id in request_ids:
                raise ReplayError(f"request IDs must be unique within case {name}")
            request_ids.add(request_id)
            arrival_time_ms = _finite_float(
                request_value.get("arrival_time_ms"),
                f"{label}.arrival_time_ms",
                minimum=0.0,
            )
            prompt_tokens = _plain_int(
                request_value.get("prompt_tokens"),
                f"{label}.prompt_tokens",
                minimum=1,
            )
            output_tokens = _plain_int(
                request_value.get("output_tokens"),
                f"{label}.output_tokens",
                minimum=1,
            )
            priority = _plain_int(request_value.get("priority", 0), f"{label}.priority")
            if prompt_tokens + output_tokens > max_model_len:
                raise ReplayError(
                    f"request {request_id} exceeds max_model_len={max_model_len}"
                )
            requests.append(
                RequestSpec(
                    request_id=request_id,
                    arrival_time_ms=arrival_time_ms,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    priority=priority,
                )
            )
        requests.sort(key=lambda request: (request.arrival_time_ms, request.request_id))
        cases.append(WorkloadCase(name=name, requests=tuple(requests), slo=slo))

    return Workload(
        path=resolved,
        sha256=_sha256_bytes(raw),
        schema_version=schema_version,
        scenario_family=scenario_value,
        profiles=tuple(profiles_value),
        cases=tuple(cases),
    )


def _validate_model(path: Path, dummy_token_id: int) -> tuple[Path, Path, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ReplayError(f"model must be a local directory: {resolved}")
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise ReplayError(f"model config is missing: {config_path}")
    config = _load_json_bytes(config_path.read_bytes(), "model config.json")
    vocab_size = config.get("vocab_size")
    text_config = config.get("text_config")
    if vocab_size is None and isinstance(text_config, dict):
        vocab_size = text_config.get("vocab_size")
    if vocab_size is not None:
        size = _plain_int(vocab_size, "model vocab_size", minimum=1)
        if dummy_token_id >= size:
            raise ReplayError(
                f"dummy token id {dummy_token_id} is outside vocab_size={size}"
            )
    return resolved, config_path, config


def _validate_candidate_path(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"candidate scheduler is not a regular file: {path}")
    if path.stat().st_size > MAX_CANDIDATE_BYTES:
        raise ReplayError("candidate scheduler exceeds the 4 MiB size limit")
    resolved = path.resolve(strict=True)
    if resolved.name != "scheduler.py":
        raise ReplayError("candidate scheduler path must end in scheduler.py")
    return resolved


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ReplayError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def _request_measurement(state: RequestState, slo: Mapping[str, float]) -> dict[str, Any]:
    required = (
        state.enqueue_started_ms,
        state.enqueue_completed_ms,
        state.first_token_time_ms,
        state.completion_time_ms,
    )
    if any(value is None for value in required):
        raise ReplayError(f"request {state.spec.request_id} has incomplete timestamps")
    if state.generated_tokens != state.spec.output_tokens:
        raise ReplayError(
            f"request {state.spec.request_id} generated {state.generated_tokens} "
            f"tokens, expected {state.spec.output_tokens}"
        )
    assert state.enqueue_started_ms is not None
    assert state.enqueue_completed_ms is not None
    assert state.first_token_time_ms is not None
    assert state.completion_time_ms is not None
    ttft_ms = state.first_token_time_ms - state.spec.arrival_time_ms
    e2e_ms = state.completion_time_ms - state.spec.arrival_time_ms
    tpot_ms = max(0.0, state.completion_time_ms - state.first_token_time_ms) / max(
        1, state.spec.output_tokens - 1
    )
    if min(ttft_ms, tpot_ms, e2e_ms) < -1e-6:
        raise ReplayError(f"request {state.spec.request_id} produced negative latency")
    slo_pass = (
        ttft_ms <= float(slo["ttft_ms"])
        and tpot_ms <= float(slo["tpot_ms"])
        and e2e_ms <= float(slo["e2e_ms"])
    )
    return {
        "request_id": state.spec.request_id,
        "arrival_time_ms": state.spec.arrival_time_ms,
        "prompt_tokens": state.spec.prompt_tokens,
        "expected_output_tokens": state.spec.output_tokens,
        "generated_tokens": state.generated_tokens,
        "priority": state.spec.priority,
        "enqueue_started_ms": state.enqueue_started_ms,
        "enqueue_completed_ms": state.enqueue_completed_ms,
        "injection_lag_ms": max(
            0.0, state.enqueue_completed_ms - state.spec.arrival_time_ms
        ),
        "first_token_time_ms": state.first_token_time_ms,
        "completion_time_ms": state.completion_time_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "finish_reason": state.finish_reason,
        "slo_pass": slo_pass,
    }


def _latency_metrics(measurements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not measurements:
        raise ReplayError("cannot summarize an empty request measurement set")
    result: dict[str, Any] = {"num_requests": len(measurements)}
    for output_name, source_name in (
        ("ttft", "ttft_ms"),
        ("tpot", "tpot_ms"),
        ("e2e", "e2e_ms"),
    ):
        values = [_finite_float(row[source_name], source_name, minimum=0.0) for row in measurements]
        result[f"mean_{output_name}_ms"] = statistics.fmean(values)
        result[f"p50_{output_name}_ms"] = _percentile(values, 0.50)
        result[f"p95_{output_name}_ms"] = _percentile(values, 0.95)
        result[f"p99_{output_name}_ms"] = _percentile(values, 0.99)
    result["p99_tbt_ms"] = result["p99_tpot_ms"]
    injection_lags = [float(row["injection_lag_ms"]) for row in measurements]
    result["p50_injection_lag_ms"] = _percentile(injection_lags, 0.50)
    result["p99_injection_lag_ms"] = _percentile(injection_lags, 0.99)
    return result


def _repeat_metrics(
    measurements: Sequence[Mapping[str, Any]],
    *,
    duration_ms: float,
    scheduler_steps: int,
    step_latencies_ms: Sequence[float],
    max_active_requests: int,
) -> dict[str, Any]:
    metrics = _latency_metrics(measurements)
    if not math.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ReplayError("measured case duration must be positive and finite")
    passed = sum(bool(row["slo_pass"]) for row in measurements)
    duration_seconds = duration_ms / 1000.0
    metrics.update(
        {
            "completion_rate": 1.0,
            "request_slo_pass_rate": passed / len(measurements),
            "duration_ms": duration_ms,
            "throughput_rps": len(measurements) / duration_seconds,
            "goodput_rps": passed / duration_seconds,
            "scheduler_steps": scheduler_steps,
            "mean_step_latency_ms": statistics.fmean(step_latencies_ms),
            "p99_step_latency_ms": _percentile(step_latencies_ms, 0.99),
            "max_active_requests": max_active_requests,
        }
    )
    return metrics


def _aggregate_case_repeats(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not repeats:
        raise ReplayError("case has no repeat results")
    measurements = [
        row
        for repeat in repeats
        for row in repeat["requests"]
    ]
    metrics = _latency_metrics(measurements)
    repeat_metrics = [repeat["metrics"] for repeat in repeats]
    for name in (
        "duration_ms",
        "throughput_rps",
        "goodput_rps",
        "request_slo_pass_rate",
        "scheduler_steps",
        "mean_step_latency_ms",
        "p99_step_latency_ms",
        "max_active_requests",
    ):
        metrics[f"median_repeat_{name}"] = statistics.median(
            float(value[name]) for value in repeat_metrics
        )
    metrics["repeats"] = len(repeats)
    metrics["request_measurements"] = len(measurements)
    metrics["completion_rate"] = min(
        float(value["completion_rate"]) for value in repeat_metrics
    )
    metrics["request_slo_pass_rate"] = statistics.fmean(
        float(value["request_slo_pass_rate"]) for value in repeat_metrics
    )
    metrics["throughput_rps"] = statistics.median(
        float(value["throughput_rps"]) for value in repeat_metrics
    )
    metrics["goodput_rps"] = statistics.median(
        float(value["goodput_rps"]) for value in repeat_metrics
    )
    return metrics


def _aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ReplayError("report has no case results")
    metrics = [case["aggregate_metrics"] for case in cases]
    goodputs = [float(value["goodput_rps"]) for value in metrics]
    robust_goodput = (
        math.exp(statistics.fmean(math.log(value) for value in goodputs))
        if all(value > 0.0 for value in goodputs)
        else 0.0
    )
    return {
        "profile_count": len(cases),
        "repeat_count": sum(int(value["repeats"]) for value in metrics),
        "request_measurements": sum(
            int(value["request_measurements"]) for value in metrics
        ),
        "completion_rate": min(float(value["completion_rate"]) for value in metrics),
        "request_slo_pass_rate": min(
            float(value["request_slo_pass_rate"]) for value in metrics
        ),
        "robust_goodput_rps": robust_goodput,
        "goodput_rps": robust_goodput,
        "worst_profile_goodput_rps": min(goodputs),
        "p99_ttft_ms": max(float(value["p99_ttft_ms"]) for value in metrics),
        "p99_tpot_ms": max(float(value["p99_tpot_ms"]) for value in metrics),
        "p99_tbt_ms": max(float(value["p99_tbt_ms"]) for value in metrics),
        "p99_e2e_ms": max(float(value["p99_e2e_ms"]) for value in metrics),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReplayError(f"cannot run {' '.join(command)}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReplayError(
            f"command {' '.join(command)} failed: {detail[:2000]}"
        )
    return completed.stdout.strip()


def _nvidia_smi_device(host_gpu_index: int) -> dict[str, Any]:
    output = _run_checked(
        [
            "nvidia-smi",
            "-i",
            str(host_gpu_index),
            "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ReplayError("nvidia-smi did not return exactly one complete GPU row")
    values = [value.strip() for value in rows[0]]
    try:
        index = int(values[0])
        memory_mib = int(values[4])
    except ValueError as exc:
        raise ReplayError("nvidia-smi returned invalid numeric GPU fields") from exc
    if index != host_gpu_index:
        raise ReplayError(
            f"nvidia-smi selected GPU {index}, expected host index {host_gpu_index}"
        )
    return {
        "host_gpu_index": index,
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": memory_mib,
        "pci_bus_id": values[5],
    }


def _git_provenance() -> dict[str, Any]:
    try:
        commit = _run_checked(["git", "rev-parse", "HEAD"], cwd=ROOT)
        status = _run_checked(["git", "status", "--porcelain"], cwd=ROOT)
    except ReplayError as exc:
        return {"available": False, "error": str(exc)}
    lines = status.splitlines() if status else []
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(lines),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
        "status_lines": lines,
    }


def _load_candidate_scheduler(
    candidate_path: Path,
    baseline_module: Any,
    engine_core_module: Any,
) -> type[Any]:
    module_name = "vllm.v1.core.sched.scheduler"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ReplayError("could not create a module spec for candidate scheduler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (KeyboardInterrupt, GeneratorExit):
        sys.modules[module_name] = baseline_module
        raise
    except BaseException as exc:
        sys.modules[module_name] = baseline_module
        raise ReplayError(f"candidate scheduler import failed: {exc}") from exc
    scheduler_type = getattr(module, "Scheduler", None)
    if not isinstance(scheduler_type, type):
        sys.modules[module_name] = baseline_module
        raise ReplayError("candidate scheduler.py must define a Scheduler class")

    # vLLM 0.11.0 resolves scheduler_cls inside EngineCore.__init__. Setting
    # both aliases makes the requested monkeypatch explicit; passing this class
    # as EngineArgs.scheduler_cls below guarantees that it is actually used.
    engine_core_module.Scheduler = scheduler_type
    engine_core_module.V1Scheduler = scheduler_type
    return scheduler_type


def _effective_scheduler(engine: Any) -> Any:
    client = getattr(engine, "engine_core", None)
    core = getattr(client, "engine_core", None)
    scheduler = getattr(core, "scheduler", None)
    if scheduler is None:
        raise ReplayError(
            "V1 in-process EngineCore scheduler is inaccessible; multiprocessing "
            "may have been enabled unexpectedly"
        )
    return scheduler


def _assert_engine_drained(engine: Any, label: str) -> None:
    scheduler = _effective_scheduler(engine)
    try:
        unfinished = int(scheduler.get_num_unfinished_requests())
        tracked = len(scheduler.requests)
        kv_usage = float(scheduler.kv_cache_manager.usage())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ReplayError(f"cannot audit scheduler state after {label}: {exc}") from exc
    if unfinished != 0 or tracked != 0:
        raise ReplayError(
            f"scheduler retained requests after {label}: "
            f"unfinished={unfinished}, tracked={tracked}"
        )
    if not math.isfinite(kv_usage) or abs(kv_usage) > 1e-12:
        raise ReplayError(f"scheduler retained KV cache after {label}: usage={kv_usage}")


def _actual_engine_runtime(
    engine: Any, torch: Any, args: argparse.Namespace
) -> dict[str, Any]:
    config = engine.get_vllm_config()
    model_config = config.model_config
    scheduler_config = config.scheduler_config
    cache_config = config.cache_config
    parallel_config = config.parallel_config
    checks = {
        "dtype": model_config.dtype == torch.bfloat16,
        "enforce_eager": model_config.enforce_eager is True,
        "model_max_len": model_config.max_model_len == args.max_model_len,
        "scheduler_max_model_len": scheduler_config.max_model_len
        == args.max_model_len,
        "tensor_parallel_size": parallel_config.tensor_parallel_size == 1,
        "pipeline_parallel_size": parallel_config.pipeline_parallel_size == 1,
        "prefix_caching": cache_config.enable_prefix_caching is False,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens
        == args.max_num_batched_tokens,
        "max_num_seqs": scheduler_config.max_num_seqs == args.max_num_seqs,
        "max_num_partial_prefills": scheduler_config.max_num_partial_prefills
        == args.max_num_partial_prefills,
        "max_long_partial_prefills": scheduler_config.max_long_partial_prefills
        == args.max_long_partial_prefills,
        "long_prefill_token_threshold": scheduler_config.long_prefill_token_threshold
        == args.long_prefill_token_threshold,
        "chunked_prefill": scheduler_config.chunked_prefill_enabled
        is args.enable_chunked_prefill,
        "scheduling_policy": scheduler_config.policy == args.scheduling_policy,
        "block_size": cache_config.block_size == args.block_size,
    }
    if args.num_gpu_blocks_override is not None:
        checks["num_gpu_blocks_override"] = (
            cache_config.num_gpu_blocks == args.num_gpu_blocks_override
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ReplayError(
            "vLLM effective configuration differs from the fixed replay "
            f"configuration: {', '.join(failed)}"
        )
    num_gpu_blocks = cache_config.num_gpu_blocks
    if isinstance(num_gpu_blocks, bool) or not isinstance(num_gpu_blocks, int):
        raise ReplayError("vLLM did not expose an integer GPU KV block count")
    if num_gpu_blocks < 1:
        raise ReplayError("vLLM initialized no GPU KV cache blocks")

    client = getattr(engine, "engine_core", None)
    core = getattr(client, "engine_core", None)
    executor = getattr(core, "model_executor", None)
    driver_worker = getattr(executor, "driver_worker", None)
    model_runner = getattr(driver_worker, "model_runner", None)
    groups = getattr(model_runner, "attn_groups", None)
    if not isinstance(groups, list) or not groups:
        raise ReplayError("cannot resolve the effective vLLM attention backend")
    attention_backends: set[str] = set()
    for group_list in groups:
        for group in group_list:
            backend = getattr(group, "backend", None)
            if not isinstance(backend, type):
                raise ReplayError("vLLM exposed an invalid attention backend")
            full_name = getattr(backend, "full_cls_name", None)
            identity = full_name() if callable(full_name) else None
            if (
                isinstance(identity, tuple)
                and len(identity) == 2
                and all(isinstance(value, str) for value in identity)
            ):
                name = ".".join(identity)
            else:
                name = f"{backend.__module__}.{backend.__qualname__}"
            attention_backends.add(name)
    if not attention_backends:
        raise ReplayError("vLLM initialized no attention backend")
    scheduler = _effective_scheduler(engine)
    return {
        "dtype": str(model_config.dtype),
        "enforce_eager": bool(model_config.enforce_eager),
        "max_model_len": int(model_config.max_model_len),
        "tensor_parallel_size": int(parallel_config.tensor_parallel_size),
        "pipeline_parallel_size": int(parallel_config.pipeline_parallel_size),
        "distributed_executor_backend": str(
            parallel_config.distributed_executor_backend
        ),
        "prefix_caching": bool(cache_config.enable_prefix_caching),
        "kv_cache_dtype": str(cache_config.cache_dtype),
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": int(cache_config.block_size),
        "attention_backends": sorted(attention_backends),
        "executor_class": (
            f"{executor.__class__.__module__}.{executor.__class__.__qualname__}"
        ),
        "worker_class": f"{driver_worker.worker.__class__.__module__}."
        f"{driver_worker.worker.__class__.__qualname__}",
        "scheduler_class": f"{scheduler.__class__.__module__}."
        f"{scheduler.__class__.__qualname__}",
        "load_format": str(config.load_config.load_format),
        "quantization": model_config.quantization,
        "checks": checks,
    }


def _wait_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 0.001:
            time.sleep(remaining - 0.0005)
        else:
            time.sleep(0)


def _run_warmup(
    engine: Any,
    torch: Any,
    sampling_params_type: Any,
    output_kind_delta: Any,
    *,
    warmup_requests: int,
    max_model_len: int,
    dummy_token_id: int,
    max_progress_steps: int,
) -> dict[str, Any]:
    prompt_tokens = min(128, max_model_len - 8)
    output_tokens = min(8, max_model_len - prompt_tokens)
    expected: dict[str, int] = {}
    for index in range(warmup_requests):
        request_id = f"__hardware_replay_warmup_{index}"
        expected[request_id] = 0
        params = sampling_params_type(
            max_tokens=output_tokens,
            ignore_eos=True,
            temperature=0.0,
            detokenize=False,
            output_kind=output_kind_delta,
        )
        engine.add_request(
            request_id,
            {"prompt_token_ids": [dummy_token_id] * prompt_tokens},
            params,
            arrival_time=time.time(),
            priority=0,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    steps = 0
    while engine.has_unfinished_requests():
        torch.cuda.synchronize()
        outputs = engine.step()
        torch.cuda.synchronize()
        steps += 1
        if steps > max_progress_steps:
            raise ReplayError("warmup exceeded the progress-step limit")
        for output in outputs:
            if output.request_id not in expected:
                raise ReplayError(f"warmup returned unknown request {output.request_id}")
            if len(output.outputs) != 1:
                raise ReplayError("warmup expected exactly one completion per request")
            expected[output.request_id] += len(output.outputs[0].token_ids)
    torch.cuda.synchronize()
    duration_ms = (time.perf_counter() - started) * 1000.0
    if any(value != output_tokens for value in expected.values()):
        raise ReplayError("warmup did not generate the requested token count")
    return {
        "requests": warmup_requests,
        "prompt_tokens_per_request": prompt_tokens,
        "output_tokens_per_request": output_tokens,
        "scheduler_steps": steps,
        "duration_ms": duration_ms,
    }


def _replay_case(
    engine: Any,
    torch: Any,
    sampling_params_type: Any,
    output_kind_delta: Any,
    case: WorkloadCase,
    *,
    case_index: int,
    repeat_index: int,
    dummy_token_id: int,
    max_progress_steps: int,
) -> dict[str, Any]:
    states: list[RequestState] = []
    by_internal_id: dict[str, RequestState] = {}
    for request_index, request in enumerate(case.requests):
        internal_id = f"hw-c{case_index:04d}-r{repeat_index:04d}-q{request_index:06d}"
        state = RequestState(spec=request, internal_id=internal_id)
        states.append(state)
        by_internal_id[internal_id] = state

    torch.cuda.synchronize()
    wall_start = time.time()
    perf_start = time.perf_counter()
    pending_index = 0
    active: set[str] = set()
    scheduler_steps = 0
    step_latencies_ms: list[float] = []
    max_active_requests = 0

    while pending_index < len(states) or active or engine.has_unfinished_requests():
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0
        while (
            pending_index < len(states)
            and states[pending_index].spec.arrival_time_ms <= elapsed_ms + 1e-9
        ):
            state = states[pending_index]
            state.enqueue_started_ms = (time.perf_counter() - perf_start) * 1000.0
            params = sampling_params_type(
                max_tokens=state.spec.output_tokens,
                ignore_eos=True,
                temperature=0.0,
                detokenize=False,
                output_kind=output_kind_delta,
            )
            engine.add_request(
                state.internal_id,
                {"prompt_token_ids": [dummy_token_id] * state.spec.prompt_tokens},
                params,
                arrival_time=wall_start + state.spec.arrival_time_ms / 1000.0,
                priority=state.spec.priority,
            )
            state.enqueue_completed_ms = (time.perf_counter() - perf_start) * 1000.0
            active.add(state.internal_id)
            pending_index += 1
        max_active_requests = max(max_active_requests, len(active))

        if not active:
            if engine.has_unfinished_requests():
                raise ReplayError("engine reports unfinished requests absent from replay state")
            if pending_index < len(states):
                target = perf_start + states[pending_index].spec.arrival_time_ms / 1000.0
                _wait_until(target)
            continue

        torch.cuda.synchronize()
        step_started = time.perf_counter()
        outputs = engine.step()
        torch.cuda.synchronize()
        step_completed = time.perf_counter()
        scheduler_steps += 1
        if scheduler_steps > max_progress_steps:
            raise ReplayError(
                f"case {case.name} repeat {repeat_index} exceeded the progress-step limit"
            )
        step_latencies_ms.append((step_completed - step_started) * 1000.0)
        timestamp_ms = (step_completed - perf_start) * 1000.0

        for output in outputs:
            state = by_internal_id.get(output.request_id)
            if state is None or output.request_id not in active:
                raise ReplayError(
                    f"engine returned unknown or completed request {output.request_id}"
                )
            if len(output.outputs) != 1:
                raise ReplayError("replay requires exactly one completion per request")
            completion = output.outputs[0]
            new_tokens = len(completion.token_ids)
            if new_tokens:
                if state.first_token_time_ms is None:
                    state.first_token_time_ms = timestamp_ms
                state.generated_tokens += new_tokens
                if state.generated_tokens > state.spec.output_tokens:
                    raise ReplayError(
                        f"request {state.spec.request_id} generated too many tokens"
                    )
            if output.finished:
                if state.first_token_time_ms is None:
                    raise ReplayError(
                        f"request {state.spec.request_id} finished without a token"
                    )
                state.completion_time_ms = timestamp_ms
                state.finish_reason = completion.finish_reason
                active.remove(output.request_id)

        if engine.has_unfinished_requests() != bool(active):
            raise ReplayError("engine unfinished-request state diverged from replay state")

    torch.cuda.synchronize()
    measurements = [_request_measurement(state, case.slo) for state in states]
    first_arrival_ms = min(state.spec.arrival_time_ms for state in states)
    last_completion_ms = max(
        state.completion_time_ms for state in states if state.completion_time_ms is not None
    )
    duration_ms = last_completion_ms - first_arrival_ms
    metrics = _repeat_metrics(
        measurements,
        duration_ms=duration_ms,
        scheduler_steps=scheduler_steps,
        step_latencies_ms=step_latencies_ms,
        max_active_requests=max_active_requests,
    )
    return {
        "repeat_index": repeat_index,
        "metrics": metrics,
        "requests": measurements,
    }


def _runtime_imports() -> dict[str, Any]:
    try:
        import torch
        import vllm
        from vllm import EngineArgs, SamplingParams
        from vllm.engine.llm_engine import LLMEngine
        from vllm.sampling_params import RequestOutputKind
    except Exception as exc:
        raise ReplayError(f"cannot import the vLLM hardware runtime: {exc}") from exc
    return {
        "torch": torch,
        "vllm": vllm,
        "EngineArgs": EngineArgs,
        "SamplingParams": SamplingParams,
        "LLMEngine": LLMEngine,
        "RequestOutputKind": RequestOutputKind,
    }


def _build_success_report(args: argparse.Namespace, started_at_utc: str) -> dict[str, Any]:
    workload_path = Path(args.workload)
    model_path = Path(args.model)
    candidate_path = (
        _validate_candidate_path(Path(args.candidate_scheduler))
        if args.candidate_scheduler
        else None
    )
    workload = _load_workload(workload_path, args.max_model_len)
    model_root, model_config_path, _ = _validate_model(
        model_path, args.dummy_token_id
    )
    output_path = Path(args.output).resolve()
    protected_paths = {workload.path, model_config_path.resolve()}
    if candidate_path is not None:
        protected_paths.add(candidate_path)
    if output_path in protected_paths:
        raise ReplayError("output path must not overwrite an input artifact")
    immutable_input_hashes = {
        workload.path: workload.sha256,
        model_config_path: _sha256(model_config_path),
        Path(__file__).resolve(): _sha256(Path(__file__).resolve()),
    }
    if candidate_path is not None:
        immutable_input_hashes[candidate_path] = _sha256(candidate_path)
    git_provenance = _git_provenance()

    prior_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    runtime = _runtime_imports()
    torch = runtime["torch"]
    vllm = runtime["vllm"]
    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise ReplayError(
            f"vLLM {EXPECTED_VLLM_VERSION} is required, found {vllm.__version__}"
        )
    if not torch.cuda.is_available():
        raise ReplayError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise ReplayError(
            "exactly one logical GPU must be visible after selecting --gpu-index"
        )
    torch.cuda.set_device(0)
    smi_device = _nvidia_smi_device(args.gpu_index)
    device = torch.cuda.get_device_properties(0)

    baseline_module = importlib.import_module("vllm.v1.core.sched.scheduler")
    engine_core_module = importlib.import_module("vllm.v1.engine.core")
    baseline_scheduler_type = baseline_module.Scheduler
    baseline_scheduler_path = Path(inspect.getsourcefile(baseline_scheduler_type) or "")
    if not baseline_scheduler_path.is_file():
        raise ReplayError("cannot resolve the installed baseline scheduler source")
    immutable_input_hashes[baseline_scheduler_path] = _sha256(
        baseline_scheduler_path
    )
    engine_core_path = Path(engine_core_module.__file__)
    immutable_input_hashes[engine_core_path] = _sha256(engine_core_path)
    candidate_scheduler_type: type[Any] | None = None
    if candidate_path is not None:
        candidate_scheduler_type = _load_candidate_scheduler(
            candidate_path, baseline_module, engine_core_module
        )

    engine_kwargs: dict[str, Any] = {
        "model": str(model_root),
        "skip_tokenizer_init": True,
        "load_format": "dummy",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_num_partial_prefills": args.max_num_partial_prefills,
        "max_long_partial_prefills": args.max_long_partial_prefills,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "scheduling_policy": args.scheduling_policy,
        "block_size": args.block_size,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": True,
        "generation_config": "vllm",
        "use_tqdm_on_load": False,
    }
    if candidate_scheduler_type is not None:
        engine_kwargs["scheduler_cls"] = candidate_scheduler_type

    engine = None
    try:
        engine_args = runtime["EngineArgs"](**engine_kwargs)
        engine = runtime["LLMEngine"].from_engine_args(
            engine_args,
            enable_multiprocessing=False,
        )
        scheduler = _effective_scheduler(engine)
        expected_type = candidate_scheduler_type or baseline_scheduler_type
        if scheduler.__class__ is not expected_type:
            raise ReplayError(
                "effective scheduler class differs from the requested scheduler class"
            )
        effective_source = Path(inspect.getsourcefile(scheduler.__class__) or "")
        if not effective_source.is_file():
            raise ReplayError("cannot resolve effective scheduler source")
        if candidate_path is not None and effective_source.resolve() != candidate_path:
            raise ReplayError("candidate scheduler was loaded but is not the effective source")
        actual_engine_runtime = _actual_engine_runtime(engine, torch, args)

        warmup = _run_warmup(
            engine,
            torch,
            runtime["SamplingParams"],
            runtime["RequestOutputKind"].DELTA,
            warmup_requests=args.warmup_requests,
            max_model_len=args.max_model_len,
            dummy_token_id=args.dummy_token_id,
            max_progress_steps=args.max_progress_steps,
        )
        _assert_engine_drained(engine, "warmup")

        case_results: list[dict[str, Any]] = []
        partial_path = output_path.with_suffix(output_path.suffix + ".partial")
        for case_index, case in enumerate(workload.cases):
            repeat_results: list[dict[str, Any]] = []
            for repeat_index in range(args.repeats):
                repeat_results.append(
                    _replay_case(
                        engine,
                        torch,
                        runtime["SamplingParams"],
                        runtime["RequestOutputKind"].DELTA,
                        case,
                        case_index=case_index,
                        repeat_index=repeat_index,
                        dummy_token_id=args.dummy_token_id,
                        max_progress_steps=args.max_progress_steps,
                    )
                )
                _assert_engine_drained(
                    engine, f"case {case.name} repeat {repeat_index}"
                )
                _atomic_write_json(
                    partial_path,
                    {
                        "schema_version": REPORT_SCHEMA_VERSION,
                        "artifact_kind": "vllm_hardware_workload_replay_partial",
                        "valid": False,
                        "status": "in_progress",
                        "workload_sha256": workload.sha256,
                        "completed_cases": case_results,
                        "active_case": {
                            "name": case.name,
                            "slo": dict(case.slo),
                            "repeats": repeat_results,
                        },
                    },
                )
            case_results.append(
                {
                    "name": case.name,
                    "slo": dict(case.slo),
                    "requests_per_repeat": len(case.requests),
                    "repeats": repeat_results,
                    "aggregate_metrics": _aggregate_case_repeats(repeat_results),
                }
            )

        for immutable_path, expected_hash in immutable_input_hashes.items():
            if not immutable_path.is_file() or _sha256(immutable_path) != expected_hash:
                raise ReplayError(
                    f"input artifact changed during replay: {immutable_path}"
                )

        scheduler_config = {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_num_partial_prefills": args.max_num_partial_prefills,
            "max_long_partial_prefills": args.max_long_partial_prefills,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "policy": args.scheduling_policy,
            "block_size": args.block_size,
            "num_gpu_blocks_override": args.num_gpu_blocks_override,
        }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": "vllm_hardware_workload_replay",
            "valid": True,
            "status": "ok",
            "started_at_utc": started_at_utc,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime": {
                "vllm_version": vllm.__version__,
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "cuda_runtime_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "dtype": "bfloat16",
                "load_format": "dummy",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "enforce_eager": True,
                "prefix_caching": False,
                "v1_engine": True,
                "v1_multiprocessing": False,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "seed": args.seed,
                "dummy_token_id": args.dummy_token_id,
                "scheduler_config": scheduler_config,
                "effective_engine_config": actual_engine_runtime,
                "effective_scheduler": {
                    "class": scheduler.__class__.__qualname__,
                    "module": scheduler.__class__.__module__,
                    "source": str(effective_source.resolve()),
                    "sha256": _sha256(effective_source),
                    "candidate": candidate_path is not None,
                    "core_scheduler_attribute_patched": candidate_path is not None,
                },
            },
            "device": {
                **smi_device,
                "logical_gpu_index": 0,
                "torch_name": device.name,
                "torch_total_memory_bytes": int(device.total_memory),
                "compute_capability": [int(device.major), int(device.minor)],
                "multi_processor_count": int(device.multi_processor_count),
            },
            "workload": {
                "path": str(workload.path),
                "sha256": workload.sha256,
                "schema_version": workload.schema_version,
                "scenario_family": workload.scenario_family,
                "profiles": list(workload.profiles),
                "case_count": len(workload.cases),
            },
            "measurement": {
                "clock": "time.perf_counter",
                "arrival_replay": "real_time_add_request_then_synchronous_step",
                "cuda_synchronize_before_after_step": True,
                "request_output_kind": "delta",
                "ttft_definition": "first_token_completion - scheduled_arrival",
                "tpot_definition": "(completion - first_token) / max(1, output_tokens - 1)",
                "e2e_definition": "completion - scheduled_arrival",
                "percentile_method": "nearest_index_round_q_times_n_minus_1",
                "warmup": warmup,
                "repeats_per_case": args.repeats,
            },
            "provenance": {
                "script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": immutable_input_hashes[Path(__file__).resolve()],
                },
                "model": {
                    "path": str(model_root),
                    "config_path": str(model_config_path),
                    "config_sha256": immutable_input_hashes[model_config_path],
                },
                "baseline_scheduler": {
                    "path": str(baseline_scheduler_path.resolve()),
                    "sha256": immutable_input_hashes[baseline_scheduler_path],
                },
                "candidate_scheduler": (
                    {
                        "path": str(candidate_path),
                        "sha256": immutable_input_hashes[candidate_path],
                    }
                    if candidate_path is not None
                    else None
                ),
                "vllm_package": {
                    "path": str(Path(vllm.__file__).resolve()),
                    "engine_core_sha256": immutable_input_hashes[engine_core_path],
                },
                "git": git_provenance,
                "host": {
                    "hostname": platform.node(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                },
                "environment": {
                    "prior_cuda_visible_devices": prior_cuda_visible_devices,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
                    "vllm_use_v1": os.environ["VLLM_USE_V1"],
                    "vllm_enable_v1_multiprocessing": os.environ[
                        "VLLM_ENABLE_V1_MULTIPROCESSING"
                    ],
                    "vllm_use_flashinfer_sampler": os.environ[
                        "VLLM_USE_FLASHINFER_SAMPLER"
                    ],
                },
                "arguments": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"output"}
                },
            },
            "aggregate_metrics": _aggregate_cases(case_results),
            "cases": case_results,
        }
        return report
    finally:
        if engine is not None:
            client = getattr(engine, "engine_core", None)
            if client is not None:
                client.shutdown()


def _failure_report(
    args: argparse.Namespace,
    started_at_utc: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": "vllm_hardware_workload_replay",
        "valid": False,
        "status": "failed",
        "started_at_utc": started_at_utc,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc)[:8000],
            "traceback": traceback.format_exc()[-16000:],
        },
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key != "output"
        },
        "script_sha256": _sha256(Path(__file__).resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = Path(args.output).resolve()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        report = _build_success_report(args, started_at_utc)
        _atomic_write_json(output_path, report)
    except Exception as exc:
        failure = _failure_report(args, started_at_utc, exc)
        try:
            _atomic_write_json(output_path, failure)
        except Exception as write_exc:
            print(
                f"hardware replay failed: {exc}; cannot write failure JSON: {write_exc}",
                file=sys.stderr,
            )
            return 2
        print(f"hardware replay failed: {exc}", file=sys.stderr)
        return 1
    output_path.with_suffix(output_path.suffix + ".partial").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "valid": True,
                "profiles": report["aggregate_metrics"]["profile_count"],
                "request_measurements": report["aggregate_metrics"][
                    "request_measurements"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
