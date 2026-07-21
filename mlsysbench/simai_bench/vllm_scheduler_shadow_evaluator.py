"""Trusted shadow executor for patches to the real vLLM V1 scheduler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import select
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any


MAX_MESSAGE_BYTES = 1_048_576
SAFE_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_time_model_module():
    path = pathlib.Path(__file__).with_name("vllm_batch_time_model.py")
    spec = importlib.util.spec_from_file_location("mlsysbench_vllm_batch_time_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load calibrated batch-time model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class RequestMirror:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    priority: int
    remaining_prompt_tokens: int
    remaining_output_tokens: int
    computed_tokens: int = 0
    first_token_ms: float | None = None
    completed_ms: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RequestMirror":
        request_id = str(value["request_id"])
        arrival_ms = _finite_float(value["arrival_time_ms"], "arrival_time_ms")
        prompt_tokens = _positive_int(value["prompt_tokens"], "prompt_tokens")
        output_tokens = _positive_int(value["output_tokens"], "output_tokens")
        priority = _plain_int(value.get("priority", 0), "priority")
        return cls(
            request_id=request_id,
            arrival_ms=arrival_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            priority=priority,
            remaining_prompt_tokens=prompt_tokens,
            remaining_output_tokens=output_tokens,
        )


class CandidateProcess:
    def __init__(self, solution_dir: pathlib.Path, runtime_config: pathlib.Path):
        driver = pathlib.Path(__file__).with_name("vllm_scheduler_candidate_driver.py")
        command = [
            sys.executable,
            "-I",
            str(driver),
            "--solution-dir",
            str(solution_dir),
            "--runtime-config",
            str(runtime_config),
        ]
        root = pathlib.Path(tempfile.mkdtemp(prefix="mlsysbench-vllm-candidate-"))
        python_bin = str(pathlib.Path(sys.executable).resolve().parent)
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "PATH": f"{python_bin}:/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "TMPDIR": str(root),
            "VLLM_LOGGING_LEVEL": "ERROR",
        }
        self._root = root
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )

    def call(self, payload: dict[str, Any], timeout_seconds: float = 5.0) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(self._exit_detail("candidate process exited unexpectedly"))
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(rendered.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise RuntimeError("candidate request exceeded the RPC size limit")
        self._process.stdin.write(rendered + "\n")
        self._process.stdin.flush()
        ready, _, _ = select.select([self._process.stdout], [], [], timeout_seconds)
        if not ready:
            raise RuntimeError("candidate scheduler exceeded the per-command timeout")
        line = self._process.stdout.readline(MAX_MESSAGE_BYTES + 1)
        if not line or len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise RuntimeError("candidate scheduler returned no bounded response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("candidate scheduler returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("candidate scheduler response must be an object")
        if response.get("ok") is not True:
            error = response.get("error", "candidate scheduler command failed")
            raise RuntimeError(str(error)[:4000])
        return response

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.call({"command": "close"}, timeout_seconds=1.0)
            except Exception:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        try:
            self._root.rmdir()
        except OSError:
            pass

    def _exit_detail(self, prefix: str) -> str:
        if self._process.stderr is None:
            return prefix
        detail = self._process.stderr.read(4000).strip()
        return f"{prefix}: {detail}" if detail else prefix


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _resolve_workload(commitment_path: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path]:
    payload = _load_json(commitment_path)
    bundle = payload.get("private_bundle")
    if bundle is None:
        return payload, commitment_path.parent
    if not isinstance(bundle, dict):
        raise ValueError("private_bundle must contain an object")
    bundle_id = bundle.get("bundle_id")
    expected_hash = bundle.get("workload_sha256")
    if not isinstance(bundle_id, str) or not SAFE_BUNDLE_ID.fullmatch(bundle_id):
        raise ValueError("private bundle_id is invalid")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("private workload commitment requires a SHA-256 hash")
    root_value = os.environ.get("MLSYSBENCH_PRIVATE_BUNDLE_ROOT")
    if not root_value:
        raise ValueError("private workload bundle root is unavailable")
    root = pathlib.Path(root_value).resolve()
    bundle_root = (root / bundle_id).resolve()
    if root not in bundle_root.parents:
        raise ValueError("private workload bundle escapes its root")
    workload_path = bundle_root / "workload.json"
    if not workload_path.is_file() or _sha256(workload_path) != expected_hash:
        raise ValueError("private workload bundle is missing or does not match commitment")
    return _load_json(workload_path), bundle_root


def _resolve_inside(root: pathlib.Path, value: str) -> pathlib.Path:
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("workload trace path escapes its bundle")
    return candidate


def _load_case_requests(case: dict[str, Any], root: pathlib.Path) -> list[RequestMirror]:
    time_scale = _finite_float(case.get("arrival_time_scale", 1.0), "arrival_time_scale")
    if time_scale <= 0:
        raise ValueError("arrival_time_scale must be positive")
    requests_value = case.get("requests")
    rows: list[dict[str, Any]] = []
    if requests_value is not None:
        if not isinstance(requests_value, list):
            raise ValueError("case.requests must be a list")
        rows = requests_value
    else:
        trace_value = case.get("trace_file")
        if not isinstance(trace_value, str) or not trace_value:
            raise ValueError("case requires requests or trace_file")
        trace_path = _resolve_inside(root, trace_value)
        with trace_path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                rows.append(
                    {
                        "request_id": f"r{index:05d}",
                        "arrival_time_ms": float(row["arrived_at"]) * 1000.0,
                        "prompt_tokens": int(row["num_prefill_tokens"]),
                        "output_tokens": int(row["num_decode_tokens"]),
                        "priority": int(row.get("priority", 0) or 0),
                    }
                )
    if not rows:
        raise ValueError("workload case contains no requests")
    first_arrival = min(float(row["arrival_time_ms"]) for row in rows)
    normalized: list[RequestMirror] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every request must be an object")
        item = dict(row)
        item["arrival_time_ms"] = (
            float(item["arrival_time_ms"]) - first_arrival
        ) * time_scale
        normalized.append(RequestMirror.from_dict(item))
    normalized.sort(key=lambda request: (request.arrival_ms, request.request_id))
    identifiers = [request.request_id for request in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("request IDs must be unique within a case")
    expected = case.get("expected_requests")
    if expected is not None and _positive_int(expected, "expected_requests") != len(normalized):
        raise ValueError("workload case request count does not match commitment")
    return normalized


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def _prediction_value(prediction: Any, name: str) -> float:
    if isinstance(prediction, dict):
        value = prediction[name]
    else:
        value = getattr(prediction, name)
    return _finite_float(value, name)


def _validate_kv_state(response: dict[str, Any]) -> tuple[float, int]:
    usage = _finite_float(response.get("kv_cache_usage"), "kv_cache_usage")
    if usage < 0.0 or usage > 1.0:
        raise ValueError("candidate reported invalid KV-cache usage")
    blocks = response.get("kv_cache_blocks")
    if not isinstance(blocks, dict):
        raise ValueError("candidate response requires kv_cache_blocks")
    used = _plain_int(blocks.get("used_blocks"), "kv_cache_blocks.used_blocks")
    free = _plain_int(blocks.get("free_blocks"), "kv_cache_blocks.free_blocks")
    capacity = _positive_int(
        blocks.get("capacity_blocks"), "kv_cache_blocks.capacity_blocks"
    )
    if used < 0 or free < 0 or used + free != capacity:
        raise ValueError("candidate reported inconsistent KV-cache block accounting")
    if (used == 0) != (usage == 0.0):
        raise ValueError("candidate KV-cache usage disagrees with block accounting")
    return usage, used


def _simulate_case(
    *,
    case: dict[str, Any],
    root: pathlib.Path,
    solution_dir: pathlib.Path,
    runtime_config_path: pathlib.Path,
    runtime: dict[str, Any],
    timing_model: Any,
) -> dict[str, Any]:
    requests = _load_case_requests(case, root)
    max_model_len = _positive_int(runtime["max_model_len"], "max_model_len")
    for request in requests:
        if request.prompt_tokens + request.output_tokens > max_model_len:
            raise ValueError(
                f"request {request.request_id} exceeds the fixed model length"
            )

    max_batch_tokens = _positive_int(
        runtime["max_num_batched_tokens"], "max_num_batched_tokens"
    )
    max_batch_size = _positive_int(runtime["max_num_seqs"], "max_num_seqs")
    max_steps = _positive_int(runtime.get("max_progress_steps", 1_000_000), "max_progress_steps")
    max_scheduler_wall_ms = _finite_float(
        runtime.get("max_scheduler_wall_ms", 100.0), "max_scheduler_wall_ms"
    )
    slo = case.get("slo") or runtime.get("slo")
    if not isinstance(slo, dict):
        raise ValueError("workload case requires SLO thresholds")
    ttft_slo = _finite_float(slo["ttft_ms"], "slo.ttft_ms")
    tpot_slo = _finite_float(slo["tpot_ms"], "slo.tpot_ms")
    e2e_slo = _finite_float(slo["e2e_ms"], "slo.e2e_ms")

    pending = list(requests)
    by_id = {request.request_id: request for request in requests}
    active: set[str] = set()
    completed: set[str] = set()
    now_ms = pending[0].arrival_ms
    decision_rows: list[dict[str, Any]] = []
    max_ood_distance = 0.0
    max_uncertainty_ms = 0.0
    steps = 0
    final_kv_usage = math.nan
    final_kv_used_blocks = -1
    candidate = CandidateProcess(solution_dir, runtime_config_path)
    try:
        candidate.call({"command": "init"}, timeout_seconds=15.0)
        while len(completed) < len(requests):
            arrivals: list[RequestMirror] = []
            while pending and pending[0].arrival_ms <= now_ms + 1e-9:
                request = pending.pop(0)
                active.add(request.request_id)
                arrivals.append(request)
            if arrivals:
                candidate.call(
                    {
                        "command": "arrive",
                        "now_ms": now_ms,
                        "requests": [
                            {
                                "request_id": request.request_id,
                                "arrival_time_ms": request.arrival_ms,
                                "prompt_tokens": request.prompt_tokens,
                                "output_tokens": request.output_tokens,
                                "priority": request.priority,
                            }
                            for request in arrivals
                        ],
                    }
                )
            if not active:
                if not pending:
                    break
                now_ms = pending[0].arrival_ms
                idle_response = candidate.call({"command": "step", "now_ms": now_ms})
                if idle_response.get("scheduled") != [] or _plain_int(
                    idle_response.get("total_scheduled_tokens"),
                    "total_scheduled_tokens",
                ) != 0:
                    raise ValueError("scheduler produced work while no requests were active")
                final_kv_usage, final_kv_used_blocks = _validate_kv_state(idle_response)
                if final_kv_usage != 0.0 or final_kv_used_blocks != 0:
                    raise ValueError("scheduler retained KV-cache state across an idle gap")
                continue

            response = candidate.call({"command": "step", "now_ms": now_ms})
            scheduled = response.get("scheduled")
            if not isinstance(scheduled, list) or not scheduled:
                raise ValueError("scheduler made no progress while requests were active")
            if len(scheduled) > max_batch_size:
                raise ValueError("scheduler exceeded max_num_seqs")
            scheduler_wall_ms = _finite_float(
                response.get("scheduler_wall_ms"), "scheduler_wall_ms"
            )
            if scheduler_wall_ms > max_scheduler_wall_ms:
                raise ValueError("scheduler exceeded the per-step CPU-time gate")
            final_kv_usage, final_kv_used_blocks = _validate_kv_state(response)

            total_tokens = 0
            prefill_tokens = 0
            decode_tokens = 0
            contexts: list[int] = []
            updates: list[tuple[RequestMirror, int, int]] = []
            seen: set[str] = set()
            normalized_decisions: list[dict[str, Any]] = []
            for row in scheduled:
                if not isinstance(row, dict):
                    raise ValueError("scheduled entries must be objects")
                request_id = str(row.get("request_id"))
                if request_id in seen or request_id not in active:
                    raise ValueError("scheduler selected a duplicate or unavailable request")
                seen.add(request_id)
                request = by_id[request_id]
                tokens = _positive_int(row.get("num_scheduled_tokens"), "num_scheduled_tokens")
                computed_before = _plain_int(row.get("num_computed_before"), "num_computed_before")
                computed_after = _plain_int(row.get("num_computed_after"), "num_computed_after")
                output_before = _plain_int(row.get("output_tokens_before"), "output_tokens_before")
                emitted = _plain_int(row.get("emitted_tokens"), "emitted_tokens")
                if computed_before != request.computed_tokens or computed_after != computed_before + tokens:
                    raise ValueError("candidate scheduler state diverged from the trusted token mirror")
                if output_before != request.output_tokens - request.remaining_output_tokens:
                    raise ValueError("candidate output state diverged from the trusted request mirror")
                if request.remaining_prompt_tokens > 0:
                    if tokens > request.remaining_prompt_tokens:
                        raise ValueError("scheduler crossed prefill/decode within one model step")
                    one_prefill = tokens
                    one_decode = 0
                    expected_emitted = int(tokens == request.remaining_prompt_tokens)
                else:
                    if tokens != 1:
                        raise ValueError("non-speculative decode must schedule one token per request")
                    one_prefill = 0
                    one_decode = 1
                    expected_emitted = 1
                if emitted != expected_emitted:
                    raise ValueError("candidate emitted-token summary violates vLLM semantics")
                total_tokens += tokens
                prefill_tokens += one_prefill
                decode_tokens += one_decode
                contexts.append(computed_before)
                updates.append((request, tokens, emitted))
                normalized_decisions.append(
                    {
                        "request_id": request_id,
                        "tokens": tokens,
                        "computed_before": computed_before,
                        "emitted": emitted,
                    }
                )

            if total_tokens != _plain_int(
                response.get("total_scheduled_tokens"), "total_scheduled_tokens"
            ):
                raise ValueError("scheduler total token count is inconsistent")
            if total_tokens > max_batch_tokens:
                raise ValueError("scheduler exceeded max_num_batched_tokens")
            descriptor = {
                "batch_size": len(scheduled),
                "total_tokens": total_tokens,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "context_tokens": sum(contexts),
                "max_context_tokens": max(contexts, default=0),
            }
            prediction = timing_model.predict(descriptor)
            latency_ms = _prediction_value(prediction, "latency_ms")
            distance = _prediction_value(prediction, "normalized_distance")
            uncertainty_ms = _prediction_value(prediction, "uncertainty_ms")
            if latency_ms <= 0:
                raise ValueError("timing model predicted a non-positive latency")
            max_ood_distance = max(max_ood_distance, distance)
            max_uncertainty_ms = max(max_uncertainty_ms, uncertainty_ms)
            completed_at = now_ms + latency_ms
            finished_this_step: set[str] = set()
            for request, tokens, emitted in updates:
                request.computed_tokens += tokens
                if request.remaining_prompt_tokens > 0:
                    request.remaining_prompt_tokens -= tokens
                if emitted:
                    if request.first_token_ms is None:
                        request.first_token_ms = completed_at
                    request.remaining_output_tokens -= emitted
                    if request.remaining_output_tokens < 0:
                        raise ValueError("candidate emitted too many output tokens")
                    if request.remaining_output_tokens == 0:
                        request.completed_ms = completed_at
                        active.remove(request.request_id)
                        completed.add(request.request_id)
                        finished_this_step.add(request.request_id)
            reported_finished = response.get("finished_request_ids", [])
            if not isinstance(reported_finished, list) or set(map(str, reported_finished)) != finished_this_step:
                raise ValueError("candidate finished-request set diverged from trusted mirror")
            decision_rows.append(
                {
                    "step": steps,
                    "now_ms": round(now_ms, 9),
                    "decisions": normalized_decisions,
                    "descriptor": descriptor,
                    "latency_ms": latency_ms,
                }
            )
            now_ms = completed_at
            steps += 1
            if steps > max_steps:
                raise ValueError("scheduler exceeded the progress-step limit")
    finally:
        candidate.close()

    if len(completed) != len(requests):
        raise ValueError("not all requests completed")
    if any(request.first_token_ms is None or request.completed_ms is None for request in requests):
        raise ValueError("completed requests are missing latency timestamps")
    if final_kv_usage != 0.0 or final_kv_used_blocks != 0:
        raise ValueError("scheduler did not release all KV-cache blocks after completion")

    ttft = [float(request.first_token_ms - request.arrival_ms) for request in requests]
    e2e = [float(request.completed_ms - request.arrival_ms) for request in requests]
    tpot = [
        max(0.0, float(request.completed_ms - request.first_token_ms))
        / max(1, request.output_tokens - 1)
        for request in requests
    ]
    good = [
        one_ttft <= ttft_slo and one_tpot <= tpot_slo and one_e2e <= e2e_slo
        for one_ttft, one_tpot, one_e2e in zip(ttft, tpot, e2e)
    ]
    duration_seconds = max(
        1e-9,
        (max(float(request.completed_ms) for request in requests) - requests[0].arrival_ms)
        / 1000.0,
    )
    decision_digest = hashlib.sha256(
        json.dumps(decision_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "name": str(case["name"]),
        "metrics": {
            "num_requests": float(len(requests)),
            "completion_rate": 1.0,
            "request_slo_pass_rate": sum(good) / len(good),
            "throughput_rps": len(requests) / duration_seconds,
            "goodput_rps": sum(good) / duration_seconds,
            "p50_ttft_ms": statistics.median(ttft),
            "p99_ttft_ms": _percentile(ttft, 0.99),
            "p99_tpot_ms": _percentile(tpot, 0.99),
            "p99_tbt_ms": _percentile(tpot, 0.99),
            "p99_e2e_ms": _percentile(e2e, 0.99),
            "scheduler_steps": float(steps),
            "max_timing_ood_distance": max_ood_distance,
            "max_timing_uncertainty_ms": max_uncertainty_ms,
        },
        "decision_sha256": decision_digest,
    }


def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, float]:
    goodputs = [float(result["metrics"]["goodput_rps"]) for result in case_results]
    robust = (
        math.exp(statistics.fmean(math.log(value) for value in goodputs))
        if all(value > 0 for value in goodputs)
        else 0.0
    )
    metrics: dict[str, float] = {
        "profile_count": float(len(case_results)),
        "num_requests": sum(result["metrics"]["num_requests"] for result in case_results),
        "completion_rate": min(result["metrics"]["completion_rate"] for result in case_results),
        "request_slo_pass_rate": min(
            result["metrics"]["request_slo_pass_rate"] for result in case_results
        ),
        "robust_goodput_rps": robust,
        "goodput_rps": robust,
        "worst_profile_goodput_rps": min(goodputs),
        "p99_ttft_ms": max(result["metrics"]["p99_ttft_ms"] for result in case_results),
        "p99_tpot_ms": max(result["metrics"]["p99_tpot_ms"] for result in case_results),
        "p99_tbt_ms": max(result["metrics"]["p99_tbt_ms"] for result in case_results),
        "p99_e2e_ms": max(result["metrics"]["p99_e2e_ms"] for result in case_results),
        "max_timing_ood_distance": max(
            result["metrics"]["max_timing_ood_distance"] for result in case_results
        ),
        "max_timing_uncertainty_ms": max(
            result["metrics"]["max_timing_uncertainty_ms"] for result in case_results
        ),
    }
    for result in case_results:
        name = "".join(character if character.isalnum() else "_" for character in result["name"])
        for metric_name in (
            "goodput_rps",
            "request_slo_pass_rate",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "p99_tbt_ms",
            "p99_e2e_ms",
        ):
            metrics[f"profile_{name}_{metric_name}"] = float(
                result["metrics"][metric_name]
            )
    return metrics


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _plain_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-profile", required=True)
    parser.add_argument("--runtime-config", required=True)
    args = parser.parse_args()
    output_path = pathlib.Path(args.output)
    try:
        workload, workload_root = _resolve_workload(pathlib.Path(args.workload).resolve())
        json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        runtime_path = pathlib.Path(args.runtime_config).resolve()
        runtime = _load_json(runtime_path)
        timing_module = _load_time_model_module()
        identity_payload = runtime.get("timing_profile_identity")
        if not isinstance(identity_payload, dict):
            raise ValueError("runtime config requires timing_profile_identity")
        expected_identity = timing_module.BatchTimeProfileIdentity.from_mapping(
            identity_payload
        )
        timing_model = timing_module.CalibratedBatchTimeModel.from_path(
            pathlib.Path(args.timing_profile).resolve(),
            expected_identity=expected_identity,
        )
        cases = workload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("workload cases must be a non-empty list")
        names = [case.get("name") for case in cases if isinstance(case, dict)]
        if len(names) != len(cases) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every workload case requires a name")
        if len(names) != len(set(names)):
            raise ValueError("workload case names must be unique")

        case_results: list[dict[str, Any]] = []
        for case in cases:
            first = _simulate_case(
                case=case,
                root=workload_root,
                solution_dir=pathlib.Path(args.solution_dir).resolve(),
                runtime_config_path=runtime_path,
                runtime=runtime,
                timing_model=timing_model,
            )
            second = _simulate_case(
                case=case,
                root=workload_root,
                solution_dir=pathlib.Path(args.solution_dir).resolve(),
                runtime_config_path=runtime_path,
                runtime=runtime,
                timing_model=timing_model,
            )
            if first["decision_sha256"] != second["decision_sha256"]:
                raise ValueError(f"scheduler decisions are nondeterministic for case {case['name']}")
            if first["metrics"] != second["metrics"]:
                raise ValueError(f"scheduler metrics are nondeterministic for case {case['name']}")
            case_results.append(first)
        metrics = _aggregate(case_results)
        result = {
            "valid": True,
            "failures": [],
            "metrics": metrics,
            "case_results": case_results,
        }
    except Exception as exc:  # noqa: BLE001 - task failures are structured results.
        result = {
            "valid": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "metrics": {},
            "case_results": [],
        }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
