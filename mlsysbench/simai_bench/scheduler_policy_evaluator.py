"""Deterministic multi-profile serving-policy simulator for protocol tasks.

This evaluator is intentionally a proxy, not a hardware model. Task provenance
must label results accordingly until a separate calibration bundle demonstrates
that the relevant decisions transfer to real systems.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import select
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_TIMING_MODEL = {
    "launch_ms": 0.38,
    "prefill_linear_ms": 0.014,
    "prefill_quadratic_ms": 0.00004,
    "decode_launch_ms": 0.11,
    "decode_per_request_ms": 0.065,
    "mixed_interference_ms": 0.0012,
    "batch_member_ms": 0.012,
}


@dataclass
class RequestState:
    id: int
    arrived_at_ms: float
    prompt_tokens: int
    output_tokens: int
    remaining_prefill_tokens: int
    remaining_decode_tokens: int
    tenant: str
    priority: int
    first_token_at_ms: float | None = None
    completed_at_ms: float | None = None


class Candidate:
    def __init__(self, solution_dir: pathlib.Path):
        bwrap = shutil.which("bwrap")
        driver = pathlib.Path(__file__).with_name("scheduler_candidate_driver.py")
        if bwrap is not None and _bwrap_supported(bwrap):
            command = [
                bwrap,
                "--die-with-parent",
                "--new-session",
                "--unshare-all",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind-try",
                "/lib",
                "/lib",
                "--ro-bind-try",
                "/lib64",
                "/lib64",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/solution",
                "--ro-bind",
                str(solution_dir),
                "/solution",
                "--ro-bind",
                str(driver),
                "/candidate_driver.py",
                "--chdir",
                "/solution",
                "--",
                "/usr/bin/python3",
                "-I",
                "/candidate_driver.py",
                "/solution",
                "--already-isolated",
            ]
        else:
            command = [sys.executable, "-I", str(driver), str(solution_dir)]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def schedule(self, requests: list[dict[str, Any]], limits: dict[str, int]) -> Any:
        if self.process.poll() is not None:
            detail = self.process.stderr.read(4000).strip() if self.process.stderr else ""
            raise RuntimeError(
                "candidate scheduler process exited unexpectedly"
                + (f": {detail}" if detail else "")
            )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(
            json.dumps({"requests": requests, "limits": limits}, separators=(",", ":"))
            + "\n"
        )
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 2.0)
        if not ready:
            raise RuntimeError("candidate scheduler exceeded the per-decision timeout")
        line = self.process.stdout.readline(65_537)
        if not line or len(line.encode("utf-8")) > 65_536:
            raise RuntimeError("candidate scheduler returned no bounded response")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("candidate scheduler response must be an object")
        if "error" in response:
            raise RuntimeError(f"candidate scheduler failed: {response['error']}")
        return response.get("decisions")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def _bwrap_supported(executable: str) -> bool:
    try:
        completed = subprocess.run(
            [executable, "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _observations(states: list[RequestState], now_ms: float) -> list[dict[str, Any]]:
    return [
        {
            "id": state.id,
            "arrived_at_ms": state.arrived_at_ms,
            "waiting_ms": max(0.0, now_ms - state.arrived_at_ms),
            "remaining_prefill_tokens": state.remaining_prefill_tokens,
            "remaining_decode_tokens": state.remaining_decode_tokens,
            "first_token_emitted": state.first_token_at_ms is not None,
            "tenant": state.tenant,
            "priority": state.priority,
        }
        for state in states
        if state.arrived_at_ms <= now_ms and state.completed_at_ms is None
    ]


def _validate_decisions(
    decisions: Any,
    available: list[RequestState],
    limits: dict[str, Any],
) -> list[tuple[RequestState, str, int]]:
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("scheduler must return work when requests are available")
    if len(decisions) > limits["max_batch_size"]:
        raise ValueError("scheduler exceeded max_batch_size")
    by_id = {state.id: state for state in available}
    seen: set[int] = set()
    validated: list[tuple[RequestState, str, int]] = []
    total_tokens = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("each scheduling decision must be an object")
        request_id = decision.get("request_id")
        tokens = decision.get("tokens")
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise ValueError("request_id must be an integer")
        if request_id in seen or request_id not in by_id:
            raise ValueError("request is duplicated or unavailable")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        state = by_id[request_id]
        if state.remaining_prefill_tokens > 0:
            if tokens > state.remaining_prefill_tokens:
                raise ValueError("prefill decision exceeds remaining tokens")
            phase = "prefill"
        else:
            if tokens != 1:
                raise ValueError("decode decisions must advance exactly one token")
            phase = "decode"
        total_tokens += tokens
        if total_tokens > limits["max_batch_tokens"]:
            raise ValueError("scheduler exceeded max_batch_tokens")
        seen.add(request_id)
        validated.append((state, phase, tokens))
    return validated


def _batch_duration_ms(
    validated: list[tuple[RequestState, str, int]],
    timing: dict[str, float],
) -> float:
    prefill_tokens = sum(tokens for _, phase, tokens in validated if phase == "prefill")
    decode_requests = sum(1 for _, phase, _ in validated if phase == "decode")
    duration = timing["launch_ms"]
    if prefill_tokens:
        duration += (
            timing["prefill_linear_ms"] * prefill_tokens
            + timing["prefill_quadratic_ms"] * prefill_tokens * prefill_tokens
        )
    if decode_requests:
        duration += timing["decode_launch_ms"] + timing["decode_per_request_ms"] * decode_requests
    if prefill_tokens and decode_requests:
        duration += timing["mixed_interference_ms"] * prefill_tokens * decode_requests
    duration += timing["batch_member_ms"] * max(0, len(validated) - 1)
    return duration


def simulate(workload: dict[str, Any], candidate: Candidate) -> dict[str, float]:
    limits = workload["limits"]
    timing = dict(DEFAULT_TIMING_MODEL)
    timing.update({key: float(value) for key, value in workload.get("timing_model", {}).items()})
    unknown_timing = sorted(set(timing) - set(DEFAULT_TIMING_MODEL))
    if unknown_timing:
        raise ValueError("unknown timing-model fields: " + ", ".join(unknown_timing))
    states = [
        RequestState(
            id=index,
            arrived_at_ms=float(request["arrived_at_ms"]),
            prompt_tokens=int(request["prompt_tokens"]),
            output_tokens=int(request["output_tokens"]),
            remaining_prefill_tokens=int(request["prompt_tokens"]),
            remaining_decode_tokens=int(request["output_tokens"]),
            tenant=str(request.get("tenant", "default")),
            priority=int(request.get("priority", 0)),
        )
        for index, request in enumerate(workload["requests"])
    ]
    if not states:
        raise ValueError("workload requires at least one request")
    now_ms = min(state.arrived_at_ms for state in states)
    steps = 0
    while any(state.completed_at_ms is None for state in states):
        observations = _observations(states, now_ms)
        if not observations:
            now_ms = min(
                state.arrived_at_ms
                for state in states
                if state.completed_at_ms is None and state.arrived_at_ms > now_ms
            )
            continue
        ids = {item["id"] for item in observations}
        available = [state for state in states if state.id in ids]
        decisions = candidate.schedule(
            observations,
            {
                "max_batch_tokens": int(limits["max_batch_tokens"]),
                "max_batch_size": int(limits["max_batch_size"]),
            },
        )
        validated = _validate_decisions(decisions, available, limits)
        completed_at = now_ms + _batch_duration_ms(validated, timing)
        for state, phase, tokens in validated:
            if phase == "prefill":
                state.remaining_prefill_tokens -= tokens
                if state.remaining_prefill_tokens == 0:
                    state.first_token_at_ms = completed_at
            else:
                state.remaining_decode_tokens -= 1
                if state.remaining_decode_tokens == 0:
                    state.completed_at_ms = completed_at
        now_ms = completed_at
        steps += 1
        if steps > 100_000:
            raise ValueError("scheduler exceeded the progress-step limit")

    ttft = [float(state.first_token_at_ms - state.arrived_at_ms) for state in states]
    tbt = [
        float(state.completed_at_ms - state.first_token_at_ms) / max(state.output_tokens, 1)
        for state in states
    ]
    e2e = [float(state.completed_at_ms - state.arrived_at_ms) for state in states]
    good_flags = [
        one_ttft <= float(limits["ttft_slo_ms"])
        and one_tbt <= float(limits["tbt_slo_ms"])
        and one_e2e <= float(limits["e2e_slo_ms"])
        for one_ttft, one_tbt, one_e2e in zip(ttft, tbt, e2e)
    ]
    duration_seconds = max(
        1e-9,
        (max(float(state.completed_at_ms) for state in states) - min(state.arrived_at_ms for state in states))
        / 1000.0,
    )
    tenant_rates = []
    for tenant in sorted({state.tenant for state in states}):
        flags = [flag for state, flag in zip(states, good_flags) if state.tenant == tenant]
        tenant_rates.append(sum(flags) / len(flags))
    fairness = (
        sum(tenant_rates) ** 2 / (len(tenant_rates) * sum(value * value for value in tenant_rates))
        if any(tenant_rates)
        else 0.0
    )
    return {
        "num_requests": float(len(states)),
        "throughput_rps": len(states) / duration_seconds,
        "goodput_rps": sum(good_flags) / duration_seconds,
        "request_slo_pass_rate": sum(good_flags) / len(good_flags),
        "tenant_fairness_jain": fairness,
        "p50_ttft_ms": statistics.median(ttft),
        "p99_ttft_ms": _percentile(ttft, 0.99),
        "p99_tbt_ms": _percentile(tbt, 0.99),
        "p99_e2e_ms": _percentile(e2e, 0.99),
        "scheduler_steps": float(steps),
    }


def _aggregate(cases: list[tuple[str, dict[str, float]]]) -> dict[str, float]:
    goodputs = [metrics["goodput_rps"] for _, metrics in cases]
    robust_goodput = (
        math.exp(sum(math.log(value) for value in goodputs) / len(goodputs))
        if all(value > 0 for value in goodputs)
        else 0.0
    )
    aggregated = {
        "num_requests": sum(metrics["num_requests"] for _, metrics in cases),
        "profile_count": float(len(cases)),
        "robust_goodput_rps": robust_goodput,
        "goodput_rps": robust_goodput,
        "worst_profile_goodput_rps": min(goodputs),
        "throughput_rps": statistics.fmean(metrics["throughput_rps"] for _, metrics in cases),
        "request_slo_pass_rate": statistics.fmean(
            metrics["request_slo_pass_rate"] for _, metrics in cases
        ),
        "tenant_fairness_jain": min(metrics["tenant_fairness_jain"] for _, metrics in cases),
        "p99_ttft_ms": max(metrics["p99_ttft_ms"] for _, metrics in cases),
        "p50_ttft_ms": statistics.fmean(metrics["p50_ttft_ms"] for _, metrics in cases),
        "p99_tbt_ms": max(metrics["p99_tbt_ms"] for _, metrics in cases),
        "p99_e2e_ms": max(metrics["p99_e2e_ms"] for _, metrics in cases),
        "scheduler_steps": sum(metrics["scheduler_steps"] for _, metrics in cases),
    }
    for name, metrics in cases:
        safe_name = "".join(character if character.isalnum() else "_" for character in name)
        for metric_name in (
            "goodput_rps",
            "request_slo_pass_rate",
            "tenant_fairness_jain",
            "p99_ttft_ms",
            "p99_tbt_ms",
            "p99_e2e_ms",
        ):
            aggregated[f"profile_{safe_name}_{metric_name}"] = metrics[metric_name]
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    candidates: list[Candidate] = []
    try:
        payload = json.loads(pathlib.Path(args.workload).read_text(encoding="utf-8"))
        json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        raw_cases = payload.get("cases")
        if raw_cases is None:
            raw_cases = [payload]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("workload cases must be a non-empty list")
        case_metrics: list[tuple[str, dict[str, float]]] = []
        for index, case in enumerate(raw_cases):
            if not isinstance(case, dict):
                raise ValueError("each workload case must be an object")
            candidate = Candidate(pathlib.Path(args.solution_dir))
            candidates.append(candidate)
            case_metrics.append((str(case.get("name", f"case_{index}")), simulate(case, candidate)))
            candidate.close()
        metrics = _aggregate(case_metrics)
        result = {"valid": True, "failures": [], "metrics": metrics}
    except Exception as exc:  # noqa: BLE001 - convert task failures to structured output.
        result = {
            "valid": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "metrics": {},
        }
    finally:
        for candidate in candidates:
            candidate.close()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
