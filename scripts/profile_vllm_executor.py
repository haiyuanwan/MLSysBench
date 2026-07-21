#!/usr/bin/env python3
"""Collect real vLLM executor latency samples for the shadow scheduler task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_VLLM_VERSION = "0.11.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local HF config/model directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suite",
        choices=("calibration", "validation"),
        default="calibration",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_model_len < 128:
        parser.error("--max-model-len must be at least 128")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cases(
    max_model_len: int, max_num_seqs: int, suite: str = "calibration"
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Cover the scheduling envelope without taking the full Cartesian product:
    # long-context, large-batch cases otherwise replay the same 512-token chunk
    # hundreds of times and add little information.
    calibration_grid = (
        (1, 32),
        (1, 128),
        (1, 512),
        (1, 2_048),
        (1, 7_168),
        (2, 32),
        (2, 256),
        (2, 2_048),
        (4, 32),
        (4, 128),
        (4, 1_024),
        (8, 32),
        (8, 64),
        (8, 512),
        (16, 32),
        (16, 128),
        (16, 512),
        (32, 32),
        (32, 128),
        (32, 512),
    )
    validation_grid = (
        (1, 96),
        (1, 1_024),
        (1, 4_096),
        (3, 64),
        (3, 384),
        (3, 1_536),
        (6, 48),
        (6, 192),
        (6, 768),
        (12, 48),
        (12, 192),
        (24, 48),
        (24, 192),
    )
    uniform_grid = calibration_grid if suite == "calibration" else validation_grid
    for batch_size, prompt_tokens in uniform_grid:
        if batch_size > max_num_seqs or prompt_tokens + 8 > max_model_len:
            continue
        cases.append(
            {
                "name": f"uniform_b{batch_size}_p{prompt_tokens}",
                "prompt_lengths": [prompt_tokens] * batch_size,
                "output_tokens": 4,
            }
        )

    mixed_templates = (
        (
            [32, 64, 128, 256, 512, 1024],
            [48, 48, 96, 192, 384, 768, 1280],
            [32, 256, 64, 512, 96, 1024, 128, 1536],
        )
        if suite == "calibration"
        else (
            [40, 80, 160, 320, 640, 1280, 2560],
            [72, 144, 288, 576, 1152],
        )
    )
    for index, template in enumerate(mixed_templates):
        lengths = [value for value in template if value + 8 <= max_model_len]
        lengths = (lengths * ((max_num_seqs + len(lengths) - 1) // len(lengths)))[
            : min(max_num_seqs, 24)
        ]
        if lengths:
            cases.append(
                {
                    "name": f"mixed_{index}",
                    "prompt_lengths": lengths,
                    "output_tokens": 8,
                }
            )
    if suite == "calibration":
        decode_templates = (
            [128, 256, 512, 1_024, 2_048, 4_096],
            [64, 128, 256, 512, 1_024, 2_048],
        )
        for index, template in enumerate(decode_templates):
            lengths = (
                template
                * ((max_num_seqs + len(template) - 1) // len(template))
            )[:max_num_seqs]
            cases.append(
                {
                    "name": f"decode_mixed_{index}",
                    "prompt_lengths": lengths,
                    "output_tokens": 512,
                }
            )
        for batch_size, prompt_tokens, output_tokens in (
            (1, 8_000, 64),
            (8, 1_024, 64),
            (16, 2_048, 128),
            (24, 3_072, 256),
            (32, 3_800, 400),
        ):
            if (
                batch_size <= max_num_seqs
                and prompt_tokens + output_tokens <= max_model_len
            ):
                cases.append(
                    {
                        "name": (
                            f"context_b{batch_size}_p{prompt_tokens}_o{output_tokens}"
                        ),
                        "prompt_lengths": [prompt_tokens] * batch_size,
                        "output_tokens": output_tokens,
                    }
                )
    return cases


def _write_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    checkpoint = path.with_suffix(path.suffix + ".partial")
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps({"schema_version": 1, "raw_samples": records}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint)


def _aggregate_points(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    sources: dict[tuple[int, ...], set[str]] = defaultdict(set)
    fields = (
        "batch_size",
        "total_tokens",
        "prefill_tokens",
        "decode_tokens",
        "context_tokens",
        "max_context_tokens",
    )
    for record in records:
        key = tuple(int(record[field]) for field in fields)
        grouped[key].append(float(record["latency_ms"]))
        sources[key].add(str(record["case"]))

    points: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        point = {field: value for field, value in zip(fields, key)}
        point.update(
            {
                "latency_ms": statistics.median(values),
                "latency_min_ms": min(values),
                "latency_max_ms": max(values),
                "sample_count": len(values),
                "source_cases": sorted(sources[key]),
            }
        )
        points.append(point)
    return points


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"Expected vLLM {EXPECTED_VLLM_VERSION}, found {vllm.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model_dir = Path(args.model).resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing local model config: {config_path}")

    llm = LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        seed=args.seed,
    )
    engine_core = llm.llm_engine.engine_core.engine_core
    scheduler = engine_core.scheduler
    original_execute_model = engine_core.model_executor.execute_model
    records: list[dict[str, Any]] = []
    active_case = {"name": "warmup", "repeat": -1, "record": False}

    def measured_execute_model(scheduler_output):
        request_rows: list[dict[str, int]] = []
        for request_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            request = scheduler.requests[request_id]
            computed_before = request.num_computed_tokens - scheduled_tokens
            prompt_remaining = max(0, request.num_prompt_tokens - computed_before)
            prefill_tokens = min(scheduled_tokens, prompt_remaining)
            request_rows.append(
                {
                    "scheduled_tokens": int(scheduled_tokens),
                    "prefill_tokens": int(prefill_tokens),
                    "decode_tokens": int(scheduled_tokens - prefill_tokens),
                    "context_tokens": int(computed_before),
                }
            )

        torch.cuda.synchronize()
        started = time.perf_counter()
        output = original_execute_model(scheduler_output)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if active_case["record"]:
            contexts = [row["context_tokens"] for row in request_rows]
            records.append(
                {
                    "case": active_case["name"],
                    "repeat": active_case["repeat"],
                    "batch_size": len(request_rows),
                    "total_tokens": sum(row["scheduled_tokens"] for row in request_rows),
                    "prefill_tokens": sum(row["prefill_tokens"] for row in request_rows),
                    "decode_tokens": sum(row["decode_tokens"] for row in request_rows),
                    "context_tokens": sum(contexts),
                    "max_context_tokens": max(contexts, default=0),
                    "latency_ms": latency_ms,
                }
            )
        return output

    engine_core.model_executor.execute_model = measured_execute_model
    cases = _cases(args.max_model_len, args.max_num_seqs, args.suite)
    warmup = cases[min(2, len(cases) - 1)]
    output_path = Path(args.output).resolve()

    def run_case(case: dict[str, Any], repeat: int, record: bool) -> None:
        active_case.update(name=case["name"], repeat=repeat, record=record)
        prompts = [
            {"prompt_token_ids": [1 + (index % 127)] * length}
            for index, length in enumerate(case["prompt_lengths"])
        ]
        params = SamplingParams(
            max_tokens=int(case["output_tokens"]),
            ignore_eos=True,
            temperature=0.0,
        )
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM did not complete every profiling request")

    run_case(warmup, -1, False)
    for case in cases:
        for repeat in range(args.repeats):
            run_case(case, repeat, True)
            _write_checkpoint(output_path, records)

    device = torch.cuda.get_device_properties(0)
    scheduler_path = Path(__import__(scheduler.__class__.__module__, fromlist=["x"]).__file__)
    payload = {
        "schema_version": 1,
        "profile_kind": "vllm_executor_batch_latency",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "model_config_sha256": _sha256(config_path),
            "scheduler_sha256": _sha256(scheduler_path),
            "dtype": "bfloat16",
            "load_format": "dummy",
            "enforce_eager": True,
            "prefix_caching": False,
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "seed": args.seed,
        },
        "device": {
            "name": device.name,
            "total_memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor],
            "logical_gpu_index": 0,
            "requested_host_gpu_index": args.gpu_index,
        },
        "measurement": {
            "clock": "time.perf_counter",
            "cuda_synchronize_before_after": True,
            "warmup_case": warmup["name"],
            "repeats": args.repeats,
            "case_count": len(cases),
            "raw_sample_count": len(records),
        },
        "feature_names": [
            "batch_size",
            "total_tokens",
            "prefill_tokens",
            "decode_tokens",
            "context_tokens",
            "max_context_tokens",
        ],
        "points": _aggregate_points(records),
        "raw_samples": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".partial").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "raw_samples": len(records),
                "points": len(payload["points"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
