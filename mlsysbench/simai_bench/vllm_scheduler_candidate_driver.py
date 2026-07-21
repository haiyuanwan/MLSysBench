"""JSONL driver for an untrusted vLLM 0.11.0 scheduler implementation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

MAX_CONFIG_BYTES = 65_536
MAX_INPUT_BYTES = 1_048_576
MAX_OUTPUT_BYTES = 65_536
MAX_ERROR_CHARS = 2_048
MAX_REQUEST_ID_CHARS = 128
EXPECTED_VLLM_VERSION = "0.11.0"
PROTOCOL_VERSION = 1
SCHEDULER_RELATIVE_PATH = Path("vllm/v1/core/sched/scheduler.py")
_MONOTONIC_NS = time.monotonic_ns


class ProtocolError(ValueError):
    """Raised for malformed or out-of-order protocol commands."""


class Driver:
    def __init__(
        self,
        solution_dir: Path,
        runtime_config: dict[str, Any],
        sandbox: dict[str, Any],
    ) -> None:
        self.solution_dir = solution_dir
        self.runtime_config = runtime_config
        self.sandbox = sandbox
        self.scheduler: Any = None
        self.request_type: Any = None
        self.sampling_params_type: Any = None
        self.model_runner_output_type: Any = None
        self.request_descriptors: dict[str, dict[str, Any]] = {}
        self.finished_request_ids: set[str] = set()
        self.now_ms = _finite_number(
            runtime_config.get("initial_now_ms", 0.0),
            "runtime initial_now_ms",
        )
        self.dummy_token_id = _bounded_int(
            runtime_config.get("dummy_token_id", 1),
            "runtime dummy_token_id",
            minimum=0,
            maximum=2**31 - 1,
        )
        self.max_model_len = 0
        self.max_num_batched_tokens = 0
        self.max_num_seqs = 0
        self.candidate_sha256 = _sha256_file(
            self.solution_dir / SCHEDULER_RELATIVE_PATH
        )

    def dispatch(self, payload: Any) -> tuple[dict[str, Any], bool]:
        command = _command_name(payload)
        if command == "init":
            return self._init(payload), False
        if command == "close":
            return self._close(), True
        if self.scheduler is None:
            raise ProtocolError("init must succeed before arrive or step")
        if command == "arrive":
            return self._arrive(payload), False
        if command == "step":
            return self._step(payload), False
        raise ProtocolError(f"unknown command: {command}")

    def _init(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.scheduler is not None:
            raise ProtocolError("init may be called only once")
        if "now_ms" in payload:
            self.now_ms = _finite_number(payload["now_ms"], "init now_ms")

        runtime = _import_vllm_runtime()
        version = runtime["version"]
        expected_version = self.runtime_config.get(
            "expected_vllm_version", EXPECTED_VLLM_VERSION
        )
        if expected_version != EXPECTED_VLLM_VERSION:
            raise ProtocolError(
                "this driver is fixed to vLLM 0.11.0; "
                f"runtime requested {expected_version!r}"
            )
        if version != EXPECTED_VLLM_VERSION:
            raise RuntimeError(
                f"vLLM runtime mismatch: expected 0.11.0, found {version}"
            )

        scheduler_config_values = _mapping(
            self.runtime_config.get("scheduler_config", {}),
            "runtime scheduler_config",
        )
        cache_config_values = _mapping(
            self.runtime_config.get("cache_config", {}),
            "runtime cache_config",
        )
        kv_config_values = _mapping(
            self.runtime_config.get("kv_cache_config", {}),
            "runtime kv_cache_config",
        )

        self.max_num_batched_tokens = _runtime_int(
            self.runtime_config,
            scheduler_config_values,
            "max_num_batched_tokens",
            2_048,
            minimum=1,
            maximum=1_000_000,
        )
        self.max_num_seqs = _runtime_int(
            self.runtime_config,
            scheduler_config_values,
            "max_num_seqs",
            64,
            minimum=1,
            maximum=256,
        )
        self.max_model_len = _runtime_int(
            self.runtime_config,
            scheduler_config_values,
            "max_model_len",
            8_192,
            minimum=2,
            maximum=10_000_000,
        )
        enable_chunked_prefill = _runtime_bool(
            self.runtime_config,
            scheduler_config_values,
            "enable_chunked_prefill",
            True,
        )
        policy = _runtime_value(
            self.runtime_config, scheduler_config_values, "policy", "fcfs"
        )
        if policy not in {"fcfs", "priority"}:
            raise ProtocolError("runtime scheduler policy must be fcfs or priority")

        scheduler_config = runtime["SchedulerConfig"](
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_partial_prefills=_runtime_int(
                self.runtime_config,
                scheduler_config_values,
                "max_num_partial_prefills",
                1,
                minimum=1,
                maximum=256,
            ),
            max_long_partial_prefills=_runtime_int(
                self.runtime_config,
                scheduler_config_values,
                "max_long_partial_prefills",
                1,
                minimum=1,
                maximum=256,
            ),
            long_prefill_token_threshold=_runtime_int(
                self.runtime_config,
                scheduler_config_values,
                "long_prefill_token_threshold",
                0,
                minimum=0,
                maximum=self.max_model_len,
            ),
            num_lookahead_slots=0,
            enable_chunked_prefill=enable_chunked_prefill,
            is_multimodal_model=False,
            policy=policy,
            disable_chunked_mm_input=True,
            async_scheduling=False,
        )

        block_size = _runtime_int(
            self.runtime_config,
            cache_config_values,
            "block_size",
            16,
            minimum=1,
            maximum=128,
        )
        if block_size not in {1, 8, 16, 32, 64, 128}:
            raise ProtocolError("runtime block_size is not supported by vLLM 0.11.0")
        prefix_caching = _runtime_bool(
            self.runtime_config,
            cache_config_values,
            "enable_prefix_caching",
            False,
        )
        if prefix_caching:
            raise ProtocolError("prefix caching is disabled for this shadow executor")
        num_gpu_blocks = _first_int(
            (
                kv_config_values.get("num_blocks"),
                cache_config_values.get("num_gpu_blocks"),
                self.runtime_config.get("num_gpu_blocks"),
            ),
            "runtime num_gpu_blocks",
            default=4_096,
            minimum=2,
            maximum=10_000_000,
        )
        cache_config = runtime["CacheConfig"](
            block_size=block_size,
            enable_prefix_caching=False,
            swap_space=0,
            cpu_offload_gb=0,
        )
        cache_config.num_gpu_blocks = num_gpu_blocks

        num_layers = _runtime_int(
            self.runtime_config,
            kv_config_values,
            "num_layers",
            1,
            minimum=1,
            maximum=1_024,
        )
        num_kv_heads = _runtime_int(
            self.runtime_config,
            kv_config_values,
            "num_kv_heads",
            8,
            minimum=1,
            maximum=1_024,
        )
        head_size = _runtime_int(
            self.runtime_config,
            kv_config_values,
            "head_size",
            128,
            minimum=1,
            maximum=65_536,
        )
        dtype_name = _runtime_value(
            self.runtime_config, kv_config_values, "dtype", "float16"
        )
        dtype_map = {
            "float16": runtime["torch"].float16,
            "bfloat16": runtime["torch"].bfloat16,
            "float32": runtime["torch"].float32,
        }
        if dtype_name not in dtype_map:
            raise ProtocolError(
                "runtime KV dtype must be float16, bfloat16, or float32"
            )
        layer_names = [f"model.layers.{index}.self_attn" for index in range(num_layers)]
        attention_spec = runtime["FullAttentionSpec"](
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype_map[dtype_name],
        )
        kv_cache_config = runtime["KVCacheConfig"](
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                runtime["KVCacheGroupSpec"](
                    layer_names=layer_names,
                    kv_cache_spec=attention_spec,
                )
            ],
        )

        parallel_config = SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
        )
        model_config = SimpleNamespace(
            is_encoder_decoder=False,
            is_multimodal_model=False,
            max_model_len=self.max_model_len,
        )
        vllm_config = SimpleNamespace(
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            model_config=model_config,
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            kv_events_config=None,
            virtual_now_ms=self.now_ms,
        )
        structured_output_manager = SimpleNamespace(
            should_advance=lambda _request: False,
            grammar_bitmask=lambda *_args, **_kwargs: None,
        )

        scheduler_type = _load_candidate_scheduler(
            self.solution_dir / SCHEDULER_RELATIVE_PATH
        )
        self.scheduler = scheduler_type(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            include_finished_set=False,
            log_stats=False,
        )
        self.scheduler.virtual_now_ms = self.now_ms
        self.request_type = runtime["Request"]
        self.sampling_params_type = runtime["SamplingParams"]
        self.model_runner_output_type = runtime["ModelRunnerOutput"]

        return {
            "ok": True,
            "command": "init",
            "protocol_version": PROTOCOL_VERSION,
            "vllm_version": version,
            "candidate_sha256": self.candidate_sha256,
            "now_ms": self.now_ms,
            "runtime": {
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
                "max_model_len": self.max_model_len,
                "block_size": block_size,
                "num_gpu_blocks": num_gpu_blocks,
                "prefix_caching": False,
                "speculative_decoding": False,
                "multimodal": False,
                "pipeline_parallel_size": 1,
            },
            "sandbox": self.sandbox,
            "state_semantics": {
                "num_computed_before": "immediately before Scheduler.schedule",
                "num_computed_after": (
                    "immediately after Scheduler.schedule and before "
                    "Scheduler.update_from_output"
                ),
                "num_computed_after_update": (
                    "immediately after Scheduler.update_from_output"
                ),
                "required_transition": (
                    "num_computed_after = num_computed_before + " "num_scheduled_tokens"
                ),
            },
        }

    def _arrive(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._synchronize_now(payload.get("now_ms"), "arrive")
        requests = payload.get("requests")
        if not isinstance(requests, list):
            raise ProtocolError("arrive requests must be a list")
        normalized: list[dict[str, Any]] = []
        batch_ids: set[str] = set()
        for index, value in enumerate(requests):
            descriptor = _mapping(value, f"arrive requests[{index}]")
            request_id = descriptor.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ProtocolError(
                    f"arrive requests[{index}].request_id must be a string"
                )
            if len(request_id) > MAX_REQUEST_ID_CHARS or "\x00" in request_id:
                raise ProtocolError(f"arrive request_id {request_id!r} is invalid")
            if (
                request_id in batch_ids
                or request_id in self.request_descriptors
                or request_id in self.finished_request_ids
            ):
                raise ProtocolError(f"duplicate request_id: {request_id}")
            prompt_tokens = _bounded_int(
                descriptor.get("prompt_tokens"),
                f"arrive {request_id} prompt_tokens",
                minimum=1,
                maximum=self.max_model_len - 1,
            )
            output_tokens = _bounded_int(
                descriptor.get("output_tokens"),
                f"arrive {request_id} output_tokens",
                minimum=1,
                maximum=self.max_model_len - 1,
            )
            if prompt_tokens + output_tokens > self.max_model_len:
                raise ProtocolError(
                    f"arrive {request_id} exceeds max_model_len={self.max_model_len}"
                )
            arrival_time_ms = _finite_number(
                descriptor.get("arrival_time_ms"),
                f"arrive {request_id} arrival_time_ms",
            )
            if arrival_time_ms > self.now_ms:
                raise ProtocolError(
                    f"request {request_id} has not arrived at now_ms={self.now_ms}"
                )
            priority = _bounded_int(
                descriptor.get("priority", 0),
                f"arrive {request_id} priority",
                minimum=-(2**31),
                maximum=2**31 - 1,
            )
            normalized.append(
                {
                    "request_id": request_id,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "arrival_time_ms": arrival_time_ms,
                    "priority": priority,
                }
            )
            batch_ids.add(request_id)

        for descriptor in normalized:
            request_id = descriptor["request_id"]
            sampling_params = self.sampling_params_type(
                temperature=0,
                ignore_eos=True,
                max_tokens=descriptor["output_tokens"],
            )
            request = self.request_type(
                request_id=request_id,
                prompt_token_ids=[0] * descriptor["prompt_tokens"],
                sampling_params=sampling_params,
                pooling_params=None,
                eos_token_id=None,
                arrival_time=descriptor["arrival_time_ms"] / 1_000.0,
                priority=descriptor["priority"],
            )
            self.scheduler.add_request(request)
            self.request_descriptors[request_id] = descriptor

        running, waiting = self.scheduler.get_request_counts()
        return {
            "ok": True,
            "command": "arrive",
            "accepted_requests": len(normalized),
            "now_ms": self.now_ms,
            "running_requests": int(running),
            "waiting_requests": int(waiting),
        }

    def _step(self, payload: dict[str, Any]) -> dict[str, Any]:
        now_ms = self._synchronize_now(payload.get("now_ms"), "step")

        before: dict[str, dict[str, int]] = {}
        for request_id, request in self.scheduler.requests.items():
            before[request_id] = {
                "num_computed_tokens": int(request.num_computed_tokens),
                "num_output_tokens": int(request.num_output_tokens),
            }

        started_ns = _MONOTONIC_NS()
        scheduler_output = self.scheduler.schedule()
        scheduler_wall_ms = (_MONOTONIC_NS() - started_ns) / 1_000_000.0
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        if not isinstance(scheduled_tokens, dict):
            raise RuntimeError(
                "candidate SchedulerOutput.num_scheduled_tokens is not a dict"
            )

        scheduled: list[dict[str, Any]] = []
        request_ids: list[str] = []
        sampled_token_ids: list[list[int]] = []
        summaries_by_id: dict[str, dict[str, Any]] = {}
        scheduled_requests: dict[str, Any] = {}
        recomputed_total = 0
        for request_id, value in scheduled_tokens.items():
            if not isinstance(request_id, str) or request_id not in before:
                raise RuntimeError(
                    f"candidate scheduled an unknown request: {request_id!r}"
                )
            num_scheduled = _bounded_int(
                value,
                f"SchedulerOutput tokens for {request_id}",
                minimum=1,
                maximum=self.max_num_batched_tokens,
            )
            request = self.scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(
                    f"candidate removed scheduled request {request_id} before model output"
                )
            num_computed_after = int(request.num_computed_tokens)
            descriptor = self.request_descriptors.get(request_id)
            if descriptor is None:
                raise RuntimeError(
                    f"candidate scheduled an unadmitted request: {request_id}"
                )
            num_computed_before = before[request_id]["num_computed_tokens"]
            output_tokens_before = before[request_id]["num_output_tokens"]
            known_tokens_before = descriptor["prompt_tokens"] + output_tokens_before
            if (
                int(request.num_tokens) != known_tokens_before
                or int(request.num_output_tokens) != output_tokens_before
            ):
                raise RuntimeError(
                    f"candidate request {request_id} mutated known-token state "
                    "during scheduling"
                )
            if num_computed_before < 0 or num_computed_before > known_tokens_before:
                raise RuntimeError(
                    f"candidate request {request_id} has invalid pre-schedule "
                    "computed-token state"
                )
            if num_computed_after != num_computed_before + num_scheduled:
                raise RuntimeError(
                    f"candidate request {request_id} did not advance computed "
                    "tokens by num_scheduled_tokens"
                )
            if num_computed_after > known_tokens_before:
                raise RuntimeError(
                    f"candidate request {request_id} scheduled beyond its known tokens"
                )
            emitted = (
                [self.dummy_token_id]
                if num_computed_after == known_tokens_before
                and int(request.num_output_tokens) < int(request.max_tokens)
                else []
            )
            summary = {
                "request_id": request_id,
                "num_scheduled_tokens": num_scheduled,
                "num_computed_before": num_computed_before,
                "num_computed_after": num_computed_after,
                "num_computed_after_update": None,
                "prompt_tokens": descriptor["prompt_tokens"],
                "known_tokens_before": known_tokens_before,
                "output_tokens_before": output_tokens_before,
                "output_tokens_after": None,
                "emitted_tokens": 0,
                "emitted_token_ids": [],
            }
            scheduled.append(summary)
            summaries_by_id[request_id] = summary
            scheduled_requests[request_id] = request
            request_ids.append(request_id)
            sampled_token_ids.append(emitted)
            recomputed_total += num_scheduled

        total_scheduled = _bounded_int(
            scheduler_output.total_num_scheduled_tokens,
            "SchedulerOutput total_num_scheduled_tokens",
            minimum=0,
            maximum=self.max_num_batched_tokens,
        )
        if total_scheduled != recomputed_total:
            raise RuntimeError(
                "candidate SchedulerOutput total_num_scheduled_tokens does not match its map"
            )
        resources_after_schedule = self._scheduler_state()

        model_runner_output = self.model_runner_output_type(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )
        engine_outputs = self.scheduler.update_from_output(
            scheduler_output, model_runner_output
        )

        finished_this_step = set(scheduler_output.finished_req_ids)
        for client_outputs in engine_outputs.values():
            for output in client_outputs.outputs:
                summary = summaries_by_id.get(output.request_id)
                if summary is not None:
                    token_ids = [int(token) for token in output.new_token_ids]
                    summary["emitted_tokens"] += len(token_ids)
                    summary["emitted_token_ids"].extend(token_ids)
                if output.finished:
                    finished_this_step.add(output.request_id)
            if client_outputs.finished_requests:
                finished_this_step.update(client_outputs.finished_requests)
        finished_this_step.update(getattr(self.scheduler, "finished_req_ids", ()))
        newly_finished = finished_this_step - self.finished_request_ids
        self.finished_request_ids.update(newly_finished)

        for request_id, summary in summaries_by_id.items():
            request = scheduled_requests[request_id]
            num_computed_after_update = int(request.num_computed_tokens)
            output_tokens_after = int(request.num_output_tokens)
            if num_computed_after_update != summary["num_computed_after"]:
                raise RuntimeError(
                    f"candidate request {request_id} changed computed tokens during "
                    "non-speculative model update"
                )
            if output_tokens_after != (
                summary["output_tokens_before"] + summary["emitted_tokens"]
            ):
                raise RuntimeError(
                    f"candidate request {request_id} output-token state does not "
                    "match emitted_tokens"
                )
            summary["num_computed_after_update"] = num_computed_after_update
            summary["output_tokens_after"] = output_tokens_after

        resources = self._scheduler_state()
        return {
            "ok": True,
            "command": "step",
            "now_ms": self.now_ms,
            "scheduled": scheduled,
            "total_scheduled_tokens": total_scheduled,
            "finished_request_ids": sorted(newly_finished),
            "scheduler_wall_ms": scheduler_wall_ms,
            "kv_cache_usage_after_schedule": resources_after_schedule["kv_cache_usage"],
            "kv_cache_blocks_after_schedule": resources_after_schedule[
                "kv_cache_blocks"
            ],
            **resources,
        }

    def _close(self) -> dict[str, Any]:
        if self.scheduler is not None:
            try:
                self.scheduler.shutdown()
            except BaseException as exc:
                resources = self._scheduler_state()
                self.scheduler = None
                message = str(exc).replace("\n", " ")[:MAX_ERROR_CHARS]
                return {
                    "ok": False,
                    "command": "close",
                    "error": f"{type(exc).__name__}: {message}",
                    **resources,
                }
            resources = self._scheduler_state()
            self.scheduler = None
            return {"ok": True, "command": "close", **resources}
        return {
            "ok": True,
            "command": "close",
            "initialized": False,
            "leak_free": True,
        }

    def _synchronize_now(self, value: Any, command: str) -> float:
        now_ms = _finite_number(value, f"{command} now_ms")
        if now_ms < self.now_ms:
            raise ProtocolError(
                f"{command} now_ms must be monotonic: {now_ms} < {self.now_ms}"
            )
        self.now_ms = now_ms
        self.scheduler.virtual_now_ms = now_ms
        self.scheduler.vllm_config.virtual_now_ms = now_ms
        return now_ms

    def _scheduler_state(self) -> dict[str, Any]:
        block_pool = self.scheduler.kv_cache_manager.block_pool
        free_blocks = int(block_pool.get_num_free_blocks())
        raw_total_blocks = int(block_pool.num_gpu_blocks)
        capacity_blocks = raw_total_blocks - 1
        if capacity_blocks < 0 or not 0 <= free_blocks <= capacity_blocks:
            raise RuntimeError("candidate KV block-pool accounting is invalid")
        used_blocks = capacity_blocks - free_blocks
        computed_usage = used_blocks / capacity_blocks if capacity_blocks else 0.0
        reported_usage = float(self.scheduler.kv_cache_manager.usage)
        if (
            not math.isfinite(reported_usage)
            or not 0.0 <= reported_usage <= 1.0
            or not math.isclose(
                reported_usage,
                computed_usage,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "candidate KV usage disagrees with block-pool accounting"
            )

        running_requests = len(self.scheduler.running)
        waiting_requests = len(self.scheduler.waiting)
        live_request_ids = sorted(str(value) for value in self.scheduler.requests)
        quiescent = (
            running_requests == 0 and waiting_requests == 0 and not live_request_ids
        )
        kv_cache_released = used_blocks == 0 and computed_usage == 0.0
        return {
            "initialized": True,
            "kv_cache_usage": computed_usage,
            "kv_cache_blocks": {
                "used_blocks": used_blocks,
                "free_blocks": free_blocks,
                "capacity_blocks": capacity_blocks,
            },
            "running_requests": running_requests,
            "waiting_requests": waiting_requests,
            "live_request_ids": live_request_ids,
            "quiescent": quiescent,
            "kv_cache_released": kv_cache_released,
            "leak_free": quiescent and kv_cache_released,
        }


def _import_vllm_runtime() -> dict[str, Any]:
    import torch
    import vllm
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    return {
        "version": vllm.__version__,
        "torch": torch,
        "CacheConfig": CacheConfig,
        "SchedulerConfig": SchedulerConfig,
        "SamplingParams": SamplingParams,
        "FullAttentionSpec": FullAttentionSpec,
        "KVCacheConfig": KVCacheConfig,
        "KVCacheGroupSpec": KVCacheGroupSpec,
        "ModelRunnerOutput": ModelRunnerOutput,
        "Request": Request,
    }


def _load_candidate_scheduler(candidate_path: Path) -> Any:
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise RuntimeError(
            f"candidate scheduler is not a regular file: {candidate_path}"
        )
    module_name = "vllm.v1.core.sched.scheduler"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create a module spec for candidate scheduler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    scheduler_type = getattr(module, "Scheduler", None)
    if not isinstance(scheduler_type, type):
        raise RuntimeError("candidate scheduler.py must define Scheduler")
    return scheduler_type


def _command_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ProtocolError("command payload must be a JSON object")
    command = payload.get("command", payload.get("op"))
    if not isinstance(command, str) or not command:
        raise ProtocolError("command payload requires command")
    return command


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be an object")
    return value


def _bounded_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ProtocolError(f"{name} must be between {minimum} and {maximum}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def _runtime_value(
    root: dict[str, Any], nested: dict[str, Any], key: str, default: Any
) -> Any:
    if key in nested:
        return nested[key]
    return root.get(key, default)


def _runtime_int(
    root: dict[str, Any],
    nested: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    return _bounded_int(
        _runtime_value(root, nested, key, default),
        f"runtime {key}",
        minimum=minimum,
        maximum=maximum,
    )


def _runtime_bool(
    root: dict[str, Any],
    nested: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = _runtime_value(root, nested, key, default)
    if not isinstance(value, bool):
        raise ProtocolError(f"runtime {key} must be a boolean")
    return value


def _first_int(
    values: tuple[Any, ...],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    selected = next((value for value in values if value is not None), default)
    return _bounded_int(selected, name, minimum=minimum, maximum=maximum)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runtime_config(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"runtime config is not a regular file: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ProtocolError("runtime config exceeds 65536 bytes")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return _mapping(value, "runtime config")


def _validate_solution_dir(path: Path) -> Path:
    solution_dir = path.resolve(strict=True)
    if not solution_dir.is_dir():
        raise ProtocolError(f"solution dir is not a directory: {solution_dir}")
    if any(path.name == ".git" for path in solution_dir.rglob(".git")):
        raise ProtocolError("solution dir must not contain .git")
    candidate = solution_dir / SCHEDULER_RELATIVE_PATH
    if candidate.is_symlink() or not candidate.is_file():
        raise ProtocolError(f"missing candidate file: {SCHEDULER_RELATIVE_PATH}")
    try:
        candidate.resolve(strict=True).relative_to(solution_dir)
    except ValueError as exc:
        raise ProtocolError("candidate scheduler escapes solution dir") from exc
    return solution_dir


def _install_resource_limits(config: dict[str, Any]) -> None:
    values = _mapping(config.get("resource_limits", {}), "runtime resource_limits")
    limits = (
        (resource.RLIMIT_AS, "address_space_bytes", 8 * 1024**3, 512 * 1024**2),
        (resource.RLIMIT_CPU, "cpu_seconds", 300, 1),
        (resource.RLIMIT_FSIZE, "file_size_bytes", 1024**2, 1),
        (resource.RLIMIT_NOFILE, "open_files", 128, 16),
    )
    for resource_id, key, default, minimum in limits:
        value = _bounded_int(
            values.get(key, default),
            f"runtime resource_limits.{key}",
            minimum=minimum,
            maximum=2**63 - 1,
        )
        current_soft, current_hard = resource.getrlimit(resource_id)
        if current_hard != resource.RLIM_INFINITY:
            value = min(value, current_hard)
        resource.setrlimit(resource_id, (value, value))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(resource, "RLIMIT_NPROC"):
        processes = _bounded_int(
            values.get("processes", 256),
            "runtime resource_limits.processes",
            minimum=1,
            maximum=4_096,
        )
        _, current_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if current_hard != resource.RLIM_INFINITY:
            processes = min(processes, current_hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))


def _sanitize_environment(scratch_dir: Path) -> None:
    preserved = {
        key: value
        for key, value in os.environ.items()
        if key in {"LANG", "LC_ALL", "TZ"}
    }
    os.environ.clear()
    os.environ.update(preserved)
    os.environ.update(
        {
            "BLIS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": str(scratch_dir),
            "JOBLIB_MULTIPROCESSING": "0",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": str(scratch_dir),
            "VECLIB_MAXIMUM_THREADS": "1",
            "VLLM_LOGGING_LEVEL": "ERROR",
        }
    )
    tempfile.tempdir = str(scratch_dir)


def _make_scratch_dir() -> Path:
    inherited = os.environ.get("TMPDIR")
    if inherited:
        candidate = Path(inherited)
        try:
            resolved = candidate.resolve(strict=True)
            metadata = candidate.lstat()
            temporary_roots = [
                root.resolve(strict=True)
                for root in (Path("/tmp"), Path("/var/tmp"))
                if root.exists()
            ]
            inside_temporary_root = any(
                resolved != root and root in resolved.parents
                for root in temporary_roots
            )
            if (
                not candidate.is_symlink()
                and resolved.is_dir()
                and metadata.st_uid == os.getuid()
                and metadata.st_mode & 0o077 == 0
                and inside_temporary_root
                and not any(resolved.iterdir())
            ):
                return resolved
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="mlsysbench-vllm-driver-"))


# Landlock is implemented locally so the candidate never needs repository access.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _landlock_abi_version() -> int | None:
    if sys.platform != "linux":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result >= 0:
        return int(result)
    if ctypes.get_errno() in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        return None
    return None


def _install_landlock(
    solution_dir: Path, scratch_dir: Path, mode: str
) -> dict[str, Any]:
    if mode not in {"auto", "required", "off"}:
        raise ProtocolError("runtime landlock must be auto, required, or off")
    if mode == "off":
        return {"landlock": "off", "landlock_abi": None}
    abi = _landlock_abi_version()
    if abi is None:
        if mode == "required":
            raise RuntimeError("Landlock is required but unavailable")
        return {"landlock": "unavailable", "landlock_abi": None}

    handled = (
        _FS_EXECUTE
        | _FS_WRITE_FILE
        | _FS_READ_FILE
        | _FS_READ_DIR
        | _FS_REMOVE_DIR
        | _FS_REMOVE_FILE
        | _FS_MAKE_CHAR
        | _FS_MAKE_DIR
        | _FS_MAKE_REG
        | _FS_MAKE_SOCK
        | _FS_MAKE_FIFO
        | _FS_MAKE_BLOCK
        | _FS_MAKE_SYM
    )
    if abi >= 2:
        handled |= _FS_REFER
    if abi >= 3:
        handled |= _FS_TRUNCATE
    libc = ctypes.CDLL(None, use_errno=True)
    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        _raise_landlock_errno("landlock_create_ruleset")
    try:
        paths = [
            (solution_dir, False),
            (scratch_dir, True),
            (Path(sys.prefix), False),
            (Path(sys.base_prefix), False),
            (Path("/usr"), False),
            (Path("/lib"), False),
            (Path("/lib64"), False),
            (Path("/dev/null"), False),
            (Path("/dev/urandom"), False),
            (Path("/proc/cpuinfo"), False),
        ]
        seen: set[str] = set()
        for path, writable in paths:
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError:
                continue
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            path_fd = os.open(resolved, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = _FS_EXECUTE | _FS_READ_FILE
                if resolved.is_dir():
                    allowed |= _FS_READ_DIR
                elif resolved == Path("/dev/null"):
                    allowed |= _FS_WRITE_FILE
                if writable:
                    allowed = handled
                rule = _PathBeneathAttr(
                    allowed_access=allowed & handled,
                    parent_fd=path_fd,
                )
                result = libc.syscall(
                    ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                )
                if result != 0:
                    _raise_landlock_errno(f"landlock_add_rule({resolved})")
            finally:
                os.close(path_fd)
        if (
            libc.prctl(
                ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            )
            != 0
        ):
            _raise_landlock_errno("prctl(PR_SET_NO_NEW_PRIVS)")
        result = libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
        if result != 0:
            _raise_landlock_errno("landlock_restrict_self")
    finally:
        os.close(ruleset_fd)
    return {"landlock": "enabled", "landlock_abi": abi}


def _raise_landlock_errno(operation: str) -> None:
    error_number = ctypes.get_errno()
    raise RuntimeError(f"{operation} failed: {os.strerror(error_number)}")


def _open_protocol_writer() -> tuple[TextIO, TextIO]:
    protocol = os.fdopen(
        os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8"
    )
    devnull = open(os.devnull, "w", encoding="utf-8")
    os.dup2(devnull.fileno(), sys.stdout.fileno())
    return protocol, devnull


def _write_response(writer: TextIO, payload: dict[str, Any]) -> None:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        rendered = json.dumps(
            {"ok": False, "error": f"response serialization failed: {exc}"},
            ensure_ascii=True,
            separators=(",", ":"),
        )
    if len(rendered.encode("utf-8")) > MAX_OUTPUT_BYTES:
        rendered = json.dumps(
            {"ok": False, "error": "response exceeds 65536 bytes"},
            separators=(",", ":"),
        )
    writer.write(rendered + "\n")
    writer.flush()


def _error_response(exc: BaseException, command: str | None = None) -> dict[str, Any]:
    message = str(exc).replace("\n", " ")[:MAX_ERROR_CHARS]
    response: dict[str, Any] = {
        "ok": False,
        "error": f"{type(exc).__name__}: {message}",
    }
    if command:
        response["command"] = command
    return response


def _read_input_line(stream: Any) -> tuple[bytes | None, bool]:
    line = stream.readline(MAX_INPUT_BYTES + 1)
    if not line:
        return None, False
    oversized = len(line) > MAX_INPUT_BYTES
    if oversized and not line.endswith(b"\n"):
        while True:
            remainder = stream.readline(MAX_INPUT_BYTES + 1)
            if not remainder or remainder.endswith(b"\n"):
                break
    return line, oversized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", required=True)
    parser.add_argument("--runtime-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, devnull = _open_protocol_writer()
    try:
        runtime_config = _load_runtime_config(Path(args.runtime_config))
        solution_dir = _validate_solution_dir(Path(args.solution_dir))
        scratch_dir = _make_scratch_dir()
        _install_resource_limits(runtime_config)
        _sanitize_environment(scratch_dir)
        sandbox = _install_landlock(
            solution_dir,
            scratch_dir,
            str(runtime_config.get("landlock", "auto")),
        )
        driver = Driver(solution_dir, runtime_config, sandbox)
    except BaseException as exc:
        _write_response(protocol, _error_response(exc, "startup"))
        protocol.close()
        devnull.close()
        raise SystemExit(1) from None

    while True:
        line, oversized = _read_input_line(sys.stdin.buffer)
        if line is None:
            break
        if oversized:
            _write_response(
                protocol,
                {"ok": False, "error": "input command exceeds 1048576 bytes"},
            )
            continue
        command: str | None = None
        try:
            payload = json.loads(line.decode("utf-8"))
            command = _command_name(payload)
            response, should_close = driver.dispatch(payload)
        except BaseException as exc:
            response = _error_response(exc, command)
            should_close = False
        _write_response(protocol, response)
        if should_close:
            break

    if driver.scheduler is not None:
        try:
            driver.scheduler.shutdown()
        except BaseException:
            pass
    protocol.close()
    devnull.close()


if __name__ == "__main__":
    main()
