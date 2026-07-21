#!/usr/bin/env python3
"""Build a strict calibrated batch-time profile from raw vLLM measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsysbench.simai_bench.io import write_json
from mlsysbench.simai_bench.vllm_batch_time_model import (
    ALLOWED_FEATURE_TRANSFORMS,
    FEATURE_NAMES,
    PROFILE_SCHEMA_VERSION,
    BatchDescriptor,
    BatchTimeProfileIdentity,
    CalibratedBatchTimeModel,
)


RAW_SCHEMA_VERSION = 1
RAW_PROFILE_KIND = "vllm_executor_batch_latency"
RAW_TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile_kind",
    "created_at_utc",
    "runtime",
    "device",
    "measurement",
    "feature_names",
    "points",
    "raw_samples",
}


class RawProfileError(ValueError):
    """Raised when a profiler artifact cannot support a calibrated profile."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate raw vLLM executor samples into the strict MLSysBench "
            "batch-time profile schema."
        )
    )
    parser.add_argument("--input", required=True, help="Raw profiler JSON artifact")
    parser.add_argument("--output", required=True, help="Calibrated profile JSON")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-revision", required=True)
    parser.add_argument("--device-vendor", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--vllm-revision", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--clock-policy",
        required=True,
        help="Pinned GPU clock/power policy used while collecting the raw artifact",
    )
    parser.add_argument("--neighbors", type=int, required=True)
    parser.add_argument("--distance-power", type=float, required=True)
    parser.add_argument("--max-normalized-distance", type=float, required=True)
    parser.add_argument("--max-relative-uncertainty", type=float, required=True)
    parser.add_argument(
        "--feature-weights-json",
        type=_parse_feature_weights,
        required=True,
        metavar="JSON",
        help="JSON object assigning a positive weight to each batch feature",
    )
    parser.add_argument(
        "--feature-transforms-json",
        type=_parse_feature_transforms,
        required=True,
        metavar="JSON",
        help="JSON object assigning linear or log1p to each batch feature",
    )
    args = parser.parse_args()
    for name in (
        "profile_id",
        "profile_revision",
        "device_vendor",
        "driver_version",
        "cuda_version",
        "vllm_revision",
        "attention_backend",
        "model_name",
        "model_revision",
        "clock_policy",
    ):
        value = getattr(args, name)
        if not value or value != value.strip():
            parser.error(f"--{name.replace('_', '-')} must be a non-empty trimmed string")
    return args


def _parse_feature_weights(value: str) -> dict[str, float]:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (json.JSONDecodeError, RawProfileError) as exc:
        raise argparse.ArgumentTypeError(f"invalid feature weights JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("feature weights must be a JSON object")
    try:
        _require_exact_keys(payload, set(FEATURE_NAMES), "feature weights")
        return {
            name: _require_number(
                payload[name], f"feature weights.{name}", minimum_exclusive=0.0
            )
            for name in FEATURE_NAMES
        }
    except RawProfileError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_feature_transforms(value: str) -> dict[str, str]:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (json.JSONDecodeError, RawProfileError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid feature transforms JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("feature transforms must be a JSON object")
    try:
        _require_exact_keys(payload, set(FEATURE_NAMES), "feature transforms")
        transforms: dict[str, str] = {}
        for name in FEATURE_NAMES:
            transform = _require_string(payload[name], f"feature transforms.{name}")
            if transform not in ALLOWED_FEATURE_TRANSFORMS:
                allowed = ", ".join(sorted(ALLOWED_FEATURE_TRANSFORMS))
                raise RawProfileError(
                    f"feature transforms.{name} must be one of: {allowed}"
                )
            transforms[name] = transform
        return transforms
    except RawProfileError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_raw_profile(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise RawProfileError(f"cannot read raw profile {path}: {exc}") from exc
    try:
        payload = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawProfileError(f"cannot decode raw profile {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RawProfileError("raw profile must be a JSON object")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _validate_raw_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(payload, RAW_TOP_LEVEL_FIELDS, "raw profile")
    if _require_int(payload["schema_version"], "raw profile.schema_version", minimum=1) != 1:
        raise RawProfileError(
            f"unsupported raw profile schema {payload['schema_version']!r}; "
            f"expected {RAW_SCHEMA_VERSION}"
        )
    if payload["profile_kind"] != RAW_PROFILE_KIND:
        raise RawProfileError(
            f"raw profile.profile_kind must be {RAW_PROFILE_KIND!r}"
        )
    _require_string(payload["created_at_utc"], "raw profile.created_at_utc")
    if not isinstance(payload["feature_names"], list) or tuple(
        payload["feature_names"]
    ) != FEATURE_NAMES:
        raise RawProfileError(
            f"raw profile.feature_names must be exactly {list(FEATURE_NAMES)!r}"
        )
    if not isinstance(payload["points"], list):
        raise RawProfileError("raw profile.points must be an array")

    runtime = _require_mapping(payload["runtime"], "raw profile.runtime")
    _require_exact_keys(
        runtime,
        {
            "vllm_version",
            "torch_version",
            "python_version",
            "model_config_sha256",
            "scheduler_sha256",
            "dtype",
            "load_format",
            "enforce_eager",
            "prefix_caching",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "seed",
        },
        "raw profile.runtime",
    )
    for name in (
        "vllm_version",
        "torch_version",
        "python_version",
        "dtype",
        "load_format",
    ):
        _require_string(runtime[name], f"raw profile.runtime.{name}")
    for name in ("model_config_sha256", "scheduler_sha256"):
        _require_sha256(runtime[name], f"raw profile.runtime.{name}")
    for name in ("enforce_eager", "prefix_caching"):
        _require_bool(runtime[name], f"raw profile.runtime.{name}")
    for name in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        _require_int(runtime[name], f"raw profile.runtime.{name}", minimum=1)
    _require_int(runtime["seed"], "raw profile.runtime.seed", minimum=0)
    if runtime["load_format"] != "dummy":
        raise RawProfileError("raw profile.runtime.load_format must be 'dummy'")

    device = _require_mapping(payload["device"], "raw profile.device")
    _require_exact_keys(
        device,
        {
            "name",
            "total_memory_bytes",
            "compute_capability",
            "logical_gpu_index",
            "requested_host_gpu_index",
        },
        "raw profile.device",
    )
    _require_string(device["name"], "raw profile.device.name")
    _require_int(
        device["total_memory_bytes"],
        "raw profile.device.total_memory_bytes",
        minimum=1,
    )
    capability = device["compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise RawProfileError(
            "raw profile.device.compute_capability must contain major and minor integers"
        )
    for index, value in enumerate(capability):
        _require_int(
            value,
            f"raw profile.device.compute_capability[{index}]",
            minimum=0,
        )
    _require_int(device["logical_gpu_index"], "raw profile.device.logical_gpu_index", minimum=0)
    _require_int(
        device["requested_host_gpu_index"],
        "raw profile.device.requested_host_gpu_index",
        minimum=0,
    )

    measurement = _require_mapping(payload["measurement"], "raw profile.measurement")
    _require_exact_keys(
        measurement,
        {
            "clock",
            "cuda_synchronize_before_after",
            "warmup_case",
            "repeats",
            "case_count",
            "raw_sample_count",
        },
        "raw profile.measurement",
    )
    _require_string(measurement["clock"], "raw profile.measurement.clock")
    _require_string(measurement["warmup_case"], "raw profile.measurement.warmup_case")
    synchronized = _require_bool(
        measurement["cuda_synchronize_before_after"],
        "raw profile.measurement.cuda_synchronize_before_after",
    )
    if not synchronized:
        raise RawProfileError(
            "raw profile must synchronize CUDA before and after each latency sample"
        )
    repeats = _require_int(measurement["repeats"], "raw profile.measurement.repeats", minimum=3)
    case_count = _require_int(
        measurement["case_count"], "raw profile.measurement.case_count", minimum=1
    )
    raw_sample_count = _require_int(
        measurement["raw_sample_count"],
        "raw profile.measurement.raw_sample_count",
        minimum=1,
    )

    raw_samples = payload["raw_samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise RawProfileError("raw profile.raw_samples must be a non-empty array")
    if raw_sample_count != len(raw_samples):
        raise RawProfileError(
            "raw profile.measurement.raw_sample_count does not match raw_samples length"
        )

    sample_fields = {"case", "repeat", *FEATURE_NAMES, "latency_ms"}
    validated_samples: list[tuple[BatchDescriptor, float]] = []
    cases: set[str] = set()
    for index, raw_sample in enumerate(raw_samples):
        label = f"raw profile.raw_samples[{index}]"
        sample = _require_mapping(raw_sample, label)
        _require_exact_keys(sample, sample_fields, label)
        cases.add(_require_string(sample["case"], f"{label}.case"))
        repeat = _require_int(sample["repeat"], f"{label}.repeat", minimum=0)
        if repeat >= repeats:
            raise RawProfileError(
                f"{label}.repeat must be smaller than measurement.repeats ({repeats})"
            )
        descriptor = BatchDescriptor.from_mapping(
            {name: sample[name] for name in FEATURE_NAMES}
        )
        descriptor.validate(max_model_len=int(runtime["max_model_len"]))
        latency = _require_number(
            sample["latency_ms"], f"{label}.latency_ms", minimum_exclusive=0.0
        )
        validated_samples.append((descriptor, latency))
    if len(cases) != case_count:
        raise RawProfileError(
            "raw profile.measurement.case_count does not match distinct raw sample cases"
        )

    return {
        "runtime": runtime,
        "device": device,
        "measurement": measurement,
        "samples": validated_samples,
    }


def _aggregate_points(
    samples: list[tuple[BatchDescriptor, float]],
    *,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for descriptor, latency_ms in samples:
        grouped[descriptor.feature_tuple()].append(latency_ms)
    if len(grouped) < 2:
        raise RawProfileError("raw samples must contain at least two distinct descriptors")

    points: list[dict[str, Any]] = []
    for descriptor_values in sorted(grouped):
        latencies = grouped[descriptor_values]
        if len(latencies) < minimum_samples:
            descriptor_text = ", ".join(
                f"{name}={value}"
                for name, value in zip(FEATURE_NAMES, descriptor_values)
            )
            raise RawProfileError(
                f"descriptor ({descriptor_text}) has {len(latencies)} samples; "
                f"at least {minimum_samples} are required"
            )
        descriptor_payload = dict(zip(FEATURE_NAMES, descriptor_values))
        canonical_descriptor = json.dumps(
            descriptor_payload, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        points.append(
            {
                "point_id": "batch_" + hashlib.sha256(canonical_descriptor).hexdigest(),
                **descriptor_payload,
                "latency_ms": statistics.fmean(latencies),
                "latency_stddev_ms": statistics.stdev(latencies),
                "sample_count": len(latencies),
            }
        )
    return points


def build_profile(
    raw_payload: Mapping[str, Any],
    *,
    raw_artifact_sha256: str,
    profile_id: str,
    profile_revision: str,
    device_vendor: str,
    driver_version: str,
    cuda_version: str,
    vllm_revision: str,
    attention_backend: str,
    model_name: str,
    model_revision: str,
    clock_policy: str,
    neighbors: int,
    distance_power: float,
    max_normalized_distance: float,
    max_relative_uncertainty: float,
    feature_weights: Mapping[str, float],
    feature_transforms: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and deterministically convert one raw profiler artifact."""

    raw = _validate_raw_profile(raw_payload)
    runtime = raw["runtime"]
    device = raw["device"]
    measurement = raw["measurement"]
    points = _aggregate_points(
        raw["samples"], minimum_samples=int(measurement["repeats"])
    )
    training_ranges = {
        name: {
            "min": min(int(point[name]) for point in points),
            "max": max(int(point[name]) for point in points),
        }
        for name in FEATURE_NAMES
    }
    capability = device["compute_capability"]
    identity_payload = {
        "profile_id": profile_id,
        "profile_revision": profile_revision,
        "device": {
            "vendor": device_vendor,
            "model": device["name"],
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "gpu_count": runtime["tensor_parallel_size"],
            "total_memory_bytes": device["total_memory_bytes"],
        },
        "runtime": {
            "driver_version": driver_version,
            "cuda_version": cuda_version,
            "python_version": runtime["python_version"],
            "torch_version": runtime["torch_version"],
            "vllm_revision": vllm_revision,
            "attention_backend": attention_backend,
            "dtype": runtime["dtype"],
            "enforce_eager": runtime["enforce_eager"],
        },
        "model": {
            "name": model_name,
            "revision": model_revision,
            "max_model_len": runtime["max_model_len"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
        },
    }
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "identity": identity_payload,
        "measurement": {
            "latency_unit": "ms",
            "latency_statistic": "mean",
            "timer": measurement["clock"] + "+cuda_synchronize_before_after",
            "warmup_iterations": 1,
            "minimum_measured_iterations": measurement["repeats"],
            "clock_policy": clock_policy,
            "raw_artifact_sha256": raw_artifact_sha256,
        },
        "feature_order": list(FEATURE_NAMES),
        "training_ranges": training_ranges,
        "interpolation": {
            "method": "inverse_distance_weighting_v1",
            "neighbors": neighbors,
            "distance_power": distance_power,
            "max_normalized_distance": max_normalized_distance,
            "max_relative_uncertainty": max_relative_uncertainty,
            "feature_weights": dict(feature_weights),
            "feature_transforms": dict(feature_transforms),
        },
        "points": points,
    }

    identity = BatchTimeProfileIdentity.from_mapping(identity_payload)
    CalibratedBatchTimeModel.from_dict(profile, expected_identity=identity)
    return profile


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise SystemExit("--input and --output must refer to different files")
    try:
        raw_payload, raw_hash = _load_raw_profile(input_path)
        profile = build_profile(
            raw_payload,
            raw_artifact_sha256=raw_hash,
            profile_id=args.profile_id,
            profile_revision=args.profile_revision,
            device_vendor=args.device_vendor,
            driver_version=args.driver_version,
            cuda_version=args.cuda_version,
            vllm_revision=args.vllm_revision,
            attention_backend=args.attention_backend,
            model_name=args.model_name,
            model_revision=args.model_revision,
            clock_policy=args.clock_policy,
            neighbors=args.neighbors,
            distance_power=args.distance_power,
            max_normalized_distance=args.max_normalized_distance,
            max_relative_uncertainty=args.max_relative_uncertainty,
            feature_weights=args.feature_weights_json,
            feature_transforms=args.feature_transforms_json,
        )
    except (OSError, RawProfileError, ValueError) as exc:
        raise SystemExit(f"cannot build batch-time profile: {exc}") from exc

    write_json(output_path, profile)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "points": len(profile["points"]),
                "raw_artifact_sha256": raw_hash,
                "samples": sum(point["sample_count"] for point in profile["points"]),
            },
            sort_keys=True,
        )
    )
    return 0


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RawProfileError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise RawProfileError(f"{label} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise RawProfileError(f"{label} has invalid fields: {'; '.join(details)}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RawProfileError(f"{label} must be a non-empty trimmed string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RawProfileError(f"{label} must be boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RawProfileError(f"{label} must be an integer")
    if value < minimum:
        raise RawProfileError(f"{label} must be at least {minimum}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum_exclusive: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawProfileError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= minimum_exclusive:
        raise RawProfileError(f"{label} must be finite and greater than {minimum_exclusive}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise RawProfileError(f"{label} must be 64 lowercase hexadecimal characters")
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawProfileError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_number(value: str) -> None:
    raise RawProfileError(f"non-standard JSON number {value!r} is not allowed")


if __name__ == "__main__":
    raise SystemExit(main())
