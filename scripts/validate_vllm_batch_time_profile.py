#!/usr/bin/env python3
"""Validate a calibrated vLLM batch-time profile on an independent holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsysbench.simai_bench.io import write_json
from mlsysbench.simai_bench.vllm_batch_time_model import (
    FEATURE_NAMES,
    BatchDescriptor,
    BatchTimeModelError,
    CalibratedBatchTimeModel,
    PredictionRejectedError,
)


REPORT_SCHEMA_VERSION = 1
RAW_PROFILE_KIND = "vllm_executor_batch_latency"
MIN_COVERAGE = 0.95
MAX_MEDIAN_APE = 0.08
MAX_P95_APE = 0.15
MIN_KENDALL_TAU_B = 0.80

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
RAW_RUNTIME_FIELDS = {
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
}
RAW_DEVICE_FIELDS = {
    "name",
    "total_memory_bytes",
    "compute_capability",
    "logical_gpu_index",
    "requested_host_gpu_index",
}
RAW_MEASUREMENT_FIELDS = {
    "clock",
    "cuda_synchronize_before_after",
    "warmup_case",
    "repeats",
    "case_count",
    "raw_sample_count",
}


class HoldoutValidationError(ValueError):
    """Raised when the holdout artifact cannot support a valid comparison."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a strict vLLM batch-time profile against an independent "
            "profile_vllm_executor raw JSON artifact."
        )
    )
    parser.add_argument("--profile", required=True, help="Strict calibrated profile JSON")
    parser.add_argument("--holdout-raw", required=True, help="Independent raw profiler JSON")
    parser.add_argument("--output", required=True, help="Validation report JSON")
    return parser.parse_args()


def _load_json_artifact(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise HoldoutValidationError(f"cannot read {label} {path}: {exc}") from exc
    try:
        payload = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldoutValidationError(f"cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HoldoutValidationError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _validate_and_aggregate_holdout(
    payload: Mapping[str, Any],
    model: CalibratedBatchTimeModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_exact_keys(payload, RAW_TOP_LEVEL_FIELDS, "holdout")
    if _require_int(payload["schema_version"], "holdout.schema_version", minimum=1) != 1:
        raise HoldoutValidationError("holdout.schema_version must be 1")
    if payload["profile_kind"] != RAW_PROFILE_KIND:
        raise HoldoutValidationError(
            f"holdout.profile_kind must be {RAW_PROFILE_KIND!r}"
        )
    _require_string(payload["created_at_utc"], "holdout.created_at_utc")
    if not isinstance(payload["feature_names"], list) or tuple(
        payload["feature_names"]
    ) != FEATURE_NAMES:
        raise HoldoutValidationError(
            f"holdout.feature_names must be exactly {list(FEATURE_NAMES)!r}"
        )
    if not isinstance(payload["points"], list):
        raise HoldoutValidationError("holdout.points must be an array")

    runtime = _require_mapping(payload["runtime"], "holdout.runtime")
    device = _require_mapping(payload["device"], "holdout.device")
    measurement = _require_mapping(payload["measurement"], "holdout.measurement")
    _require_exact_keys(runtime, RAW_RUNTIME_FIELDS, "holdout.runtime")
    _require_exact_keys(device, RAW_DEVICE_FIELDS, "holdout.device")
    _require_exact_keys(measurement, RAW_MEASUREMENT_FIELDS, "holdout.measurement")

    for name in (
        "vllm_version",
        "torch_version",
        "python_version",
        "dtype",
        "load_format",
    ):
        _require_string(runtime[name], f"holdout.runtime.{name}")
    for name in ("model_config_sha256", "scheduler_sha256"):
        _require_sha256(runtime[name], f"holdout.runtime.{name}")
    for name in ("enforce_eager", "prefix_caching"):
        _require_bool(runtime[name], f"holdout.runtime.{name}")
    for name in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        _require_int(runtime[name], f"holdout.runtime.{name}", minimum=1)
    _require_int(runtime["seed"], "holdout.runtime.seed", minimum=0)
    if runtime["load_format"] != "dummy":
        raise HoldoutValidationError("holdout.runtime.load_format must be 'dummy'")
    if runtime["prefix_caching"] is not False:
        raise HoldoutValidationError("holdout.runtime.prefix_caching must be false")

    _require_string(device["name"], "holdout.device.name")
    _require_int(device["total_memory_bytes"], "holdout.device.total_memory_bytes", minimum=1)
    capability = device["compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise HoldoutValidationError(
            "holdout.device.compute_capability must contain major and minor integers"
        )
    for index, value in enumerate(capability):
        _require_int(value, f"holdout.device.compute_capability[{index}]", minimum=0)
    _require_int(device["logical_gpu_index"], "holdout.device.logical_gpu_index", minimum=0)
    _require_int(
        device["requested_host_gpu_index"],
        "holdout.device.requested_host_gpu_index",
        minimum=0,
    )

    _require_string(measurement["clock"], "holdout.measurement.clock")
    _require_string(measurement["warmup_case"], "holdout.measurement.warmup_case")
    if _require_bool(
        measurement["cuda_synchronize_before_after"],
        "holdout.measurement.cuda_synchronize_before_after",
    ) is not True:
        raise HoldoutValidationError(
            "holdout must synchronize CUDA before and after each latency sample"
        )
    repeats = _require_int(measurement["repeats"], "holdout.measurement.repeats", minimum=3)
    case_count = _require_int(
        measurement["case_count"], "holdout.measurement.case_count", minimum=1
    )
    raw_sample_count = _require_int(
        measurement["raw_sample_count"],
        "holdout.measurement.raw_sample_count",
        minimum=1,
    )

    identity = model.identity
    comparable_identity = {
        "device.model": (identity.device_model, device["name"]),
        "device.compute_capability": (
            identity.compute_capability,
            f"{capability[0]}.{capability[1]}",
        ),
        "device.total_memory_bytes": (
            identity.total_memory_bytes,
            device["total_memory_bytes"],
        ),
        "runtime.python_version": (identity.python_version, runtime["python_version"]),
        "runtime.torch_version": (identity.torch_version, runtime["torch_version"]),
        "runtime.dtype": (identity.dtype, runtime["dtype"]),
        "runtime.enforce_eager": (identity.enforce_eager, runtime["enforce_eager"]),
        "model.max_model_len": (identity.max_model_len, runtime["max_model_len"]),
        "model.tensor_parallel_size": (
            identity.tensor_parallel_size,
            runtime["tensor_parallel_size"],
        ),
        "device.gpu_count": (identity.gpu_count, runtime["tensor_parallel_size"]),
    }
    identity_mismatches = [
        f"{name}: profile has {expected!r}, holdout has {actual!r}"
        for name, (expected, actual) in comparable_identity.items()
        if expected != actual
    ]
    if identity_mismatches:
        raise HoldoutValidationError(
            "holdout execution identity mismatch: " + "; ".join(identity_mismatches)
        )

    raw_samples = payload["raw_samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise HoldoutValidationError("holdout.raw_samples must be a non-empty array")
    if len(raw_samples) != raw_sample_count:
        raise HoldoutValidationError(
            "holdout.measurement.raw_sample_count does not match raw_samples length"
        )

    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    cases: set[str] = set()
    sample_fields = {"case", "repeat", *FEATURE_NAMES, "latency_ms"}
    for index, raw_sample in enumerate(raw_samples):
        label = f"holdout.raw_samples[{index}]"
        sample = _require_mapping(raw_sample, label)
        _require_exact_keys(sample, sample_fields, label)
        cases.add(_require_string(sample["case"], f"{label}.case"))
        repeat = _require_int(sample["repeat"], f"{label}.repeat", minimum=0)
        if repeat >= repeats:
            raise HoldoutValidationError(
                f"{label}.repeat must be smaller than measurement.repeats ({repeats})"
            )
        descriptor = BatchDescriptor.from_mapping(
            {name: sample[name] for name in FEATURE_NAMES}
        )
        descriptor.validate(max_model_len=identity.max_model_len)
        latency_ms = _require_number(
            sample["latency_ms"], f"{label}.latency_ms", minimum_exclusive=0.0
        )
        grouped[descriptor.feature_tuple()].append(latency_ms)
    if len(cases) != case_count:
        raise HoldoutValidationError(
            "holdout.measurement.case_count does not match distinct raw sample cases"
        )

    points: list[dict[str, Any]] = []
    for feature_values in sorted(grouped):
        latencies = grouped[feature_values]
        if len(latencies) < repeats:
            raise HoldoutValidationError(
                f"descriptor {feature_values!r} has {len(latencies)} samples; "
                f"at least {repeats} are required"
            )
        points.append(
            {
                "descriptor": dict(zip(FEATURE_NAMES, feature_values)),
                "sample_count": len(latencies),
                "actual_latency_ms": statistics.fmean(latencies),
                "actual_latency_stddev_ms": statistics.stdev(latencies),
            }
        )

    raw_metadata = {
        "vllm_version": runtime["vllm_version"],
        "model_config_sha256": runtime["model_config_sha256"],
        "scheduler_sha256": runtime["scheduler_sha256"],
        "clock": measurement["clock"],
        "repeats": repeats,
        "case_count": case_count,
        "raw_sample_count": raw_sample_count,
    }
    return points, raw_metadata


def validate_holdout(
    model: CalibratedBatchTimeModel,
    holdout_points: Sequence[Mapping[str, Any]],
    *,
    profile_artifact_sha256: str,
    holdout_artifact_sha256: str,
    raw_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all unique holdout descriptors and apply the fixed acceptance gates."""

    if holdout_artifact_sha256 == model.measurement["raw_artifact_sha256"]:
        raise HoldoutValidationError(
            "holdout raw artifact is identical to the profile calibration artifact"
        )

    point_results: list[dict[str, Any]] = []
    accepted_apes: list[float] = []
    accepted_actual: list[float] = []
    accepted_predicted: list[float] = []
    for point in holdout_points:
        descriptor = dict(point["descriptor"])
        actual_latency_ms = float(point["actual_latency_ms"])
        result: dict[str, Any] = {
            "descriptor": descriptor,
            "sample_count": int(point["sample_count"]),
            "actual_latency_ms": actual_latency_ms,
            "actual_latency_stddev_ms": float(point["actual_latency_stddev_ms"]),
        }
        try:
            prediction = model.predict(descriptor)
        except PredictionRejectedError as exc:
            diagnostics = dict(exc.diagnostics)
            result.update(
                {
                    "status": "rejected",
                    "predicted_latency_ms": None,
                    "signed_error_ms": None,
                    "signed_relative_error": None,
                    "absolute_percentage_error": None,
                    "normalized_distance": diagnostics.get(
                        "normalized_distance",
                        diagnostics.get("nearest_normalized_distance"),
                    ),
                    "weighted_normalized_distance": diagnostics.get(
                        "weighted_normalized_distance"
                    ),
                    "uncertainty_ms": diagnostics.get("uncertainty_ms"),
                    "relative_uncertainty": diagnostics.get("relative_uncertainty"),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "diagnostics": diagnostics,
                    },
                }
            )
        except BatchTimeModelError as exc:
            result.update(
                {
                    "status": "rejected",
                    "predicted_latency_ms": None,
                    "signed_error_ms": None,
                    "signed_relative_error": None,
                    "absolute_percentage_error": None,
                    "normalized_distance": None,
                    "weighted_normalized_distance": None,
                    "uncertainty_ms": None,
                    "relative_uncertainty": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        else:
            signed_error_ms = prediction.latency_ms - actual_latency_ms
            signed_relative_error = signed_error_ms / actual_latency_ms
            ape = abs(signed_relative_error)
            result.update(
                {
                    "status": "accepted",
                    "predicted_latency_ms": prediction.latency_ms,
                    "signed_error_ms": signed_error_ms,
                    "signed_relative_error": signed_relative_error,
                    "absolute_percentage_error": ape,
                    "normalized_distance": prediction.normalized_distance,
                    "weighted_normalized_distance": (
                        prediction.weighted_normalized_distance
                    ),
                    "uncertainty_ms": prediction.uncertainty_ms,
                    "relative_uncertainty": prediction.relative_uncertainty,
                    "exact_match": prediction.exact_match,
                    "prediction_diagnostics": prediction.diagnostics(),
                    "error": None,
                }
            )
            accepted_apes.append(ape)
            accepted_actual.append(actual_latency_ms)
            accepted_predicted.append(prediction.latency_ms)
        point_results.append(result)

    total_count = len(point_results)
    accepted_count = len(accepted_apes)
    rejected_count = total_count - accepted_count
    coverage = accepted_count / total_count if total_count else 0.0
    median_ape = _percentile(accepted_apes, 0.50)
    p90_ape = _percentile(accepted_apes, 0.90)
    p95_ape = _percentile(accepted_apes, 0.95)
    tau_details = _kendall_tau_b_details(accepted_actual, accepted_predicted)
    spearman_rho = _spearman_rho(accepted_actual, accepted_predicted)

    gates = {
        "coverage": _gate(coverage, MIN_COVERAGE, ">="),
        "median_ape": _gate(median_ape, MAX_MEDIAN_APE, "<="),
        "p95_ape": _gate(p95_ape, MAX_P95_APE, "<="),
        "kendall_tau_b": _gate(tau_details["value"], MIN_KENDALL_TAU_B, ">="),
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validation_kind": "vllm_batch_time_profile_hardware_holdout",
        "passed": passed,
        "definitions": {
            "ape_unit": "fraction",
            "signed_error": "predicted_minus_actual",
            "percentile_method": "linear_r7",
            "rank_metrics_population": "accepted_unique_descriptors",
            "coverage_denominator": "all_unique_holdout_descriptors",
        },
        "profile": {
            "profile_id": model.identity.profile_id,
            "profile_revision": model.identity.profile_revision,
            "profile_artifact_sha256": profile_artifact_sha256,
            "calibration_raw_artifact_sha256": model.measurement[
                "raw_artifact_sha256"
            ],
        },
        "holdout": {
            "raw_artifact_sha256": holdout_artifact_sha256,
            "independent_from_calibration_artifact": True,
            **dict(raw_metadata),
        },
        "thresholds": {
            "minimum_coverage": MIN_COVERAGE,
            "maximum_median_ape": MAX_MEDIAN_APE,
            "maximum_p95_ape": MAX_P95_APE,
            "minimum_kendall_tau_b": MIN_KENDALL_TAU_B,
        },
        "summary": {
            "unique_descriptor_count": total_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "coverage": coverage,
            "median_ape": median_ape,
            "p90_ape": p90_ape,
            "p95_ape": p95_ape,
            "kendall_tau_b": tau_details["value"],
            "kendall_tau_b_pairs": {
                key: value for key, value in tau_details.items() if key != "value"
            },
            "spearman_rho": spearman_rho,
        },
        "gates": gates,
        "points": point_results,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _kendall_tau_b_details(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, int | float | None]:
    if len(actual) != len(predicted):
        raise ValueError("Kendall inputs must have equal length")
    concordant = 0
    discordant = 0
    tied_actual_only = 0
    tied_predicted_only = 0
    tied_both = 0
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            actual_delta = actual[left] - actual[right]
            predicted_delta = predicted[left] - predicted[right]
            if actual_delta == 0 and predicted_delta == 0:
                tied_both += 1
            elif actual_delta == 0:
                tied_actual_only += 1
            elif predicted_delta == 0:
                tied_predicted_only += 1
            elif actual_delta * predicted_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_actual_only)
        * (concordant + discordant + tied_predicted_only)
    )
    value = (concordant - discordant) / denominator if denominator else None
    return {
        "value": value,
        "concordant": concordant,
        "discordant": discordant,
        "tied_actual_only": tied_actual_only,
        "tied_predicted_only": tied_predicted_only,
        "tied_both": tied_both,
    }


def _kendall_tau_b(
    actual: Sequence[float], predicted: Sequence[float]
) -> float | None:
    value = _kendall_tau_b_details(actual, predicted)["value"]
    return None if value is None else float(value)


def _spearman_rho(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(actual) != len(predicted):
        raise ValueError("Spearman inputs must have equal length")
    if len(actual) < 2:
        return None
    actual_ranks = _average_ranks(actual)
    predicted_ranks = _average_ranks(predicted)
    actual_mean = statistics.fmean(actual_ranks)
    predicted_mean = statistics.fmean(predicted_ranks)
    numerator = sum(
        (left - actual_mean) * (right - predicted_mean)
        for left, right in zip(actual_ranks, predicted_ranks)
    )
    actual_squared = sum((value - actual_mean) ** 2 for value in actual_ranks)
    predicted_squared = sum((value - predicted_mean) ** 2 for value in predicted_ranks)
    denominator = math.sqrt(actual_squared * predicted_squared)
    return numerator / denominator if denominator else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _gate(value: float | None, threshold: float, operator: str) -> dict[str, Any]:
    if operator == ">=":
        passed = value is not None and value >= threshold
    elif operator == "<=":
        passed = value is not None and value <= threshold
    else:
        raise ValueError(f"unsupported gate operator {operator!r}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _fatal_report(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validation_kind": "vllm_batch_time_profile_hardware_holdout",
        "passed": False,
        "fatal_error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> int:
    args = _parse_args()
    profile_path = Path(args.profile).resolve()
    holdout_path = Path(args.holdout_raw).resolve()
    output_path = Path(args.output).resolve()
    if output_path in {profile_path, holdout_path}:
        raise SystemExit("--output must not overwrite either input artifact")

    try:
        model = CalibratedBatchTimeModel.from_path(profile_path)
        profile_artifact_sha256 = _sha256_file(profile_path)
        holdout_payload, holdout_artifact_sha256 = _load_json_artifact(
            holdout_path, "holdout raw artifact"
        )
        holdout_points, raw_metadata = _validate_and_aggregate_holdout(
            holdout_payload, model
        )
        report = validate_holdout(
            model,
            holdout_points,
            profile_artifact_sha256=profile_artifact_sha256,
            holdout_artifact_sha256=holdout_artifact_sha256,
            raw_metadata=raw_metadata,
        )
    except (HoldoutValidationError, BatchTimeModelError) as exc:
        report = _fatal_report(exc)

    write_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "passed": report["passed"],
                "summary": report.get("summary"),
                "fatal_error": report.get("fatal_error"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HoldoutValidationError(f"cannot hash profile artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HoldoutValidationError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise HoldoutValidationError(f"{label} keys must be strings")
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
        raise HoldoutValidationError(f"{label} has invalid fields: {'; '.join(details)}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HoldoutValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise HoldoutValidationError(f"{label} must be boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HoldoutValidationError(f"{label} must be an integer")
    if value < minimum:
        raise HoldoutValidationError(f"{label} must be at least {minimum}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum_exclusive: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HoldoutValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= minimum_exclusive:
        raise HoldoutValidationError(
            f"{label} must be finite and greater than {minimum_exclusive}"
        )
    return result


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise HoldoutValidationError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_number(value: str) -> None:
    raise HoldoutValidationError(f"non-standard JSON number {value!r} is not allowed")


if __name__ == "__main__":
    raise SystemExit(main())
