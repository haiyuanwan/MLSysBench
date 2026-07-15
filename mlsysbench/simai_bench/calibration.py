"""Analyze paired simulator and hardware measurements for decision fidelity."""

from __future__ import annotations

import itertools
import math
import statistics
from pathlib import Path
from typing import Any

from mlsysbench.simai_bench.io import ConfigError, load_structured


def analyze_calibration(path: str | Path) -> dict[str, Any]:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ConfigError("calibration input must contain an object")
    metric = payload.get("metric")
    direction = payload.get("direction", "maximize")
    records = payload.get("records")
    top_k = int(payload.get("top_k", 3))
    if not isinstance(metric, str) or not metric:
        raise ConfigError("calibration metric must be a non-empty string")
    if direction not in {"maximize", "minimize"}:
        raise ConfigError("calibration direction must be maximize or minimize")
    if not isinstance(records, list) or len(records) < 2:
        raise ConfigError("calibration requires at least two paired records")
    if top_k <= 0:
        raise ConfigError("calibration top_k must be positive")

    normalized = [_normalize_record(record, metric) for record in records]
    simulator = [record["simulator"] for record in normalized]
    hardware = [record["hardware_mean"] for record in normalized]
    errors = [sim - real for sim, real in zip(simulator, hardware)]
    relative_errors = [abs(error) / abs(real) for error, real in zip(errors, hardware) if real != 0]
    simulator_ranks = _ranks(simulator, direction)
    hardware_ranks = _ranks(hardware, direction)
    spearman = _pearson(simulator_ranks, hardware_ranks)
    kendall, decision_agreement = _pairwise_agreement(simulator, hardware, direction)
    k = min(top_k, len(normalized))
    simulator_top = _top_ids(normalized, "simulator", direction, k)
    hardware_top = _top_ids(normalized, "hardware_mean", direction, k)
    repeated = [record for record in normalized if len(record["hardware_values"]) > 1]
    return {
        "schema_version": 1,
        "metric": metric,
        "direction": direction,
        "paired_configurations": len(normalized),
        "mean_absolute_error": statistics.fmean(abs(error) for error in errors),
        "mean_absolute_percentage_error": (
            statistics.fmean(relative_errors) if relative_errors else None
        ),
        "mean_signed_error": statistics.fmean(errors),
        "spearman_rank_correlation": spearman,
        "kendall_tau_b": kendall,
        "pairwise_decision_agreement": decision_agreement,
        "top_k": k,
        "simulator_top_k": simulator_top,
        "hardware_top_k": hardware_top,
        "top_k_overlap": len(set(simulator_top) & set(hardware_top)) / k,
        "repeated_hardware_configurations": len(repeated),
        "mean_hardware_coefficient_of_variation": (
            statistics.fmean(record["hardware_cv"] for record in repeated)
            if repeated
            else None
        ),
        "records": normalized,
    }


def _normalize_record(record: Any, metric: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ConfigError("each calibration record must contain an object")
    config_id = record.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ConfigError("each calibration record requires config_id")
    simulator = _metric_value(record.get("simulator"), metric, "simulator")
    hardware_payload = record.get("hardware")
    if isinstance(hardware_payload, list):
        hardware_values = [float(value) for value in hardware_payload]
    elif isinstance(hardware_payload, dict):
        values = hardware_payload.get("values")
        if isinstance(values, list):
            hardware_values = [float(value) for value in values]
        else:
            hardware_values = [_metric_value(hardware_payload, metric, "hardware")]
    else:
        hardware_values = [float(hardware_payload)]
    if not hardware_values or not all(math.isfinite(value) for value in hardware_values):
        raise ConfigError(f"hardware measurements for {config_id} must be finite")
    hardware_mean = statistics.fmean(hardware_values)
    hardware_cv = (
        statistics.stdev(hardware_values) / abs(hardware_mean)
        if len(hardware_values) > 1 and hardware_mean != 0
        else 0.0
    )
    return {
        "config_id": config_id,
        "simulator": simulator,
        "hardware_values": hardware_values,
        "hardware_mean": hardware_mean,
        "hardware_cv": hardware_cv,
    }


def _metric_value(payload: Any, metric: str, label: str) -> float:
    if isinstance(payload, dict):
        payload = payload.get(metric)
    try:
        value = float(payload)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} requires numeric metric {metric!r}") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{label} metric {metric!r} must be finite")
    return value


def _ranks(values: list[float], direction: str) -> list[float]:
    oriented = values if direction == "maximize" else [-value for value in values]
    ordered = sorted(range(len(values)), key=lambda index: oriented[index], reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and oriented[ordered[end]] == oriented[ordered[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[ordered[offset]] = rank
        position = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _pairwise_agreement(
    simulator: list[float],
    hardware: list[float],
    direction: str,
) -> tuple[float | None, float]:
    concordant = discordant = ties_sim = ties_hardware = 0
    for left, right in itertools.combinations(range(len(simulator)), 2):
        sim_delta = simulator[left] - simulator[right]
        hardware_delta = hardware[left] - hardware[right]
        if direction == "minimize":
            sim_delta = -sim_delta
            hardware_delta = -hardware_delta
        if sim_delta == 0 and hardware_delta == 0:
            ties_sim += 1
            ties_hardware += 1
        elif sim_delta == 0:
            ties_sim += 1
        elif hardware_delta == 0:
            ties_hardware += 1
        elif sim_delta * hardware_delta > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_sim)
        * (concordant + discordant + ties_hardware)
    )
    tau = (concordant - discordant) / denominator if denominator else None
    decisive = concordant + discordant
    agreement = concordant / decisive if decisive else 0.0
    return tau, agreement


def _top_ids(
    records: list[dict[str, Any]],
    field: str,
    direction: str,
    k: int,
) -> list[str]:
    return [
        record["config_id"]
        for record in sorted(
            records,
            key=lambda item: item[field],
            reverse=direction == "maximize",
        )[:k]
    ]
