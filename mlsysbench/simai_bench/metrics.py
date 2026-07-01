"""Metric extraction for Vidur/SimAI benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import quantiles
from typing import Any


def load_metrics_json(data: dict[str, Any]) -> dict[str, float]:
    metrics = data.get("metrics", data)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def parse_vidur_output(output_dir: str | Path, slo: Any = None) -> dict[str, float]:
    output_dir = Path(output_dir)
    request_metrics = output_dir / "request_metrics.csv"
    if not request_metrics.exists():
        request_metrics = output_dir / "plots" / "request_metrics.csv"
    if not request_metrics.exists():
        raise FileNotFoundError(f"Could not find request_metrics.csv under {output_dir}")

    rows = _read_csv_rows(request_metrics)
    if not rows:
        raise ValueError(f"No rows in {request_metrics}")

    metrics: dict[str, float] = {"num_requests": float(len(rows))}

    e2e_values = _column_values(rows, "request_e2e_time")
    prefill_values = _column_values(rows, "prefill_e2e_time")
    decode_values = _column_values(rows, "decode_time")
    decode_tokens = _column_values(rows, "request_num_decode_tokens")

    if e2e_values:
        metrics["p50_e2e_ms"] = _percentile(e2e_values, 0.50) * 1000.0
        metrics["p95_e2e_ms"] = _percentile(e2e_values, 0.95) * 1000.0
        metrics["p99_e2e_ms"] = _percentile(e2e_values, 0.99) * 1000.0
    if prefill_values:
        metrics["p50_ttft_ms"] = _percentile(prefill_values, 0.50) * 1000.0
        metrics["p95_ttft_ms"] = _percentile(prefill_values, 0.95) * 1000.0
        metrics["p99_ttft_ms"] = _percentile(prefill_values, 0.99) * 1000.0
    if decode_values and decode_tokens and len(decode_values) == len(decode_tokens):
        tbt_values = [
            decode / max(tokens, 1.0)
            for decode, tokens in zip(decode_values, decode_tokens)
        ]
        metrics["p99_tbt_ms"] = _percentile(tbt_values, 0.99) * 1000.0

    arrived_at = _column_values(rows, "arrived_at")
    completed_at = _column_values(rows, "completed_at")
    if arrived_at and completed_at:
        duration = max(completed_at) - min(arrived_at)
        if duration > 0:
            metrics["throughput_rps"] = len(rows) / duration
            metrics["goodput_rps"] = _goodput(rows, duration, slo)

    return metrics


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    # Python's quantiles excludes endpoints; nearest-rank is stable for small task traces.
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def _goodput(rows: list[dict[str, str]], duration: float, slo: Any) -> float:
    if slo is None:
        return len(rows) / duration
    good = 0
    for row in rows:
        if _row_meets_slo(row, slo):
            good += 1
    return good / duration


def _row_meets_slo(row: dict[str, str], slo: Any) -> bool:
    checks = [
        ("prefill_e2e_time", getattr(slo, "p99_ttft_ms", None), 1000.0),
        ("request_e2e_time", getattr(slo, "p99_e2e_ms", None), 1000.0),
    ]
    for column, limit_ms, scale in checks:
        if limit_ms is None:
            continue
        try:
            if float(row[column]) * scale > limit_ms:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True

