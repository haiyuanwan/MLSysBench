#!/usr/bin/env python3
"""Build precommitted public/private workloads from pinned inference traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsysbench.simai_bench.io import write_json


PROFILES = ["mixed_prompt_output", "mixed_concurrency"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-trace", required=True)
    parser.add_argument("--conversation-trace", required=True)
    parser.add_argument("--public-output", required=True)
    parser.add_argument("--private-bundle-root", required=True)
    parser.add_argument("--commitment-output", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--public-start", type=int, required=True)
    parser.add_argument("--hidden-code-start", type=int, required=True)
    parser.add_argument("--hidden-conversation-start", type=int, required=True)
    parser.add_argument("--requests-per-case", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--e2e-slo-ms", type=float, required=True)
    args = parser.parse_args()
    if not args.bundle_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in args.bundle_id
    ):
        parser.error("--bundle-id must be a non-empty safe identifier")
    if args.requests_per_case < 16:
        parser.error("--requests-per-case must be at least 16")
    for name in ("ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trace(path: Path, max_model_len: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "arrived_at",
            "num_prefill_tokens",
            "num_decode_tokens",
        ]:
            raise ValueError(f"unexpected trace schema in {path}")
        for source_index, row in enumerate(reader):
            arrived_at = float(row["arrived_at"])
            prompt_tokens = int(row["num_prefill_tokens"])
            output_tokens = int(row["num_decode_tokens"])
            if not math.isfinite(arrived_at) or arrived_at < 0:
                raise ValueError(f"invalid arrival at source row {source_index}")
            if prompt_tokens <= 0 or output_tokens <= 0:
                raise ValueError(f"non-positive token count at source row {source_index}")
            if prompt_tokens + output_tokens > max_model_len:
                continue
            rows.append(
                {
                    "source_index": source_index,
                    "arrival_time_ms": arrived_at * 1000.0,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                }
            )
    return rows


def _case(
    *,
    name: str,
    rows: list[dict[str, Any]],
    start: int,
    count: int,
    arrival_time_scale: float,
    slo: dict[str, float],
    expose_selector: bool,
) -> dict[str, Any]:
    selected = rows[start : start + count]
    if len(selected) != count:
        raise ValueError(f"case {name} cannot select {count} rows at offset {start}")
    first_arrival = selected[0]["arrival_time_ms"]
    requests = [
        {
            "request_id": f"{name}_r{index:04d}",
            "arrival_time_ms": round(
                (float(row["arrival_time_ms"]) - first_arrival) * arrival_time_scale,
                9,
            ),
            "prompt_tokens": int(row["prompt_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "priority": 0,
        }
        for index, row in enumerate(selected)
    ]
    result: dict[str, Any] = {
        "name": name,
        "expected_requests": count,
        "slo": slo,
        "requests": requests,
    }
    if expose_selector:
        result["public_selector"] = {
            "legal_row_offset": start,
            "row_count": count,
            "arrival_time_scale": arrival_time_scale,
            "source_indices": [int(row["source_index"]) for row in selected],
        }
    return result


def main() -> int:
    args = _parse_args()
    code_path = Path(args.code_trace).resolve()
    conversation_path = Path(args.conversation_trace).resolve()
    code_rows = _load_trace(code_path, args.max_model_len)
    conversation_rows = _load_trace(conversation_path, args.max_model_len)
    slo = {
        "ttft_ms": args.ttft_slo_ms,
        "tpot_ms": args.tpot_slo_ms,
        "e2e_ms": args.e2e_slo_ms,
    }
    source = {
        "code_trace_sha256": _sha256(code_path),
        "conversation_trace_sha256": _sha256(conversation_path),
        "filter": f"prompt_tokens + output_tokens <= {args.max_model_len}",
    }
    public = {
        "schema_version": 1,
        "scenario_family": "balanced",
        "profiles": PROFILES,
        "source": source,
        "cases": [
            _case(
                name="public_code_native",
                rows=code_rows,
                start=args.public_start,
                count=args.requests_per_case,
                arrival_time_scale=1.0,
                slo=slo,
                expose_selector=True,
            ),
            _case(
                name="public_code_burst",
                rows=code_rows,
                start=args.public_start,
                count=args.requests_per_case,
                arrival_time_scale=0.35,
                slo=slo,
                expose_selector=True,
            ),
        ],
    }
    private = {
        "schema_version": 1,
        "scenario_family": "balanced",
        "profiles": PROFILES,
        "source": source,
        "cases": [
            _case(
                name="hidden_code_holdout",
                rows=code_rows,
                start=args.hidden_code_start,
                count=args.requests_per_case,
                arrival_time_scale=0.55,
                slo=slo,
                expose_selector=False,
            ),
            _case(
                name="hidden_conversation_native",
                rows=conversation_rows,
                start=args.hidden_conversation_start,
                count=args.requests_per_case,
                arrival_time_scale=1.0,
                slo=slo,
                expose_selector=False,
            ),
            _case(
                name="hidden_conversation_burst",
                rows=conversation_rows,
                start=args.hidden_conversation_start,
                count=args.requests_per_case,
                arrival_time_scale=0.4,
                slo=slo,
                expose_selector=False,
            ),
        ],
    }

    public_path = Path(args.public_output).resolve()
    private_path = (
        Path(args.private_bundle_root).resolve() / args.bundle_id / "workload.json"
    )
    commitment_path = Path(args.commitment_output).resolve()
    write_json(public_path, public)
    write_json(private_path, private)
    private_hash = _sha256(private_path)
    commitment = {
        "schema_version": 1,
        "scenario_family": "balanced",
        "profiles": PROFILES,
        "private_bundle": {
            "bundle_id": args.bundle_id,
            "workload_sha256": private_hash,
        },
    }
    write_json(commitment_path, commitment)
    print(
        json.dumps(
            {
                "public_output": str(public_path),
                "public_sha256": _sha256(public_path),
                "private_output": str(private_path),
                "private_sha256": private_hash,
                "commitment_output": str(commitment_path),
                "requests_per_case": args.requests_per_case,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
