#!/usr/bin/env python3
"""Create deterministic replay windows from the audited Azure/Vidur traces."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDUR_TRACES = ROOT / "third_party/SimAI/vidur-alibabacloud/data/processed_traces"
TASK = ROOT / "tasks/simai_gym/azure2023_chunked_prefill_transfer"
WINDOWS = (
    (
        "development_interactive",
        VIDUR_TRACES / "splitwise_code.csv",
        TASK / "public/traces/code_window_2048_2175.csv",
        2048,
        128,
    ),
    (
        "final_interactive",
        VIDUR_TRACES / "splitwise_conv.csv",
        TASK / "hidden/traces/conversation_window_8192_8447.csv",
        8192,
        256,
    ),
    (
        "development_confirmation",
        VIDUR_TRACES / "splitwise_code.csv",
        TASK / "public/traces/code_window_2048_3071.csv",
        2048,
        1024,
    ),
    (
        "final_confirmation",
        VIDUR_TRACES / "splitwise_conv.csv",
        TASK / "hidden/traces/conversation_window_8192_10239.csv",
        8192,
        2048,
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    name: str,
    source: Path,
    destination: Path,
    start: int,
    count: int,
) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[start : start + count]
    if len(selected) != count:
        raise SystemExit(f"{name}: source has too few rows for [{start}:{start + count}]")
    origin = float(selected[0]["arrived_at"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["arrived_at", "num_prefill_tokens", "num_decode_tokens"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "arrived_at": f"{float(row['arrived_at']) - origin:.6f}",
                    "num_prefill_tokens": int(row["num_prefill_tokens"]),
                    "num_decode_tokens": int(row["num_decode_tokens"]),
                }
            )
    print(
        f"{name}: rows={count} source_slice=[{start}:{start + count}] "
        f"sha256={sha256(destination)} path={destination.relative_to(ROOT)}"
    )


def main() -> None:
    for args in WINDOWS:
        prepare(*args)


if __name__ == "__main__":
    main()
