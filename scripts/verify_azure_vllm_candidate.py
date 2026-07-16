#!/usr/bin/env python3
"""Verify the raw Azure 2023 traces and the checked-in Vidur transformation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "code": {
        "sha256": "54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6",
        "records": 8819,
        "processed": ROOT
        / "third_party/SimAI/vidur-alibabacloud/data/processed_traces/splitwise_code.csv",
    },
    "conversation": {
        "sha256": "2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8",
        "records": 19366,
        "processed": ROOT
        / "third_party/SimAI/vidur-alibabacloud/data/processed_traces/splitwise_conv.csv",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify(name: str, raw_path: Path) -> None:
    expected = EXPECTED[name]
    actual_hash = _sha256(raw_path)
    if actual_hash != expected["sha256"]:
        raise SystemExit(
            f"{name}: raw SHA-256 mismatch: {actual_hash} != {expected['sha256']}"
        )

    raw = _read(raw_path)
    processed_path = Path(expected["processed"])
    processed = _read(processed_path)
    if len(raw) != expected["records"] or len(processed) != len(raw):
        raise SystemExit(
            f"{name}: record mismatch: raw={len(raw)} processed={len(processed)} "
            f"expected={expected['records']}"
        )

    first_timestamp = datetime.fromisoformat(raw[0]["TIMESTAMP"])
    previous = first_timestamp
    for index, (raw_row, processed_row) in enumerate(zip(raw, processed)):
        timestamp = datetime.fromisoformat(raw_row["TIMESTAMP"])
        if timestamp < previous:
            raise SystemExit(f"{name}: timestamps are not sorted at row {index + 2}")
        previous = timestamp
        expected_arrival = (timestamp - first_timestamp).total_seconds()
        if abs(float(processed_row["arrived_at"]) - expected_arrival) > 1e-9:
            raise SystemExit(f"{name}: arrived_at mismatch at row {index + 2}")
        if int(processed_row["num_prefill_tokens"]) != int(raw_row["ContextTokens"]):
            raise SystemExit(f"{name}: prefill token mismatch at row {index + 2}")
        if int(processed_row["num_decode_tokens"]) != int(raw_row["GeneratedTokens"]):
            raise SystemExit(f"{name}: decode token mismatch at row {index + 2}")

    print(
        f"{name}: ok; raw_sha256={actual_hash}; records={len(raw)}; "
        f"processed_sha256={_sha256(processed_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-raw", type=Path, required=True)
    parser.add_argument("--conversation-raw", type=Path, required=True)
    args = parser.parse_args()
    verify("code", args.code_raw)
    verify("conversation", args.conversation_raw)


if __name__ == "__main__":
    main()
