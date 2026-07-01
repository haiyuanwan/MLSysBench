"""Command line interface for the SimAI benchmark evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlsysbench.simai_bench.evaluator import evaluate_and_write, evaluate_submission


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mlsysbench.simai_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one submission")
    evaluate_parser.add_argument("--task", required=True, help="Task directory")
    evaluate_parser.add_argument("--submission", required=True, help="Submission JSON/YAML file")
    evaluate_parser.add_argument("--output", help="Optional result JSON path")

    args = parser.parse_args()
    if args.command == "evaluate":
        if args.output:
            result = evaluate_and_write(args.task, args.submission, args.output)
        else:
            result = evaluate_submission(args.task, args.submission)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

