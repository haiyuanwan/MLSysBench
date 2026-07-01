"""Command line interface for the SimAI benchmark evaluator."""

from __future__ import annotations

import argparse
import json

from mlsysbench.simai_bench.agent_runner import run_agent_once
from mlsysbench.simai_bench.evaluator import evaluate_and_write, evaluate_submission
from mlsysbench.simai_bench.model_client import make_model_client


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mlsysbench.simai_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one submission")
    evaluate_parser.add_argument("--task", required=True, help="Task directory")
    evaluate_parser.add_argument("--submission", required=True, help="Submission JSON/YAML file")
    evaluate_parser.add_argument("--output", help="Optional result JSON path")

    agent_parser = subparsers.add_parser(
        "run-agent",
        help="Call a model API to generate a submission, then evaluate it",
    )
    agent_parser.add_argument("--task", required=True, help="Task directory")
    agent_parser.add_argument("--output-dir", required=True, help="Directory for prompt/submission/result")
    agent_parser.add_argument(
        "--provider",
        choices=["dry-run", "openai-compatible"],
        default="dry-run",
        help="Model provider",
    )
    agent_parser.add_argument("--model", help="Model name for API providers")
    agent_parser.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. https://api.example.com/v1")
    agent_parser.add_argument("--api-key", help="API key literal. Prefer --api-key-env for shared runs.")
    agent_parser.add_argument("--api-key-env", help="Environment variable containing API key")
    agent_parser.add_argument("--timeout-seconds", type=int, default=120, help="Model API timeout")

    args = parser.parse_args()
    if args.command == "evaluate":
        if args.output:
            result = evaluate_and_write(args.task, args.submission, args.output)
        else:
            result = evaluate_submission(args.task, args.submission)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "run-agent":
        client = make_model_client(args)
        agent_result = run_agent_once(args.task, args.output_dir, client)
        print(json.dumps(agent_result.evaluation.to_dict(), indent=2, sort_keys=True))
