"""Command line interface for the SimAI benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mlsysbench.simai_bench.agent_runner import run_agent_loop, run_agent_once
from mlsysbench.simai_bench.cli_agent import run_cli_agent
from mlsysbench.simai_bench.codex_ccswitch import (
    CodexCCSwitchSpec,
    install_runtime_assets,
    run_isolated_codex,
)
from mlsysbench.simai_bench.evaluator import evaluate_and_write, evaluate_submission
from mlsysbench.simai_bench.io import ConfigError, write_json
from mlsysbench.simai_bench.model_client import load_dotenv, make_model_client
from mlsysbench.simai_bench.search import run_search
from mlsysbench.simai_bench.task_validation import validate_task


def _add_model_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=["dry-run", "openai-compatible"],
        default="dry-run",
        help="Model provider",
    )
    parser.add_argument("--model", help="Model name for API providers")
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL, e.g. https://api.siliconflow.cn/v1",
    )
    parser.add_argument("--api-key", help="API key literal. Prefer environment variables.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Model API timeout")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request OpenAI-compatible JSON object output",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable provider reasoning mode",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Reasoning token budget (SiliconFlow supports 128-32768)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mlsysbench.simai_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one submission")
    evaluate_parser.add_argument("--task", required=True, help="Task directory")
    evaluate_parser.add_argument("--submission", required=True, help="Submission JSON/YAML file")
    evaluate_parser.add_argument("--output", help="Optional result JSON path")

    validate_parser = subparsers.add_parser(
        "validate-task",
        help="Check task invariants before publication",
    )
    validate_parser.add_argument("--task", required=True, help="Task directory")
    validate_parser.add_argument("--output", help="Optional validation report JSON path")
    validate_parser.add_argument(
        "--run-real-baseline",
        action="store_true",
        help="Execute development and final baseline replays for non-mock runners",
    )

    agent_parser = subparsers.add_parser(
        "run-agent",
        help="Call a model API to generate a submission, then evaluate it",
    )
    agent_parser.add_argument("--task", required=True, help="Task directory")
    agent_parser.add_argument("--output-dir", required=True, help="Directory for prompt/submission/result")
    _add_model_client_args(agent_parser)

    loop_parser = subparsers.add_parser(
        "run-agent-loop",
        help="Run a multi-step optimization trajectory with measured feedback",
    )
    loop_parser.add_argument("--task", required=True, help="Task directory")
    loop_parser.add_argument("--output-dir", required=True, help="Directory for trajectory artifacts")
    loop_parser.add_argument("--max-steps", type=int, help="Override the task step budget")
    _add_model_client_args(loop_parser)

    search_parser = subparsers.add_parser(
        "search",
        help="Run a matched-budget non-agent search baseline",
    )
    search_parser.add_argument("--task", required=True, help="Task directory")
    search_parser.add_argument("--output-dir", required=True, help="Directory for trajectory artifacts")
    search_parser.add_argument(
        "--method",
        choices=["grid", "random", "tpe", "smac"],
        required=True,
    )
    search_parser.add_argument("--budget", type=int, required=True)
    search_parser.add_argument("--seed", type=int, default=0)
    search_parser.add_argument(
        "--wall-time-seconds",
        type=float,
        help="Optional development-search wall-clock limit",
    )

    runtime_parser = subparsers.add_parser(
        "prepare-codex-runtime",
        help="Install pinned isolated Codex CLI and CC Switch assets",
    )
    runtime_parser.add_argument("--asset-dir", help="Override the runtime asset directory")
    runtime_parser.add_argument("--codex-source", help="Pinned Codex 0.144.3 binary to copy")
    runtime_parser.add_argument("--cc-switch-source", help="Verified CC Switch AppImage to copy")
    runtime_parser.add_argument(
        "--download-cc-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the pinned official CC Switch AppImage when absent",
    )

    isolated_codex_parser = subparsers.add_parser(
        "run-isolated-codex",
        help="Run or resume Codex through an isolated CC Switch sidecar",
    )
    isolated_codex_parser.add_argument("--workspace", default=".", help="Codex workspace")
    isolated_codex_parser.add_argument("--output-dir", required=True, help="New log directory")
    isolated_codex_parser.add_argument("--session-id", help="Host Codex session UUID to copy")
    isolated_codex_parser.add_argument("--model", help="SiliconFlow model")
    isolated_codex_parser.add_argument("--base-url", help="SiliconFlow API base URL")
    isolated_codex_parser.add_argument("--max-output-tokens", type=int)
    isolated_codex_parser.add_argument("--context-window", type=int)
    isolated_codex_parser.add_argument("--thinking-budget", type=int)
    isolated_codex_parser.add_argument("--asset-dir", help="Pinned runtime asset directory")
    isolated_codex_parser.add_argument("--host-codex-home", help="Source CODEX_HOME")
    isolated_codex_parser.add_argument("--timeout-seconds", type=int, default=3600)
    isolated_codex_parser.add_argument(
        "--prompt",
        default=(
            "Continue the session's current work in this isolated environment. "
            "Inspect the workspace, complete the active task, and verify your changes."
        ),
    )

    cli_agent_parser = subparsers.add_parser(
        "run-cli-agent",
        help="Run a filesystem-capable CLI agent with a budgeted development evaluator",
    )
    cli_agent_parser.add_argument("--task", required=True, help="Task directory")
    cli_agent_parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for the public workspace and private run artifacts",
    )
    cli_agent_parser.add_argument(
        "--agent-command",
        help=(
            "Agent command string. Optional placeholders: {prompt}, {prompt_file}, "
            "and {workspace}"
        ),
    )
    cli_agent_parser.add_argument(
        "--agent-profile",
        choices=["custom", "chat-completions", "longcat", "codex"],
        default="codex",
        help=(
            "Agent scaffold. Benchmark mode defaults to the isolated Codex CLI + "
            "CC Switch profile"
        ),
    )
    cli_agent_parser.add_argument(
        "--agent-mode",
        choices=["benchmark", "debug"],
        default="benchmark",
        help=(
            "Benchmark mode requires Codex CLI + CC Switch; debug mode permits "
            "custom and direct Chat Completions agents"
        ),
    )
    cli_agent_parser.add_argument("--model", help="Override MODEL_NAME for the chat agent")
    cli_agent_parser.add_argument("--base-url", help="Override MODEL_BASE_URL for the chat agent")
    cli_agent_parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override MODEL_MAX_TOKENS for each chat completion",
    )
    cli_agent_parser.add_argument(
        "--context-window",
        type=int,
        help="Record and communicate the model context window",
    )
    cli_agent_parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        help="Timeout for each model API call",
    )
    cli_agent_parser.add_argument(
        "--thinking-budget",
        type=int,
        help="SiliconFlow reasoning token budget (default: 32768)",
    )
    cli_agent_parser.add_argument(
        "--codex-session-id",
        help="Copy and resume one host Codex session inside the isolated runtime",
    )
    cli_agent_parser.add_argument(
        "--codex-asset-dir",
        help="Directory containing the pinned Codex and CC Switch assets",
    )
    cli_agent_parser.add_argument(
        "--wall-time-seconds",
        type=int,
        default=3600,
        help="Total wall-time budget for the agent and development queries",
    )
    cli_agent_parser.add_argument(
        "--max-queries",
        type=int,
        help="Development query budget; defaults to task constraints.max_steps",
    )
    cli_agent_parser.add_argument(
        "--isolation",
        choices=["landlock", "bwrap", "none"],
        default="bwrap",
        help="Agent isolation backend; benchmark mode requires bwrap",
    )
    cli_agent_parser.add_argument(
        "--agent-read-path",
        action="append",
        default=[],
        help="Extra read-only path visible to a Landlock-isolated custom agent",
    )

    args = parser.parse_args()
    if args.command == "evaluate":
        if args.output:
            result = evaluate_and_write(args.task, args.submission, args.output)
        else:
            result = evaluate_submission(args.task, args.submission)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "validate-task":
        validation = validate_task(
            args.task,
            run_real_baseline=args.run_real_baseline,
        )
        payload = validation.to_dict()
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not validation.valid:
            raise SystemExit(1)
    elif args.command == "run-agent":
        client = make_model_client(args)
        agent_result = run_agent_once(args.task, args.output_dir, client)
        print(json.dumps(agent_result.evaluation.to_dict(), indent=2, sort_keys=True))
    elif args.command == "run-agent-loop":
        client = make_model_client(args)
        loop_result = run_agent_loop(
            args.task,
            args.output_dir,
            client,
            max_steps=args.max_steps,
        )
        print(json.dumps(loop_result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "search":
        search_result = run_search(
            args.task,
            args.output_dir,
            method=args.method,
            budget=args.budget,
            seed=args.seed,
            wall_time_seconds=args.wall_time_seconds,
        )
        print(json.dumps(search_result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "prepare-codex-runtime":
        manifest = install_runtime_assets(
            asset_dir=args.asset_dir,
            codex_source=args.codex_source,
            cc_switch_source=args.cc_switch_source,
            download_cc_switch=args.download_cc_switch,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "run-isolated-codex":
        load_dotenv()
        spec = CodexCCSwitchSpec(
            model=args.model or os.environ.get("MODEL_NAME", "meituan-longcat/LongCat-2.0"),
            base_url=args.base_url
            or os.environ.get("MODEL_BASE_URL", "https://api.siliconflow.cn/v1"),
            max_output_tokens=args.max_output_tokens
            if args.max_output_tokens is not None
            else int(os.environ.get("MODEL_MAX_TOKENS", "131072")),
            context_window=args.context_window
            if args.context_window is not None
            else int(os.environ.get("MODEL_CONTEXT_WINDOW", "1048576")),
            thinking_budget=args.thinking_budget
            if args.thinking_budget is not None
            else int(os.environ.get("MODEL_THINKING_BUDGET", "32768")),
            session_id=args.session_id,
            asset_dir=args.asset_dir,
            host_codex_home=args.host_codex_home,
            prompt=args.prompt,
            last_message_path=Path(args.output_dir) / "last_message.txt",
        )
        result = run_isolated_codex(
            workspace=args.workspace,
            output_dir=args.output_dir,
            spec=spec,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "completed":
            raise SystemExit(1)
    elif args.command == "run-cli-agent":
        load_dotenv()
        if args.agent_mode == "benchmark" and args.agent_profile != "codex":
            raise ConfigError(
                "--agent-mode benchmark requires --agent-profile codex; "
                "use --agent-mode debug for other profiles"
            )
        agent_read_paths = list(args.agent_read_path)
        agent_environment = {}
        if args.model:
            agent_environment["MODEL_NAME"] = args.model
        if args.base_url:
            agent_environment["MODEL_BASE_URL"] = args.base_url
        if args.max_output_tokens is not None:
            agent_environment["MODEL_MAX_TOKENS"] = str(args.max_output_tokens)
        if args.context_window is not None:
            agent_environment["MODEL_CONTEXT_WINDOW"] = str(args.context_window)
        if args.model_timeout_seconds is not None:
            agent_environment["MODEL_TIMEOUT_SECONDS"] = str(args.model_timeout_seconds)
        if args.thinking_budget is not None:
            agent_environment["MODEL_THINKING_BUDGET"] = str(args.thinking_budget)
        agent_runtime = None
        if args.agent_profile in {"longcat", "chat-completions"}:
            if args.agent_command:
                raise ConfigError(
                    "--agent-command cannot be combined with a chat agent profile"
                )
            chat_agent_path = Path(__file__).with_name("chat_cli_agent.py").resolve()
            agent_command = [sys.executable, str(chat_agent_path)]
            agent_read_paths.append(str(chat_agent_path))
        elif args.agent_profile == "codex":
            if args.agent_command:
                raise ConfigError("--agent-command cannot be combined with the codex profile")
            if args.isolation == "none":
                raise ConfigError("The codex profile requires Landlock or bwrap isolation")
            model = args.model or os.environ.get("MODEL_NAME", "meituan-longcat/LongCat-2.0")
            base_url = args.base_url or os.environ.get(
                "MODEL_BASE_URL", "https://api.siliconflow.cn/v1"
            )
            max_output_tokens = (
                args.max_output_tokens
                if args.max_output_tokens is not None
                else int(os.environ.get("MODEL_MAX_TOKENS", "131072"))
            )
            context_window = (
                args.context_window
                if args.context_window is not None
                else int(os.environ.get("MODEL_CONTEXT_WINDOW", "1048576"))
            )
            thinking_budget = (
                args.thinking_budget
                if args.thinking_budget is not None
                else int(os.environ.get("MODEL_THINKING_BUDGET", "32768"))
            )
            agent_runtime = CodexCCSwitchSpec(
                model=model,
                base_url=base_url,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
                thinking_budget=thinking_budget,
                session_id=args.codex_session_id,
                asset_dir=args.codex_asset_dir,
            )
            agent_command = None
        else:
            if not args.agent_command:
                raise ConfigError("--agent-command is required for --agent-profile custom")
            agent_command = args.agent_command
        cli_agent_result = run_cli_agent(
            args.task,
            args.output_dir,
            agent_command,
            wall_time_seconds=args.wall_time_seconds,
            max_queries=args.max_queries,
            isolation=args.isolation,
            agent_read_paths=agent_read_paths,
            agent_environment=agent_environment,
            agent_runtime=agent_runtime,
            agent_scaffold=(
                "codex-cli+cc-switch"
                if args.agent_profile == "codex"
                else args.agent_profile
            ),
            benchmark_mode=args.agent_mode == "benchmark",
        )
        print(json.dumps(cli_agent_result.to_dict(), indent=2, sort_keys=True))
