"""Minimal tool-using Chat Completions agent for a public benchmark workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "meituan-longcat/LongCat-2.0"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 131_072
DEFAULT_CONTEXT_WINDOW = 1_048_576
MAX_TOOL_OUTPUT_CHARS = 40_000
WORKSPACE = Path.cwd().resolve()


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the public workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 200000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the public workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories below a public workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one shell command in the public workspace. Use this to call "
                "evaluate_dev.py and inspect its measured response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["command"],
            },
        },
    },
]


def main() -> int:
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        print("MODEL_API_KEY is required", file=sys.stderr)
        return 2
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    base_url = os.environ.get("MODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    max_tokens = _environment_int("MODEL_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    context_window = _environment_int("MODEL_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)
    timeout_seconds = _environment_int("MODEL_TIMEOUT_SECONDS", 600)
    max_turns = _environment_int("MODEL_AGENT_MAX_TURNS", 64)
    enable_thinking = _environment_bool("MODEL_ENABLE_THINKING", True)
    thinking_budget = _environment_int("MODEL_THINKING_BUDGET", 32_768)
    if not 128 <= thinking_budget <= 32_768:
        raise RuntimeError("MODEL_THINKING_BUDGET must be between 128 and 32768")

    prompt_path = Path(os.environ.get("MLSYSBENCH_PROMPT_FILE", "MISSION.md"))
    prompt = prompt_path.read_text(encoding="utf-8")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an autonomous inference-systems engineer operating in a restricted "
                "public workspace. Use the supplied tools to inspect files, run controlled "
                "development evaluations, compare measured results, and write "
                "final_submission.json before finishing. Never invent evaluator results."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The configured model context window is {context_window} tokens.\n\n{prompt}"
            ),
        },
    ]
    calls: list[dict[str, Any]] = []
    nudges = 0

    for turn in range(1, max_turns + 1):
        started = time.perf_counter()
        response = _chat_completion(
            url=base_url + "/chat/completions",
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        latency = round(time.perf_counter() - started, 6)
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            raise RuntimeError("Model response did not contain an assistant message")
        tool_calls = message.get("tool_calls") or []
        call_record = {
            "turn": turn,
            "request_id": response.get("id"),
            "response_model": response.get("model", model),
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "latency_seconds": latency,
            "usage": response.get("usage"),
            "tool_names": [
                item.get("function", {}).get("name")
                for item in tool_calls
                if isinstance(item, dict)
            ],
        }
        calls.append(call_record)
        _write_json(WORKSPACE / "chat_agent_stats.json", _summarize_calls(model, calls))

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content") or "",
        }
        if isinstance(message.get("reasoning_content"), str):
            assistant_message["reasoning_content"] = message["reasoning_content"]
        if isinstance(tool_calls, list) and tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if tool_calls:
            for tool_call in tool_calls:
                tool_result = _execute_tool_call(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", "")),
                        "content": json.dumps(tool_result, sort_keys=True),
                    }
                )
            _write_json(WORKSPACE / "chat_agent_transcript.json", messages)
            continue

        if (WORKSPACE / "final_submission.json").is_file():
            _write_json(WORKSPACE / "chat_agent_transcript.json", messages)
            print(f"completed after {turn} model turns", flush=True)
            return 0
        if nudges >= 2:
            break
        nudges += 1
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have not created final_submission.json. Continue using tools and "
                    "finish the measured benchmark task before replying without tool calls."
                ),
            }
        )

    _write_json(WORKSPACE / "chat_agent_transcript.json", messages)
    print("model stopped without creating final_submission.json", file=sys.stderr)
    return 2


def _chat_completion(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: int,
    enable_thinking: bool = True,
    thinking_budget: int = 32_768,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_thinking:
        payload["enable_thinking"] = True
        payload["thinking_budget"] = thinking_budget
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Model API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Model API request failed: {error.reason}") from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Model API response must be a JSON object")
    return parsed


def _execute_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {"ok": False, "error": "invalid tool call"}
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return {"ok": False, "error": "tool call has no function"}
    name = function.get("name")
    arguments_value = function.get("arguments", "{}")
    try:
        arguments = (
            json.loads(arguments_value) if isinstance(arguments_value, str) else arguments_value
        )
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if name == "read_file":
            return _read_file(arguments)
        if name == "write_file":
            return _write_file(arguments)
        if name == "list_files":
            return _list_files(arguments)
        if name == "run_command":
            return _run_command(arguments)
        return {"ok": False, "error": f"unknown tool {name!r}"}
    except Exception as error:  # noqa: BLE001 - return tool failures to the model.
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def _read_file(arguments: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_path(str(arguments["path"]))
    max_chars = max(1, min(int(arguments.get("max_chars", 100_000)), 200_000))
    content = path.read_text(encoding="utf-8")
    truncated = len(content) > max_chars
    return {
        "ok": True,
        "path": str(path.relative_to(WORKSPACE)),
        "content": content[:max_chars],
        "truncated": truncated,
    }


def _write_file(arguments: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_path(str(arguments["path"]))
    content = arguments["content"]
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "path": str(path.relative_to(WORKSPACE)),
        "bytes_written": len(content.encode("utf-8")),
    }


def _list_files(arguments: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_path(str(arguments.get("path", ".")))
    max_entries = max(1, min(int(arguments.get("max_entries", 500)), 1000))
    entries: list[str] = []
    if root.is_file():
        entries.append(str(root.relative_to(WORKSPACE)))
    else:
        for path in sorted(root.rglob("*")):
            entries.append(str(path.relative_to(WORKSPACE)) + ("/" if path.is_dir() else ""))
            if len(entries) >= max_entries:
                break
    return {"ok": True, "entries": entries, "truncated": len(entries) >= max_entries}


def _run_command(arguments: dict[str, Any]) -> dict[str, Any]:
    command = arguments["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    timeout_seconds = max(1, min(int(arguments.get("timeout_seconds", 120)), 300))
    environment = os.environ.copy()
    for key in list(environment):
        upper = key.upper()
        if any(marker in upper for marker in ("API_KEY", "PASSWORD", "SECRET")):
            environment.pop(key, None)
    try:
        completed = subprocess.run(
            command,
            cwd=WORKSPACE,
            env=environment,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-MAX_TOOL_OUTPUT_CHARS:],
            "stderr": completed.stderr[-MAX_TOOL_OUTPUT_CHARS:],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "error": f"command timed out after {timeout_seconds} seconds",
            "stdout": _tail_text(error.stdout),
            "stderr": _tail_text(error.stderr),
        }


def _workspace_path(value: str) -> Path:
    candidate = (WORKSPACE / value).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError("path escapes the public workspace")
    return candidate


def _summarize_calls(model: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, int | float] = {}
    for call in calls:
        call_usage = call.get("usage")
        if not isinstance(call_usage, dict):
            continue
        for key, value in call_usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return {
        "model": model,
        "api_calls": len(calls),
        "latency_seconds": round(sum(call["latency_seconds"] for call in calls), 6),
        "usage": usage,
        "calls": calls,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def _tail_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-MAX_TOOL_OUTPUT_CHARS:]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - preserve a concise CLI failure in stderr.
        print(f"chat CLI agent failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
