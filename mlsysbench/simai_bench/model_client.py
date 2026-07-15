"""Model API clients for benchmark agents."""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlsysbench.simai_bench.io import ConfigError


MODEL_METADATA_KEY = "_model_metadata"
DEFAULT_MAX_OUTPUT_TOKENS = 131_072


class ModelClient(Protocol):
    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DryRunClient:
    """Deterministic client for testing the agent path without network access."""

    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        allowed = context.get("allowed_actions", {})
        changes: dict[str, Any] = {}
        history = context.get("experiment_history")
        if "replica_scheduler_config_type" in allowed:
            choices = allowed["replica_scheduler_config_type"].get("choices") or []
            if "sarathi" in choices:
                changes["replica_scheduler_config_type"] = "sarathi"
        if "cluster_config_num_replicas" in allowed:
            choices = allowed["cluster_config_num_replicas"].get("choices") or []
            if 4 in choices:
                changes["cluster_config_num_replicas"] = 4
        if "replica_config_tensor_parallel_size" in allowed:
            choices = allowed["replica_config_tensor_parallel_size"].get("choices") or []
            if 2 in choices:
                changes["replica_config_tensor_parallel_size"] = 2
        if "sarathi_scheduler_config_chunk_size" in allowed and history != []:
            choices = allowed["sarathi_scheduler_config_chunk_size"].get("choices") or []
            changes["sarathi_scheduler_config_chunk_size"] = 512 if 512 in choices else choices[0]
        if "vllm_scheduler_config_max_tokens_in_batch" in allowed and history != []:
            baseline = context.get("baseline_config", {})
            changes["vllm_scheduler_config_max_tokens_in_batch"] = baseline.get(
                "vllm_scheduler_config_max_tokens_in_batch",
                2048,
            )
        result = {
            "changes": changes,
            "notes": "Dry-run client generated a deterministic valid submission.",
        }
        if isinstance(history, list) and len(history) >= 2:
            result["stop"] = True
            replica_choices = allowed.get("cluster_config_num_replicas", {}).get("choices") or []
            tp_choices = allowed.get("replica_config_tensor_parallel_size", {}).get("choices") or []
            chunk_choices = allowed.get("sarathi_scheduler_config_chunk_size", {}).get("choices") or []
            if 4 in tp_choices and 1024 in chunk_choices:
                final_changes = {
                    "replica_config_tensor_parallel_size": 4,
                    "replica_scheduler_config_type": "sarathi",
                    "sarathi_scheduler_config_chunk_size": 1024,
                }
                if 8 in replica_choices:
                    final_changes["cluster_config_num_replicas"] = 8
                result["final_changes"] = final_changes
        return result


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client using stdlib HTTP."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    json_mode: bool = False
    enable_thinking: bool = False
    thinking_budget: int = 32_768

    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(context)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an inference-system optimization agent. "
                        "Return only valid JSON with keys changes, notes, optional stop, "
                        "and optional final_changes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.enable_thinking:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.thinking_budget
        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise ConfigError(
                f"Model API request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ConfigError(
                f"Model API request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConfigError(f"Model API request failed: {exc}") from exc

        latency_seconds = time.perf_counter() - started_at
        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ConfigError("Model API returned invalid JSON") from exc

        try:
            choice = response_payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ConfigError("Model API response did not contain a chat completion") from exc
        if not isinstance(content, str) or not content.strip():
            message = choice.get("message") if isinstance(choice, dict) else None
            has_reasoning = isinstance(message, dict) and bool(message.get("reasoning_content"))
            raise ConfigError(
                "Model API response contained no text content "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"reasoning_content={has_reasoning}, usage={response_payload.get('usage')!r})"
            )

        submission = extract_submission_json(content)
        metadata: dict[str, Any] = {
            "provider": "openai-compatible",
            "requested_model": self.model,
            "response_model": response_payload.get("model", self.model),
            "request_id": response_payload.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "latency_seconds": round(latency_seconds, 6),
        }
        usage = response_payload.get("usage")
        if isinstance(usage, dict):
            metadata["usage"] = {
                str(key): value
                for key, value in usage.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        submission[MODEL_METADATA_KEY] = metadata
        return submission


def make_model_client(args: Any) -> ModelClient:
    load_dotenv()
    if args.provider == "dry-run":
        return DryRunClient()
    if args.provider == "openai-compatible":
        api_key = getattr(args, "api_key", None)
        api_key_env = getattr(args, "api_key_env", None)
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ConfigError(f"Environment variable {api_key_env} is not set")
        elif not api_key:
            api_key = os.environ.get("MODEL_API_KEY")
        model = getattr(args, "model", None) or os.environ.get("MODEL_NAME")
        base_url = getattr(args, "base_url", None) or os.environ.get("MODEL_BASE_URL")
        if not api_key:
            raise ConfigError(
                "Missing API key. Use --api-key, --api-key-env, or MODEL_API_KEY."
            )
        if not base_url:
            raise ConfigError("--base-url or MODEL_BASE_URL is required for openai-compatible provider")
        if not model:
            raise ConfigError("--model or MODEL_NAME is required for openai-compatible provider")

        max_tokens = getattr(args, "max_output_tokens", None)
        if max_tokens is None:
            max_tokens = _env_int("MODEL_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        temperature = getattr(args, "temperature", None)
        if temperature is None:
            temperature = _env_float("MODEL_TEMPERATURE", 0.0)
        json_mode = getattr(args, "json_mode", None)
        if json_mode is None:
            json_mode = _env_bool("MODEL_JSON_MODE", False)
        enable_thinking = getattr(args, "enable_thinking", None)
        if enable_thinking is None:
            enable_thinking = _env_bool("MODEL_ENABLE_THINKING", False)
        thinking_budget = getattr(args, "thinking_budget", None)
        if thinking_budget is None:
            thinking_budget = _env_int("MODEL_THINKING_BUDGET", 32_768)
        timeout_seconds = getattr(args, "timeout_seconds", 120)

        if max_tokens <= 0:
            raise ConfigError("max output tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ConfigError("temperature must be between 0 and 2")
        if timeout_seconds <= 0:
            raise ConfigError("timeout seconds must be positive")
        if not 128 <= thinking_budget <= 32_768:
            raise ConfigError("thinking budget must be between 128 and 32768")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
    raise ConfigError(f"Unsupported provider {args.provider}")


def load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from .env without overriding the shell."""
    for env_path in _candidate_env_paths():
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def _candidate_env_paths() -> list[Path]:
    cwd = Path.cwd()
    package_root = Path(__file__).resolve().parents[2]
    return [cwd / ".env", package_root / ".env"]


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - retain the original HTTP error if reading fails.
        body = ""
    if not body:
        return str(exc.reason)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        detail = body
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or error)
        else:
            detail = str(error or payload)
    return " ".join(detail.split())[:500]


def build_prompt(context: dict[str, Any]) -> str:
    return (
        "You must optimize this SimAI/Vidur inference-serving task.\n"
        "Only use keys listed in allowed_actions. Do not invent fields.\n"
        "Do not use forbidden EP/topology/model-path actions.\n"
        "Return only JSON in this exact shape:\n"
        '{"changes": {"field": "value"}, "notes": "short rationale", '
        '"stop": false, "final_changes": null}\n\n'
        f"Task context:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )


def extract_submission_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
    else:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", content, re.DOTALL)
        if not match:
            raise ConfigError("Model response did not contain JSON")
        data = json.loads(match.group(1))
    if not isinstance(data, dict) or not isinstance(data.get("changes"), dict):
        raise ConfigError("Model submission JSON must include a changes object")
    if "stop" in data and not isinstance(data["stop"], bool):
        raise ConfigError("Model submission stop must be a boolean")
    if data.get("final_changes") is not None and not isinstance(data["final_changes"], dict):
        raise ConfigError("Model submission final_changes must be an object or null")
    data.setdefault("notes", "")
    return data
