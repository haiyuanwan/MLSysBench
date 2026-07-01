"""Model API clients for benchmark agents."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlsysbench.simai_bench.io import ConfigError


class ModelClient(Protocol):
    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DryRunClient:
    """Deterministic client for testing the agent path without network access."""

    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        allowed = context.get("allowed_actions", {})
        changes: dict[str, Any] = {}
        if "replica_scheduler_config_type" in allowed:
            choices = allowed["replica_scheduler_config_type"].get("choices") or []
            if "sarathi" in choices:
                changes["replica_scheduler_config_type"] = "sarathi"
        if "sarathi_scheduler_config_chunk_size" in allowed:
            choices = allowed["sarathi_scheduler_config_chunk_size"].get("choices") or []
            changes["sarathi_scheduler_config_chunk_size"] = 512 if 512 in choices else choices[0]
        if "vllm_scheduler_config_max_tokens_in_batch" in allowed:
            baseline = context.get("baseline_config", {})
            changes["vllm_scheduler_config_max_tokens_in_batch"] = baseline.get(
                "vllm_scheduler_config_max_tokens_in_batch",
                2048,
            )
        return {
            "changes": changes,
            "notes": "Dry-run client generated a deterministic valid submission.",
        }


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client using stdlib HTTP."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120

    def generate_submission(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(context)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an inference-system optimization agent. "
                        "Return only valid JSON with keys changes and notes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConfigError(f"Model API request failed: {exc}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        return extract_submission_json(content)


def make_model_client(args: Any) -> ModelClient:
    load_dotenv()
    if args.provider == "dry-run":
        return DryRunClient()
    if args.provider == "openai-compatible":
        api_key = args.api_key
        if args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
        model = args.model or os.environ.get("MODEL_NAME")
        base_url = args.base_url or os.environ.get("MODEL_BASE_URL")
        if not api_key:
            raise ConfigError("Missing API key. Use --api-key or --api-key-env.")
        if not base_url:
            raise ConfigError("--base-url or MODEL_BASE_URL is required for openai-compatible provider")
        if not model:
            raise ConfigError("--model or MODEL_NAME is required for openai-compatible provider")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=args.timeout_seconds,
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


def build_prompt(context: dict[str, Any]) -> str:
    return (
        "You must optimize this SimAI/Vidur inference-serving task.\n"
        "Only use keys listed in allowed_actions. Do not invent fields.\n"
        "Do not use forbidden EP/topology/model-path actions.\n"
        "Return only JSON in this exact shape:\n"
        '{"changes": {"field": "value"}, "notes": "short rationale"}\n\n'
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
    data.setdefault("notes", "")
    return data
