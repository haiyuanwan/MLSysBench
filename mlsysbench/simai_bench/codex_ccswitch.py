"""Isolated Codex CLI runtime backed by the official CC Switch proxy."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mlsysbench.simai_bench.io import ConfigError
from mlsysbench.simai_bench.landlock import (
    default_system_read_paths,
    landlock_abi_version,
    restrict_current_process,
)


CODEX_VERSION = "0.144.3"
CODEX_SHA256 = "37e6f5953f191b04f7b62cb07dae90f51d0947ad89f0355665b421fbde28700b"
CC_SWITCH_VERSION = "3.17.0"
CC_SWITCH_SHA256 = "f4ee9d1dcc0015de583c9f5e3b3436c7a351f7e781dbb746fbba96525b129eb9"
CC_SWITCH_FILENAME = f"CC-Switch-v{CC_SWITCH_VERSION}-Linux-x86_64.AppImage"
CC_SWITCH_URL = (
    "https://github.com/farion1231/cc-switch/releases/download/"
    f"v{CC_SWITCH_VERSION}/{CC_SWITCH_FILENAME}"
)
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "meituan-longcat/LongCat-2.0"
DEFAULT_CONTEXT_WINDOW = 1_048_576
DEFAULT_MAX_OUTPUT_TOKENS = 131_072
DEFAULT_THINKING_BUDGET = 32_768
SUPPORTED_MODELS = (
    "meituan-longcat/LongCat-2.0",
    "zai-org/GLM-5.2",
    "moonshotai/Kimi-K2.7-Code",
    "deepseek-ai/DeepSeek-V4-Pro",
)

_LOCAL_BASE_URL_RE = re.compile(
    r'^\s*base_url\s*=\s*"http://127\.0\.0\.1:(\d+)(?:/v1)?"\s*$',
    re.MULTILINE,
)
_LOCAL_PLACEHOLDER_RE = re.compile(
    r'^\s*experimental_bearer_token\s*=\s*"PROXY_MANAGED"\s*$',
    re.MULTILINE,
)
_API_CREDENTIAL_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_SECRET_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def default_asset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "_runtime"


def codex_asset_path(asset_dir: str | Path | None = None) -> Path:
    root = Path(asset_dir) if asset_dir is not None else default_asset_dir()
    return root / f"codex-{CODEX_VERSION}" / "codex"


def cc_switch_asset_path(asset_dir: str | Path | None = None) -> Path:
    root = Path(asset_dir) if asset_dir is not None else default_asset_dir()
    return root / CC_SWITCH_FILENAME


def install_runtime_assets(
    *,
    asset_dir: str | Path | None = None,
    codex_source: str | Path | None = None,
    cc_switch_source: str | Path | None = None,
    download_cc_switch: bool = True,
) -> dict[str, Any]:
    """Install pinned official binaries into the ignored project runtime directory."""

    root = Path(asset_dir).resolve() if asset_dir is not None else default_asset_dir().resolve()
    root.mkdir(parents=True, exist_ok=True)
    codex_target = codex_asset_path(root)
    cc_switch_target = cc_switch_asset_path(root)

    if codex_target.exists():
        _verify_sha256(codex_target, CODEX_SHA256, "Codex CLI")
    else:
        source = _resolve_codex_source(codex_source)
        _verify_sha256(source, CODEX_SHA256, "Codex CLI source")
        _copy_verified(source, codex_target, CODEX_SHA256)
    codex_target.chmod(0o755)

    if cc_switch_target.exists():
        _verify_sha256(cc_switch_target, CC_SWITCH_SHA256, "CC Switch")
    elif cc_switch_source is not None:
        source = Path(cc_switch_source).expanduser().resolve()
        _verify_sha256(source, CC_SWITCH_SHA256, "CC Switch source")
        _copy_verified(source, cc_switch_target, CC_SWITCH_SHA256)
    elif download_cc_switch:
        _download_verified(CC_SWITCH_URL, cc_switch_target, CC_SWITCH_SHA256)
    else:
        raise ConfigError(
            f"CC Switch asset is missing: {cc_switch_target}. "
            "Run prepare-codex-runtime with download enabled."
        )
    cc_switch_target.chmod(0o755)

    manifest = {
        "codex": {
            "version": CODEX_VERSION,
            "path": str(codex_target),
            "sha256": CODEX_SHA256,
        },
        "cc_switch": {
            "version": CC_SWITCH_VERSION,
            "path": str(cc_switch_target),
            "sha256": CC_SWITCH_SHA256,
            "source_url": CC_SWITCH_URL,
        },
    }
    _write_json_private(root / "runtime-assets.json", manifest, mode=0o644)
    return manifest


@dataclass(frozen=True)
class CodexCCSwitchSpec:
    """Configuration for one isolated Codex + CC Switch run."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_window: int = DEFAULT_CONTEXT_WINDOW
    thinking_budget: int = DEFAULT_THINKING_BUDGET
    api_key_env: str = "MODEL_API_KEY"
    session_id: str | None = None
    asset_dir: str | Path | None = None
    host_codex_home: str | Path | None = None
    startup_timeout_seconds: int = 45
    prompt: str | None = None
    last_message_path: str | Path | None = None

    def prepare(
        self,
        *,
        output_dir: Path,
        workspace: Path,
        mission_path: Path,
    ) -> "PreparedCodexCCSwitchRuntime":
        if self.model not in SUPPORTED_MODELS:
            raise ConfigError(
                f"Unsupported isolated Codex model {self.model!r}; expected one of "
                + ", ".join(SUPPORTED_MODELS)
            )
        for name, value in (
            ("max_output_tokens", self.max_output_tokens),
            ("context_window", self.context_window),
            ("thinking_budget", self.thinking_budget),
            ("startup_timeout_seconds", self.startup_timeout_seconds),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be positive")
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ConfigError(f"Environment variable {self.api_key_env} is not set")

        codex_binary = codex_asset_path(self.asset_dir).resolve()
        cc_switch_binary = cc_switch_asset_path(self.asset_dir).resolve()
        _verify_sha256(codex_binary, CODEX_SHA256, "isolated Codex CLI")
        _verify_sha256(cc_switch_binary, CC_SWITCH_SHA256, "isolated CC Switch")

        host_codex_home = (
            Path(self.host_codex_home).expanduser().resolve()
            if self.host_codex_home is not None
            else (Path.home() / ".codex").resolve()
        )
        runtime_root = output_dir / "runtime"
        proxy_home = runtime_root / "proxy-home"
        agent_state = runtime_root / "agent-state"
        codex_home = agent_state / "codex-home"
        agent_home = agent_state / "home"
        for directory in (
            runtime_root,
            proxy_home,
            agent_state,
            codex_home,
            agent_home,
            agent_state / "xdg-config",
            agent_state / "xdg-data",
            agent_state / "xdg-cache",
            agent_state / "xdg-state",
            agent_state / "xdg-runtime",
            agent_state / "tmp",
            proxy_home / ".cc-switch",
            runtime_root / "xdg-config",
            runtime_root / "xdg-data",
            runtime_root / "xdg-cache",
            runtime_root / "xdg-state",
            runtime_root / "xdg-runtime",
            runtime_root / "tmp",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)

        proxy_codex_home = proxy_home / ".codex"
        if proxy_codex_home.exists() or proxy_codex_home.is_symlink():
            raise ConfigError(f"Unexpected pre-existing proxy Codex path: {proxy_codex_home}")
        proxy_codex_home.symlink_to(codex_home, target_is_directory=True)

        provider_config = build_siliconflow_provider_config(
            api_key=api_key,
            model=self.model,
            base_url=self.base_url,
            max_output_tokens=self.max_output_tokens,
            context_window=self.context_window,
            thinking_budget=self.thinking_budget,
        )

        protected_paths = [host_codex_home / "config.toml", host_codex_home / "auth.json"]
        session_metadata = None
        if self.session_id is not None:
            session_metadata = snapshot_codex_session(
                host_codex_home=host_codex_home,
                isolated_codex_home=codex_home,
                session_id=self.session_id,
                workspace=workspace,
                redact_values=(api_key,),
            )
            protected_paths.append(Path(session_metadata["source_path"]))
        host_before = fingerprint_paths(protected_paths)

        prompt = self.prompt or (
            "Continue in this isolated environment. Read MISSION.md and task_context.json, "
            "use evaluate_dev.py for measured experiments, and write final_submission.json "
            "before finishing. Do not inspect files outside the current workspace."
        )
        output_message = (
            Path(self.last_message_path).expanduser().resolve()
            if self.last_message_path is not None
            else workspace / "codex_final_message.txt"
        )
        output_message.parent.mkdir(parents=True, exist_ok=True)
        if self.session_id is None:
            agent_command = [
                str(codex_binary),
                "exec",
                "--color",
                "never",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "-C",
                str(workspace),
                "-m",
                self.model,
                "-o",
                str(output_message),
                prompt,
            ]
        else:
            agent_command = [
                str(codex_binary),
                "exec",
                "--color",
                "never",
                "resume",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "-m",
                self.model,
                "-o",
                str(output_message),
                self.session_id,
                prompt,
            ]

        environment = {
            "HOME": str(agent_home),
            "CODEX_HOME": str(codex_home),
            "CODEX_SQLITE_HOME": str(codex_home),
            "XDG_CONFIG_HOME": str(agent_state / "xdg-config"),
            "XDG_DATA_HOME": str(agent_state / "xdg-data"),
            "XDG_CACHE_HOME": str(agent_state / "xdg-cache"),
            "XDG_STATE_HOME": str(agent_state / "xdg-state"),
            "XDG_RUNTIME_DIR": str(agent_state / "xdg-runtime"),
            "TMPDIR": str(agent_state / "tmp"),
            "MODEL_NAME": self.model,
            "MODEL_BASE_URL": "http://127.0.0.1",
            "MODEL_MAX_TOKENS": str(self.max_output_tokens),
            "MODEL_CONTEXT_WINDOW": str(self.context_window),
            "MODEL_ENABLE_THINKING": "true",
            "MODEL_THINKING_BUDGET": str(self.thinking_budget),
        }
        remove_environment = tuple(
            key
            for key in os.environ
            if any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
        )
        metadata = {
            "runtime": "official-codex-with-cc-switch",
            "codex_version": CODEX_VERSION,
            "codex_sha256": CODEX_SHA256,
            "cc_switch_version": CC_SWITCH_VERSION,
            "cc_switch_sha256": CC_SWITCH_SHA256,
            "model": self.model,
            "upstream_base_url": self.base_url,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "thinking": {"enabled": True, "budget": self.thinking_budget},
            "sidecar_isolation": {
                "filesystem": "landlock",
                "landlock_abi": landlock_abi_version(),
            },
            "session": session_metadata,
            "host_codex_guard": {"before": host_before, "unchanged": None},
        }
        _write_json_private(runtime_root / "runtime_manifest.json", metadata)

        return PreparedCodexCCSwitchRuntime(
            runtime_root=runtime_root,
            proxy_home=proxy_home,
            codex_home=codex_home,
            agent_state=agent_state,
            workspace=workspace,
            cc_switch_binary=cc_switch_binary,
            agent_command=agent_command,
            agent_environment=environment,
            environment_remove=remove_environment,
            protected_paths=protected_paths,
            host_before=host_before,
            metadata=metadata,
            startup_timeout_seconds=self.startup_timeout_seconds,
            secret_value=api_key,
            provider_config=provider_config,
        )


class PreparedCodexCCSwitchRuntime:
    """Prepared per-run state and lifecycle for a CC Switch sidecar."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        proxy_home: Path,
        codex_home: Path,
        agent_state: Path,
        workspace: Path,
        cc_switch_binary: Path,
        agent_command: Sequence[str],
        agent_environment: dict[str, str],
        environment_remove: Sequence[str],
        protected_paths: Sequence[Path],
        host_before: dict[str, Any],
        metadata: dict[str, Any],
        startup_timeout_seconds: int,
        secret_value: str,
        provider_config: dict[str, Any],
    ) -> None:
        self.runtime_root = runtime_root
        self.proxy_home = proxy_home
        self.codex_home = codex_home
        self.agent_state = agent_state
        self.workspace = workspace
        self.cc_switch_binary = cc_switch_binary
        self.agent_command = list(agent_command)
        self.agent_environment = dict(agent_environment)
        self.environment_remove = tuple(environment_remove)
        self.agent_read_paths = (cc_switch_binary, Path(self.agent_command[0]))
        self.agent_read_write_paths = (agent_state,)
        self.protected_paths = tuple(protected_paths)
        self.host_before = host_before
        self.metadata = metadata
        self.startup_timeout_seconds = startup_timeout_seconds
        self.secret_value = secret_value
        self.provider_config = provider_config
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any = None
        self._stopped = False

    def start(self) -> None:
        try:
            self._seed_private_config()
            self._launch_cc_switch()
            self._wait_for_database()
            self._terminate_cc_switch()
            self._enable_codex_takeover()
            self._launch_cc_switch()
            port = self._wait_for_proxy()
            self.agent_environment["MODEL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
            self.metadata["local_proxy"] = {"address": "127.0.0.1", "port": port}
            _write_json_private(self.runtime_root / "runtime_manifest.json", self.metadata)
        except Exception:
            self._terminate_cc_switch()
            raise

    def _seed_private_config(self) -> None:
        _write_json_private(
            self.proxy_home / ".cc-switch" / "config.json",
            self.provider_config,
        )
        _write_json_private(
            self.proxy_home / ".cc-switch" / "settings.json",
            {
                "showInTray": False,
                "minimizeToTrayOnClose": False,
                "silentStartup": True,
                "enableLocalProxy": True,
                "proxyConfirmed": True,
                "firstRunNoticeConfirmed": True,
                "preserveCodexOfficialAuthOnSwitch": False,
                "currentProviderCodex": "mlsysbench-siliconflow",
                "backupIntervalHours": 0,
            },
        )
        live_settings = self.provider_config["codex"]["providers"][
            "mlsysbench-siliconflow"
        ]["settingsConfig"]
        _write_json_private(self.codex_home / "auth.json", live_settings["auth"])
        _write_text_private(self.codex_home / "config.toml", live_settings["config"])

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._terminate_cc_switch()
        host_after = fingerprint_paths(self.protected_paths)
        unchanged = host_after == self.host_before
        self.metadata["host_codex_guard"] = {
            "before": self.host_before,
            "after": host_after,
            "unchanged": unchanged,
        }
        _write_json_private(
            self.codex_home / "auth.json",
            {"OPENAI_API_KEY": "PROXY_MANAGED"},
        )
        shutil.rmtree(self.proxy_home, ignore_errors=True)
        for name in (
            "xdg-config",
            "xdg-data",
            "xdg-cache",
            "xdg-state",
            "xdg-runtime",
            "tmp",
        ):
            shutil.rmtree(self.runtime_root / name, ignore_errors=True)
        _redact_file_secret(
            self.runtime_root / "cc-switch.process.log",
            self.secret_value,
        )
        credentials_removed = not _tree_contains_secret(
            self.runtime_root,
            self.secret_value,
        )
        self.metadata["credentials_removed"] = credentials_removed
        _write_json_private(self.runtime_root / "runtime_manifest.json", self.metadata)
        if not unchanged:
            raise ConfigError(
                "Host Codex configuration or source session changed during the isolated run; "
                f"see {self.runtime_root / 'runtime_manifest.json'}"
            )
        if not credentials_removed:
            raise ConfigError(
                "The upstream API key remained in isolated runtime artifacts after cleanup"
            )

    def _cc_switch_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(self.proxy_home),
                "CC_SWITCH_TEST_HOME": str(self.proxy_home),
                "CODEX_HOME": str(self.codex_home),
                "CODEX_SQLITE_HOME": str(self.codex_home),
                "XDG_CONFIG_HOME": str(self.runtime_root / "xdg-config"),
                "XDG_DATA_HOME": str(self.runtime_root / "xdg-data"),
                "XDG_CACHE_HOME": str(self.runtime_root / "xdg-cache"),
                "XDG_STATE_HOME": str(self.runtime_root / "xdg-state"),
                "XDG_RUNTIME_DIR": str(self.runtime_root / "xdg-runtime"),
                "TMPDIR": str(self.runtime_root / "tmp"),
                "GDK_BACKEND": "x11",
                "NO_AT_BRIDGE": "1",
                "APPIMAGE_EXTRACT_AND_RUN": "1",
            }
        )
        environment.pop("DBUS_SESSION_BUS_ADDRESS", None)
        environment.pop("SESSION_MANAGER", None)
        return environment

    def _launch_cc_switch(self) -> None:
        log_path = self.runtime_root / "cc-switch.process.log"
        self._log_handle = log_path.open("a", encoding="utf-8")
        sidecar_read_paths = [
            *default_system_read_paths(),
            self.cc_switch_binary,
        ]
        for candidate in (Path("/proc"), Path("/tmp/.X11-unix")):
            if candidate.exists():
                sidecar_read_paths.append(candidate)
        xauthority = os.environ.get("XAUTHORITY")
        if xauthority and Path(xauthority).is_file():
            sidecar_read_paths.append(Path(xauthority))

        def install_sidecar_landlock() -> None:
            restrict_current_process(
                read_only_paths=sidecar_read_paths,
                read_write_paths=[self.runtime_root],
            )

        try:
            self.process = subprocess.Popen(
                [str(self.cc_switch_binary)],
                cwd=self.runtime_root,
                env=self._cc_switch_environment(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                preexec_fn=install_sidecar_landlock,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._log_handle.close()
            self._log_handle = None
            raise ConfigError(f"CC Switch could not start: {exc}") from exc

    def _terminate_cc_switch(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _wait_for_database(self) -> None:
        database = self.proxy_home / ".cc-switch" / "cc-switch.db"
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            self._raise_if_sidecar_exited("database initialization")
            if database.is_file():
                try:
                    with sqlite3.connect(database, timeout=0.2) as connection:
                        provider = connection.execute(
                            "SELECT COUNT(*) FROM providers "
                            "WHERE id = ? AND app_type = 'codex'",
                            ("mlsysbench-siliconflow",),
                        ).fetchone()
                        rows = connection.execute(
                            "SELECT COUNT(*) FROM proxy_config"
                        ).fetchone()
                    if provider and provider[0] == 1 and rows and rows[0] >= 3:
                        return
                except sqlite3.Error:
                    pass
            time.sleep(0.1)
        raise ConfigError(
            "CC Switch did not finish isolated database initialization; "
            f"see {self.runtime_root / 'cc-switch.process.log'}"
        )

    def _enable_codex_takeover(self) -> None:
        database = self.proxy_home / ".cc-switch" / "cc-switch.db"
        try:
            with sqlite3.connect(database, timeout=5) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE proxy_config SET listen_address = '127.0.0.1', "
                    "listen_port = 0, enable_logging = 1, "
                    "streaming_first_byte_timeout = 120, streaming_idle_timeout = 600, "
                    "non_streaming_timeout = 1200, enabled = 0"
                )
                connection.execute(
                    "UPDATE proxy_config SET enabled = 1 WHERE app_type = 'codex'"
                )
                if connection.total_changes < 4:
                    raise ConfigError("CC Switch proxy_config schema was not initialized")
                connection.commit()
        except sqlite3.Error as exc:
            raise ConfigError(f"Could not enable isolated CC Switch Codex routing: {exc}") from exc

    def _wait_for_proxy(self) -> int:
        config_path = self.codex_home / "config.toml"
        auth_path = self.codex_home / "auth.json"
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            self._raise_if_sidecar_exited("proxy startup")
            try:
                config_text = config_path.read_text(encoding="utf-8")
                auth = json.loads(auth_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
                continue
            match = _LOCAL_BASE_URL_RE.search(config_text)
            placeholder = auth.get("OPENAI_API_KEY") if isinstance(auth, dict) else None
            has_placeholder = placeholder == "PROXY_MANAGED" or bool(
                _LOCAL_PLACEHOLDER_RE.search(config_text)
            )
            if match and has_placeholder:
                port = int(match.group(1))
                if _tcp_listening("127.0.0.1", port):
                    _write_json_private(
                        auth_path,
                        {"OPENAI_API_KEY": "PROXY_MANAGED"},
                    )
                    if _tree_contains_secret(self.agent_state, self.secret_value):
                        raise ConfigError(
                            "The upstream API key remained in the Codex-visible agent state"
                        )
                    return port
            time.sleep(0.1)
        raise ConfigError(
            "CC Switch did not expose the isolated Codex proxy; "
            f"see {self.runtime_root / 'cc-switch.process.log'}"
        )

    def _raise_if_sidecar_exited(self, phase: str) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise ConfigError(
                f"CC Switch exited during {phase} with code {self.process.returncode}; "
                f"see {self.runtime_root / 'cc-switch.process.log'}"
            )


def run_isolated_codex(
    *,
    workspace: str | Path,
    output_dir: str | Path,
    spec: CodexCCSwitchSpec,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Run Codex directly while keeping CC Switch credentials outside its workspace."""

    if timeout_seconds <= 0:
        raise ConfigError("timeout_seconds must be positive")
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        raise ConfigError(f"Workspace does not exist: {workspace_path}")
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise ConfigError(f"Isolated Codex output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    private_parent = Path(tempfile.mkdtemp(prefix="mlsysbench-codex-", dir="/tmp"))
    stdout_path = output_path / "codex.stdout.jsonl"
    stderr_path = output_path / "codex.stderr.log"
    result_path = output_path / "run_result.json"
    prepared: PreparedCodexCCSwitchRuntime | None = None
    process: subprocess.Popen[str] | None = None
    started_at = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    error: str | None = None
    try:
        prepared = spec.prepare(
            output_dir=private_parent,
            workspace=workspace_path,
            mission_path=workspace_path / "MISSION.md",
        )
        prepared.start()
        environment = os.environ.copy()
        for key in prepared.environment_remove:
            environment.pop(key, None)
        environment.update(prepared.agent_environment)
        environment["PWD"] = str(workspace_path)

        read_only_paths = [
            *default_system_read_paths(),
            *(Path(path) for path in prepared.agent_read_paths),
        ]

        def install_agent_landlock() -> None:
            restrict_current_process(
                read_only_paths=read_only_paths,
                read_write_paths=[
                    workspace_path,
                    output_path,
                    *prepared.agent_read_write_paths,
                ],
            )

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                prepared.agent_command,
                cwd=workspace_path,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
                preexec_fn=install_agent_landlock,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
                if exit_code != 0:
                    error = f"Codex exited with code {exit_code}"
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                exit_code = process.returncode
    except (ConfigError, OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_group(process)
        if prepared is not None:
            try:
                prepared.stop()
            except ConfigError as exc:
                error = error or str(exc)

        elapsed = round(time.monotonic() - started_at, 6)
        metadata = prepared.metadata if prepared is not None else None
        result = {
            "status": "timed_out" if timed_out else ("failed" if error else "completed"),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "error": error,
            "elapsed_seconds": elapsed,
            "workspace": str(workspace_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "runtime": metadata,
        }
        _write_json_private(result_path, result, mode=0o644)
        shutil.rmtree(private_parent, ignore_errors=True)
    return result


def build_siliconflow_provider_config(
    *,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict[str, Any]:
    """Build CC Switch's supported v2 migration input from its SiliconFlow preset."""

    provider_name = "mlsysbench-siliconflow"
    config_toml = "\n".join(
        [
            'model_provider = "custom"',
            f"model = {json.dumps(model)}",
            'model_reasoning_effort = "high"',
            f"model_context_window = {context_window}",
            f"model_max_output_tokens = {max_output_tokens}",
            'disable_response_storage = true',
            'web_search = "disabled"',
            "",
            "[model_providers.custom]",
            f"name = {json.dumps(provider_name)}",
            f"base_url = {json.dumps(base_url.rstrip('/'))}",
            'wire_api = "responses"',
            "requires_openai_auth = true",
        ]
    )
    catalog = [
        {
            "model": model_id,
            "displayName": model_id.rsplit("/", 1)[-1],
            "contextWindow": context_window,
            "inputModalities": ["text"],
        }
        for model_id in SUPPORTED_MODELS
    ]
    provider = {
        "id": "mlsysbench-siliconflow",
        "name": "SiliconFlow (MLSysBench isolated)",
        "settingsConfig": {
            "auth": {"OPENAI_API_KEY": api_key},
            "config": config_toml,
            "modelCatalog": {"models": catalog},
        },
        "websiteUrl": "https://siliconflow.cn",
        "category": "aggregator",
        "icon": "siliconflow",
        "meta": {
            "apiFormat": "openai_chat",
            "codexChatReasoning": {
                "supportsThinking": True,
                "supportsEffort": False,
                "thinkingParam": "enable_thinking",
                "effortParam": "none",
                "outputFormat": "reasoning_content",
            },
            "localProxyRequestOverrides": {
                "body": {
                    "enable_thinking": True,
                    "thinking_budget": thinking_budget,
                    "max_tokens": max_output_tokens,
                }
            },
        },
    }
    return {
        "version": 2,
        "codex": {
            "providers": {"mlsysbench-siliconflow": provider},
            "current": "mlsysbench-siliconflow",
        },
    }


def snapshot_codex_session(
    *,
    host_codex_home: Path,
    isolated_codex_home: Path,
    session_id: str,
    workspace: Path,
    redact_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Copy one rollout into CODEX_HOME and retarget only its isolated metadata."""

    if not re.fullmatch(r"[0-9a-fA-F-]{36}", session_id):
        raise ConfigError("session_id must be a UUID")
    sessions_root = host_codex_home / "sessions"
    candidates = [
        path
        for path in sessions_root.rglob(f"*{session_id}*.jsonl")
        if path.is_file() and session_id in path.name
    ]
    if len(candidates) != 1:
        raise ConfigError(
            f"Expected exactly one host Codex rollout for {session_id}, found {len(candidates)}"
        )
    source = candidates[0].resolve()
    relative = source.relative_to(sessions_root.resolve())
    destination = isolated_codex_home / "sessions" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    replaced = False
    with source.open("r", encoding="utf-8") as source_handle, destination.open(
        "x", encoding="utf-8"
    ) as destination_handle:
        for raw_line in source_handle:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Host Codex rollout is invalid JSONL: {source}") from exc
            record = _redact_session_value(record, redact_values)
            payload = record.get("payload") if isinstance(record, dict) else None
            if not replaced and isinstance(record, dict) and record.get("type") == "session_meta":
                if not isinstance(payload, dict):
                    raise ConfigError("Host Codex rollout has invalid session_meta")
                recorded_id = payload.get("id") or payload.get("session_id")
                if recorded_id != session_id:
                    raise ConfigError("Host Codex rollout session id does not match its filename")
                payload["cwd"] = str(workspace)
                payload["model_provider"] = "custom"
                replaced = True
            destination_handle.write(
                json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
            )
    destination.chmod(0o600)
    if not replaced:
        destination.unlink(missing_ok=True)
        raise ConfigError("Host Codex rollout does not contain session_meta")
    return {
        "id": session_id,
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
        "snapshot_path": str(destination),
        "snapshot_sha256": _sha256_file(destination),
        "cwd": str(workspace),
    }


def fingerprint_paths(paths: Sequence[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        key = str(resolved)
        try:
            stat_result = resolved.stat()
        except FileNotFoundError:
            result[key] = {"exists": False}
            continue
        result[key] = {
            "exists": True,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "sha256": _sha256_file(resolved) if resolved.is_file() else None,
        }
    return result


def _resolve_codex_source(value: str | Path | None) -> Path:
    if value is not None:
        source = Path(value).expanduser().resolve()
    elif os.environ.get("MLSYSBENCH_HOST_CODEX_BINARY"):
        source = Path(os.environ["MLSYSBENCH_HOST_CODEX_BINARY"]).expanduser().resolve()
    else:
        executable = shutil.which("codex")
        if executable is None:
            raise ConfigError(
                "Pinned Codex CLI source was not found; pass --codex-source or set "
                "MLSYSBENCH_HOST_CODEX_BINARY"
            )
        source = Path(executable).resolve()
    if not source.is_file():
        raise ConfigError(f"Codex CLI source does not exist: {source}")
    return source


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".part-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        _verify_sha256(temporary, expected_sha256, destination.name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".part-{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        _verify_sha256(temporary, expected_sha256, destination.name)
        os.replace(temporary, destination)
    except Exception as exc:
        raise ConfigError(f"Could not download verified CC Switch {CC_SWITCH_VERSION}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ConfigError(f"{label} asset is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ConfigError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_private(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_private(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _redact_session_value(value: Any, redact_values: Sequence[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_session_value(item, redact_values) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_session_value(item, redact_values) for item in value]
    if not isinstance(value, str):
        return value
    redacted = value
    for secret in redact_values:
        if secret:
            redacted = redacted.replace(secret, "<redacted-api-key>")
    return _API_CREDENTIAL_RE.sub("<redacted-api-key>", redacted)


def _tree_contains_secret(root: Path, secret: str) -> bool:
    if not secret:
        return False
    needle = secret.encode("utf-8")
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                carry = b""
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    combined = carry + chunk
                    if needle in combined:
                        return True
                    carry_length = max(0, len(needle) - 1)
                    carry = combined[-carry_length:] if carry_length else b""
        except OSError:
            continue
    return False


def _redact_file_secret(path: Path, secret: str) -> None:
    if not secret or not path.is_file():
        return
    needle = secret.encode("utf-8")
    try:
        content = path.read_bytes()
    except OSError:
        return
    if needle not in content:
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_bytes(content.replace(needle, b"<redacted-api-key>"))
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tcp_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)
