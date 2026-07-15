import json
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlsysbench.simai_bench.agent_runner import run_agent_loop, run_agent_once
from mlsysbench.simai_bench.actions import validate_changes
from mlsysbench.simai_bench.cli_agent import prepare_public_workspace, run_cli_agent
from mlsysbench.simai_bench.chat_cli_agent import _chat_completion
from mlsysbench.simai_bench.codex_ccswitch import (
    CodexCCSwitchSpec,
    PreparedCodexCCSwitchRuntime,
    SUPPORTED_MODELS,
    build_siliconflow_provider_config,
    fingerprint_paths,
    run_isolated_codex,
    snapshot_codex_session,
)
from mlsysbench.simai_bench.evaluator import evaluate_changes, evaluate_submission
from mlsysbench.simai_bench.io import ConfigError, load_structured
from mlsysbench.simai_bench.metrics import parse_vidur_output
from mlsysbench.simai_bench.model_client import (
    MODEL_METADATA_KEY,
    DryRunClient,
    OpenAICompatibleClient,
    make_model_client,
)
from mlsysbench.simai_bench.runner import VidurRunner, detect_aicb_failure_or_default
from mlsysbench.simai_bench.schema import RunnerSpec, SLO, ScenarioSpec, TaskSpec
from mlsysbench.simai_bench.search import run_search
from mlsysbench.simai_bench.task_validation import validate_task


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "simai_gym" / "l1_scheduler_choice"
REAL_TASK = ROOT / "tasks" / "simai_gym" / "qwen3_next_aicb_benchmark"
SCALE_TASK = ROOT / "tasks" / "scale_up" / "mock_scale_transfer"
SCENARIO_TASKS = {
    "prefill_heavy": ROOT / "tasks" / "scenarios" / "mock_prefill_heavy",
    "decode_heavy": ROOT / "tasks" / "scenarios" / "mock_decode_heavy",
    "high_load": SCALE_TASK,
    "balanced": ROOT / "tasks" / "scenarios" / "mock_balanced",
}
SUBMISSION = ROOT / "submissions" / "examples" / "sarathi_scheduler.json"


class SimAIBenchTest(unittest.TestCase):
    def test_example_submission_scores_against_baseline(self):
        result = evaluate_submission(TASK, SUBMISSION)

        self.assertTrue(result.valid)
        self.assertEqual(result.primary_metric, "goodput_rps")
        self.assertAlmostEqual(result.ratio, 68.0 / 42.0)
        self.assertAlmostEqual(result.score, 68.0 / 42.0)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.gpu_units, 8)

    def test_forbidden_ep_action_is_rejected(self):
        task = TaskSpec.load(TASK)
        allowed_actions = task.load_allowed_actions()

        with self.assertRaises(ConfigError):
            validate_changes(
                {"replica_config_expert_model_parallel_size": 8},
                allowed_actions,
            )

    def test_unlisted_action_is_rejected(self):
        task = TaskSpec.load(TASK)
        allowed_actions = task.load_allowed_actions()

        with self.assertRaises(ConfigError):
            validate_changes({"replica_config_tensor_parallel_size": 4}, allowed_actions)

    def test_dry_run_agent_generates_and_evaluates_submission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_agent_once(TASK, tmpdir, DryRunClient())

            self.assertTrue(result.submission_path.exists())
            self.assertTrue(result.result_path.exists())
            self.assertTrue(result.evaluation.valid)

    def test_multi_step_agent_records_and_retains_best_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_agent_loop(TASK, tmpdir, DryRunClient(), max_steps=4)

            self.assertEqual(result.steps_used, 3)
            self.assertIsNotNone(result.best_evaluation)
            self.assertTrue(result.best_evaluation.valid)
            self.assertAlmostEqual(result.best_evaluation.ratio, 68.0 / 42.0)
            trajectory = json.loads(result.trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(len(trajectory), 3)
            self.assertTrue(trajectory[-1]["stop"])

    def test_agent_loop_records_and_summarizes_model_usage(self):
        class MetadataClient:
            def generate_submission(self, context):
                return {
                    "changes": {},
                    "notes": "stop at baseline",
                    "stop": True,
                    MODEL_METADATA_KEY: {
                        "requested_model": "test-model",
                        "response_model": "test-model",
                        "latency_seconds": 0.25,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_agent_loop(SCALE_TASK, tmpdir, MetadataClient(), max_steps=2)
            trajectory = json.loads(result.trajectory_path.read_text(encoding="utf-8"))

        self.assertEqual(trajectory[0]["model_call"]["usage"]["total_tokens"], 120)
        self.assertEqual(result.model_stats["api_calls"], 1)
        self.assertEqual(result.model_stats["models"], ["test-model"])
        self.assertEqual(result.model_stats["usage"]["total_tokens"], 120)

    def test_openai_compatible_client_uses_json_mode_and_records_metadata(self):
        captured = {}
        response_payload = {
            "id": "chatcmpl-test",
            "model": "meituan-longcat/LongCat-2.0",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "changes": {"replica_scheduler_config_type": "sarathi"},
                                "notes": "measured choice",
                                "stop": False,
                            }
                        )
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 16,
                "total_tokens": 96,
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        client = OpenAICompatibleClient(
            base_url="https://api.siliconflow.cn/v1",
            api_key="test-key",
            model="meituan-longcat/LongCat-2.0",
            timeout_seconds=45,
            max_tokens=512,
            temperature=0.2,
            json_mode=True,
            enable_thinking=True,
            thinking_budget=32_768,
        )
        with (
            patch(
                "mlsysbench.simai_bench.model_client.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            patch(
                "mlsysbench.simai_bench.model_client.time.perf_counter",
                side_effect=[10.0, 10.25],
            ),
        ):
            submission = client.generate_submission({"allowed_actions": {}})

        self.assertEqual(captured["timeout"], 45)
        self.assertEqual(captured["payload"]["max_tokens"], 512)
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(
            captured["payload"]["response_format"],
            {"type": "json_object"},
        )
        self.assertTrue(captured["payload"]["enable_thinking"])
        self.assertEqual(captured["payload"]["thinking_budget"], 32_768)
        metadata = submission[MODEL_METADATA_KEY]
        self.assertEqual(metadata["request_id"], "chatcmpl-test")
        self.assertEqual(metadata["latency_seconds"], 0.25)
        self.assertEqual(metadata["usage"]["total_tokens"], 96)

    def test_model_client_reads_default_environment_configuration(self):
        args = SimpleNamespace(
            provider="openai-compatible",
            api_key=None,
            api_key_env=None,
            model=None,
            base_url=None,
            timeout_seconds=30,
            max_output_tokens=None,
            temperature=None,
            json_mode=None,
        )
        environment = {
            "MODEL_API_KEY": "test-key",
            "MODEL_BASE_URL": "https://api.siliconflow.cn/v1",
            "MODEL_NAME": "meituan-longcat/LongCat-2.0",
            "MODEL_MAX_TOKENS": "768",
            "MODEL_TEMPERATURE": "0.1",
            "MODEL_JSON_MODE": "true",
            "MODEL_ENABLE_THINKING": "true",
            "MODEL_THINKING_BUDGET": "32768",
        }
        with (
            patch("mlsysbench.simai_bench.model_client.load_dotenv"),
            patch.dict(os.environ, environment, clear=True),
        ):
            client = make_model_client(args)

        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.max_tokens, 768)
        self.assertEqual(client.temperature, 0.1)
        self.assertTrue(client.json_mode)
        self.assertTrue(client.enable_thinking)
        self.assertEqual(client.thinking_budget, 32_768)

    def test_grid_search_finds_known_mock_optimum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_search(TASK, tmpdir, method="grid", budget=24)

            self.assertEqual(result.evaluations, 24)
            self.assertIsNotNone(result.best_evaluation)
            self.assertAlmostEqual(result.best_evaluation.ratio, 68.0 / 42.0)

    def test_search_wall_time_can_stop_before_query_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_search(
                TASK,
                tmpdir,
                method="grid",
                budget=24,
                wall_time_seconds=1e-12,
            )

        self.assertEqual(result.evaluations, 0)
        self.assertEqual(result.stopped_reason, "wall_time")

    def test_optional_hpo_methods_report_missing_dependency(self):
        if importlib.util.find_spec("optuna") is not None:
            self.skipTest("Optuna is installed in this environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ConfigError, "Optuna"):
                run_search(TASK, tmpdir, method="tpe", budget=3)

    def test_scale_transfer_separates_development_and_final_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_agent_loop(SCALE_TASK, tmpdir, DryRunClient(), max_steps=5)

            self.assertAlmostEqual(result.best_evaluation.ratio, 1.6)
            self.assertAlmostEqual(result.final_evaluation.ratio, 1.875)
            self.assertEqual(result.best_evaluation.merged_config["cluster_config_num_replicas"], 8)
            self.assertEqual(result.final_evaluation.merged_config["cluster_config_num_replicas"], 32)

    def test_scale_transfer_mock_surface_is_complete_and_baseline_consistent(self):
        task = TaskSpec.load(SCALE_TASK)
        baseline_config = task.load_baseline_config()
        allowed_actions = task.load_allowed_actions()

        for phase in ("development", "final"):
            result = evaluate_changes(
                task,
                baseline_config,
                allowed_actions,
                {},
                phase=phase,
            )
            self.assertTrue(result.valid)
            self.assertAlmostEqual(result.ratio, 1.0)
            self.assertEqual(result.agent_metrics, result.baseline_metrics)

        for path in (
            SCALE_TASK / "public" / "mock_metrics.json",
            SCALE_TASK / "hidden" / "mock_metrics.json",
        ):
            surface = load_structured(path)
            self.assertNotIn("default", surface)
            self.assertEqual(len(surface), 18)

    def test_canonical_scenario_fixtures_are_dense_and_valid(self):
        expected_profiles = {
            "prefill_heavy": {"short_prompt", "long_prompt"},
            "decode_heavy": {"short_output", "long_output"},
            "high_load": {"burst", "poisson", "constant"},
            "balanced": {"mixed_prompt_output", "mixed_concurrency"},
        }

        for family, task_path in SCENARIO_TASKS.items():
            with self.subTest(family=family):
                task = TaskSpec.load(task_path)
                result = validate_task(task_path)

                self.assertEqual(task.schema_version, 1)
                self.assertEqual(task.scenario.family, family)
                self.assertEqual(set(task.scenario.profiles), expected_profiles[family])
                self.assertTrue(result.valid, result.errors)
                self.assertEqual(result.warnings, [])
                self.assertEqual(
                    result.details["scenario"]["missing_canonical_profiles"], []
                )
                self.assertFalse(
                    result.details["development_mock_surface"]["has_default"]
                )
                self.assertFalse(result.details["final_mock_surface"]["has_default"])
                self.assertEqual(
                    result.details["development_mock_surface"]["missing_configurations"],
                    0,
                )
                self.assertEqual(
                    result.details["final_mock_surface"]["missing_configurations"], 0
                )

    def test_scenario_schema_rejects_cross_family_profiles(self):
        with self.assertRaisesRegex(ConfigError, "does not support profiles"):
            ScenarioSpec.from_dict(
                {
                    "family": "prefill_heavy",
                    "transfer": "workload_shift",
                    "starting_point": "framework_default",
                    "profiles": ["burst"],
                }
            )

    def test_task_schema_version_is_required_and_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_copy = Path(tmpdir) / "task"
            shutil.copytree(SCENARIO_TASKS["balanced"], task_copy)
            task_path = task_copy / "task.json"
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            payload["schema_version"] = 99
            task_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "schema_version"):
                TaskSpec.load(task_copy)

    def test_task_validator_rejects_workload_family_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_copy = Path(tmpdir) / "task"
            shutil.copytree(SCENARIO_TASKS["prefill_heavy"], task_copy)
            workload_path = task_copy / "hidden" / "eval_workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["scenario_family"] = "decode_heavy"
            workload_path.write_text(json.dumps(workload), encoding="utf-8")

            result = validate_task(task_copy)

        self.assertFalse(result.valid)
        self.assertTrue(any("does not match" in error for error in result.errors))

    def test_task_validator_rejects_incomplete_mock_surface(self):
        valid_result = validate_task(SCALE_TASK)
        self.assertTrue(valid_result.valid)
        self.assertEqual(valid_result.details["development_mock_surface"]["missing_configurations"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            task_copy = Path(tmpdir) / "task"
            shutil.copytree(SCALE_TASK, task_copy)
            surface_path = task_copy / "public" / "mock_metrics.json"
            surface = json.loads(surface_path.read_text(encoding="utf-8"))
            surface.pop(next(iter(surface)))
            surface_path.write_text(json.dumps(surface), encoding="utf-8")

            invalid_result = validate_task(task_copy)

        self.assertFalse(invalid_result.valid)
        self.assertTrue(any("missing 1 of 18" in error for error in invalid_result.errors))

    def test_scale_transfer_search_submits_public_optimum_to_hidden_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_search(SCALE_TASK, tmpdir, method="grid", budget=18)

            self.assertAlmostEqual(result.best_development_evaluation.ratio, 1.6)
            self.assertAlmostEqual(result.best_evaluation.ratio, 1.3125)

    def test_cli_agent_workspace_contains_only_public_interface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            prepare_public_workspace(
                SCALE_TASK,
                workspace,
                max_queries=3,
                wall_time_seconds=60,
            )

            self.assertEqual(
                {path.name for path in workspace.iterdir()},
                {
                    ".tmp",
                    "MISSION.md",
                    "budget.json",
                    "evaluate_dev.py",
                    "final_submission.schema.json",
                    "task_context.json",
                },
            )
            context = json.loads((workspace / "task_context.json").read_text(encoding="utf-8"))
            self.assertNotIn("hidden", context)
            self.assertNotIn("runner", context)
            self.assertNotIn(str(SCALE_TASK), json.dumps(context))
            self.assertEqual(context["schema_version"], 1)
            self.assertEqual(context["scenario"]["family"], "high_load")
            self.assertEqual(context["scenario"]["transfer"], "scale_up")

    def test_benchmark_mode_requires_canonical_codex_scaffold(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
            ConfigError, r"codex-cli\+cc-switch"
        ):
            run_cli_agent(
                SCALE_TASK,
                Path(tmpdir) / "run",
                [sys.executable, "-c", "pass"],
                wall_time_seconds=10,
                max_queries=1,
                isolation="landlock",
                agent_scaffold="custom",
                benchmark_mode=True,
            )

    def test_benchmark_mode_requires_process_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
            ConfigError, "bwrap process isolation"
        ):
            run_cli_agent(
                SCALE_TASK,
                Path(tmpdir) / "run",
                [sys.executable, "-c", "pass"],
                wall_time_seconds=10,
                max_queries=1,
                isolation="landlock",
                agent_scaffold="codex-cli+cc-switch",
                benchmark_mode=True,
            )

    def test_cli_agent_enforces_landlock_query_budget_and_clean_final_replay(self):
        private_metrics = SCALE_TASK / "hidden" / "mock_metrics.json"
        fake_agent = "\n".join(
            [
                "import json, pathlib, subprocess, sys",
                f"private_path = pathlib.Path({str(private_metrics)!r})",
                "try:",
                "    private_path.read_text(encoding='utf-8')",
                "except PermissionError:",
                "    private_status = 'blocked'",
                "else:",
                "    private_status = 'readable'",
                "pathlib.Path('private_read_status.txt').write_text(private_status, encoding='utf-8')",
                "candidate = {'changes': {'replica_config_tensor_parallel_size': 2, 'replica_scheduler_config_type': 'sarathi', 'sarathi_scheduler_config_chunk_size': 512}}",
                "pathlib.Path('candidate.json').write_text(json.dumps(candidate), encoding='utf-8')",
                "first = subprocess.run([sys.executable, 'evaluate_dev.py', 'candidate.json'], check=False)",
                "second = subprocess.run([sys.executable, 'evaluate_dev.py', 'candidate.json'], check=False)",
                "pathlib.Path('query_returncodes.json').write_text(json.dumps([first.returncode, second.returncode]), encoding='utf-8')",
                "final_submission = {'changes': {'replica_config_tensor_parallel_size': 4, 'replica_scheduler_config_type': 'sarathi', 'sarathi_scheduler_config_chunk_size': 1024}}",
                "pathlib.Path('final_submission.json').write_text(json.dumps(final_submission), encoding='utf-8')",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cli_agent(
                SCALE_TASK,
                Path(tmpdir) / "run",
                [sys.executable, "-c", fake_agent],
                wall_time_seconds=30,
                max_queries=1,
                isolation="landlock",
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            trajectory = json.loads(result.trajectory_path.read_text(encoding="utf-8"))
            private_status = (result.workspace / "private_read_status.txt").read_text(
                encoding="utf-8"
            )
            query_returncodes = json.loads(
                (result.workspace / "query_returncodes.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.queries_used, 1)
        self.assertEqual(query_returncodes, [0, 1])
        self.assertEqual(private_status, "blocked")
        self.assertEqual(len(trajectory), 1)
        self.assertAlmostEqual(trajectory[0]["evaluation"]["ratio"], 1.6)
        self.assertIsNotNone(result.final_evaluation)
        self.assertAlmostEqual(result.final_evaluation.ratio, 1.875)
        self.assertEqual(manifest["isolation"]["backend"], "landlock")
        self.assertEqual(manifest["agent"]["scaffold"], "custom")
        self.assertFalse(manifest["agent"]["benchmark_mode"])
        self.assertTrue(manifest["isolation"]["hard_filesystem_boundary"])
        self.assertFalse(manifest["isolation"]["process_isolated"])
        self.assertEqual(manifest["budgets"]["development_queries_used"], 1)
        self.assertTrue(manifest["gates"]["agent_process"])
        self.assertTrue(manifest["gates"]["overall"])
        self.assertNotEqual(
            manifest["inputs"]["development"]["workload_sha256"],
            manifest["inputs"]["final"]["workload_sha256"],
        )
        self.assertNotEqual(
            manifest["inputs"]["development"]["seeds"],
            manifest["inputs"]["final"]["seeds"],
        )

    def test_cli_agent_kills_process_group_at_wall_time_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cli_agent(
                SCALE_TASK,
                Path(tmpdir) / "run",
                [sys.executable, "-c", "import time; time.sleep(10)"],
                wall_time_seconds=1,
                max_queries=1,
                isolation="landlock",
            )

        self.assertEqual(result.status, "timed_out")
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.final_evaluation)
        self.assertIn("wall-time budget", result.final_error)

    def test_landlock_rejects_read_path_overlapping_private_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ConfigError, "overlaps the private task tree"):
                run_cli_agent(
                    SCALE_TASK,
                    Path(tmpdir) / "run",
                    [sys.executable, "-c", "pass"],
                    wall_time_seconds=10,
                    max_queries=1,
                    isolation="landlock",
                    agent_read_paths=[SCALE_TASK.parent],
                )

    def test_chat_cli_agent_requests_tools_and_maximum_configured_output(self):
        captured = {}
        response_payload = {
            "id": "chatcmpl-tool-test",
            "model": "meituan-longcat/LongCat-2.0",
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch(
            "mlsysbench.simai_bench.chat_cli_agent.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            response = _chat_completion(
                url="https://api.siliconflow.cn/v1/chat/completions",
                api_key="test-key",
                model="meituan-longcat/LongCat-2.0",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=131_072,
                timeout_seconds=600,
            )

        self.assertEqual(response["id"], "chatcmpl-tool-test")
        self.assertEqual(captured["timeout"], 600)
        self.assertEqual(captured["payload"]["max_tokens"], 131_072)
        self.assertTrue(captured["payload"]["enable_thinking"])
        self.assertEqual(captured["payload"]["thinking_budget"], 32_768)
        self.assertEqual(captured["payload"]["tool_choice"], "auto")
        self.assertEqual(
            {tool["function"]["name"] for tool in captured["payload"]["tools"]},
            {"read_file", "write_file", "list_files", "run_command"},
        )

    def test_cc_switch_provider_uses_chat_bridge_reasoning_and_full_catalog(self):
        config = build_siliconflow_provider_config(
            api_key="test-secret",
            model="meituan-longcat/LongCat-2.0",
            max_output_tokens=131_072,
            context_window=1_048_576,
            thinking_budget=32_768,
        )
        manager = config["codex"]
        provider = manager["providers"][manager["current"]]
        settings = provider["settingsConfig"]
        meta = provider["meta"]

        self.assertEqual(settings["auth"]["OPENAI_API_KEY"], "test-secret")
        self.assertIn('wire_api = "responses"', settings["config"])
        self.assertIn("model_context_window = 1048576", settings["config"])
        self.assertEqual(meta["apiFormat"], "openai_chat")
        self.assertEqual(
            meta["codexChatReasoning"]["thinkingParam"],
            "enable_thinking",
        )
        self.assertEqual(
            meta["localProxyRequestOverrides"]["body"],
            {
                "enable_thinking": True,
                "thinking_budget": 32_768,
                "max_tokens": 131_072,
            },
        )
        models = settings["modelCatalog"]["models"]
        self.assertEqual({item["model"] for item in models}, set(SUPPORTED_MODELS))
        self.assertTrue(all(item["contextWindow"] == 1_048_576 for item in models))

    def test_codex_session_snapshot_retargets_only_the_isolated_copy(self):
        session_id = "019f5e59-7b19-76d2-a785-e82d6ead1f73"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            host_home = root / "host-codex"
            source = (
                host_home
                / "sessions"
                / "2026"
                / "07"
                / "14"
                / f"rollout-test-{session_id}.jsonl"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {
                                    "id": session_id,
                                    "session_id": session_id,
                                    "cwd": "/host/workspace",
                                    "model_provider": "OpenAI",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {
                                    "message": "keep sk-abcdefghijklmnopqrstuvwxyz123456"
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            before = fingerprint_paths([source])
            isolated_home = root / "isolated-codex"
            workspace = root / "public-workspace"
            workspace.mkdir()

            metadata = snapshot_codex_session(
                host_codex_home=host_home,
                isolated_codex_home=isolated_home,
                session_id=session_id,
                workspace=workspace,
            )

            after = fingerprint_paths([source])
            snapshot = Path(metadata["snapshot_path"])
            records = [json.loads(line) for line in snapshot.read_text().splitlines()]

        self.assertEqual(before, after)
        self.assertEqual(records[0]["payload"]["cwd"], str(workspace))
        self.assertEqual(records[0]["payload"]["model_provider"], "custom")
        self.assertEqual(records[1]["payload"]["message"], "keep <redacted-api-key>")

    def test_standalone_codex_runtime_strips_parent_secrets(self):
        class FakePreparedRuntime:
            def __init__(self, private_output, workspace):
                self.state = private_output / "runtime" / "agent-state"
                self.state.mkdir(parents=True)
                script = "\n".join(
                    [
                        "import json, os, pathlib",
                        "pathlib.Path('standalone_result.json').write_text(",
                        "    json.dumps({'secret_visible': 'MODEL_API_KEY' in os.environ}),",
                        "    encoding='utf-8',",
                        ")",
                    ]
                )
                self.agent_command = [sys.executable, "-c", script]
                self.agent_environment = {"HOME": str(self.state), "TMPDIR": str(self.state)}
                self.environment_remove = ("MODEL_API_KEY",)
                self.agent_read_paths = (Path(sys.executable),)
                self.agent_read_write_paths = (self.state,)
                self.metadata = {"fake": True, "stopped": False}

            def start(self):
                self.metadata["started"] = True

            def stop(self):
                self.metadata["stopped"] = True

        class FakeSpec:
            def prepare(self, *, output_dir, workspace, mission_path):
                return FakePreparedRuntime(output_dir, workspace)

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"MODEL_API_KEY": "test-secret"}
        ):
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir()
            result = run_isolated_codex(
                workspace=workspace,
                output_dir=root / "output",
                spec=FakeSpec(),
                timeout_seconds=10,
            )
            agent_result = json.loads(
                (workspace / "standalone_result.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["status"], "completed")
        self.assertFalse(agent_result["secret_visible"])
        self.assertTrue(result["runtime"]["stopped"])

    def test_cc_switch_runtime_cleanup_removes_all_key_copies(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            proxy_home = runtime_root / "proxy-home"
            codex_home = runtime_root / "agent-state" / "codex-home"
            workspace = root / "workspace"
            for path in (proxy_home, codex_home, workspace):
                path.mkdir(parents=True)
            (proxy_home / "cc-switch.db").write_text(secret, encoding="utf-8")
            (codex_home / "auth.json").write_text(
                json.dumps({"OPENAI_API_KEY": secret}),
                encoding="utf-8",
            )
            process_log = runtime_root / "cc-switch.process.log"
            process_log.write_text(f"redact {secret}\n", encoding="utf-8")
            protected = root / "host-config.toml"
            protected.write_text("host = true\n", encoding="utf-8")
            host_before = fingerprint_paths([protected])
            metadata = {"host_codex_guard": {"before": host_before}}
            prepared = PreparedCodexCCSwitchRuntime(
                runtime_root=runtime_root,
                proxy_home=proxy_home,
                codex_home=codex_home,
                agent_state=runtime_root / "agent-state",
                workspace=workspace,
                cc_switch_binary=root / "cc-switch",
                agent_command=[sys.executable, "-c", "pass"],
                agent_environment={},
                environment_remove=(),
                protected_paths=[protected],
                host_before=host_before,
                metadata=metadata,
                startup_timeout_seconds=1,
                secret_value=secret,
                provider_config={},
            )

            prepared.stop()

            proxy_removed = not proxy_home.exists()
            auth = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
            log_text = process_log.read_text(encoding="utf-8")

        self.assertTrue(proxy_removed)
        self.assertEqual(auth["OPENAI_API_KEY"], "PROXY_MANAGED")
        self.assertNotIn(secret, log_text)
        self.assertTrue(metadata["credentials_removed"])
        self.assertTrue(metadata["host_codex_guard"]["unchanged"])

    def test_codex_prepare_failure_writes_no_credentials(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"MODEL_API_KEY": secret}
        ), patch("mlsysbench.simai_bench.codex_ccswitch._verify_sha256"):
            root = Path(tmpdir)
            asset_dir = root / "assets"
            codex_asset = asset_dir / "codex-0.144.3" / "codex"
            codex_asset.parent.mkdir(parents=True)
            codex_asset.write_bytes(b"codex")
            (asset_dir / "CC-Switch-v3.17.0-Linux-x86_64.AppImage").write_bytes(b"cc")
            workspace = root / "workspace"
            workspace.mkdir()
            output = root / "output"
            spec = CodexCCSwitchSpec(
                session_id="019f5e59-7b19-76d2-a785-e82d6ead1f73",
                asset_dir=asset_dir,
                host_codex_home=root / "missing-host-codex",
            )

            with self.assertRaisesRegex(ConfigError, "found 0"):
                spec.prepare(
                    output_dir=output,
                    workspace=workspace,
                    mission_path=workspace / "MISSION.md",
                )

            leaked = any(
                secret.encode("utf-8") in path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            )

        self.assertFalse(leaked)

    def test_gpu_budget_is_enforced_before_real_runner(self):
        submission = {
            "changes": {
                "cluster_config_num_replicas": 64,
                "replica_config_tensor_parallel_size": 8,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = Path(tmpdir) / "submission.json"
            submission_path.write_text(json.dumps(submission), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "exceeding budget 32"):
                evaluate_submission(REAL_TASK, submission_path)

    def test_real_baseline_metrics_use_corrected_time_units(self):
        metrics = TaskSpec.load(REAL_TASK).load_baseline_metrics()

        self.assertGreater(metrics["p99_e2e_ms"], 250.0)
        self.assertLess(metrics["p99_e2e_ms"], 350.0)
        self.assertGreater(metrics["p99_tbt_ms"], 25.0)

    def test_model_can_omit_unchanged_allowed_defaults(self):
        submission = {
            "changes": {
                "replica_scheduler_config_type": "sarathi",
                "sarathi_scheduler_config_chunk_size": 512,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = Path(tmpdir) / "submission.json"
            submission_path.write_text(json.dumps(submission), encoding="utf-8")

            result = evaluate_submission(TASK, submission_path)

            self.assertTrue(result.valid)
            self.assertAlmostEqual(result.ratio, 68.0 / 42.0)

    def test_vidur_runner_passes_configured_environment(self):
        task = replace(
            TaskSpec.load(TASK),
            runner=RunnerSpec(
                type="vidur",
                config={
                    "vidur_root": str(ROOT),
                    "output_dir": "runs/test_vidur_runner_env",
                    "python_bin": "python",
                    "env": {
                        "CUDA_HOME": "/usr/local/cuda-12.9",
                        "TORCH_CUDA_ARCH_LIST": "8.9",
                    },
                },
            ),
        )

        completed = type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "", "stderr": "forced failure"},
        )()

        with patch("mlsysbench.simai_bench.runner.subprocess.run", return_value=completed) as run_mock:
            result = VidurRunner().run(task, {}, {})

        self.assertFalse(result.success)
        passed_env = run_mock.call_args.kwargs["env"]
        self.assertEqual(passed_env["CUDA_HOME"], "/usr/local/cuda-12.9")
        self.assertEqual(passed_env["TORCH_CUDA_ARCH_LIST"], "8.9")

    def test_detects_aicb_default_fallback(self):
        error = detect_aicb_failure_or_default(
            "WARNING AICB data is empty, using default attention execution time",
            "",
        )

        self.assertIsNotNone(error)
        self.assertIn("AICB data is empty", error)

    def test_parse_vidur_output_finds_timestamped_metrics_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir) / "2026-07-02_22-54-03-985018"
            metrics_dir.mkdir()
            (metrics_dir / "request_metrics.csv").write_text(
                "\n".join(
                    [
                        "request_e2e_time,prefill_e2e_time,decode_time,request_num_decode_tokens,arrived_at,completed_at",
                        "0.2,0.05,0.12,4,0.0,0.2",
                    ]
                ),
                encoding="utf-8",
            )

            metrics = parse_vidur_output(tmpdir)

        self.assertEqual(metrics["num_requests"], 1.0)
        self.assertAlmostEqual(metrics["p99_e2e_ms"], 200.0)
        self.assertAlmostEqual(metrics["p99_ttft_ms"], 50.0)
        self.assertAlmostEqual(metrics["p99_tbt_ms"], 30.0)

    def test_goodput_applies_per_request_tbt_slo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "request_metrics.csv"
            metrics_path.write_text(
                "\n".join(
                    [
                        "request_e2e_time,prefill_e2e_time,decode_time,request_num_decode_tokens,arrived_at,completed_at",
                        "0.2,0.05,0.4,4,0.0,0.2",
                        "0.2,0.05,0.04,4,0.1,0.3",
                    ]
                ),
                encoding="utf-8",
            )

            metrics = parse_vidur_output(tmpdir, SLO(p99_tbt_ms=20.0))

        self.assertAlmostEqual(metrics["goodput_rps"], 1.0 / 0.3)


if __name__ == "__main__":
    unittest.main()
