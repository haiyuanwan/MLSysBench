from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    REPOSITORY_ROOT
    / "mlsysbench"
    / "simai_bench"
    / "vllm_scheduler_candidate_driver.py"
)


class VllmSchedulerCandidateDriverIntegrationTest(unittest.TestCase):
    runtime_python: Path
    scheduler_source: Path

    @classmethod
    def setUpClass(cls) -> None:
        candidates = []
        configured = os.environ.get("MLSYSBENCH_VLLM_PYTHON")
        if configured:
            candidates.append(Path(configured))
        candidates.append(REPOSITORY_ROOT / ".venv311" / "bin" / "python")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            probe = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    (
                        "import json, pathlib, vllm; "
                        "import vllm.v1.core.sched.scheduler as scheduler; "
                        "print(json.dumps({'version': vllm.__version__, "
                        "'scheduler': str(pathlib.Path(scheduler.__file__).resolve())}))"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if probe.returncode != 0:
                continue
            try:
                runtime = json.loads(probe.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                continue
            scheduler_source = Path(runtime.get("scheduler", ""))
            if runtime.get("version") == "0.11.0" and scheduler_source.is_file():
                cls.runtime_python = candidate
                cls.scheduler_source = scheduler_source
                return
        raise unittest.SkipTest(
            "vLLM 0.11.0 runtime is unavailable; set MLSYSBENCH_VLLM_PYTHON"
        )

    def _run_session(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(
            prefix="mlsysbench-driver-test-"
        ) as root_value:
            root = Path(root_value)
            solution = root / "solution"
            candidate = solution / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
            candidate.parent.mkdir(parents=True)
            shutil.copy2(self.scheduler_source, candidate)

            runtime_config = root / "runtime.json"
            runtime_config.write_text(
                json.dumps(
                    {
                        "expected_vllm_version": "0.11.0",
                        "landlock": "off",
                        "scheduler_config": {
                            "max_num_batched_tokens": 4,
                            "max_num_seqs": 2,
                            "max_model_len": 32,
                            "enable_chunked_prefill": True,
                        },
                        "cache_config": {
                            "block_size": 16,
                            "num_gpu_blocks": 16,
                            "enable_prefix_caching": False,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            scratch = root / "scratch"
            scratch.mkdir(mode=0o700)
            environment = dict(os.environ)
            environment["TMPDIR"] = str(scratch)
            payload = "".join(
                json.dumps(command, sort_keys=True, separators=(",", ":")) + "\n"
                for command in commands
            )
            completed = subprocess.run(
                [
                    str(self.runtime_python),
                    "-I",
                    str(DRIVER),
                    "--solution-dir",
                    str(solution),
                    "--runtime-config",
                    str(runtime_config),
                ],
                input=payload,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            responses = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
            self.assertEqual(len(responses), len(commands), completed.stdout)
            return responses

    def test_empty_step_advances_time_and_completed_request_releases_kv(self) -> None:
        responses = self._run_session(
            [
                {"command": "init"},
                {"command": "arrive", "requests": []},
                {
                    "command": "arrive",
                    "now_ms": 0,
                    "requests": [
                        {
                            "request_id": "future",
                            "prompt_tokens": 5,
                            "output_tokens": 2,
                            "arrival_time_ms": 25,
                            "priority": 0,
                        }
                    ],
                },
                {"command": "step", "now_ms": 25},
                {
                    "command": "arrive",
                    "now_ms": 24,
                    "requests": [
                        {
                            "request_id": "future",
                            "prompt_tokens": 5,
                            "output_tokens": 2,
                            "arrival_time_ms": 25,
                            "priority": 0,
                        }
                    ],
                },
                {
                    "command": "arrive",
                    "now_ms": 25,
                    "requests": [
                        {
                            "request_id": "future",
                            "prompt_tokens": 5,
                            "output_tokens": 2,
                            "arrival_time_ms": 25,
                            "priority": 0,
                        }
                    ],
                },
                {"command": "step", "now_ms": 25},
                {"command": "step", "now_ms": 26},
                {"command": "step", "now_ms": 27},
                {"command": "step", "now_ms": 28},
                {"command": "close"},
            ]
        )

        (
            init,
            missing_now,
            rejected,
            idle,
            non_monotonic,
            arrived,
            partial,
            first_token,
            finished,
            idle_after,
            close,
        ) = responses
        self.assertTrue(init["ok"])
        self.assertEqual(init["protocol_version"], 1)
        self.assertFalse(missing_now["ok"])
        self.assertIn("arrive now_ms must be a number", missing_now["error"])
        self.assertFalse(rejected["ok"])
        self.assertIn("has not arrived", rejected["error"])

        self.assertTrue(idle["ok"])
        self.assertEqual(idle["now_ms"], 25.0)
        self.assertEqual(idle["scheduled"], [])
        self.assertTrue(idle["leak_free"])
        self.assertFalse(non_monotonic["ok"])
        self.assertIn("must be monotonic", non_monotonic["error"])
        self.assertTrue(arrived["ok"])

        for response in (partial, first_token, finished):
            self.assertTrue(response["ok"])
            self.assertEqual(
                response["total_scheduled_tokens"],
                sum(row["num_scheduled_tokens"] for row in response["scheduled"]),
            )
            for row in response["scheduled"]:
                self.assertEqual(
                    row["num_computed_after"],
                    row["num_computed_before"] + row["num_scheduled_tokens"],
                )
                self.assertEqual(
                    row["num_computed_after_update"], row["num_computed_after"]
                )
                self.assertEqual(
                    row["output_tokens_after"],
                    row["output_tokens_before"] + row["emitted_tokens"],
                )
                self.assertLessEqual(
                    row["num_computed_after"], row["known_tokens_before"]
                )
            blocks = response["kv_cache_blocks"]
            expected_usage = blocks["used_blocks"] / blocks["capacity_blocks"]
            self.assertAlmostEqual(response["kv_cache_usage"], expected_usage)

        self.assertEqual(partial["scheduled"][0]["emitted_tokens"], 0)
        self.assertEqual(first_token["scheduled"][0]["emitted_tokens"], 1)
        self.assertEqual(finished["scheduled"][0]["emitted_tokens"], 1)
        self.assertEqual(finished["finished_request_ids"], ["future"])
        self.assertGreater(finished["kv_cache_blocks_after_schedule"]["used_blocks"], 0)
        self.assertEqual(finished["kv_cache_usage"], 0.0)
        self.assertEqual(finished["kv_cache_blocks"]["used_blocks"], 0)
        self.assertEqual(
            finished["kv_cache_blocks"]["free_blocks"],
            finished["kv_cache_blocks"]["capacity_blocks"],
        )
        self.assertEqual(finished["live_request_ids"], [])
        self.assertTrue(finished["quiescent"])
        self.assertTrue(finished["kv_cache_released"])
        self.assertTrue(finished["leak_free"])

        self.assertEqual(idle_after["scheduled"], [])
        self.assertEqual(idle_after["now_ms"], 28.0)
        self.assertTrue(idle_after["leak_free"])
        self.assertTrue(close["ok"])
        self.assertTrue(close["initialized"])
        self.assertEqual(close["kv_cache_usage"], 0.0)
        self.assertEqual(close["kv_cache_blocks"]["used_blocks"], 0)
        self.assertEqual(close["live_request_ids"], [])
        self.assertTrue(close["leak_free"])

    def test_arrive_syncs_time_after_batch_and_close_reports_live_kv(self) -> None:
        responses = self._run_session(
            [
                {"command": "init"},
                {
                    "command": "arrive",
                    "now_ms": 0,
                    "requests": [
                        {
                            "request_id": "live",
                            "prompt_tokens": 5,
                            "output_tokens": 2,
                            "arrival_time_ms": 0,
                            "priority": 0,
                        }
                    ],
                },
                {"command": "step", "now_ms": 0},
                {
                    "command": "arrive",
                    "now_ms": 52.5,
                    "requests": [
                        {
                            "request_id": "during",
                            "prompt_tokens": 2,
                            "output_tokens": 1,
                            "arrival_time_ms": 20,
                            "priority": 0,
                        }
                    ],
                },
                {"command": "close"},
            ]
        )
        arrived_during_batch = responses[-2]
        close = responses[-1]
        self.assertTrue(arrived_during_batch["ok"])
        self.assertEqual(arrived_during_batch["now_ms"], 52.5)
        self.assertEqual(arrived_during_batch["accepted_requests"], 1)
        self.assertTrue(close["ok"])
        self.assertEqual(close["live_request_ids"], ["during", "live"])
        self.assertEqual(close["running_requests"], 1)
        self.assertEqual(close["waiting_requests"], 1)
        self.assertGreater(close["kv_cache_usage"], 0.0)
        self.assertGreater(close["kv_cache_blocks"]["used_blocks"], 0)
        self.assertFalse(close["quiescent"])
        self.assertFalse(close["kv_cache_released"])
        self.assertFalse(close["leak_free"])


if __name__ == "__main__":
    unittest.main()
