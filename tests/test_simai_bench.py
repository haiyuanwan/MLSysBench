import math
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mlsysbench.simai_bench.agent_runner import run_agent_once
from mlsysbench.simai_bench.actions import validate_changes
from mlsysbench.simai_bench.evaluator import evaluate_submission
from mlsysbench.simai_bench.io import ConfigError
from mlsysbench.simai_bench.metrics import parse_vidur_output
from mlsysbench.simai_bench.model_client import DryRunClient
from mlsysbench.simai_bench.runner import VidurRunner, detect_aicb_failure_or_default
from mlsysbench.simai_bench.schema import RunnerSpec, TaskSpec


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "simai_gym" / "l1_scheduler_choice"
SUBMISSION = ROOT / "submissions" / "examples" / "sarathi_scheduler.json"


class SimAIBenchTest(unittest.TestCase):
    def test_example_submission_scores_against_baseline(self):
        result = evaluate_submission(TASK, SUBMISSION)

        self.assertTrue(result.valid)
        self.assertEqual(result.primary_metric, "goodput_rps")
        self.assertAlmostEqual(result.ratio, 68.0 / 42.0)
        self.assertAlmostEqual(result.score, math.log(68.0 / 42.0))
        self.assertEqual(result.failures, [])

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


if __name__ == "__main__":
    unittest.main()
