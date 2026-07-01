import math
import json
import tempfile
import unittest
from pathlib import Path

from mlsysbench.simai_bench.agent_runner import run_agent_once
from mlsysbench.simai_bench.actions import validate_changes
from mlsysbench.simai_bench.evaluator import evaluate_submission
from mlsysbench.simai_bench.io import ConfigError
from mlsysbench.simai_bench.model_client import DryRunClient
from mlsysbench.simai_bench.schema import TaskSpec


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


if __name__ == "__main__":
    unittest.main()
