import math
import unittest
from pathlib import Path

from mlsysbench.simai_bench.actions import validate_changes
from mlsysbench.simai_bench.evaluator import evaluate_submission
from mlsysbench.simai_bench.io import ConfigError
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


if __name__ == "__main__":
    unittest.main()
