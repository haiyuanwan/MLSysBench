import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlsysbench.simai_bench.vllm_batch_time_model import (
    FEATURE_NAMES,
    PROFILE_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_vllm_batch_time_profile.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "validate_vllm_batch_time_profile", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load holdout validator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _descriptors() -> list[dict[str, int]]:
    return [
        {
            "batch_size": 1,
            "total_tokens": 32,
            "prefill_tokens": 32,
            "decode_tokens": 0,
            "context_tokens": 0,
            "max_context_tokens": 0,
        },
        {
            "batch_size": 1,
            "total_tokens": 64,
            "prefill_tokens": 64,
            "decode_tokens": 0,
            "context_tokens": 0,
            "max_context_tokens": 0,
        },
        {
            "batch_size": 1,
            "total_tokens": 1,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "context_tokens": 32,
            "max_context_tokens": 32,
        },
        {
            "batch_size": 2,
            "total_tokens": 2,
            "prefill_tokens": 0,
            "decode_tokens": 2,
            "context_tokens": 64,
            "max_context_tokens": 32,
        },
    ]


def _profile() -> dict:
    points = []
    for index, (descriptor, latency) in enumerate(
        zip(_descriptors(), (10.0, 20.0, 30.0, 40.0))
    ):
        points.append(
            {
                "point_id": f"point-{index}",
                **descriptor,
                "latency_ms": latency,
                "latency_stddev_ms": 0.0,
                "sample_count": 3,
            }
        )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "identity": {
            "profile_id": "holdout-test",
            "profile_revision": "r1",
            "device": {
                "vendor": "NVIDIA",
                "model": "RTX PRO 5880",
                "compute_capability": "8.9",
                "gpu_count": 1,
                "total_memory_bytes": 48 * 1024**3,
            },
            "runtime": {
                "driver_version": "575.57",
                "cuda_version": "12.8",
                "python_version": "3.12.8",
                "torch_version": "2.8.0",
                "vllm_revision": "v0.11.0",
                "attention_backend": "FLASH_ATTN",
                "dtype": "bfloat16",
                "enforce_eager": True,
            },
            "model": {
                "name": "test/model",
                "revision": "model-revision",
                "max_model_len": 2048,
                "tensor_parallel_size": 1,
            },
        },
        "measurement": {
            "latency_unit": "ms",
            "latency_statistic": "mean",
            "timer": "time.perf_counter+cuda_synchronize_before_after",
            "warmup_iterations": 1,
            "minimum_measured_iterations": 3,
            "clock_policy": "application clocks locked",
            "raw_artifact_sha256": "a" * 64,
        },
        "feature_order": list(FEATURE_NAMES),
        "training_ranges": {
            name: {
                "min": min(point[name] for point in points),
                "max": max(point[name] for point in points),
            }
            for name in FEATURE_NAMES
        },
        "interpolation": {
            "method": "inverse_distance_weighting_v1",
            "neighbors": 2,
            "distance_power": 2.0,
            "max_normalized_distance": 1.0,
            "max_relative_uncertainty": 1.0,
            "feature_weights": {name: 1.0 for name in FEATURE_NAMES},
            "feature_transforms": {
                name: (
                    "log1p"
                    if name in {"context_tokens", "max_context_tokens"}
                    else "linear"
                )
                for name in FEATURE_NAMES
            },
        },
        "points": points,
    }


def _holdout(*, include_ood: bool = False) -> dict:
    descriptor_latencies = list(zip(_descriptors(), (10.0, 20.0, 30.0, 40.0)))
    if include_ood:
        descriptor_latencies.append(
            (
                {
                    "batch_size": 1,
                    "total_tokens": 128,
                    "prefill_tokens": 128,
                    "decode_tokens": 0,
                    "context_tokens": 0,
                    "max_context_tokens": 0,
                },
                50.0,
            )
        )
    samples = []
    for case_index, (descriptor, latency) in enumerate(descriptor_latencies):
        for repeat in range(3):
            samples.append(
                {
                    "case": f"case-{case_index}",
                    "repeat": repeat,
                    **descriptor,
                    "latency_ms": latency,
                }
            )
    return {
        "schema_version": 1,
        "profile_kind": "vllm_executor_batch_latency",
        "created_at_utc": "2026-07-21T00:00:00Z",
        "runtime": {
            "vllm_version": "0.11.0",
            "torch_version": "2.8.0",
            "python_version": "3.12.8",
            "model_config_sha256": "1" * 64,
            "scheduler_sha256": "2" * 64,
            "dtype": "bfloat16",
            "load_format": "dummy",
            "enforce_eager": True,
            "prefix_caching": False,
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 32,
            "seed": 0,
        },
        "device": {
            "name": "RTX PRO 5880",
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": [8, 9],
            "logical_gpu_index": 0,
            "requested_host_gpu_index": 0,
        },
        "measurement": {
            "clock": "time.perf_counter",
            "cuda_synchronize_before_after": True,
            "warmup_case": "case-0",
            "repeats": 3,
            "case_count": len(descriptor_latencies),
            "raw_sample_count": len(samples),
        },
        "feature_names": list(FEATURE_NAMES),
        "points": [],
        "raw_samples": samples,
    }


class VllmBatchTimeHoldoutValidatorTest(unittest.TestCase):
    def _run_validator(self, directory: Path, holdout: dict):
        profile_path = directory / "profile.json"
        holdout_path = directory / "holdout.json"
        output_path = directory / "validation.json"
        profile_path.write_text(
            json.dumps(_profile(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        holdout_path.write_text(
            json.dumps(holdout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR_PATH),
                "--profile",
                str(profile_path),
                "--holdout-raw",
                str(holdout_path),
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return completed, json.loads(output_path.read_text(encoding="utf-8"))

    def test_passing_holdout_reports_errors_rank_metrics_and_gates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed, report = self._run_validator(Path(tmpdir), _holdout())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["coverage"], 1.0)
        self.assertEqual(report["summary"]["median_ape"], 0.0)
        self.assertEqual(report["summary"]["p90_ape"], 0.0)
        self.assertEqual(report["summary"]["p95_ape"], 0.0)
        self.assertEqual(report["summary"]["kendall_tau_b"], 1.0)
        self.assertEqual(report["summary"]["spearman_rho"], 1.0)
        self.assertTrue(all(gate["passed"] for gate in report["gates"].values()))
        self.assertTrue(all(point["status"] == "accepted" for point in report["points"]))
        self.assertTrue(
            all(
                "uncertainty_ms" in point and "normalized_distance" in point
                for point in report["points"]
            )
        )

    def test_rejection_fails_coverage_but_still_writes_complete_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed, report = self._run_validator(
                Path(tmpdir), _holdout(include_ood=True)
            )

        self.assertEqual(completed.returncode, 2)
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["coverage"], 0.8)
        self.assertFalse(report["gates"]["coverage"]["passed"])
        rejected = [point for point in report["points"] if point["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["error"]["type"], "OutOfDistributionError")
        self.assertIsNotNone(rejected[0]["normalized_distance"])

    def test_kendall_tau_b_accounts_for_ties(self):
        validator = _load_validator_module()
        details = validator._kendall_tau_b_details(
            [1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]
        )

        self.assertAlmostEqual(details["value"], 0.8)
        self.assertEqual(details["concordant"], 4)
        self.assertEqual(details["tied_actual_only"], 1)
        self.assertEqual(details["tied_predicted_only"], 1)
        self.assertEqual(
            validator._kendall_tau_b([1.0, 1.0, 2.0], [5.0, 5.0, 4.0]),
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
