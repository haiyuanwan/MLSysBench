import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from mlsysbench.simai_bench.vllm_batch_time_model import (
    FEATURE_NAMES,
    PROFILE_JSON_SCHEMA,
    PROFILE_SCHEMA_VERSION,
    BatchDescriptor,
    BatchTimeProfileIdentity,
    CalibratedBatchTimeModel,
    OutOfDistributionError,
    ProfileIdentityError,
    ProfileSchemaError,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_vllm_batch_time_profile.py"


def _feature_transforms() -> dict[str, str]:
    return {
        name: "log1p" if name in {"context_tokens", "max_context_tokens"} else "linear"
        for name in FEATURE_NAMES
    }


def _identity_payload() -> dict:
    return {
        "profile_id": "test-profile",
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
    }


def _strict_profile() -> dict:
    points = [
        {
            "point_id": "initial-prefill",
            "batch_size": 1,
            "total_tokens": 32,
            "prefill_tokens": 32,
            "decode_tokens": 0,
            "context_tokens": 0,
            "max_context_tokens": 0,
            "latency_ms": 10.0,
            "latency_stddev_ms": 1.0,
            "sample_count": 3,
        },
        {
            "point_id": "decode",
            "batch_size": 1,
            "total_tokens": 1,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "context_tokens": 32,
            "max_context_tokens": 32,
            "latency_ms": 2.0,
            "latency_stddev_ms": 0.1,
            "sample_count": 3,
        },
        {
            "point_id": "mixed",
            "batch_size": 2,
            "total_tokens": 33,
            "prefill_tokens": 32,
            "decode_tokens": 1,
            "context_tokens": 64,
            "max_context_tokens": 48,
            "latency_ms": 12.0,
            "latency_stddev_ms": 0.5,
            "sample_count": 3,
        },
    ]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "identity": _identity_payload(),
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
            "feature_transforms": _feature_transforms(),
        },
        "points": points,
    }


def _raw_profile() -> dict:
    descriptors = (
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
            "total_tokens": 1,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "context_tokens": 32,
            "max_context_tokens": 32,
        },
    )
    samples = []
    for case, descriptor, latencies in zip(
        ("prefill", "decode"), descriptors, ((10.0, 12.0, 14.0), (20.0, 22.0, 24.0))
    ):
        for repeat, latency in enumerate(latencies):
            samples.append(
                {
                    "case": case,
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
            "warmup_case": "prefill",
            "repeats": 3,
            "case_count": 2,
            "raw_sample_count": len(samples),
        },
        "feature_names": list(FEATURE_NAMES),
        "points": [],
        "raw_samples": samples,
    }


class BatchTimeModelTest(unittest.TestCase):
    def test_initial_prefill_allows_zero_context_but_preserves_invariants(self):
        descriptor = BatchDescriptor.from_mapping(
            {
                "batch_size": 1,
                "total_tokens": 32,
                "prefill_tokens": 32,
                "decode_tokens": 0,
                "context_tokens": 0,
                "max_context_tokens": 0,
            }
        )
        self.assertEqual(descriptor.context_tokens, 0)
        self.assertEqual(
            PROFILE_JSON_SCHEMA["properties"]["points"]["items"]["properties"][
                "context_tokens"
            ]["minimum"],
            0,
        )

        for invalid in (
            replace(descriptor, max_context_tokens=1),
            BatchDescriptor(1, 1, 0, 1, 0, 0),
        ):
            with self.assertRaises(ProfileSchemaError):
                invalid.validate()

    def test_strict_profile_schema_rejects_unknown_fields(self):
        profile = _strict_profile()
        identity = BatchTimeProfileIdentity.from_mapping(profile["identity"])
        CalibratedBatchTimeModel.from_dict(profile, expected_identity=identity)

        malformed = deepcopy(profile)
        malformed["points"][0]["unmodeled_hint"] = 1
        with self.assertRaisesRegex(ProfileSchemaError, "unknown unmodeled_hint"):
            CalibratedBatchTimeModel.from_dict(malformed, expected_identity=identity)

    def test_feature_transforms_are_explicit_and_restricted(self):
        profile = _strict_profile()
        identity = BatchTimeProfileIdentity.from_mapping(profile["identity"])
        self.assertIn(
            "feature_transforms",
            PROFILE_JSON_SCHEMA["properties"]["interpolation"]["required"],
        )

        missing = deepcopy(profile)
        missing["interpolation"].pop("feature_transforms")
        with self.assertRaisesRegex(ProfileSchemaError, "missing feature_transforms"):
            CalibratedBatchTimeModel.from_dict(missing, expected_identity=identity)

        invalid = deepcopy(profile)
        invalid["interpolation"]["feature_transforms"]["context_tokens"] = "sqrt"
        with self.assertRaisesRegex(ProfileSchemaError, "must be one of"):
            CalibratedBatchTimeModel.from_dict(invalid, expected_identity=identity)

        legacy = deepcopy(profile)
        legacy["schema_version"] = 1
        with self.assertRaisesRegex(ProfileSchemaError, "expected 2"):
            CalibratedBatchTimeModel.from_dict(legacy, expected_identity=identity)

    def test_log1p_distance_expands_low_context_and_ood_uses_raw_range(self):
        profile = _strict_profile()
        identity = BatchTimeProfileIdentity.from_mapping(profile["identity"])
        log_model = CalibratedBatchTimeModel.from_dict(
            profile, expected_identity=identity
        )
        query = BatchDescriptor(1, 32, 32, 0, 8, 8)
        prediction = log_model.predict(query)
        expected_log_distance = math.sqrt(
            (
                (math.log1p(8) / math.log1p(64)) ** 2
                + (math.log1p(8) / math.log1p(48)) ** 2
            )
            / len(FEATURE_NAMES)
        )
        self.assertAlmostEqual(prediction.normalized_distance, expected_log_distance)

        linear_profile = deepcopy(profile)
        linear_profile["interpolation"]["feature_transforms"] = {
            name: "linear" for name in FEATURE_NAMES
        }
        linear_model = CalibratedBatchTimeModel.from_dict(
            linear_profile, expected_identity=identity
        )
        self.assertGreater(
            prediction.normalized_distance,
            linear_model.predict(query).normalized_distance,
        )

        with self.assertRaises(OutOfDistributionError) as caught:
            log_model.predict(BatchDescriptor(1, 32, 32, 0, 65, 48))
        self.assertTrue(
            any(
                violation.startswith("context_tokens=65 ")
                for violation in caught.exception.diagnostics["range_violations"]
            )
        )

    def test_prediction_rejects_ood_and_profile_rejects_identity_mismatch(self):
        profile = _strict_profile()
        identity = BatchTimeProfileIdentity.from_mapping(profile["identity"])
        model = CalibratedBatchTimeModel.from_dict(profile, expected_identity=identity)

        with self.assertRaises(OutOfDistributionError) as caught:
            model.predict(BatchDescriptor(1, 64, 64, 0, 0, 0))
        self.assertTrue(caught.exception.diagnostics["range_violations"])

        with self.assertRaisesRegex(ProfileIdentityError, "cuda_version"):
            CalibratedBatchTimeModel.from_dict(
                profile,
                expected_identity=replace(identity, cuda_version="different"),
            )

    def test_builder_is_deterministic_and_computes_mean_sample_stddev(self):
        raw = _raw_profile()
        raw_text = json.dumps(raw, indent=2, sort_keys=True) + "\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            raw_path = directory / "raw.json"
            first_path = directory / "first.json"
            second_path = directory / "second.json"
            raw_path.write_text(raw_text, encoding="utf-8")
            common_args = [
                sys.executable,
                str(BUILDER),
                "--input",
                str(raw_path),
                "--profile-id",
                "profile-id",
                "--profile-revision",
                "r1",
                "--device-vendor",
                "NVIDIA",
                "--driver-version",
                "575.57",
                "--cuda-version",
                "12.8",
                "--vllm-revision",
                "v0.11.0",
                "--attention-backend",
                "FLASH_ATTN",
                "--model-name",
                "test/model",
                "--model-revision",
                "model-revision",
                "--clock-policy",
                "application clocks locked",
                "--neighbors",
                "2",
                "--distance-power",
                "2",
                "--max-normalized-distance",
                "1",
                "--max-relative-uncertainty",
                "1",
                "--feature-weights-json",
                json.dumps({name: 1 for name in FEATURE_NAMES}),
                "--feature-transforms-json",
                json.dumps(_feature_transforms()),
            ]
            for output_path in (first_path, second_path):
                subprocess.run(
                    [*common_args, "--output", str(output_path)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            profile = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(
                profile["measurement"]["raw_artifact_sha256"],
                hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                profile["training_ranges"]["context_tokens"], {"min": 0, "max": 32}
            )
            self.assertEqual(
                profile["interpolation"]["feature_transforms"],
                _feature_transforms(),
            )
            by_total_tokens = {point["total_tokens"]: point for point in profile["points"]}
            self.assertEqual(by_total_tokens[32]["latency_ms"], 12.0)
            self.assertEqual(by_total_tokens[32]["latency_stddev_ms"], 2.0)
            self.assertEqual(by_total_tokens[32]["sample_count"], 3)

            identity = BatchTimeProfileIdentity.from_mapping(profile["identity"])
            loaded = CalibratedBatchTimeModel.from_path(
                first_path, expected_identity=identity
            )
            descriptor = {
                name: by_total_tokens[32][name] for name in FEATURE_NAMES
            }
            self.assertEqual(loaded.predict(descriptor).latency_ms, 12.0)


if __name__ == "__main__":
    unittest.main()
