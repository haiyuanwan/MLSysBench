from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_vllm_workload_hardware.py"
SPEC = importlib.util.spec_from_file_location(
    "test_replay_vllm_workload_hardware_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return {
        "schema_version": 1,
        "scenario_family": "balanced",
        "profiles": ["mixed_prompt_output", "mixed_concurrency"],
        "source": {"trace_sha256": "0" * 64},
        "cases": [
            {
                "name": "case_a",
                "expected_requests": 2,
                "slo": {"ttft_ms": 50.0, "tpot_ms": 25.0, "e2e_ms": 150.0},
                "requests": [
                    {
                        "request_id": "later",
                        "arrival_time_ms": 10.0,
                        "prompt_tokens": 12,
                        "output_tokens": 4,
                        "priority": 0,
                    },
                    {
                        "request_id": "first",
                        "arrival_time_ms": 0.0,
                        "prompt_tokens": 8,
                        "output_tokens": 2,
                        "priority": 1,
                    },
                ],
            }
        ],
    }


class WorkloadInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="hardware-replay-input-test-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, payload: dict, name: str = "workload.json") -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_inline_cases_sorts_arrivals_and_hashes_exact_bytes(self) -> None:
        path = self._write(_payload())

        workload = MODULE._load_workload(path, max_model_len=32)

        self.assertEqual(workload.path, path.resolve())
        self.assertEqual(
            workload.sha256,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(workload.scenario_family, "balanced")
        self.assertEqual(
            workload.profiles,
            ("mixed_prompt_output", "mixed_concurrency"),
        )
        self.assertEqual(len(workload.cases), 1)
        self.assertEqual(
            [request.request_id for request in workload.cases[0].requests],
            ["first", "later"],
        )
        self.assertEqual(workload.cases[0].requests[1].output_tokens, 4)

    def test_rejects_commitments_external_traces_duplicates_and_overlong_rows(self) -> None:
        commitment = {
            "schema_version": 1,
            "private_bundle": {
                "bundle_id": "private-v1",
                "workload_sha256": "0" * 64,
            },
        }
        trace_case = _payload()
        trace_case["cases"][0].pop("requests")
        trace_case["cases"][0]["trace_file"] = "private.csv"
        duplicate = _payload()
        duplicate["cases"][0]["requests"][1]["request_id"] = "later"
        overlong = _payload()
        overlong["cases"][0]["requests"][0]["prompt_tokens"] = 30

        fixtures = (
            (commitment, "commitment.json", "private_bundle"),
            (trace_case, "trace.json", "trace_file"),
            (duplicate, "duplicate.json", "unique"),
            (overlong, "overlong.json", "max_model_len"),
        )
        for payload, name, message in fixtures:
            with self.subTest(name=name):
                path = self._write(payload, name)
                with self.assertRaisesRegex(MODULE.ReplayError, message):
                    MODULE._load_workload(path, max_model_len=32)

    def test_rejects_duplicate_json_keys_and_non_finite_numbers(self) -> None:
        duplicate = self.root / "duplicate-keys.json"
        duplicate.write_text(
            '{"schema_version":1,"schema_version":1,"cases":[]}',
            encoding="utf-8",
        )
        non_finite = self.root / "nan.json"
        payload = json.dumps(_payload()).replace("10.0", "NaN", 1)
        non_finite.write_text(payload, encoding="utf-8")

        with self.assertRaisesRegex(MODULE.ReplayError, "duplicate JSON key"):
            MODULE._load_workload(duplicate, max_model_len=32)
        with self.assertRaisesRegex(MODULE.ReplayError, "non-standard JSON number"):
            MODULE._load_workload(non_finite, max_model_len=32)

    def test_rejects_symlinked_candidate_and_accepts_regular_scheduler_file(self) -> None:
        candidate = self.root / "scheduler.py"
        candidate.write_text("class Scheduler:\n    pass\n", encoding="utf-8")
        self.assertEqual(MODULE._validate_candidate_path(candidate), candidate.resolve())

        symlink = self.root / "candidate" / "scheduler.py"
        symlink.parent.mkdir()
        symlink.symlink_to(candidate)
        with self.assertRaisesRegex(MODULE.ReplayError, "regular file"):
            MODULE._validate_candidate_path(symlink)

    def test_main_writes_invalid_json_before_any_gpu_import_on_input_failure(self) -> None:
        workload = self._write(_payload())
        output = self.root / "failed-report.json"
        runtime_import_state = {
            name: name in sys.modules for name in ("torch", "vllm")
        }

        return_code = MODULE.main(
            [
                "--model",
                str(self.root / "missing-model"),
                "--workload",
                str(workload),
                "--output",
                str(output),
                "--max-model-len",
                "32",
            ]
        )

        self.assertEqual(return_code, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(report["valid"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["type"], "FileNotFoundError")
        self.assertEqual(
            runtime_import_state,
            {name: name in sys.modules for name in ("torch", "vllm")},
        )


class MetricTest(unittest.TestCase):
    def _measurement(
        self,
        *,
        request_id: str = "r0",
        arrival: float = 10.0,
        first: float = 40.0,
        completion: float = 100.0,
        output_tokens: int = 4,
        slo: dict | None = None,
    ) -> dict:
        state = MODULE.RequestState(
            spec=MODULE.RequestSpec(
                request_id=request_id,
                arrival_time_ms=arrival,
                prompt_tokens=8,
                output_tokens=output_tokens,
                priority=0,
            ),
            internal_id=f"internal-{request_id}",
            enqueue_started_ms=arrival + 0.5,
            enqueue_completed_ms=arrival + 1.0,
            first_token_time_ms=first,
            completion_time_ms=completion,
            generated_tokens=output_tokens,
            finish_reason="length",
        )
        return MODULE._request_measurement(
            state,
            slo or {"ttft_ms": 30.0, "tpot_ms": 20.0, "e2e_ms": 90.0},
        )

    def test_request_metric_definitions_match_shadow_evaluator(self) -> None:
        measurement = self._measurement()

        self.assertEqual(measurement["ttft_ms"], 30.0)
        self.assertEqual(measurement["tpot_ms"], 20.0)
        self.assertEqual(measurement["e2e_ms"], 90.0)
        self.assertEqual(measurement["injection_lag_ms"], 1.0)
        self.assertTrue(measurement["slo_pass"])

        one_token = self._measurement(
            request_id="one",
            first=20.0,
            completion=20.0,
            output_tokens=1,
        )
        self.assertEqual(one_token["tpot_ms"], 0.0)

    def test_request_metric_fails_closed_on_wrong_token_count(self) -> None:
        state = MODULE.RequestState(
            spec=MODULE.RequestSpec("r0", 0.0, 8, 2, 0),
            internal_id="internal",
            enqueue_started_ms=0.0,
            enqueue_completed_ms=0.1,
            first_token_time_ms=1.0,
            completion_time_ms=2.0,
            generated_tokens=1,
        )
        with self.assertRaisesRegex(MODULE.ReplayError, "generated 1 tokens"):
            MODULE._request_measurement(
                state,
                {"ttft_ms": 10.0, "tpot_ms": 10.0, "e2e_ms": 10.0},
            )

    def test_repeat_and_case_aggregation_preserve_tail_and_goodput(self) -> None:
        first = self._measurement(request_id="r0")
        second = self._measurement(
            request_id="r1",
            arrival=0.0,
            first=100.0,
            completion=200.0,
            output_tokens=2,
            slo={"ttft_ms": 20.0, "tpot_ms": 20.0, "e2e_ms": 20.0},
        )
        repeat_metrics = MODULE._repeat_metrics(
            [first, second],
            duration_ms=200.0,
            scheduler_steps=3,
            step_latencies_ms=[1.0, 2.0, 4.0],
            max_active_requests=2,
        )
        self.assertEqual(repeat_metrics["throughput_rps"], 10.0)
        self.assertEqual(repeat_metrics["goodput_rps"], 5.0)
        self.assertEqual(repeat_metrics["request_slo_pass_rate"], 0.5)
        self.assertEqual(repeat_metrics["p99_e2e_ms"], 200.0)

        repeat = {
            "repeat_index": 0,
            "metrics": repeat_metrics,
            "requests": [first, second],
        }
        second_repeat = copy.deepcopy(repeat)
        second_repeat["repeat_index"] = 1
        aggregate = MODULE._aggregate_case_repeats([repeat, second_repeat])
        self.assertEqual(aggregate["repeats"], 2)
        self.assertEqual(aggregate["request_measurements"], 4)
        self.assertEqual(aggregate["goodput_rps"], 5.0)

        cases = [
            {"name": "a", "aggregate_metrics": aggregate},
            {"name": "b", "aggregate_metrics": aggregate},
        ]
        overall = MODULE._aggregate_cases(cases)
        self.assertEqual(overall["profile_count"], 2)
        self.assertEqual(overall["repeat_count"], 4)
        self.assertEqual(overall["request_measurements"], 8)
        self.assertAlmostEqual(overall["robust_goodput_rps"], 5.0)


class ArgumentTest(unittest.TestCase):
    def test_scheduler_knobs_are_exposed_without_importing_vllm(self) -> None:
        args = MODULE._parse_args(
            [
                "--model",
                "/model",
                "--workload",
                "/workload.json",
                "--output",
                "/report.json",
                "--max-num-batched-tokens",
                "256",
                "--max-num-seqs",
                "16",
                "--max-num-partial-prefills",
                "2",
                "--max-long-partial-prefills",
                "1",
                "--long-prefill-token-threshold",
                "128",
                "--scheduling-policy",
                "priority",
                "--no-enable-chunked-prefill",
            ]
        )
        self.assertEqual(args.max_num_batched_tokens, 256)
        self.assertEqual(args.max_num_seqs, 16)
        self.assertEqual(args.max_num_partial_prefills, 2)
        self.assertEqual(args.long_prefill_token_threshold, 128)
        self.assertEqual(args.scheduling_policy, "priority")
        self.assertFalse(args.enable_chunked_prefill)


if __name__ == "__main__":
    unittest.main()
