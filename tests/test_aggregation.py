import json
import tempfile
import unittest
from pathlib import Path

from mlsysbench.simai_bench.aggregation import aggregate_runs
from mlsysbench.simai_bench.run_matrix import plan_matrix


ROOT = Path(__file__).resolve().parents[1]
SCALE_TASK = ROOT / "tasks" / "scale_up" / "mock_scale_transfer"


class AggregationTest(unittest.TestCase):
    def test_complete_three_model_matrix_keeps_failures_and_uses_paired_bonferroni(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cells: list[tuple[str, dict[str, object]]] = []
            for repeat in range(6):
                for model in ("model_a", "model_b", "model_c"):
                    cell_id = f"{model}-{repeat}"
                    cells.append((cell_id, self._dimensions(model, repeat)))
            self._write_plan(root, cells)

            for cell_id, dimensions in cells:
                model = str(dimensions["model"])
                repeat = int(dimensions["repeat"])
                if model == "model_b" and repeat == 2:
                    self._write_cell(root, cell_id, dimensions, state="failed")
                    continue
                score = {"model_a": 0.80, "model_b": 0.70, "model_c": 0.79}[model]
                self._write_cell(root, cell_id, dimensions, state="completed", score=score)

            report = aggregate_runs(root, bootstrap_samples=500)

        self.assertEqual(report["summary"]["runs"], 18)
        self.assertEqual(report["by_model"]["model_b"]["runs"], 6)
        self.assertEqual(report["by_model"]["model_b"]["execution_failures"], 1)
        self.assertAlmostEqual(
            report["by_model"]["model_b"]["mean_score_failure_as_zero"],
            3.5 / 6,
        )
        self.assertTrue(report["pairing_integrity"]["complete"])

        comparisons = {
            (item["model_a"], item["model_b"]): item
            for item in report["paired_comparisons"]
        }
        self.assertEqual(len(comparisons), 3)
        self.assertAlmostEqual(
            comparisons[("model_a", "model_b")]["confidence_level"],
            1.0 - 0.05 / 3.0,
        )
        self.assertEqual(comparisons[("model_a", "model_b")]["decision"], "better")
        self.assertEqual(comparisons[("model_a", "model_b")]["better_model"], "model_a")
        self.assertEqual(comparisons[("model_a", "model_c")]["decision"], "equivalent")
        self.assertEqual(comparisons[("model_b", "model_c")]["decision"], "better")
        self.assertEqual(comparisons[("model_b", "model_c")]["better_model"], "model_c")

        failed = next(record for record in report["runs"] if record["cell_id"] == "model_b-2")
        self.assertEqual(failed["score"], 0.0)
        self.assertEqual(failed["seed"], 7)
        self.assertEqual(failed["repeat"], 2)
        self.assertEqual(
            set(failed["dimensions"]),
            {"task", "model", "scaffold", "starting_point", "budget", "seed", "repeat"},
        )

    def test_missing_and_duplicate_pairing_is_reported_and_forces_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cells = [
                ("a-0", self._dimensions("model_a", 0)),
                ("a-0-duplicate", self._dimensions("model_a", 0)),
                ("b-0", self._dimensions("model_b", 0)),
                ("a-1", self._dimensions("model_a", 1)),
                ("b-1-missing", self._dimensions("model_b", 1)),
            ]
            self._write_plan(root, cells)
            for cell_id, dimensions in cells[:-1]:
                self._write_cell(root, cell_id, dimensions, state="completed", score=0.5)

            report = aggregate_runs(root, bootstrap_samples=100)

        comparison = report["paired_comparisons"][0]
        self.assertEqual(comparison["decision"], "inconclusive")
        self.assertFalse(comparison["pairing_complete"])
        self.assertEqual(len(comparison["duplicate_blocks"]), 1)
        self.assertEqual(len(comparison["missing_blocks"]), 1)
        self.assertEqual(report["pairing_integrity"]["duplicate_comparison_blocks"], 1)
        self.assertEqual(report["pairing_integrity"]["missing_comparison_blocks"], 1)
        self.assertEqual(
            report["pairing_integrity"]["missing_planned_cells"],
            [{"matrix_id": "fixture", "cell_id": "b-1-missing"}],
        )

    def test_artifact_subdirectory_recovers_dimensions_from_ancestor_cell_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dimensions = self._dimensions("model_a", 4)
            artifact_dir = self._write_cell(
                root,
                "model-a-4",
                dimensions,
                state="completed",
                score=0.75,
            )

            report = aggregate_runs(artifact_dir, bootstrap_samples=20)

        self.assertEqual(report["summary"]["runs"], 1)
        self.assertEqual(report["runs"][0]["model"], "model_a")
        self.assertEqual(report["runs"][0]["seed"], 7)
        self.assertEqual(report["runs"][0]["repeat"], 4)

    def test_run_level_timeout_is_zero_even_when_matrix_process_exited_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = self._write_cell(
                root,
                "timed-out",
                self._dimensions("model_a", 0),
                state="completed",
                score=0.9,
            )
            manifest_path = artifact_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "timed_out"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = aggregate_runs(root, bootstrap_samples=20)

        self.assertEqual(report["runs"][0]["status"], "timed_out")
        self.assertTrue(report["runs"][0]["execution_failed"])
        self.assertEqual(report["runs"][0]["score"], 0.0)

    def test_matrix_plan_interleaves_models_within_each_non_model_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "matrix.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "matrix_id": "interleaved",
                        "tasks": [{"id": "scale", "path": str(SCALE_TASK)}],
                        "models": [
                            {"id": "model_a", "name": "provider/a"},
                            {"id": "model_b", "name": "provider/b"},
                            {"id": "model_c", "name": "provider/c"},
                        ],
                        "scaffolds": [
                            {
                                "id": "same_agent",
                                "kind": "cli_agent",
                                "agent_profile": "custom",
                                "agent_mode": "debug",
                                "isolation": "none",
                                "agent_command": "python3 fake_agent.py",
                            }
                        ],
                        "budgets": [
                            {"id": "q1", "max_queries": 1, "wall_time_seconds": 30}
                        ],
                        "seeds": [3, 5],
                        "repeats": 2,
                    }
                ),
                encoding="utf-8",
            )

            plan = plan_matrix(manifest_path)

        self.assertEqual(len(plan.cells), 12)
        for offset in range(0, len(plan.cells), 3):
            block = plan.cells[offset : offset + 3]
            self.assertEqual(
                [cell.model["id"] for cell in block],
                ["model_a", "model_b", "model_c"],
            )
            self.assertEqual(
                len(
                    {
                        (
                            cell.task_id,
                            cell.scaffold["id"],
                            cell.budget["id"],
                            cell.seed,
                            cell.repeat,
                        )
                        for cell in block
                    }
                ),
                1,
            )

    @staticmethod
    def _dimensions(model: str, repeat: int) -> dict[str, object]:
        return {
            "task": "task_1",
            "model": model,
            "scaffold": "scaffold_1",
            "starting_point": "framework_default",
            "budget": "budget_1",
            "seed": 7,
            "repeat": repeat,
        }

    @staticmethod
    def _write_plan(root: Path, cells: list[tuple[str, dict[str, object]]]) -> None:
        (root / "matrix_plan.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "matrix_id": "fixture",
                    "cell_count": len(cells),
                    "cells": [
                        {"cell_id": cell_id, "dimensions": dimensions}
                        for cell_id, dimensions in cells
                    ],
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_cell(
        root: Path,
        cell_id: str,
        dimensions: dict[str, object],
        *,
        state: str,
        score: float | None = None,
    ) -> Path:
        cell_dir = root / "cells" / cell_id
        artifact_dir = cell_dir / "attempts" / "1" / "artifacts"
        cell_dir.mkdir(parents=True)
        (cell_dir / "cell_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "matrix_id": "fixture",
                    "cell_id": cell_id,
                    "dimensions": dimensions,
                    "artifact_root": str(cell_dir / "attempts"),
                }
            ),
            encoding="utf-8",
        )
        (cell_dir / "cell_status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cell_id": cell_id,
                    "state": state,
                    "attempt": 1,
                    "artifact_dir": str(artifact_dir),
                    "exit_code": 0 if state == "completed" else 1,
                }
            ),
            encoding="utf-8",
        )
        if state == "completed":
            assert score is not None
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task_id": "internal_task_name",
                        "status": "completed",
                        "agent": {
                            "model": "provider/model-name",
                            "scaffold": "reported-scaffold",
                        },
                        "budgets": {"development_queries_used": 0},
                        "gates": {"overall": True},
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "final_result.json").write_text(
                json.dumps(
                    {
                        "valid": True,
                        "score": score,
                        "ratio": score,
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "development_trajectory.json").write_text("[]", encoding="utf-8")
        return artifact_dir


if __name__ == "__main__":
    unittest.main()
