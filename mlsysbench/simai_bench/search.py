"""Matched-budget non-agent search baselines."""

from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mlsysbench.simai_bench.evaluator import EvaluationResult, evaluate_changes
from mlsysbench.simai_bench.io import ConfigError, write_json
from mlsysbench.simai_bench.schema import ActionSpec, TaskSpec


@dataclass(frozen=True)
class SearchResult:
    task_id: str
    method: str
    budget: int
    evaluations: int
    best_development_evaluation: EvaluationResult | None
    best_evaluation: EvaluationResult | None
    trajectory_path: Path
    wall_time_seconds: float | None = None
    elapsed_seconds: float = 0.0
    stopped_reason: str | None = None
    final_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "budget": self.budget,
            "evaluations": self.evaluations,
            "best_development_evaluation": (
                self.best_development_evaluation.to_dict()
                if self.best_development_evaluation is not None
                else None
            ),
            "best_evaluation": (
                self.best_evaluation.to_dict() if self.best_evaluation is not None else None
            ),
            "trajectory_path": str(self.trajectory_path),
            "wall_time_seconds": self.wall_time_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "stopped_reason": self.stopped_reason,
            "final_error": self.final_error,
        }


class _SearchSession:
    def __init__(
        self,
        *,
        task: TaskSpec,
        baseline_config: dict[str, Any],
        allowed_actions: dict[str, ActionSpec],
        output_dir: Path,
        query_budget: int,
        wall_time_seconds: float | None,
    ) -> None:
        self.task = task
        self.baseline_config = baseline_config
        self.allowed_actions = allowed_actions
        self.output_dir = output_dir
        self.query_budget = query_budget
        self.wall_time_seconds = wall_time_seconds
        self.started = time.monotonic()
        self.deadline = (
            self.started + wall_time_seconds if wall_time_seconds is not None else None
        )
        self.trajectory: list[dict[str, Any]] = []
        self.best: EvaluationResult | None = None
        self.best_changes: dict[str, Any] | None = None

    def can_evaluate(self) -> bool:
        if len(self.trajectory) >= self.query_budget:
            return False
        return self.deadline is None or time.monotonic() < self.deadline

    def evaluate(self, changes: dict[str, Any]) -> float:
        if not self.can_evaluate():
            return 0.0
        changes = {key: _plain_value(value) for key, value in changes.items()}
        error: str | None = None
        started = time.monotonic()
        try:
            evaluation = evaluate_changes(
                self.task,
                self.baseline_config,
                self.allowed_actions,
                changes,
                phase="development",
            )
        except Exception as exc:  # noqa: BLE001 - invalid candidates are search outcomes.
            evaluation = None
            error = str(exc)

        is_best = False
        if evaluation is not None and evaluation.valid:
            if self.best is None or evaluation.score > self.best.score:
                self.best = evaluation
                self.best_changes = changes
                is_best = True
        self.trajectory.append(
            {
                "step": len(self.trajectory) + 1,
                "elapsed_seconds": round(time.monotonic() - self.started, 6),
                "evaluation_seconds": round(time.monotonic() - started, 6),
                "changes": changes,
                "error": error,
                "evaluation": evaluation.to_dict() if evaluation is not None else None,
                "is_best_so_far": is_best,
            }
        )
        write_json(self.output_dir / "trajectory.json", self.trajectory)
        if evaluation is None or not evaluation.valid:
            return 0.0
        return evaluation.score

    def elapsed(self) -> float:
        return time.monotonic() - self.started


def run_search(
    task_dir: str | Path,
    output_dir: str | Path,
    method: str,
    budget: int,
    seed: int = 0,
    wall_time_seconds: float | None = None,
) -> SearchResult:
    if method not in {"grid", "random", "tpe", "smac"}:
        raise ConfigError("search method must be grid, random, tpe, or smac")
    if budget <= 0:
        raise ConfigError("search budget must be positive")
    if wall_time_seconds is not None and wall_time_seconds <= 0:
        raise ConfigError("wall_time_seconds must be positive")

    task = TaskSpec.load(task_dir)
    baseline_config = task.load_baseline_config()
    allowed_actions = task.load_allowed_actions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _SearchSession(
        task=task,
        baseline_config=baseline_config,
        allowed_actions=allowed_actions,
        output_dir=output_dir,
        query_budget=budget,
        wall_time_seconds=wall_time_seconds,
    )

    if method in {"grid", "random"}:
        candidates = list(_candidate_configs(allowed_actions, baseline_config))
        if method == "random":
            random.Random(seed).shuffle(candidates)
        for changes in candidates:
            if not session.can_evaluate():
                break
            session.evaluate(changes)
    elif method == "tpe":
        _run_tpe(session, seed)
    else:
        _run_smac(session, seed)

    optimization_elapsed = round(session.elapsed(), 6)
    stopped_reason = _stopped_reason(session, method)

    final_evaluation: EvaluationResult | None = None
    final_error: str | None = None
    if session.best is not None and session.best_changes is not None:
        write_json(output_dir / "best_development_result.json", session.best.to_dict())
        try:
            final_evaluation = evaluate_changes(
                task,
                baseline_config,
                allowed_actions,
                session.best_changes,
                phase="final",
            )
        except Exception as exc:  # noqa: BLE001 - report hidden-evaluation failure.
            final_evaluation = None
            final_error = str(exc)
            write_json(output_dir / "final_error.json", {"error": final_error})
        if final_evaluation is not None:
            write_json(output_dir / "final_result.json", final_evaluation.to_dict())

    write_json(
        output_dir / "search_manifest.json",
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "method": method,
            "seed": seed,
            "query_budget": budget,
            "wall_time_seconds": wall_time_seconds,
            "evaluations": len(session.trajectory),
            "optimization_elapsed_seconds": optimization_elapsed,
            "stopped_reason": stopped_reason,
            "best_development_score": session.best.score if session.best is not None else None,
            "final_score": final_evaluation.score if final_evaluation is not None else None,
            "final_valid": final_evaluation.valid if final_evaluation is not None else False,
            "final_error": final_error,
        },
    )

    return SearchResult(
        task_id=task.task_id,
        method=method,
        budget=budget,
        evaluations=len(session.trajectory),
        best_development_evaluation=session.best,
        best_evaluation=final_evaluation,
        trajectory_path=output_dir / "trajectory.json",
        wall_time_seconds=wall_time_seconds,
        elapsed_seconds=optimization_elapsed,
        stopped_reason=stopped_reason,
        final_error=final_error,
    )


def _run_tpe(session: _SearchSession, seed: int) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ConfigError(
            "TPE search requires Optuna; install the hpo extra with "
            "`python3 -m pip install -e '.[hpo]'`"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        if not session.can_evaluate():
            raise optuna.TrialPruned("search budget exhausted")
        changes = {
            name: _suggest_optuna_value(trial, name, spec, session.baseline_config.get(name))
            for name, spec in session.allowed_actions.items()
        }
        return session.evaluate(changes)

    study.optimize(
        objective,
        n_trials=session.query_budget,
        timeout=session.wall_time_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )


def _suggest_optuna_value(
    trial: Any,
    name: str,
    spec: ActionSpec,
    baseline: Any,
) -> Any:
    if spec.choices is not None:
        return trial.suggest_categorical(name, list(spec.choices))
    if spec.type == "bool":
        return trial.suggest_categorical(name, [False, True])
    if spec.type == "int" and spec.minimum is not None and spec.maximum is not None:
        return trial.suggest_int(name, math.ceil(spec.minimum), math.floor(spec.maximum))
    if spec.type == "float" and spec.minimum is not None and spec.maximum is not None:
        return trial.suggest_float(name, spec.minimum, spec.maximum)
    values = _candidate_values(spec, baseline)
    if not values:
        raise ConfigError(f"Action {name} has no finite TPE search domain")
    return trial.suggest_categorical(name, values)


def _run_smac(session: _SearchSession, seed: int) -> None:
    try:
        from ConfigSpace import (
            CategoricalHyperparameter,
            ConfigurationSpace,
            UniformFloatHyperparameter,
            UniformIntegerHyperparameter,
        )
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as exc:
        raise ConfigError(
            "SMAC search requires SMAC3 and ConfigSpace; install the hpo extra with "
            "`python3 -m pip install -e '.[hpo]'`"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - dependency imports may inspect host process state.
        raise ConfigError(f"SMAC dependencies failed to initialize: {exc}") from exc

    configspace = ConfigurationSpace(seed=seed)
    hyperparameters = []
    for name, spec in session.allowed_actions.items():
        baseline = session.baseline_config.get(name)
        if spec.choices is not None:
            choices = list(spec.choices)
            default = baseline if baseline in choices else choices[0]
            hyperparameter = CategoricalHyperparameter(
                name,
                choices=choices,
                default_value=default,
            )
        elif spec.type == "bool":
            hyperparameter = CategoricalHyperparameter(
                name,
                choices=[False, True],
                default_value=bool(baseline) if baseline is not None else False,
            )
        elif spec.type == "int" and spec.minimum is not None and spec.maximum is not None:
            lower = math.ceil(spec.minimum)
            upper = math.floor(spec.maximum)
            default = int(baseline) if baseline is not None else None
            hyperparameter = UniformIntegerHyperparameter(
                name,
                lower=lower,
                upper=upper,
                default_value=default,
            )
        elif spec.type == "float" and spec.minimum is not None and spec.maximum is not None:
            default = float(baseline) if baseline is not None else None
            hyperparameter = UniformFloatHyperparameter(
                name,
                lower=spec.minimum,
                upper=spec.maximum,
                default_value=default,
            )
        else:
            choices = _candidate_values(spec, baseline)
            if not choices:
                raise ConfigError(f"Action {name} has no finite SMAC search domain")
            hyperparameter = CategoricalHyperparameter(
                name,
                choices=choices,
                default_value=baseline if baseline in choices else choices[0],
            )
        hyperparameters.append(hyperparameter)
    configspace.add(hyperparameters)

    scenario_options: dict[str, Any] = {
        "configspace": configspace,
        "name": f"mlsysbench_{session.task.task_id}",
        "output_directory": session.output_dir / "smac",
        "deterministic": True,
        "n_trials": session.query_budget,
        "use_default_config": True,
        "seed": seed,
        "n_workers": 1,
    }
    if session.wall_time_seconds is not None:
        scenario_options["walltime_limit"] = session.wall_time_seconds
    scenario = Scenario(**scenario_options)

    def target(config: Any, seed: int = 0) -> float:
        del seed
        if not session.can_evaluate():
            return 0.0
        score = session.evaluate(dict(config))
        return -score

    facade = HyperparameterOptimizationFacade(
        scenario,
        target,
        overwrite=True,
    )
    facade.optimize()


def _stopped_reason(session: _SearchSession, method: str) -> str:
    if len(session.trajectory) >= session.query_budget:
        return "query_budget"
    if session.deadline is not None and time.monotonic() >= session.deadline:
        return "wall_time"
    if method in {"grid", "random"}:
        return "search_space_exhausted"
    return "optimizer_stopped"


def _plain_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def _candidate_configs(
    allowed_actions: dict[str, ActionSpec],
    baseline_config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    names = list(allowed_actions)
    values = [
        _candidate_values(spec, baseline_config.get(name))
        for name, spec in allowed_actions.items()
    ]
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def _candidate_values(spec: ActionSpec, baseline: Any) -> list[Any]:
    if spec.choices is not None:
        return list(spec.choices)
    if spec.type == "bool":
        return [False, True]

    values: list[Any] = []
    if baseline is not None:
        values.append(baseline)
    if spec.minimum is not None:
        values.append(spec.minimum)
    if spec.minimum is not None and spec.maximum is not None:
        values.append((spec.minimum + spec.maximum) / 2)
    if spec.maximum is not None:
        values.append(spec.maximum)

    if spec.type == "int":
        values = [int(round(value)) for value in values]
    elif spec.type == "float":
        values = [float(value) for value in values]
    elif spec.type == "str" and baseline is not None:
        values = [str(baseline)]
    return list(dict.fromkeys(values))
