"""Baseline-relative scoring for SimAI benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mlsysbench.simai_bench.schema import Objective, SLO


@dataclass(frozen=True)
class ScoreBreakdown:
    valid: bool
    score: float
    ratio: float
    failures: list[str]


def score_metrics(
    baseline_metrics: dict[str, float],
    agent_metrics: dict[str, float],
    objective: Objective,
    slo: SLO,
    runner_success: bool,
) -> ScoreBreakdown:
    failures: list[str] = []
    if not runner_success:
        failures.append("runner_failed")

    baseline_value = baseline_metrics.get(objective.primary_metric)
    agent_value = agent_metrics.get(objective.primary_metric)
    if baseline_value is None:
        failures.append(f"missing baseline metric {objective.primary_metric}")
    if agent_value is None:
        failures.append(f"missing agent metric {objective.primary_metric}")

    slo_ok, slo_failures = slo.check(agent_metrics)
    failures.extend(slo_failures)

    valid = not failures and slo_ok
    if not valid or baseline_value is None or agent_value is None:
        return ScoreBreakdown(False, 0.0, 0.0, failures)

    if baseline_value <= 0 or agent_value <= 0:
        return ScoreBreakdown(False, 0.0, 0.0, ["objective metric must be positive"])

    if objective.direction == "maximize":
        ratio = agent_value / baseline_value
    else:
        ratio = baseline_value / agent_value

    score = max(0.0, min(ratio, objective.score_cap))
    return ScoreBreakdown(valid=True, score=score, ratio=ratio, failures=[])
