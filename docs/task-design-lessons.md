# Task Design Lessons from CLI-Agent Protocol Runs

## Scope

This note turns the first filesystem-capable agent runs into task-design
requirements. The four model runs are protocol smoke tests, not capability
estimates: each model was run once on one synthetic surface.

The evaluated sparse fixture used five development queries and a 900-second
wall-time limit. LongCat-2.0, GLM-5.2, Kimi-K2.7-Code, and DeepSeek-V4-Pro all
eventually submitted the same hidden-optimal configuration. They required 15
to 22 model calls and 67,750 to 193,081 total API tokens.

## What the Runs Exposed

1. **The fixture dominated the result.** The evaluated revision omitted the
   baseline signature and mapped only two of eighteen configurations. Every
   other configuration returned the same invalid metrics. This turned the task
   into combination guessing rather than performance engineering.
2. **Useful interactions appeared late.** All four agents spent their first
   queries on a baseline or partial changes. The first valid interaction was
   found only on query four or five.
3. **Budget conclusions were unstable.** LongCat failed under three CLI queries
   but succeeded under five. A single budget point cannot support a ranking.
4. **Public optimization and transfer are different abilities.** No model
   measured the public optimum, but all selected a lower-scoring public design
   that was optimal at the disclosed larger target scale.
5. **Efficiency varied substantially despite identical final scores.** Kimi
   finished in 44.65 seconds and 67,750 tokens; GLM took 263.32 seconds and
   193,081 tokens. Final score alone hides important agent behavior and cost.

The fixture is now baseline-consistent, has metrics for all eighteen legal
configurations, gives graded local feedback, and fails closed on unknown
signatures. With the same five-query Random protocol, the mean hidden ratio
over twenty seeds changed from 0.796875 on the sparse fixture (invalid runs
counted as zero) to 1.428125 on the dense fixture. That difference is a task
artifact, not a new search capability.

## Invariants for Every Task

Task publication should fail unless all applicable invariants pass:

- evaluating an empty change set reproduces the declared baseline metrics;
- every legal mock configuration has an explicit entry and unknown signatures
  fail closed;
- development and final workload files have different hashes and explicit
  seeds;
- immutable workload, model, and resource fields cannot be submitted;
- baseline, candidate, and oracle configurations satisfy the same quality and
  integrity checks;
- final evaluation is launched from fresh state after development feedback is
  disabled;
- simulator output directories are unique and stale metrics cannot be reused;
- task manifests record code revision, dirty state, workload hashes, model,
  token usage, wall time, query count, and actual isolation level.

Mock tasks are protocol tests only. Paper results must use calibrated simulator
or real-system runs, and should never treat a catch-all mock default as a
performance measurement.

## Task Matrix

The benchmark needs a matrix of task families rather than more variants of one
hand-authored scheduler surface.

| Family | Development condition | Final shift | Capability isolated |
|---|---|---|---|
| IID configuration | Same hardware and load class | New request seeds | Black-box optimization |
| Scale-up transfer | 8-32 GPUs | 128-1024 GPUs | Parallelism extrapolation |
| Network transfer | High-bandwidth fabric | Constrained scale-out fabric | Communication reasoning |
| Workload transfer | Balanced requests | Prefill-heavy or decode-heavy | Scheduler generalization |
| SLO robustness | One public load point | Several hidden load points | Robust design, not peak tuning |
| Diagnosis | Slow valid baseline plus symptoms | Held-out workload | Root-cause-directed repair |
| Open scheduler | Public tests and traces | Hidden correctness and load tests | Code modification and integrity |

Each scale-transfer mechanism needs counterexamples. For example, some tasks
should reward higher TP at the target scale, while network-bound tasks should
penalize it. Otherwise a model can learn a benchmark shortcut such as "choose
the largest TP for hidden scale" without using measurements.

## Feedback and Search Space

- Use dense, physically grounded response surfaces. Single-factor experiments
  should reveal direction while still leaving meaningful interactions.
- Represent conditional parameters explicitly. A Sarathi chunk size should not
  create three distinct vLLM configurations in Random/Grid accounting.
- Include memory, communication, utilization, and queueing observations when
  the corresponding mechanism is part of the task. Goodput alone is often
  underdetermined.
- Keep target deployment metadata public when scale transfer is the intended
  capability, but keep target workload samples, seeds, and measured surface
  private.
- Use multiple final seeds or load points and aggregate them. One hidden point
  is too easy to overfit accidentally.

## Scoring and Gates

Retain raw baseline-relative ratios for interpretability, but add normalized
regret against an evaluator-owned oracle:

```text
progress = (candidate - baseline) / (oracle - baseline)
```

Apply the direction of the objective and report unclipped values separately.
The aggregate score should be zero unless validity, resource, SLO, quality, and
integrity gates all pass. Open-code tasks need deterministic correctness tests
and a quality floor before performance is considered.

Report at least these trajectory measures in addition to final score:

- unique configurations and duplicate-query rate;
- first improvement and first valid interaction query;
- best-development score and final transfer gap;
- whether the best measured state was retained or rolled back;
- number of parameters changed per experiment;
- API calls, tokens, model latency, evaluator time, and estimated cost.

## Experimental Protocol

InferenceBench's strongest reusable idea is matched resource accounting. Every
agent, Random, Grid, Optuna TPE, and SMAC3 run should share both a simulator
query cap and a wall-clock cap. Report budget curves, not one endpoint: a useful
initial set is 3, 5, 10, and 20 development queries.

Use repeated runs for stochastic agents and search methods. Five repetitions
per model-task-budget cell is a minimum for pilot work; paper claims should use
enough repetitions to report confidence intervals and paired task-level tests.
The structured hypothesis/measure/retain prompt should be an ablation against a
minimal mission prompt, because the harness prompt itself can create the
behavior being measured.

The implementation borrows these mechanisms from
[InferenceBench](https://github.com/aisa-group/InferenceBench): real CLI-agent
execution, wall-clock accounting, clean final relaunch, held-out seeds, matched
HPO baselines, and trajectory analysis. It intentionally does not copy the
monolithic shell orchestration or treat a read-only mount as hidden data.
Inspection of InferenceBench commit `24cdf88` confirms that its Codex agents run
through native `codex exec`/`resume` inside Apptainer; the repository contains
no CC Switch integration. MLSysBench therefore treats Codex CLI as the
canonical scaffold and documents CC Switch strictly as its SiliconFlow protocol
bridge.

## Next Implementation Order

The repository-wide implementation order is defined by the
[benchmark protocol](benchmark-protocol.md) and [roadmap](status-and-roadmap.md):

1. define the four canonical scenario families;
2. create genuinely different development/final pairs;
3. establish the four-tier baseline ladder;
4. implement repeated run-matrix execution and aggregation;
5. move final replay into a fresh container or ephemeral worker;
6. add starting-point ablations;
7. calibrate simulator decisions against real hardware;
8. open the server/code modification track only after the preceding gates pass.

Conditional actions, canonical configuration hashing, stronger validation,
portable runtime profiles, optional-search adapters, and CI are cross-cutting
foundations. They should be added when the stage that first depends on them is
implemented rather than treated as a competing roadmap.
