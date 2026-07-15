# Status and Roadmap

This document records the implementation boundary verified on 2026-07-15 and
maps it to the ordered design in [Benchmark protocol](benchmark-protocol.md).
“Implemented” means behavior exists in code and is covered by a local test or
task validator.

## Verification snapshot

- `python3 -m unittest discover -s tests -v`: 50 tests passed at the last update.
- `python3 -m compileall -q mlsysbench tests`: passed.
- Four canonical scenario fixtures validate without warnings: `prefill_heavy`,
  `decode_heavy`, `high_load`, and `balanced`. Their development/final mock
  surfaces are complete and fail closed.
- `scale_up/mock_scale_transfer`: 18/18 explicit development configurations,
  18/18 final configurations, and valid baseline replay in both phases.
- `simai_gym/l1_scheduler_choice`: validation failed. It maps 1/36 legal
  configurations, uses a catch-all default, lacks an explicit baseline
  signature, and its declared baseline does not match or pass the SLO.
- Both Qwen3-Next Vidur+AICB tasks pass structural validation when real baseline
  replay is skipped, but neither has a separate development phase.

The real summaries were generated on the local RTX5880-Ada-48Q host. Raw
`runs/` paths are ignored, so the checked-in JSON files are evidence summaries,
not complete portable reproduction bundles.

## Capability matrix

| Capability | State | Evidence or boundary |
|---|---|---|
| Task/action parsing and resource gates | Implemented | Schema, action, and GPU-budget tests |
| Development/final split | Partial | Four synthetic fixtures; absent from real tasks |
| SLO and baseline-relative scoring | Implemented | Metric/scoring tests |
| One-shot and multi-step API agents | Implemented | Dry-run and mocked-client tests |
| Filesystem-capable CLI agents | Implemented on Linux | Benchmark mode requires pinned Codex CLI + CC Switch and bwrap; direct chat/custom are debug-only |
| Grid and random search | Implemented | Finite discretized candidate space |
| TPE and SMAC3 | Optional adapters | Default suite tests only missing-dependency behavior |
| Vidur+AICB runner | Experimental | Local smoke and 32-request summaries |
| Scenario-family metadata | Implemented | Required schema, objective/profile validation, public agent context, four fixtures |
| Bounded scheduler-code protocol | Prototype implemented | Schema v2 source bundle, editable allowlist, clean starter reconstruction, deterministic hidden evaluator |
| Patch/policy/multi-fidelity protocol | Fixture implemented | Schema-v3 provenance, multi-profile policy simulator, robust/worst/fairness metrics, cost-accounted fidelities |
| Four-tier baseline ladder | Not implemented | Individual baselines/search exist, not tiered per task |
| Run matrix and result aggregation | Partial | `aggregate-results` covers failures, bootstrap interval, transfer/process/cost metrics; declarative run-matrix execution remains pending |
| Containerized final replay | Not implemented | Landlock/process separation only |
| Starting-point ablations | Not implemented | No task/run dimension |
| Simulator-to-hardware calibration | Analysis implemented, evidence missing | `analyze-calibration` reports error, Spearman/Kendall, top-k, pairwise agreement and hardware CV; no real paired bundle yet |
| Open server/code track | Partial protocol only | Synthetic patch/policy fixtures exist; no arbitrary upstream patch, model-quality gate, or real-hardware score |
| Continuous integration | Not implemented | No CI workflow checked in |

## Ordered implementation roadmap

Stages are sequential because each consumes artifacts from the previous one.
Stages 1-6 build the benchmark protocol; publication claims based on simulation
remain blocked until stage 7 calibration succeeds.

### Stage 1: canonical scenario families

**Status: implemented for protocol fixtures.** Real-backend scenario coverage
and publication evidence remain Stage 2 and Stage 7 work.

Define `prefill_heavy`, `decode_heavy`, `high_load`, and `balanced` as
schema-level metadata. Specify primary/secondary metrics, workload profiles,
legal transfer mechanisms, and counterexamples for every family.

Deliverables:

- versioned scenario metadata schema;
- task validator checks for family/profile consistency;
- one dependency-free dense fixture per family;
- removal or explicit archival of the invalid L1 scheduler prototype.

Exit criteria: all four fixtures replay valid baselines, fail closed, and pass
task validation.

### Stage 2: paired development/final tasks

Turn each scenario family into a small public development environment and a
shifted final environment. At least one task per family uses scale shift; the
others cover workload, load-profile, network, or hardware shift.

Deliverables:

- distinct public/final workload hashes and seeds;
- development and final baseline provenance;
- immutable final overrides and resource budgets;
- multi-profile final evaluation for `high_load`;
- portable runtime profiles replacing absolute host paths.

Exit criteria: no publication candidate falls back to the same development and
final specification.

### Stage 3: baseline ladder

Add named `naive`, `framework_default`, `expert_recipe`, and `matched_search`
tiers to every candidate task. Search results distinguish restricted and
full-space search.

Deliverables:

- tiered baseline manifests and replay commands;
- Random, Grid where feasible, TPE, and SMAC3 under matched budgets;
- expert recipe provenance;
- common accounting for query, time, GPU/simulator, token, and monetary cost.

Exit criteria: every task report compares agents against all applicable tiers
and identifies the score denominator explicitly.

### Stage 4: run matrix and aggregation

**Status: aggregation implemented; declarative matrix execution pending.**

Implement declarative execution over task, model, scaffold, starting point,
budget, seed, and repeat.

Deliverables:

- `run-matrix` orchestration with resumable, immutable cell manifests;
- `aggregate-results` with uncertainty and paired comparisons;
- reliability, gate, behavioral, generalization, and cost metrics;
- machine-readable result schema and leaderboard ingestion checks.

Exit criteria: a synthetic full matrix can be regenerated from one manifest,
and failed runs appear explicitly in aggregates.

### Stage 5: fresh final replay

Replace same-host Landlock-only final evaluation with a supervised fresh
container or ephemeral worker.

Deliverables:

- agent process-group termination;
- read-only evaluator bundle and private final inputs;
- restricted network and declared resource limits;
- final-artifact allowlist and hashes;
- container image/runtime digest and reachability checks.

Exit criteria: final scores reproduce after all development processes and
caches are removed, and private files are never mounted in the agent workspace.

### Stage 6: starting-point ablations

Run each canonical scenario from `from_scratch`, `framework_default`, and
`expert_template` starting points with otherwise matched conditions.

Deliverables:

- starting point in task/run manifests;
- prompt and scaffold ablations;
- regression metric for damaging a strong starting configuration;
- framework/action diversity analysis.

Exit criteria: reports separate model capability, agent scaffold contribution,
and prior configuration quality.

### Stage 7: simulator calibration

**Status: analysis command implemented; real paired evidence absent.**

Compare identical configurations in simulation and on hardware across scenario
families, schedulers, profiles, sequence lengths, parallelism choices, and at
least two hardware environments.

Deliverables:

- repeated real measurements with variance;
- absolute/relative error, rank correlation, top-k overlap, and pairwise
  decision agreement;
- calibrated regions and explicit unsupported regions;
- task-level decision on whether simulation may support model ranking.

Exit criteria: every simulator-backed publication task has a documented
calibration result for its relevant decision region. A backend smoke run is not
sufficient.

### Stage 8: open server/code track

Only after stages 1-7 are stable, allow open framework choice, server launchers,
scheduler or routing patches, kernels, quantization, and memory-policy changes.

Deliverables:

- standardized final server/patch artifact contract;
- deterministic correctness tests;
- hidden baseline-relative quality gate;
- deterministic integrity checks plus a validated semantic judge if needed;
- clean container relaunch and real-hardware scoring.

Exit criteria: an artifact cannot score through model substitution, evaluator
tampering, external inference, pre-generated responses, stale processes, or
quality regression.

## Cross-cutting work

These items run throughout all stages instead of forming a separate stage:

- repair or retire invalid fixtures rather than filling them with arbitrary
  synthetic defaults;
- move machine-specific paths and CUDA settings into runtime profiles;
- add CI, formatting/linting, type checks, coverage, package smoke tests, and
  internal-link/result-provenance checks;
- version task/result schemas and define migrations;
- document Codex/CC Switch update and verification policy;
- retain raw ratios, failure causes, and manifests even when presenting an
  aggregate score.

## Current claim boundary

Today the repository can claim that the synthetic scale-transfer protocol,
agent/search interfaces, budgets, bounded scheduler source submission, and
clean replay logic execute as tested. It
can also claim that the local Vidur+AICB backend has produced real summaries on
one hardware class.

It cannot yet claim calibrated simulator fidelity, model rankings, robust
cross-scale generalization, production-grade isolation, arbitrary patch
security, or open-ended system optimization capability.
