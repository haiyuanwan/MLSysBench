# Benchmark Protocol

## Positioning

MLSysBench evaluates whether AI agents can conduct disciplined
inference-systems experiments and transfer conclusions beyond the workloads,
models, hardware, and scales they directly evaluate. Artifacts may be measured
configurations, bounded online policies, or reproducible system patches.

The benchmark is complementary to
[InferenceBench](https://github.com/aisa-group/InferenceBench): InferenceBench
tests open-ended optimization of a real single-H100 inference server;
MLSysBench uses controlled simulation and real anchor runs to test transfer and
the allocation of experiment budget between cheap biased feedback and scarce
real measurements.

The benchmark targets four questions:

1. Can an agent diagnose poor TTFT, TBT, goodput, memory pressure, or
   communication overhead from measured observations?
2. Under matched budgets, does it outperform non-agent search and expert
   recipes?
3. Can it explore a smaller public deployment and generalize to a larger or
   shifted final deployment?
4. Does cheaper experimentation improve discipline, or merely enable more
   repeated and uncontrolled trials?
5. Can an agent choose informative high-fidelity probes and correct a biased
   simulator more effectively than matched multi-fidelity search?

Negative results are meaningful. Search dominating agents exposes an
experiment-management gap; agents helping primarily under distribution shift
identifies where systems knowledge adds value beyond black-box optimization.

## Canonical scenario families

Tasks are organized first by bottleneck, then by transfer mechanism. The first
release should implement four scenario families inspired by InferenceBench but
evaluated at multiple scales:

| Family | Primary pressure | Primary metric | Required workload profiles |
|---|---|---|---|
| `prefill_heavy` | Long input and prompt processing | TTFT or TTFT-constrained goodput | Short/long prompt mixture |
| `decode_heavy` | Long generation and KV-cache traffic | TBT/TPOT or decode goodput | Short/long output mixture |
| `high_load` | Queueing and admission pressure | Request throughput or goodput | Burst, Poisson, and constant-rate |
| `balanced` | Joint latency and throughput | Geometric mean of normalized components | Mixed prompt/output and concurrency |

Each family must contain counterexamples. For example, some scale shifts should
reward higher tensor parallelism while constrained-network shifts penalize it.
Otherwise models can learn one benchmark shortcut without interpreting
feedback.

Schema-v3 publication-oriented task metadata additionally records provenance:

```json
{
  "schema_version": 3,
  "track": "policy_transfer",
  "scenario": {
    "family": "high_load",
    "transfer": "scale_up",
    "starting_point": "framework_default",
    "profiles": ["burst", "poisson", "constant"]
  },
  "provenance": {
    "source_type": "upstream_pr",
    "source_url": "https://github.com/ORG/REPO/pull/NUMBER",
    "source_revision": "PARENT_COMMIT_SHA",
    "license": "Apache-2.0",
    "task_authors": ["curator"],
    "validators": ["upstream maintainer"],
    "publication_status": "pilot",
    "calibration_status": "partially_calibrated"
  }
}
```

These fields are required and validated by the current task schema. The
`starting_point` value is recorded now; executing it as a matched experimental
dimension remains Stage 6 work.

## Development and final phases

Every benchmark task defines two genuinely different phases:

1. **Development:** public workload, baseline, action space, simulator feedback,
   and fixed query/wall-time/token budgets.
2. **Final:** one submission evaluated on evaluator-owned seeds, load, hardware,
   network, or deployment scale.

The development deployment should be smaller or cheaper than final for
transfer tasks. A recommended first matrix is:

| Family | Development | Final shift |
|---|---|---|
| `prefill_heavy` | 8 GPUs, public prompt distribution | 128 GPUs, longer hidden prompts |
| `decode_heavy` | Small concurrency | Larger concurrency or slower interconnect |
| `high_load` | One public Poisson point | Burst + Poisson + constant-rate profiles |
| `balanced` | Single-node deployment | Multi-node hidden topology and workload mix |

Development and final workload files must differ in hash and seed. Final
`config_overrides` are applied after submitted changes. The final evaluator is
called exactly once after development ends. `final_changes` may differ from the
best public configuration so scale transfer measures reasoned extrapolation,
not just copying a public optimum.

Files under `hidden/` are local fixtures, not secrecy by themselves. Production
secrecy comes from a dedicated evaluation boundary that never mounts private
inputs into the agent workspace.

## Multi-fidelity development

A task may expose named development fidelities. Each declares a semantic
`kind` (`simulator`, `hardware_proxy`, or `real_hardware`), positive cost, and
optional query cap. All calls share `constraints.max_development_cost_units`.
The agent selects a fidelity on each development request; final evaluation does
not accept a fidelity selector.

`hardware_proxy` means a deliberately different simulator surface used to test
budget accounting and calibration behavior. It is never evidence of a physical
measurement. Manifests record fidelity names, kinds, costs, and counts.

Final workloads may contain multiple evaluator-owned cases. The built-in
policy fixture reports geometric-mean robust goodput, worst-profile goodput,
request SLO pass rate, tenant fairness, maximum tail latencies, and all raw
per-profile values. A zero-goodput profile remains zero and cannot be hidden by
aggregation.

## Provenance and evidence status

Schema-v3 tasks declare their source, revision, license, curators, validators,
contamination cutoff, publication status, and calibration status. Validation
enforces these rules:

- a hand-authored task can only be a `fixture`;
- an externally sourced task needs URL, revision, license, and a validator;
- a publication `candidate` cannot be `uncalibrated` or `proxy_only`.

The complete intake procedure, trace requirements, blind-task protocol, and
promotion checklist are in
[Task intake and publication gates](task-intake-and-publication-gates.md).

## Baseline ladder

Every publication task must define four baseline tiers:

| Tier | Purpose |
|---|---|
| `naive` | Correct, valid, deliberately untuned reference |
| `framework_default` | Default behavior of the selected simulator/framework |
| `expert_recipe` | Reproducible human-designed strong configuration |
| `matched_search` | Random, Grid where feasible, TPE, and SMAC3 under the same budget |

The declared score denominator must name its tier. Reports show all tiers; they
must not imply agent capability merely because an agent beats a weak naïve
baseline.

Search comparisons distinguish:

- **restricted search:** searches one named framework or action subset;
- **full-space search:** searches every action available to the agent;
- **oracle analysis:** evaluator-only best known result, never exposed during
  development;
- **agent-guided search:** agent proposes or narrows a space and an optimizer
  executes it.

All comparisons report query count, wall time, simulator/GPU time, token/API
cost, and number of distinct valid evaluations. Equal wall time alone is not a
sufficient fairness condition.

## Run matrix and aggregation

One run is protocol validation, not a model result. The target orchestration
layer expands a declarative matrix over:

```text
task x model x agent scaffold x starting point x budget x seed x repeat
```

Pilot work uses at least three independent repeats per cell; paper claims use
enough repeats for confidence intervals and paired task-level tests. Final
workloads contain multiple seeds or load points.

The implemented `run-matrix` command writes one immutable cell manifest plus a
separate mutable status record per cell, resumes completed cells, retains
failures, and rejects credentials embedded in matrix manifests. Starting point
is currently inherited from each task definition; matched starting-point
variants remain Stage 6 work. `aggregate-results` reports:

- mean, median, spread, and confidence interval;
- validity, SLO, quality, integrity, and reachability pass rates;
- capped and uncapped baseline-relative ratios;
- development-to-final generalization gap;
- best-public versus final-submitted gap;
- unique configurations and duplicate-query rate;
- query/time to first valid improvement;
- invalid, OOM, crash, and timeout rates;
- rollback/best-state retention behavior;
- framework, scheduler, and action diversity;
- token, simulator, wall-clock, GPU, and estimated monetary cost.

Aggregate task scores use geometric means of normalized, higher-is-better
components. Failure rate remains a separate first-class metric; failed runs are
never silently removed from aggregates.

## Canonical agent scaffold

InferenceBench installs each native CLI agent in its evaluation container. Its
Codex runs invoke `codex exec`, continue with `codex exec resume`, enforce the
remaining wall-clock budget, and record Codex as the scaffold being evaluated.
It does not use CC Switch.

MLSysBench follows the same scaffold principle. The canonical current matrix
uses a pinned Codex CLI for every evaluated model and records
`codex-cli+cc-switch` in each run manifest. CC Switch is an MLSysBench-specific
protocol bridge: it translates Codex Responses traffic to the SiliconFlow Chat
Completions endpoints used by LongCat, GLM, Kimi, and DeepSeek. It is not
credited as an InferenceBench component.

`run-cli-agent` has two modes:

- `benchmark`: requires the Codex CLI + CC Switch scaffold and bwrap process
  isolation;
- `debug`: permits the direct Chat Completions helper or a custom command, but
  its results are excluded from the canonical model matrix.

The manifest records the scaffold and mode. Future native Claude Code, Gemini,
or OpenCode tracks must be separate scaffold factors rather than silently
mixed with Codex results.

## Final replay and trust boundary

Development feedback and final scoring are different services. After the agent
exits, the harness must:

1. terminate the agent and its process group;
2. stop the development evaluator and invalidate its token;
3. create a fresh container or ephemeral worker;
4. copy only the declared final artifact and pinned runtime inputs;
5. mount evaluator code and final data read-only and outside the agent-visible
   workspace;
6. disable or restrict network egress;
7. execute one supervised final replay;
8. record hashes, image digest, resource limits, and reachability outcome.

Landlock remains useful defense in depth but is not the production boundary.
Deterministic checks, isolation, and hashes are primary. An agentic integrity
judge may inspect transcripts and artifacts for model substitution, external
inference, generated-answer replay, evaluator tampering, or other semantic
abuse, but it must be validated and must not replace deterministic controls.

## Starting-point ablations

Every canonical scenario supports three starting points:

| Starting point | What it measures |
|---|---|
| `from_scratch` | End-to-end system construction and framework choice |
| `framework_default` | Optimization from a working default |
| `expert_template` | Ability to improve or correctly retain a strong starting point |

Starting point is an experimental factor, not hidden task metadata. Prompts,
available tools, time budgets, and final gates remain otherwise identical.
This separates model capability from scaffold quality and exposes regressions
where an agent damages an already strong configuration.

## Gates and scoring

Workload fields listed in `constraints.immutable_fields` cannot be submitted.
Resource use is currently:

```text
GPU units = replicas * tensor_parallel_size * pipeline_parallel_size
```

The gate order is:

```text
schema -> immutable/resource -> runner/reachability -> correctness
       -> model quality -> SLO/custom metric gates -> integrity -> performance score
```

Configuration-only tasks with fixed model weights may mark correctness and
quality as not applicable. Any track that changes quantization, weights,
framework behavior, kernels, or scheduler code must define them. Quality floors
are baseline-relative and evaluated on final-only samples.

`metric_gates` provides deterministic minimum or maximum thresholds for
runner-emitted metrics that are not latency SLO fields. The policy-transfer
fixture uses it to require both request SLO pass rate and Jain tenant fairness,
so aggregate goodput cannot compensate for starving one tenant.

The current per-task score is capped and baseline-relative:

```text
maximize: ratio = candidate_metric / baseline_metric
minimize: ratio = baseline_metric / candidate_metric
score = min(ratio, score_cap)
```

Raw ratios, gate failures, and reliability rates are always retained.
Normalized regret against an evaluator-owned oracle may be added as a secondary
metric, but the oracle is never exposed to the agent.

## Simulator calibration gate

Stages before calibration build protocol and tooling; they do not authorize
real capability claims. Every real-backed scenario family must pass a
simulator-calibration gate before entering a paper leaderboard.

Calibration compares identical configurations in simulation and on hardware
across schedulers, load profiles, sequence lengths, parallelism choices, and at
least two hardware environments. It reports:

- absolute and relative metric error;
- Spearman/Kendall ranking correlation;
- top-k configuration overlap and pairwise decision agreement;
- repeated hardware variance and confidence intervals;
- explicit unsupported or low-confidence regions.

Simulation is acceptable for decision evaluation only where it preserves the
ranking or decision boundary relevant to the task. A successful backend run is
not calibration.

## Tracks

### Configuration optimization

Agents vary scheduler, batching, parallelism, replica, chunked-prefill, and
related fields under matched budgets. Small spaces require exhaustive-search
comparisons and are evidence of search behavior, not automatically systems
reasoning.

### Infrastructure diagnosis

Agents receive a slow but valid system and measured symptoms, then repair it.
Scoring uses measured behavior rather than textual root-cause matching. This
track now has a schema-v2 workload-aware scheduler fixture using a synthetic
deterministic timing model; it is not yet a publication-ready Vidur or hardware
task.

### Scale and workload transfer

Agents explore a smaller or cheaper public environment and submit for a larger
or shifted final environment. This remains MLSysBench's primary contribution.

### Open server and code optimization

Only after the configuration benchmark is calibrated and operationally secure
does the benchmark admit open framework selection, server launchers, scheduler
patches, routing, memory policy, kernels, or quantization. Final artifacts must
relaunch in a clean container and pass correctness, quality, integrity, and
performance gates.

The current bounded scheduler fixture is intentionally narrower than this open
track: it permits one allowlisted Python file behind a JSON decision interface.
It validates artifact handling and hidden workload replay, not arbitrary
framework modification.

## Ordered implementation plan

The normative implementation order is:

1. define the four canonical scenario families;
2. create paired small-development and shifted-final tasks for each family;
3. implement the naïve/default/expert/search baseline ladder;
4. add repeated `run-matrix` orchestration and `aggregate-results` reporting;
5. move final replay into a fresh container or ephemeral worker;
6. add starting-point ablations;
7. calibrate simulator decisions against repeated real hardware runs;
8. add the open server/code track with correctness, quality, and integrity
   gates.

Portability, CI, versioned schemas, and repair/retirement of invalid fixtures
are cross-cutting requirements throughout the eight stages. No simulator-based
model ranking is publication-ready before stage 7 passes.

## Task publication checklist

A task is publication-ready only when:

- it declares scenario family, transfer mechanism, starting point, and baseline
  tier;
- `validate-task` passes without integrity errors;
- development and final workloads differ and their provenance is recorded;
- the declared baseline exactly replays to a valid ratio of `1.0`;
- every legal mock configuration is explicitly mapped, or a real runner fails
  closed on unsupported configurations;
- resource and immutable-field gates are tested;
- simulator/default fallback paths are rejected;
- all applicable baseline tiers are reproduced;
- repeated final seeds and failure handling are defined;
- final replay occurs in the declared isolation level;
- calibration status and allowed claim are explicit;
- result manifests state source revision, image/runtime digest, hardware,
  software, seeds, hashes, budgets, gates, and caveats.
