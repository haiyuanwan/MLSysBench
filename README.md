# MLSysBench

MLSysBench is a simulator-grounded benchmark for studying whether AI agents can
act as inference-systems engineers. The benchmark gives an agent a model, a
hardware and cost envelope, a public development workload, a simulator, and a
limited experiment budget. The agent must run experiments, diagnose
bottlenecks, retain the best observed design, and submit a final configuration
for a hidden workload or a larger deployment scale.

The current repository is an executable research scaffold for configuration,
scheduling, and scale-transfer tasks built on Vidur, SimAI, and AICB. It does
not yet claim complete kernel, algorithm, and end-to-end serving coverage.

## Research Story

AI coding agents increasingly participate in ML systems work, but it is still
unclear whether they can conduct disciplined performance engineering. A model
may know that tensor parallelism, chunked prefill, batching, and prefill-decode
disaggregation matter without being able to:

- isolate a bottleneck;
- propose diverse candidate systems;
- compare candidates under controlled workloads;
- roll back regressions;
- preserve the best configuration found so far;
- transfer conclusions to a different workload or deployment scale.

[InferenceBench](https://github.com/aisa-group/InferenceBench) studies this
problem on a real single-H100 server. It shows that frontier agents improve a
naive server but still lose to matched-budget Random, TPE, and SMAC3 search.
Its central result is that agents often know relevant techniques but perform
shallow exploration and fail to manage experiments reliably.

MLSysBench asks the complementary question:

> If simulation makes large-scale inference experiments cheap and repeatable,
> can agents learn system behavior and make decisions that transfer beyond the
> configurations and scales they directly evaluated?

The intended distinction is:

| Dimension | InferenceBench | MLSysBench |
|---|---|---|
| Execution environment | Real single H100 | Calibrated inference simulator |
| Main artifact | Running inference server | Cluster configuration or system patch |
| Primary capability | Open-ended engineering and experiment execution | Diagnosis, budgeted exploration, and scale transfer |
| Development scale | Same machine as final evaluation | Public smaller scale |
| Final evaluation | Held-out requests | Held-out requests, load, hardware, or larger scale |
| Cost envelope | Two-hour GPU session | Wall time and simulator-query budget |

The benchmark should not be presented as a simulator clone of InferenceBench.
Its scientific value depends on large-scale decisions, controlled
counterfactual experiments, and development-to-final generalization.

## Research Questions

The benchmark is organized around four research questions.

1. **Diagnosis:** Can agents infer the cause of poor TTFT, TBT, goodput, memory
   pressure, or communication overhead from system observations?
2. **Budgeted optimization:** Under the same simulator-query budget, do agents
   outperform Random, grid, TPE, SMAC3, Bayesian optimization, and expert
   heuristics?
3. **Scale transfer:** Can an agent explore at 8-32 GPUs and submit a strong
   configuration for a hidden 128-9216 GPU deployment?
4. **Simulator use:** Does cheaper experimentation lead to broader, more
   disciplined exploration, or do agents still repeat measurements and remain
   close to familiar recipes?

These questions permit either a positive or negative result. A useful paper
does not require agents to win. A result showing that search dominates agents
despite cheap simulation would expose an important limitation. A result where
agents mainly help under distribution or scale shift would identify where
systems knowledge adds value beyond black-box optimization.

## Evaluation Protocol

Each task has a public development phase and a hidden final phase.

```text
Public development phase
  model and hardware description
  public workload or symptoms
  allowed system actions
  simulator feedback
  fixed step/query budget
             |
             v
Agent trajectory
  propose -> simulate -> compare -> retain/rollback
             |
             v
Final submission
  one configuration or code patch
             |
             v
Hidden final phase
  hidden seed/workload/load/scale
  fixed resource budget
  validity and SLO gates
  baseline-relative score
```

Development evaluations are included in the agent trajectory. The hidden
final evaluator is called only after the trajectory ends. On the stopping
step, an agent may provide `final_changes` that differs from its most recent
development experiment. This allows the scale-transfer track to measure
reasoned extrapolation instead of merely copying the best public configuration.

The bundled hidden files are local protocol fixtures. `run-cli-agent` creates a
separate public workspace and keeps the task in an evaluator process. On Linux,
the default Landlock policy prevents the agent and its child processes from
reading the repository while allowing writes only in that workspace. Final
evaluation starts only after the development service has stopped. Production
runs should additionally use a container or dedicated worker for process and
network isolation.

### Fixed Evaluation Conditions

Workload and resource conditions are evaluator-owned. A submission cannot
change request count, arrival rate, sequence lengths, model identity, or other
fields listed in `constraints.immutable_fields`. Hidden
`config_overrides` are applied after the submitted changes are merged.

The evaluator computes resource use as:

```text
GPU units = replicas * tensor_parallel_size * pipeline_parallel_size
```

Configurations exceeding the development or final GPU budget are rejected
before the simulator runs. This prevents a submission from improving goodput
by silently buying more hardware.

### Gates

The current configuration tracks use three gates:

- **Validity gate:** schema-valid configuration and successful simulator run.
- **Resource gate:** GPU use remains within the task budget.
- **SLO gate:** required P99 TTFT, TBT, and E2E limits are satisfied.

Future open-code tasks also require correctness, quality, and integrity gates.
For real serving tasks, model quality should remain at least 95% of a fixed
baseline accuracy, following the general pattern used by InferenceBench.

### Metrics

The primary metric is task-specific. Typical choices are:

- SLO goodput;
- request throughput;
- inverse P99 TTFT or TBT;
- goodput per GPU or per dollar;
- normalized regret against a best-known search result;
- public-to-hidden generalization gap.

The implemented score is a capped baseline-relative ratio:

```text
maximize: ratio = agent_metric / baseline_metric
minimize: ratio = baseline_metric / agent_metric
score = min(ratio, score_cap)
```

A baseline-equivalent valid result scores `1.0`; invalid results score `0`.
Paper-level aggregate reporting should use geometric means across tasks and
show uncapped ratios as well as validity rates.

## Task Tracks

### 1. Budgeted Optimization

The agent repeatedly evaluates legal configurations under a fixed public
workload and query budget. Actions may include:

- TP, PP, and replica allocation;
- vLLM-, ORCA-, or Sarathi-style scheduling;
- batch-token limits and chunked-prefill size;
- prefill-decode allocation and transfer precision;
- routing, admission-control, and memory-policy parameters.

This track must always be compared with matched-budget non-agent search. A
small action space is a search problem, not evidence of systems reasoning.

### 2. Infra Diagnosis

The agent receives a slow but valid system, symptoms, and a public metrics
report. It must identify and repair issues such as excessive TP, poor batching,
memory pressure, scheduler mismatch, or prefill/decode interference. Scoring
uses measured final behavior rather than matching a textual root-cause label.

### 3. Scale Transfer

The agent explores a smaller public deployment and submits a configuration for
a larger hidden deployment. Example shifts include:

- 8 GPUs to 128 GPUs;
- 4 replicas to 64 replicas;
- balanced traffic to prefill-heavy traffic;
- H100 scale-up to H20 scale-out;
- low communication pressure to a constrained network fabric.

This is the primary differentiator from real single-node serving benchmarks.

### 4. Open System Design

The future open track allows scheduler or routing code modifications, such as
adaptive chunked prefill, SLO-aware admission control, or memory-aware routing.
This track is not implemented in the current scaffold and is not part of the
current empirical claim.

## Agent and Search Interfaces

### Multi-Step Agent

`run-agent-loop` sends the model the public task context and measured
development history. Every step is stored in `trajectory.json` with:

- proposed changes and rationale;
- validity and failure information;
- development metrics and score;
- whether the step became best-so-far;
- optional stopping and final-transfer decision;
- model, API latency, finish reason, and token usage when returned by the provider.

```bash
python -m mlsysbench.simai_bench run-agent-loop \
  --task tasks/scale_up/mock_scale_transfer \
  --provider dry-run \
  --output-dir runs/experiments/scale_transfer_agent
```

The `dry-run` provider is deterministic protocol validation, not an AI model
result. SiliconFlow uses the OpenAI-compatible provider. The ignored `.env`
file can be configured from `.env.example`:

```dotenv
MODEL_API_KEY=replace-with-your-api-key
MODEL_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=meituan-longcat/LongCat-2.0
MODEL_MAX_TOKENS=131072
MODEL_CONTEXT_WINDOW=1048576
MODEL_ENABLE_THINKING=true
MODEL_THINKING_BUDGET=32768
MODEL_TEMPERATURE=0
MODEL_JSON_MODE=false
```

The API key, endpoint, model, and generation settings are loaded automatically:

```bash
python -m mlsysbench.simai_bench run-agent-loop \
  --task tasks/scale_up/mock_scale_transfer \
  --provider openai-compatible \
  --output-dir runs/experiments/model_scale_transfer
```

CLI flags can override every model setting. LongCat uses prompt-enforced JSON
here because its reasoning output can consume a short response before any final
`message.content` is emitted. Although LongCat advertises a 1024K context,
SiliconFlow currently enforces `131072` as this model's maximum completion
length, so that value is used by default. Other SiliconFlow language models can
use `MODEL_JSON_MODE=true` when they support `response_format` reliably.

### Filesystem-Capable CLI Agent

`run-cli-agent` exposes only `MISSION.md`, `task_context.json`, budget metadata,
the final-submission schema, and `evaluate_dev.py`. Development calls use a
file-queue RPC to a private evaluator process, so the harness works on compute
nodes that prohibit local sockets. Query count and wall time are both hard
budgets. The resulting manifest records source revision, workload hashes and
seeds, model usage, isolation level, and final gate outcomes.

The bundled Chat Completions profile supports any SiliconFlow tool-calling
model without third-party packages:

```bash
python3 -m mlsysbench.simai_bench run-cli-agent \
  --task tasks/scale_up/mock_scale_transfer \
  --agent-profile chat-completions \
  --model meituan-longcat/LongCat-2.0 \
  --context-window 1048576 \
  --max-output-tokens 131072 \
  --max-queries 5 \
  --wall-time-seconds 900 \
  --isolation landlock \
  --output-dir runs/experiments/longcat_cli
```

External agents such as OpenCode can use `--agent-profile custom
--agent-command '...'`.

### Isolated Codex and CC Switch

Current Codex releases speak the Responses API while these SiliconFlow models
use Chat Completions. The `codex` profile runs the official Codex CLI through
the official CC Switch v3.17.0 bridge. Both binaries are pinned by SHA-256 and
stored only in the ignored `runs/_runtime` directory:

```bash
python3 -m mlsysbench.simai_bench prepare-codex-runtime \
  --codex-source "$(command -v codex)"
```

To resume an existing Codex session without touching the host Codex state:

```bash
python3 -m mlsysbench.simai_bench run-isolated-codex \
  --workspace . \
  --session-id SESSION_UUID \
  --model meituan-longcat/LongCat-2.0 \
  --output-dir runs/isolated-codex/session-run
```

The source rollout is copied and its `cwd` is retargeted only in the isolated
snapshot. API-key-shaped values in history are redacted. CC Switch receives the
real key in a private Landlock domain; Codex sees only `PROXY_MANAGED`, its own
isolated `CODEX_HOME`, and the selected workspace. Hashes of the host
`config.toml`, `auth.json`, and source rollout are checked before and after the
run.

The same runtime can execute a benchmark task:

```bash
python3 -m mlsysbench.simai_bench run-cli-agent \
  --task tasks/scale_up/mock_scale_transfer \
  --agent-profile codex \
  --model meituan-longcat/LongCat-2.0 \
  --context-window 1048576 \
  --max-output-tokens 131072 \
  --thinking-budget 32768 \
  --max-queries 5 \
  --wall-time-seconds 900 \
  --output-dir runs/experiments/longcat_codex
```

The configured catalog also accepts `zai-org/GLM-5.2`,
`moonshotai/Kimi-K2.7-Code`, and `deepseek-ai/DeepSeek-V4-Pro`. Each run uses a
fresh CC Switch database and a kernel-assigned loopback port.
`scripts/run_codex_siliconflow_matrix.sh` runs all four models under identical
budgets for a matched comparison.

### Matched-Budget Search

Grid and random remain dependency-free. Optuna TPE and SMAC3 are available
through the `hpo` extra. All methods stop when either their query budget or
their wall-clock budget is exhausted, then submit the best development
configuration once to the final evaluator.

```bash
python -m mlsysbench.simai_bench search \
  --task tasks/scale_up/mock_scale_transfer \
  --method grid \
  --budget 18 \
  --output-dir runs/experiments/scale_transfer_grid

python -m mlsysbench.simai_bench search \
  --task tasks/scale_up/mock_scale_transfer \
  --method random \
  --budget 6 \
  --wall-time-seconds 900 \
  --seed 3 \
  --output-dir runs/experiments/scale_transfer_random
```

```bash
python3 -m pip install -e '.[hpo]'
python3 -m mlsysbench.simai_bench search \
  --task tasks/scale_up/mock_scale_transfer \
  --method tpe \
  --budget 18 \
  --wall-time-seconds 900 \
  --output-dir runs/experiments/scale_transfer_tpe
```

## Protocol Validation Experiment

The bundled `mock_scale_transfer` task is a synthetic test of the evaluation
protocol. Its performance surface is manually constructed so that the public
8-replica optimum differs from the hidden 32-replica optimum.

The current deterministic run produces:

| Method | Development queries | Best development ratio | Hidden final ratio |
|---|---:|---:|---:|
| Deterministic agent protocol | 3 | 1.60x | 1.875x |
| Grid search | 18 | 1.60x | 1.3125x |
| Random search, seed 1 | 6 | 1.60x | 1.3125x |
| Random search, seed 2 | 6 | 1.34x | 1.875x |
| Random search, seed 3 | 6 | 1.30x | 1.6875x |

The deterministic client explicitly demonstrates use of `final_changes`; it is
not evidence that language-model agents outperform search. The experiment only
shows that:

- development and final workloads are separated;
- query budgets are recorded;
- search and agents use the same evaluator;
- a final configuration may extrapolate beyond the public optimum;
- complete trajectories and best-so-far state are reproducible.

An initial SiliconFlow run with `meituan-longcat/LongCat-2.0` produced a useful
budget-sensitivity check on an earlier synthetic fixture revision:

| Budget | API calls | Total tokens | Best development | Hidden final |
|---:|---:|---:|---:|---:|
| 1 step | 1 | 2,091 | 1.34x | 1.875x |
| 5 steps | 5 | 14,481 | 1.60x | 1.3125x |

The five-step run optimized the public workload more successfully but
transferred worse than the one-step decision. This is evidence that the
protocol can expose public-workload overfitting, not evidence of a general
LongCat capability: the performance surface is deliberately constructed. The
full API and trajectory summary is stored in
[`longcat_initial_results.json`](benchmarks/protocol/longcat_initial_results.json).

### CLI Agent Smoke Test

A matched five-query run on the corrected dense fixture exercised LongCat-2.0,
GLM-5.2, Kimi-K2.7-Code, and DeepSeek-V4-Pro:

| Model | Wall time | API calls | Total tokens | Best development | Hidden final |
|---|---:|---:|---:|---:|---:|
| LongCat-2.0 | 83.32 s | 15 | 76,609 | 1.60x | 1.3125x |
| GLM-5.2 | 653.61 s | 17 | 313,369 | 1.60x | 1.46875x |
| Kimi-K2.7-Code | 48.42 s | 15 | 64,696 | 1.34x | 1.875x |
| DeepSeek-V4-Pro | 151.94 s | 18 | 113,906 | 1.60x | 1.3125x |

Kimi selected the hidden-optimal `TP4 + Sarathi + chunk-1024` despite its lower
public score. GLM partially extrapolated the chunk size. LongCat and DeepSeek
submitted the measured public optimum and transferred worse. This is the
intended distinction between public optimization and scale-transfer reasoning,
but one run per model on a synthetic task is not a model ranking.

At the same five-query budget, twenty-seed mean hidden ratios were 1.428x for
Random, 1.463x for Optuna TPE, and 1.447x for SMAC3. The exact model manifests,
input hashes, usage, HPO versions, and caveats are summarized in
[`cli_agent_matched_5_dense_fixture.json`](benchmarks/protocol/cli_agent_matched_5_dense_fixture.json).

An earlier sparse fixture made all four models score 1.875x, but it omitted the
baseline signature and mapped only two of eighteen configurations. That flawed
result is retained as a task-design counterexample in
[`cli_agent_matched_5_sparse_fixture.json`](benchmarks/protocol/cli_agent_matched_5_sparse_fixture.json).
The corrected fixture maps every legal configuration and fails closed for
unknown signatures. The resulting requirements and next task matrix are in
[`task-design-lessons.md`](docs/task-design-lessons.md).

## Real SimAI, Vidur, and AICB Backend

The real runner executes `python -m vidur.main`, parses request-level metrics,
and can use AICB-backed layer timings. Every simulator call now receives a
unique output directory, preventing a successful process from accidentally
reusing a stale `request_metrics.csv` from an earlier run.

The AICB integration previously converted per-layer nanoseconds to seconds and
then multiplied the accumulated model time by another `1e-3`. Existing
Qwen3-Next baselines were therefore approximately 1000 times too small. This
repository fixes that conversion. The 1-request and 32-request baselines were
regenerated on two RTX5880-Ada-48Q GPUs using the real CUDA bf16 fallback path.

The corrected 32-request baseline reports P99 E2E `293.99 ms`, P99 TTFT
`67.19 ms`, P99 TBT `30.49 ms`, and goodput `60.49 req/s`. The former baseline
reported sub-millisecond request latency and must not be used.

A fixed-resource scheduler comparison found that vLLM increased measured
goodput by only `0.5%` while worsening P99 E2E by `79%` and P99 TBT by about
`2x`. This exposed an overly loose initial SLO. The real benchmark now uses
P99 TTFT <= `80 ms` and P99 TBT <= `40 ms`, under which the Sarathi baseline is
valid and the vLLM variant is rejected.

A two-step LongCat integration run retained `16 replicas x TP2 + Sarathi` after
its TP4 experiment failed AICB workload generation. The retained configuration
reported `60.5500 req/s`, or `1.00106x` the baseline, using 8,728 model tokens.
This is only an end-to-end API/GPU harness check: TP communication is not yet
modeled, and a 0.106% simulated difference is not a defensible optimization
claim.

Run a regenerated real task with:

```bash
python -m mlsysbench.simai_bench evaluate \
  --task tasks/simai_gym/qwen3_next_aicb_benchmark \
  --submission submissions/examples/qwen3_next_aicb_benchmark_baseline.json
```

Baseline metadata records the corrected run directory and timing provenance.

## Simulator Validation Requirements

Simulation makes the benchmark affordable, but simulation fidelity becomes
part of benchmark validity. The SimAI NSDI paper validates large-scale training
simulation; the newer multi-request inference path requires separate
validation. Before making model-ranking claims, the benchmark should compare at
least 20-30 representative configurations against real systems and report:

- absolute error for TTFT, TBT, throughput, and memory;
- Spearman and Kendall rank correlation;
- top-k configuration agreement;
- whether agent/search rankings remain stable across simulation and hardware;
- calibration error across model, batch, sequence, and parallelism regimes.

Ranking fidelity is more important than a low average absolute error when the
benchmark is used to choose among configurations.

Known fidelity limitations in the current backend include:

- AICB TP collective communication is not yet modeled in its current path;
- PP support for the AICB model is incomplete;
- force-BS1 profiling does not capture all batch-dependent kernel behavior;
- PD transfer and some MoE communication paths use parameterized estimates;
- the local Ada fallback is not equivalent to the official Hopper/Blackwell
  AICB environment.

Tasks must restrict their action spaces to validated simulator behavior.

## Behavioral Analysis

A paper-scale evaluation should analyze trajectories, not only final scores.
Recommended measurements include:

- number of distinct valid configurations;
- fraction of single-variable controlled experiments;
- repeated evaluation rate;
- invalid, OOM, and timeout rate;
- simulator-budget utilization;
- time or queries to first valid improvement;
- best-found versus final-submitted gap;
- rollback frequency;
- search-space coverage;
- development-to-hidden generalization gap;
- token cost and wall-clock cost.

The intended headline question is whether agents know the techniques but fail
to execute disciplined experiments, or whether their systems knowledge helps
specifically when the hidden environment differs from development.

## Repository Layout

```text
mlsysbench/simai_bench/
  schema.py          task, workload, budget, and action schemas
  evaluator.py       development/final evaluation and scoring
  runner.py          mock and Vidur backends
  agent_runner.py    one-shot and multi-step agent execution
  cli_agent.py       isolated CLI-agent orchestration and run manifests
  chat_cli_agent.py  tool-using OpenAI-compatible CLI agent
  landlock.py        Linux filesystem isolation for agent processes
  search.py          grid, random, Optuna TPE, and SMAC3 search
  metrics.py         TTFT, TBT, throughput, and goodput extraction

tasks/
  simai_gym/         configuration and scheduling tasks
  scale_up/          public-to-hidden scale-transfer tasks

tests/
  test_simai_bench.py
```

## Testing

The dependency-free test suite covers schema validation, immutable workload
conditions, GPU budgets, SLO goodput, multi-step trajectories, search baselines,
public-workspace isolation, query and wall-time enforcement, clean final replay,
mock-surface completeness, Vidur command construction, and AICB fallback detection.

```bash
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scale_up/mock_scale_transfer
python -m unittest discover -s tests -v
python -m compileall -q mlsysbench
```

## Current Status

| Component | Status |
|---|---|
| Task schema and allowed-action validation | Implemented |
| Hidden workload overrides | Implemented |
| Development/final split | Implemented |
| GPU budget gate | Implemented |
| Per-request TTFT/TBT goodput | Implemented |
| Multi-step agent trajectory | Implemented |
| Isolated filesystem-capable CLI agent | Implemented with Landlock; bwrap optional |
| Clean final replay and workload manifest | Implemented |
| Grid and random matched-budget baselines | Implemented |
| Optuna TPE and SMAC3 adapters | Implemented as optional `hpo` extra |
| Unique Vidur run isolation | Implemented |
| Mock scale-transfer task | Implemented |
| AICB unit correction | Implemented |
| Regenerated real AICB baselines | Implemented on RTX5880-Ada-48Q |
| Real simulator-to-hardware validation | Pending |
| Multi-model CLI protocol smoke test | Implemented on one synthetic fixture |
| Repeated frontier-agent evaluation | Pending |
| Multi-model, multi-hardware task suite | Pending |
| Open scheduler-code track | Pending |

The default filesystem sandbox did not expose the GPU, but host execution
verified driver `570.172.18`, two RTX5880-Ada-48Q devices, CUDA-enabled PyTorch,
and successful real AICB smoke and 32-request benchmark runs.

## Paper Plan

The paper should make a focused system-agent claim rather than a broad
full-stack claim.

1. **Introduction:** real single-node agent benchmarks cannot economically
   cover large-scale cluster decisions.
2. **Insight:** simulation enables controlled, fast counterfactual experiments,
   but agents must still use it effectively and generalize beyond public runs.
3. **Benchmark:** diagnosis, budgeted optimization, and scale-transfer tasks
   with hidden final evaluation.
4. **Validation:** simulator ranking correlation against real deployments.
5. **Evaluation:** frontier agents versus matched-budget search and expert
   heuristics across models, workloads, hardware, and seeds.
6. **Behavior:** explain final performance through exploration diversity,
   rollback discipline, best-state retention, and generalization gaps.

A suitable thesis statement is:

> InferenceBench asks whether agents can optimize a real single-GPU server;
> MLSysBench asks whether agents can use simulation to reason about inference
> systems at scales that cannot be explored directly on real hardware.

## Related Work

- [InferenceBench](https://github.com/aisa-group/InferenceBench): open-ended
  inference server optimization by CLI agents.
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench): correct and
  efficient GPU kernel generation.
- [SimAI](https://github.com/aliyun/SimAI): large-scale AI system simulation and
  the underlying inference simulation components used here.
- [Vidur](https://github.com/microsoft/vidur): LLM inference performance
  simulation and scheduling analysis.
- [CompilerGym](https://github.com/facebookresearch/CompilerGym): interactive
  optimization environments and agent evaluation.

## License

The project uses the MIT license. Vendored third-party components retain their
original licenses.
