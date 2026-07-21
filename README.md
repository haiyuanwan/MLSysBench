# MLSysBench

**Paper title:** *Toward Self-Improving ML Systems: Benchmarking AI Agents as Systems Optimizers*

MLSysBench is a simulator-grounded benchmark for studying whether AI agents
can work as inference-systems engineers. An agent receives a public task,
allowed configuration actions, simulator feedback, and a fixed experiment
budget. It must explore deliberately and submit one final configuration or a
bounded source artifact for a held-out workload or deployment scale.

The repository currently provides an executable research prototype for
configuration search, scheduler selection, and scale transfer using mock
surfaces and Vidur+AICB. It also includes schema-v3 PatchTransfer,
PolicyTransfer, and MultiFidelity protocol fixtures. It is not yet a complete or publication-ready
benchmark suite. In particular, the real tasks still need simulator-to-hardware
calibration, a true development/final split, broader task coverage, and repeated
agent runs. See [Status and roadmap](docs/status-and-roadmap.md) for the exact
boundary.

## What works today

- versioned task directories with baseline configurations, allowed actions,
  workload overrides, resource budgets, and SLO gates;
- required, validated scenario-family metadata for prefill-heavy, decode-heavy,
  high-load, and balanced workloads;
- baseline-relative scoring for throughput, goodput, and latency objectives;
- separate development and final evaluation for the scale-transfer fixture;
- one-shot, multi-step, and filesystem-capable agent interfaces;
- matched-budget grid, random, Optuna TPE, and SMAC3 search adapters;
- Linux Landlock isolation, query budgets, wall-time limits, and clean final
  replay for CLI agents;
- a real Vidur+AICB execution path with fallback detection and checked-in
  baseline summaries;
- schema-v2 code submissions with editable-file allowlists, development
  evaluation, clean starter reconstruction, and a deterministic scheduler task;
- schema-v3 provenance gates that prevent hand-authored or proxy-only tasks
  from being labeled as publication candidates;
- multi-profile robust-goodput, worst-profile, tenant-fairness, and SLO metrics;
- cost-accounted development fidelities with per-fidelity query caps and a
  tested simulator-versus-hardware-proxy allocation task;
- result aggregation for failures, uncertainty, transfer gaps, duplicate
  experiments, fidelity use and cost, plus paired simulator/hardware
  calibration analysis;
- dependency-free unit tests and task pre-publication validation.

## Quick start

MLSysBench requires Python 3.10 or newer. The mock protocol and test suite have
no runtime dependencies.

```bash
python3 -m mlsysbench.simai_bench --help

# Validate the canonical synthetic protocol fixture.
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scale_up/mock_scale_transfer

# Run a complete dependency-free search baseline.
python3 -m mlsysbench.simai_bench search \
  --task tasks/scale_up/mock_scale_transfer \
  --method grid \
  --budget 18 \
  --output-dir runs/quickstart/grid

# Exercise the multi-step agent protocol without an external model API.
python3 -m mlsysbench.simai_bench run-agent-loop \
  --task tasks/scale_up/mock_scale_transfer \
  --provider dry-run \
  --output-dir runs/quickstart/dry-run-agent

# Run the repository checks.
scripts/check_repo.sh
```

For editable installation and optional search dependencies:

```bash
python3 -m pip install -e '.[test]'
python3 -m pip install -e '.[hpo]'  # optional TPE and SMAC3 adapters
```

The real Vidur+AICB path needs a separate CUDA environment. Follow the
[implementation and CUDA runbook](docs/simai-benchmark-code.md); do not treat
the smoke task as benchmark evidence.

## Evaluation protocol

```text
Public task + allowed actions + development budget
                       |
                       v
            propose -> evaluate -> retain
                       |
                       v
              one final submission
                       |
                       v
     hidden workload/scale + gates + scoring
```

Workload and resource conditions are evaluator-owned. A submission cannot
change fields listed in `constraints.immutable_fields`. Hidden
`config_overrides` are applied after submission changes, and configurations
that exceed the task's GPU budget are rejected before simulation.

The implemented score is a capped baseline-relative ratio:

```text
maximize: ratio = candidate_metric / baseline_metric
minimize: ratio = baseline_metric / candidate_metric
score = min(ratio, score_cap)
```

Validity, resource, SLO, and task-specific metric gates are applied first.
Invalid candidates score zero. A valid baseline-equivalent result scores
`1.0`.

The four synthetic scenario fixtures keep development and final workloads and
performance surfaces separate. The final evaluator is called once after the
development budget ends. They validate the protocol; their synthetic numbers
are not model-capability evidence.

For the full scientific motivation, invariants, and comparison methodology,
read [Benchmark protocol](docs/benchmark-protocol.md).

## Task readiness

| Task | Backend | Purpose | Readiness |
|---|---|---|---|
| `scenarios/mock_prefill_heavy` | Mock | Short/long prompt and TTFT behavior | Valid Stage 1 fixture; synthetic only |
| `scenarios/mock_decode_heavy` | Mock | Short/long output and TBT behavior | Valid Stage 1 fixture; synthetic only |
| `scale_up/mock_scale_transfer` | Mock | Burst/Poisson/constant high-load scale transfer | Valid Stage 1 fixture; synthetic only |
| `scenarios/mock_balanced` | Mock | Mixed prompt/output and concurrency behavior | Valid Stage 1 fixture; synthetic only |
| `code_scheduler/workload_aware_chunked_prefill` | Python scheduler simulator | Modify batching code under held-out load shift | Valid code-protocol fixture; synthetic timing model |
| `patch_transfer/adaptive_chunk_patch` | Multi-profile scheduler proxy | Patch transfer across burst and long-prompt profiles | Schema-v3 fixture; proxy only |
| `policy_transfer/nonstationary_fair_scheduler` | Multi-profile scheduler proxy | Online policy, priority and tenant fairness | Schema-v3 fixture; synthetic trace |
| `multifidelity/scheduler_probe_allocation` | Biased simulator + hardware proxy | Allocate a shared cost budget across fidelities | Schema-v3 fixture; no physical GPU measurements |
| `simai_gym/l1_scheduler_choice` | Mock | Early scheduler-choice example used by unit tests | Legacy prototype; fails publication validation |
| `simai_gym/qwen3_next_aicb_smoke` | Vidur+AICB | CUDA/AICB health check | Smoke test only; no separate development phase |
| `simai_gym/qwen3_next_aicb_benchmark` | Vidur+AICB | Real multi-request scheduler comparison | Experimental; one host and no hardware calibration |

Run `validate-task` before using any task in an experiment. For real runners,
`--run-real-baseline` additionally executes the expensive baseline replay.
The detailed validation results and remediation order are maintained in
[Status and roadmap](docs/status-and-roadmap.md).

The code scheduler fixture executes untrusted candidate logic behind a narrow
JSON observation/action interface. Bubblewrap is preferred; Landlock is a
filesystem-only fallback on development hosts without user namespaces. The
fixture validates the code-artifact protocol, not Vidur or hardware fidelity.

## Interfaces

| Command | Role |
|---|---|
| `evaluate` | Validate and score one JSON/YAML submission |
| `validate-task` | Check task invariants before publication |
| `run-agent` | Generate and evaluate a one-shot model submission |
| `run-agent-loop` | Run a measured multi-step optimization trajectory |
| `search` | Run grid, random, TPE, or SMAC3 under a matched budget |
| `validate-baseline-ladder` | Validate four named tiers, provenance, matched budgets, and human-expert records |
| `run-baseline-ladder` | Replay static tiers and every declared matched-search seed |
| `run-matrix` | Plan, execute, and resume a declarative experiment matrix |
| `aggregate-results` | Aggregate run validity, uncertainty, transfer behavior, process metrics, and cost |
| `analyze-calibration` | Compare paired simulator and repeated hardware measurements |
| `run-cli-agent` | Run a filesystem-capable agent in a public workspace |
| `prepare-codex-runtime` | Prepare pinned Codex and CC Switch runtime assets |
| `run-isolated-codex` | Run or resume Codex against an isolated workspace |

Use `python3 -m mlsysbench.simai_bench COMMAND --help` as the authoritative
option reference.

The current research positioning and all primary sources consulted on
2026-07-16 are preserved in [Related work](docs/related-work.md) and
[the machine-readable source catalog](docs/related-work-sources.json).

### Model-backed agents

Copy `.env.example` to the ignored `.env` file and set the provider values:

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

Do not pass production credentials with `--api-key`, because command-line
arguments may be visible to other processes. Prefer `.env` or a dedicated
environment variable.

Example multi-step run:

```bash
python3 -m mlsysbench.simai_bench run-agent-loop \
  --task tasks/scale_up/mock_scale_transfer \
  --provider openai-compatible \
  --output-dir runs/experiments/model-agent
```

Canonical filesystem-capable benchmark run:

```bash
python3 -m mlsysbench.simai_bench prepare-codex-runtime

python3 -m mlsysbench.simai_bench run-cli-agent \
  --task tasks/scale_up/mock_scale_transfer \
  --agent-mode benchmark \
  --agent-profile codex \
  --max-queries 5 \
  --wall-time-seconds 900 \
  --isolation bwrap \
  --output-dir runs/experiments/codex-agent
```

`run-cli-agent` defaults to this benchmark mode. The pinned Codex CLI is the
agent scaffold, matching InferenceBench's treatment of Codex as a real CLI
agent. InferenceBench invokes Codex directly; MLSysBench additionally uses the
pinned CC Switch sidecar only to translate Codex's Responses API traffic to the
SiliconFlow Chat Completions models used by the current matrix.

Direct `chat-completions` and custom agents are debugging scaffolds and must be
selected explicitly with `--agent-mode debug`. Their results are not comparable
to the canonical Codex-scaffold matrix. Landlock-only execution is likewise a
debug boundary; benchmark mode requires bwrap today and a dedicated fresh
container/worker remains required before publication.

## Repository map

```text
mlsysbench/simai_bench/   evaluator, runners, agents, isolation, and search
tasks/                    versioned task fixtures and backend configuration
submissions/examples/     example configuration diffs
benchmarks/baselines/     checked-in real baseline summaries
benchmarks/protocol/      protocol experiments and explicit caveats
benchmarks/calibration/   paired simulator/hardware calibration bundles
scripts/                  repository checks and reproducible run wrappers
docs/                     design, architecture, runbooks, surveys, and status
tests/                    dependency-free unit and integration-style tests
third_party/SimAI/        vendored simulator stack and local compatibility work
```

Start at the [documentation index](docs/README.md). The
[architecture guide](docs/architecture.md) explains module ownership, data
flow, trust boundaries, and generated artifacts.

## Evidence and reproducibility

Checked-in result summaries are separated by intent:

- [`benchmarks/baselines/`](benchmarks/baselines/) contains real Vidur+AICB
  baseline metadata;
- [`benchmarks/protocol/`](benchmarks/protocol/) contains harness experiments,
  including synthetic fixtures with explicit caveats;
- [`benchmarks/matrices/`](benchmarks/matrices/) contains the declarative matrix
  format and a synthetic protocol example;
- `runs/` is ignored and contains local trajectories, raw simulator output,
  isolated workspaces, and runtime assets.

The current real measurements were produced on two NVIDIA
RTX5880-Ada-48Q devices. They establish that the local execution path runs;
they do not establish cross-hardware simulator fidelity or model rankings.

## Known limitations

The current repository has four canonical scenario fixtures, four bounded
scheduler/transfer protocol fixtures, and two experimental real-runner tasks,
but it does not yet support
publication-grade model rankings. Real tasks still use host-specific runtime
paths, lack genuine development/final separation, and have not been calibrated
across hardware. The legacy `l1_scheduler_choice` fixture is intentionally
reported as invalid.

Implementation proceeds in this order:

1. canonical prefill-heavy, decode-heavy, high-load, and balanced scenarios
   (implemented for protocol fixtures);
2. paired public development and shifted final workloads;
3. naïve, framework-default, expert, and matched-search baselines (contract and
   replay implemented; real candidate evidence pending);
4. repeated run-matrix execution and aggregate reporting (local resumable
   orchestration implemented; publication matrix and paired reporting pending);
5. fresh-container final replay;
6. from-scratch/default/expert-template starting-point ablations;
7. simulator-to-hardware calibration;
8. an open server/code track with correctness, quality, and integrity gates.

Portability, invalid-fixture cleanup, schema versioning, CI, linting, coverage,
and provenance checks are cross-cutting requirements throughout these stages.

See the [benchmark protocol](docs/benchmark-protocol.md) for the normative
design and [status and roadmap](docs/status-and-roadmap.md) for evidence and
stage acceptance criteria.

## License

MLSysBench is licensed under the MIT License. Vendored third-party components
retain their original licenses.
