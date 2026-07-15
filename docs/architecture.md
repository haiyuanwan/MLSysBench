# Architecture

## Scope

MLSysBench evaluates configuration diffs and one bounded source-bundle format
against task-owned workloads and metrics. The implementation supports mock
performance surfaces, Vidur+AICB experimental runs, and a deterministic
scheduler-code protocol fixture. It does not yet evaluate arbitrary server,
kernel, or framework patches.

## Data flow

```text
task.json + baseline_config + allowed_actions
                    |
                    v
              TaskSpec.load
                    |
submission.changes -> validate -> merge -> hidden overrides -> resource gate
                                                        |
                                                        v
                                      mock, Vidur, or Python-code runner
                                                        |
                                                        v
                                              normalized metrics
                                                        |
                                                        v
                                      validity + SLO gate + ratio score
```

Development evaluation uses the task's `development` specification when one is
present. Final evaluation always uses `hidden`. CLI-agent runs stop the public
development service before executing one clean final replay.

For a schema-v2 or schema-v3 code task, the public starter is copied to `solution/` and only
declared editable files are bundled with each query. The Python-code runner
reconstructs a fresh temporary starter, overlays those files, and invokes a
trusted task evaluator. The scheduler fixture runs candidate decisions through
a JSON-lines adapter so candidate code never receives the hidden workload file.

Schema-v3 tasks add machine-readable source provenance and evidence status.
Their development phase may expose named fidelities with different
`cost_units`, per-fidelity query limits, and one shared cost budget. The final
phase never accepts a fidelity selector. The built-in scheduler-policy
evaluator supports multiple workload cases and reports robust/worst-profile
goodput, SLO pass rate, tenant fairness, and per-profile diagnostics; it remains
a deterministic proxy until a calibration bundle establishes decision fidelity.

## Module ownership

| Module | Responsibility |
|---|---|
| `schema.py` | Parse scenario, objective, SLO, runner, development, hidden, and constraint fields |
| `actions.py` | Validate submission diffs and merge them with the baseline |
| `evaluator.py` | Coordinate phase selection, workload overrides, runner execution, and scoring |
| `runner.py` | Dispatch mock, Vidur, and bounded Python-code backends; detect AICB fallback/default behavior |
| `metrics.py` | Parse request metrics and calculate TTFT/TBT goodput |
| `scoring.py` | Apply validity/SLO gates and baseline-relative scoring |
| `task_validation.py` | Check scenario consistency, mock-surface completeness, phase separation, and baseline replay |
| `agent_runner.py` | Run one-shot and multi-step API agents |
| `model_client.py` | Configure dry-run and OpenAI-compatible model clients |
| `search.py` | Run matched-budget grid, random, TPE, and SMAC3 baselines |
| `cli_agent.py` | Build public workspaces, enforce canonical/debug scaffold policy, and record run manifests |
| `chat_cli_agent.py` | Provide the dependency-free debug-only Chat Completions profile |
| `landlock.py` | Apply Linux filesystem restrictions to agent processes |
| `codex_ccswitch.py` | Prepare the canonical pinned Codex scaffold and MLSysBench-specific CC Switch protocol bridge |
| `cli.py` | Expose repository workflows through one command-line entry point |

## Task layout

```text
task-name/
  README.md
  task.json                scenario, objective, phases, constraints, runner
  baseline_config.json
  allowed_actions.json
  public/                  optional development-only data
    baseline_metrics.json
    dev_workload.json
    mock_metrics.json
    starter/               code tasks only; copied into the agent workspace
  hidden/
    baseline_metrics.json
    eval_workload.json
    mock_metrics.json      mock runners only
```

`task.json` is the source of truth. Paths are resolved relative to the task
directory, except runner paths that intentionally point at repository or host
runtime resources. The current real tasks contain absolute host paths; this is
a known portability issue, not a recommended task-authoring pattern.

## Agent trust boundaries

The one-shot and multi-step API agents receive serialized public context from
the harness. A filesystem-capable CLI agent instead receives a generated
workspace containing only the mission, task context, budget metadata, final
submission schema, development evaluator helper, and any declared public
starter source.

On Linux, Landlock denies repository reads and limits writes to the public
workspace. The development evaluator remains in a separate process and owns
private task files. This protects accidental filesystem access, but it does
not provide complete process, syscall, or network isolation. Benchmark-grade
runs should use a dedicated container or worker in addition to Landlock.
The scheduler fixture prefers bubblewrap for its candidate subprocess and uses
Landlock only as a filesystem-isolation fallback for restricted development
hosts; that fallback does not provide network namespace isolation.

## Search-space semantics

`choices` actions are enumerated exactly. Boolean actions use both values.
Range-based grid and random search discretize an action to its baseline,
minimum, midpoint, and maximum (deduplicated and converted to the declared
type). Therefore `grid` means exhaustive search over this generated finite
candidate set, not every integer or real value in a declared range. TPE and
SMAC3 use their native range representations when the optional dependencies
are installed.

## Artifacts

All mutable run output belongs under ignored `runs/`:

- agent prompts, responses, trajectories, and submissions;
- public CLI-agent workspaces and private evaluator queues;
- final evaluation and reproducibility manifests;
- raw Vidur output and request metrics;
- downloaded isolated runtime assets.

Compact, reviewed summaries may be checked into `benchmarks/baselines/` or
`benchmarks/protocol/`. Each summary must state whether the data are synthetic,
smoke-only, or real, and must include enough provenance to avoid presenting a
protocol fixture as model-capability evidence.
