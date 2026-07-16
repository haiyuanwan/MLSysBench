# Task Catalog

Each task is a self-contained evaluation definition. A directory is not
automatically publication-ready merely because it is checked into `tasks/`;
always consult its README and run the validator.

## Current tasks

| Task | Type | Maturity |
|---|---|---|
| [`scenarios/mock_prefill_heavy`](scenarios/mock_prefill_heavy/) | Synthetic prefill-heavy scenario | Valid Stage 1 fixture |
| [`scenarios/mock_decode_heavy`](scenarios/mock_decode_heavy/) | Synthetic decode-heavy scenario | Valid Stage 1 fixture |
| [`scale_up/mock_scale_transfer`](scale_up/mock_scale_transfer/) | Synthetic high-load scale transfer | Valid Stage 1 fixture |
| [`scenarios/mock_balanced`](scenarios/mock_balanced/) | Synthetic balanced scenario | Valid Stage 1 fixture |
| [`code_scheduler/workload_aware_chunked_prefill`](code_scheduler/workload_aware_chunked_prefill/) | Executable scheduler-code task | Valid protocol fixture; synthetic timing model |
| [`patch_transfer/adaptive_chunk_patch`](patch_transfer/adaptive_chunk_patch/) | Cross-workload code patch | Schema-v3 protocol fixture; proxy only |
| [`policy_transfer/nonstationary_fair_scheduler`](policy_transfer/nonstationary_fair_scheduler/) | Nonstationary online policy | Schema-v3 protocol fixture; synthetic trace |
| [`simai_gym/azure2023_chunked_prefill_transfer`](simai_gym/azure2023_chunked_prefill_transfer/) | Real-trace chunked-prefill transfer | Selected intake; baseline/calibration/validation pending |
| [`multifidelity/scheduler_probe_allocation`](multifidelity/scheduler_probe_allocation/) | Cost-accounted multi-fidelity optimization | Schema-v3 protocol fixture; hardware proxy is simulated |
| [`simai_gym/l1_scheduler_choice`](simai_gym/l1_scheduler_choice/) | Synthetic scheduler choice | Legacy invalid prototype |
| [`simai_gym/qwen3_next_aicb_smoke`](simai_gym/qwen3_next_aicb_smoke/) | Real Vidur+AICB health check | Smoke-only |
| [`simai_gym/qwen3_next_aicb_benchmark`](simai_gym/qwen3_next_aicb_benchmark/) | Real multi-request configuration task | Experimental |

## Required files

Every task contains:

- `task.json`: schema version, scenario metadata, objective, SLO, phases,
  constraints, and runner;
- `baseline_config.json`: complete starting configuration;
- `allowed_actions.json`: legal submission fields and value domains;
- `hidden/baseline_metrics.json`: final baseline denominator;
- `hidden/eval_workload.json`: evaluator-owned final overrides;
- `README.md`: purpose, prerequisites, maturity, and allowed claim.

Publication candidates additionally declare `baseline_ladder` in `task.json`.
The referenced manifest records `naive`, `framework_default`, `expert_recipe`,
and `matched_search` tiers, the score denominator, matched budgets/seeds, and a
human-expert result record. Candidate ladders must also reference a complete
replay `result_bundle`; declarations without result evidence do not pass. See
the contract example in
[`benchmarks/protocol/baseline_ladder.schema.json`](../benchmarks/protocol/baseline_ladder.schema.json).
Simulator-backed candidates also declare `calibration_bundle`; candidates
whose final score is measured directly on hardware declare
`real_hardware_evidence`. Validation checks the referenced evidence rather than
trusting `provenance.calibration_status` by itself.

A separate `development` section and public files are required for tasks that
claim held-out generalization. Mock tasks should explicitly map every legal
finite candidate and fail closed on unknown signatures.

Schema-v2 code tasks additionally declare `submission.type: code`, a public
`starter_dir`, and an exact editable-file allowlist. The CLI workspace receives
the starter under `solution/`; development queries bundle only allowlisted text
files, and final replay reconstructs a fresh starter before applying them.
`python_code` evaluators are trusted task components and must isolate candidate
code from evaluator-owned workloads.

Schema-v3 tasks additionally declare machine-readable provenance and evidence
status. Multi-fidelity development phases name each fidelity's kind, cost and
per-fidelity query limit, plus a total `max_development_cost_units` budget.
Hand-authored tasks are validator-enforced fixtures and cannot be promoted to a
paper candidate. See the
[task intake and publication gates](../docs/task-intake-and-publication-gates.md).

Every task must declare a supported scenario family, transfer mechanism,
workload profile list, and starting point. The loader rejects unknown or
cross-family profiles, and the validator checks the objective and final
workload metadata. The named-baseline schema, validation, and replay command are
implemented; populating them with real expert/search/human evidence remains
Stage 3 work. See the [benchmark protocol](../docs/benchmark-protocol.md).

## Validation

```bash
python3 -m mlsysbench.simai_bench validate-task --task TASK_DIRECTORY
```

For a real runner, add `--run-real-baseline` on a prepared host. Publication
requires a valid baseline replay with ratio `1.0`, distinct development and
final inputs, and documented provenance. See the
[benchmark protocol](../docs/benchmark-protocol.md) and
[status document](../docs/status-and-roadmap.md).

Validate or replay a declared ladder with:

```bash
python3 -m mlsysbench.simai_bench validate-baseline-ladder --task TASK_DIRECTORY
python3 -m mlsysbench.simai_bench run-baseline-ladder \
  --task TASK_DIRECTORY --output-dir runs/baselines/TASK_ID
```
