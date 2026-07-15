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

A separate `development` section and public files are required for tasks that
claim held-out generalization. Mock tasks should explicitly map every legal
finite candidate and fail closed on unknown signatures.

Every task must declare a supported scenario family, transfer mechanism,
workload profile list, and starting point. The loader rejects unknown or
cross-family profiles, and the validator checks the objective and final
workload metadata. Named baseline tiers remain Stage 3 work. See the
[benchmark protocol](../docs/benchmark-protocol.md).

## Validation

```bash
python3 -m mlsysbench.simai_bench validate-task --task TASK_DIRECTORY
```

For a real runner, add `--run-real-baseline` on a prepared host. Publication
requires a valid baseline replay with ratio `1.0`, distinct development and
final inputs, and documented provenance. See the
[benchmark protocol](../docs/benchmark-protocol.md) and
[status document](../docs/status-and-roadmap.md).
