# Result Catalog

This directory stores compact, reviewed evidence. Raw trajectories, simulator
outputs, caches, downloaded runtimes, and temporary workspaces belong under the
ignored `runs/` directory.

## `baselines/`

Real-backend baseline summaries:

- `qwen3_next_aicb_smoke.json`: one-request CUDA/AICB health check;
- `qwen3_next_aicb_benchmark.json`: 32-request Vidur+AICB baseline.

These files record the local software and hardware context, but the referenced
raw `runs/` directories are not checked in. They demonstrate backend execution,
not cross-hardware simulator calibration.

## `protocol/`

Harness and methodology experiments:

- `mock_scale_transfer_results.json`: deterministic and search sanity checks;
- `longcat_initial_results.json`: initial model-protocol exercise;
- `cli_agent_matched_5_sparse_fixture.json`: retained invalid sparse-fixture
  experiment with an explicit caveat;
- `cli_agent_matched_5_dense_fixture.json`: validator-clean synthetic fixture
  runs;
- `real_scheduler_comparison.json`: local Sarathi/vLLM comparison summary.

Single-run and synthetic files must not be presented as model rankings.

## `calibration/`

The directory contains the paired-measurement input format for
`analyze-calibration`. The checked-in example is synthetic schema
documentation, not simulator-fidelity evidence. Real bundles must retain
hardware repeats and artifact hashes and report error, rank correlation, top-k
overlap, pairwise decision agreement, and measurement variance.

## Admission rules

A checked-in result summary should include:

- whether the workload and performance surface are synthetic or real;
- task identity and source revision;
- workload, task, and surface hashes where applicable;
- model/provider and generation settings for agent runs;
- query, wall-time, token, hardware, and software budgets;
- seeds and repeat count;
- metric units, gate outcomes, and baseline definition;
- an explicit caveat describing what the result does not establish.

Do not check in credentials, full environment dumps, model transcripts with
secrets, or large raw runtime directories.
