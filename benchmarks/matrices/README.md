# Declarative Run Matrices

`run-matrix` expands a checked-in manifest over task, model, scaffold, budget,
seed, and repeat. A task's declared `scenario.starting_point` is copied into
every cell; separate task variants are required for starting-point ablations.

Preview a matrix without creating run artifacts:

```bash
python3 -m mlsysbench.simai_bench run-matrix \
  --manifest benchmarks/matrices/protocol.schema.json \
  --dry-run
```

Execute or resume it:

```bash
python3 -m mlsysbench.simai_bench run-matrix \
  --manifest benchmarks/matrices/protocol.schema.json \
  --output-dir runs/matrices/protocol
```

Each cell has an immutable `cell_manifest.json`, a mutable
`cell_status.json`, and preserved `attempts/N/` directories containing separate
stdout/stderr logs plus executor-owned artifacts. Completed cells are skipped
on resume. Failed cells are retained and require `--retry-failed` for another
attempt.

The example is synthetic protocol documentation. It is not a paper result.
Credentials are never accepted in a matrix manifest; runners obtain them from
their normal environment/runtime mechanism.
