# SimAI Benchmark Code Scaffold

The first code scaffold implements the benchmark protocol described in
`docs/simai-benchmark-proposals.md`:

- task specification loading
- allowed action validation
- baseline config + submission diff merging
- runner abstraction (`mock` and `vidur`)
- Vidur metrics parsing
- SLO gates
- baseline-relative scoring
- CLI evaluation

Run the bundled smoke test task:

```bash
python -m mlsysbench.simai_bench evaluate \
  --task tasks/simai_gym/l1_scheduler_choice \
  --submission submissions/examples/sarathi_scheduler.json
```

Real SimAI/Vidur tasks should use `runner.type = "vidur"` in `task.json` and
provide a valid `vidur_root`, timeout, and output directory. The evaluator will
run `python -m vidur.main` with the merged flat config.

The first supported action space is intentionally limited to the code-verified
surface:

```text
TP / PP / replica count / scheduler / batch cap / Sarathi chunk size /
PD on-off / P:D ratio / PD bandwidth sensitivity / QPS under SLO
```

EP search, EP/AllToAll network scoring, topology-aware PD congestion,
heterogeneous scheduling, and prefix-aware routing are future extensions.

