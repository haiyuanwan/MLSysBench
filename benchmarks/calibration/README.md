# Simulator Calibration Bundles

Each input pairs identical configuration IDs measured by a simulator and by
repeated hardware runs. Analyze a bundle with:

```bash
python3 -m mlsysbench.simai_bench analyze-calibration \
  --input benchmarks/calibration/example.schema.json
```

`example.schema.json` demonstrates the input shape only; its values are
synthetic and are not calibration evidence. Publication bundles must identify
the task revision, hardware, model, framework, workload, seeds, warmup policy,
raw-measurement artifact hashes, and supported/unsupported decision regions.
Every configuration needs at least three hardware repeats, and a task claiming
`calibration_status: calibrated` must reference the bundle through
`calibration_bundle` in `task.json`.
