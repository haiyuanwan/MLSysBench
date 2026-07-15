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
and raw-measurement artifact hashes.
