# SimAI-Gym L1: Scheduler Choice

> **Legacy protocol prototype:** this task is retained because it exercises the
> compact evaluation path in unit tests, but it currently fails
> `validate-task`. Its mock surface is incomplete and its declared baseline is
> not a valid replay under the configured SLO. Do not use its score as
> benchmark evidence; see `docs/status-and-roadmap.md` for the repair criteria.

This example task models a long-prefill workload where the baseline vLLM-style
scheduler over-batches prompts and misses the TTFT SLO.

The agent submits a configuration diff in `changes`. The evaluator validates
that every changed key is listed in `allowed_actions.json`, merges it with
`baseline_config.json`, runs the configured runner, checks SLO gates, and scores
the result relative to `hidden/baseline_metrics.json`.

This bundled task uses the `mock` runner so the benchmark harness can be tested
without building SimAI. Real tasks should switch `runner.type` to `vidur`.
