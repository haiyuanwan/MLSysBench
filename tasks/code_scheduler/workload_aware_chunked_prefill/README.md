# Workload-aware chunked-prefill scheduler

Optimize the runnable scheduler in `solution/scheduler.py`. This is a code task:
the evaluator executes the scheduling decisions returned by your implementation,
not a hand-authored configuration lookup table.

The scheduler receives only requests that have already arrived. Each request
observation contains its id, waiting time, remaining prefill/decode work, and
whether its first token has been emitted. Return a list of `{request_id, tokens}`
decisions within the supplied token and batch-size limits. Prefill work may be
chunked; decode requests must advance by exactly one token per batch.

The public workload contains mixed prompt/output lengths. Final evaluation uses
held-out burst and mixed-concurrency traffic. All requests must complete exactly
once, the scheduler must make progress, and every decision must respect the
resource limits. Invalid code or scheduling decisions score zero.

Run a development experiment with:

```bash
python3 evaluate_dev.py candidate.json --output candidate_result.json
```

where `candidate.json` contains `{"changes": {}}`. The helper automatically
evaluates the current `solution/scheduler.py`.
