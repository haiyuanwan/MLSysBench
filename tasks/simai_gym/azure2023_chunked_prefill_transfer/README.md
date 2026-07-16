# Azure 2023 chunked-prefill transfer

This is the first externally grounded task package under construction. It uses
the Azure 2023 code trace for development and the higher-QPS conversation trace
for final transfer evaluation. The editable decision is the scheduler,
Sarathi-style chunk size, and batch cap; workload, model, hardware profile, and
arrival timing are immutable.

Interactive evaluation uses deterministic contiguous windows: code rows
`[2048:2176]` (128 requests) for development and conversation rows
`[8192:8448]` (256 requests) for final evaluation. Confirmation windows retain
1,024 and 2,048 requests respectively. The window generator and hashes are
checked in. Full-trace replay remains a promotion gate and is not silently
substituted by either shorter task layer.

The task is intentionally marked `publication_status: intake`. The task runner
uses Vidur's Sarathi scheduler as a simulator proxy for the chunked-prefill
mechanism upstreamed in vLLM PRs #3853 and #3884. That proxy relationship must
be calibrated against the pinned vLLM expert revision on real hardware before
the task can become a paper candidate.

The original traces contain no prompt text. Raw and processed checksums,
license, revision, trace statistics, alternative candidates, and the unresolved
November 11/16 source-date discrepancy are recorded in
[`benchmarks/intake/azure2023_vllm_chunked_prefill/candidate.json`](../../../benchmarks/intake/azure2023_vllm_chunked_prefill/candidate.json).

The checked-in baseline metric files are invalid sentinels until the framework
default is replayed at least three times. They prevent an unreproduced number
from silently becoming the score denominator.
