# Upstream candidate intake

This directory records benchmark candidates before they become runnable tasks.
An intake record is evidence of source selection, not permission to report a
paper result. In particular, `selected` does not mean `pilot` or `candidate`
under the task schema.

Each candidate must pin the upstream code and data revisions, record licenses
and checksums, compare at least one plausible alternative, and list unresolved
promotion blockers. A runnable task is created only after its transformation
and replay contract are deterministic. External validation, hardware evidence,
and the baseline ladder remain separate publication gates.

Current selection:

- [`azure2023_vllm_chunked_prefill`](azure2023_vllm_chunked_prefill/README.md)
  combines the vLLM chunked-prefill PR series with the Azure 2023 code and
  conversation production traces.

For proposed serving and training tasks built from the same intake model, see
[source-grounded task cards](task-cards.md). These are design-review artifacts,
not runnable tasks or publication candidates.
