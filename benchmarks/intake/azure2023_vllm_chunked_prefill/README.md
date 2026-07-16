# Azure 2023 × vLLM chunked-prefill intake

Status: **selected intake; not yet a publication candidate**.

The selected systems question is whether a chunked-prefill scheduling choice
developed on the Azure code trace transfers to a higher-arrival-rate,
longer-decode conversation workload. This is grounded in the vLLM chunked
prefill RFC and its scheduler and end-to-end implementation PRs, rather than in
a hand-authored performance equation.

## Why this is the first candidate

The code trace contains 8,819 requests at 2.5667 native QPS. Its P95 context is
7,302 tokens while its P50 generation is only 13 tokens. The conversation
trace contains 19,366 requests at 5.5304 native QPS, with P50/P95 generation
lengths of 129/451. That creates a useful development-to-final shift in both
arrival pressure and the prefill/decode mix.

The upstream change boundary is deliberately the full series:

- vLLM PR #3853, merge `18de88348954b7e535a62c0b7e55004f253e9f21`,
  adds the scheduler;
- vLLM PR #3884, merge `67b4221a61ace91a79aff507df0a95a01978300e`,
  is the first change explicitly described as end-to-end.

Using #3853 alone would incorrectly treat an incomplete implementation as the
expert artifact.

## Trace audit

The Azure source publishes timestamps and input/output token counts but no
prompt content. Raw inputs are CC-BY-4.0 and pinned to AzurePublicDataset
revision `207bed67dd10090b28ad4f745b2cfd41a11aace4`. Raw and processed hashes,
summary statistics, and exact source URLs are in `candidate.json`.

The checked-in Vidur `splitwise_code.csv` and `splitwise_conv.csv` files retain
all records. Their transformation is limited to relative arrival time and
column renaming. Run the verifier from the repository root after downloading
the two raw files:

```bash
python scripts/verify_azure_vllm_candidate.py \
  --code-raw /path/to/AzureLLMInferenceTrace_code.csv \
  --conversation-raw /path/to/AzureLLMInferenceTrace_conv.csv
```

There is one unresolved source discrepancy: the dataset description says the
trace was collected on November 11, 2023, but all released rows are timestamped
November 16, 2023. It is recorded as a promotion blocker for maintainer
clarification.

## What is and is not complete

Step 1 is complete when this intake and its verifier pass. Step 2 will create a
runnable task package and reproduce its simulator baseline/expert boundary.
The package must remain below paper-candidate status until independent review,
paired real-hardware calibration, a complete baseline ladder, and the human
expert study exist.
