# Proposed Source-Grounded Task Cards

These are design-review task cards, not runnable tasks or publication
candidates. They make the proposed benchmark unit concrete before we spend
effort implementing a runner, collecting calibration data, or sealing final
inputs.

Each card has three independent anchors:

1. a real system mechanism, such as an upstream change or measured training
   framework behavior;
2. a real workload source, such as a licensed serving trace or an AICB workload
   generated from a pinned framework configuration; and
3. a fixed SimAI/AICB revision that makes development experiments cheap.

Public and final phases always use the same mechanism and simulator revision.
They differ only in declared deployment variables: workload windows, cluster
scale, topology, or hardware profile. The final evaluator owns exact windows,
seeds, and measurements.

## Common Contract

For every promoted card:

- the agent receives public deployment metadata, an unoptimized baseline,
  allowed actions, and a bounded development evaluation API;
- it may not read or invoke the final workload;
- it submits one configuration or bounded patch after development;
- the final evaluator runs multiple private cases, including at least one
  operationally plausible counterexample;
- the task is accepted only after the baseline, expert artifact, and a
  pre-registered calibration configuration set are replayed on hardware.

The expert artifact selects and validates a real mechanism. It is never an
answer key: the final cases must include conditions where blindly copying it is
neutral or harmful.

## S1: Chunked-Prefill Domain Transfer

**Status:** nearest pilot. The source intake already exists at
[`azure2023_vllm_chunked_prefill`](azure2023_vllm_chunked_prefill/).

| Field | Definition |
|---|---|
| Real mechanism | vLLM's end-to-end chunked-prefill change, anchored by PRs #3853 and #3884. |
| Workload anchor | Azure 2023 code and conversation inference traces, revision-pinned and content-free. |
| Agent decision | Scheduler family, chunk size, and batch cap. The model, trace replay semantics, and GPU profile stay immutable. |
| Public development | Several private-to-the-agent evaluator windows from the code domain. The task statement discloses that final traffic has higher arrival pressure and a different prefill/decode mix, but not final rows or seeds. |
| Hidden final | Multiple conversation-domain windows at native timing, plus a low-load counterexample window. |
| Main score | Robust SLO-bounded goodput across final windows. Report P99 TTFT and TBT separately. |

Why the phases are one problem: both replay real LLM serving traces through the
same chunked-prefill scheduler mechanism. The final does not invent a different
reward function; it changes the operational mix that makes the chunk-size
tradeoff matter.

Required split validation before agent runs:

1. sweep the legal scheduler settings over many code and conversation windows;
2. verify that long-prefill fraction, arrival pressure, and decode length
   predict the chunk-size tradeoff;
3. retain cases where the public optimum is not always final-optimal, but a
   robust policy can outperform a fixed recipe;
4. run the pre-registered sweep subset on the pinned vLLM revision and hardware.

Current limitation: both source traces are public historical data. This is a
reproducible transfer task, not a contamination-resistant headline task. Exact
final selectors must still be evaluator-private.

## T1: Megatron Parallelism Scale Transfer

**Status:** near-term training task. AICB already supports Megatron training
workload generation and AIOB computation profiles.

| Field | Definition |
|---|---|
| Real mechanism | The interaction of data, tensor, and pipeline parallelism with communication and pipeline bubbles in Megatron-style training. |
| Workload anchor | AICB workload generated from a pinned Megatron configuration and an AIOB profile captured on the target GPU class. |
| Agent decision | Valid `(DP, TP, PP, micro_batch_size, gradient_accumulation)` decomposition with fixed model, sequence length, global batch, optimizer semantics, and world size. |
| Public development | 32-GPU, high-bandwidth topology; several sequence-length and micro-batch observations under a fixed model and global batch. |
| Hidden final | 128-GPU topology with a declared but not fully specified network constraint, plus held-out sequence-length and micro-batch cases. |
| Main score | Geometric mean iteration-time improvement across final cases, subject to memory feasibility and fixed global-batch/correctness invariants. |

Why the phases are one problem: the training graph and numerical training
semantics are unchanged. Only scale and topology change, so the agent must
reason about the communication-to-compute ratio rather than discover a new
algorithm.

Required calibration:

- generate the exact AICB workload from the pinned Megatron revision;
- replay baseline and a stratified set of parallelism plans on 32 and 128 GPUs;
- report whether simulator and hardware agree on pairwise plan selection, not
  merely absolute iteration time;
- reject plans that change global batch or silently change numerical semantics.

This is the shortest path to a real training track because the vendored AICB
stack already generates Megatron training workloads. It still requires a
training runner, topology manifests, and hardware evidence before it is a
benchmark task.

## T2: MoE Expert-Parallel Topology Transfer

**Status:** source-grounded design; simulator telemetry work required.

| Field | Definition |
|---|---|
| Real mechanism | MoE token dispatch and all-to-all communication compete with tensor/pipeline communication; AICB includes DeepSeek training support and updated Megatron MoE communication. |
| Workload anchor | A pinned DeepSeek or Megatron-MoE training configuration, with router-load histograms captured from a real training run. |
| Agent decision | Valid `(EP, TP, PP, DP, micro_batch_size)` plan and expert placement policy while model architecture, top-k routing, global batch, and optimizer remain fixed. |
| Public development | 64 GPUs with balanced expert load and a high-bisection topology. |
| Hidden final | 256 GPUs with evaluator-owned expert-load skew and a constrained inter-node topology. |
| Main score | Worst-case step time across final load/topology cases, with an OOM and global-batch validity gate. |

Why the phases are one problem: the model and routing semantics stay fixed;
only deployment scale, topology, and observed load skew change. A strategy that
always maximizes expert parallelism should win on some cases and lose on others.

Blocking gap: the task must not fabricate expert skew with a hand-written
timing equation. Intake needs real router-load histograms, and the simulator
adapter must consume them. Until then this remains a design card, not a fixture.

## S2: Coding-Agent Prefix-Cache Policy Transfer

**Status:** source-grounded serving design; Vidur cache-reuse support required.

| Field | Definition |
|---|---|
| Real mechanism | Prefix-cache retention and eviction under long coding-agent sessions. |
| Workload anchor | TraceLab coding-agent sessions, subject to license, revision, and redistribution review. |
| Agent decision | Cache budget partition, retention/eviction policy, and batching policy. The serving model and request semantics remain fixed. |
| Public development | A sampled set of coding-agent sessions with disclosed aggregate prefix-reuse and human-gap statistics. |
| Hidden final | Time-separated sessions and an evaluator-owned mixture with low reuse and bursty tool-call returns. |
| Main score | Robust goodput under TTFT/TBT SLOs, with a cache-capacity gate. |

Why the phases are one problem: all cases are coding-agent serving sessions
under the same cache mechanism. The final changes the reuse and idle-gap
distribution, which is precisely the operational uncertainty a cache policy
must handle.

Blocking gap: the current Vidur path does not model the required prefix reuse,
cache eviction, or human-paced gap behavior. First implement and validate that
mechanism against a real serving system; do not approximate it with a hidden
lookup table.

## Recommended Build Order

1. Promote S1 from intake to a sealed multi-window pilot after simulator-to-vLLM
   calibration.
2. Build T1 as the first training runner and run its calibration sweep on the
   available cluster.
3. Add T2 only after real MoE routing telemetry is available.
4. Add S2 only after a cache-aware serving model is validated.

This creates one source-grounded serving task and one source-grounded training
task before broadening coverage. It avoids treating hand-authored response
surfaces as benchmark evidence.
