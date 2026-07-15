# Related Work and Benchmark Positioning

Last literature review: **2026-07-16**. The machine-readable companion is
[`related-work-sources.json`](related-work-sources.json). It records the primary
paper or project URL and the specific design evidence used here; it is intended
to be the starting point for a paper bibliography, not a substitute for
checking the final camera-ready versions.

## Positioning

MLSysBench should not claim novelty from inference tuning, real performance
patches, kernel generation, agentic workloads, or real traces in isolation.
Each is already represented by strong contemporary benchmarks. The defensible
research question is their intersection:

> Can an autonomous agent discover and ship inference-system optimizations
> that remain effective under workload, model, hardware, and scale shifts,
> while allocating a fixed experimental budget between cheap biased simulation
> and scarce real-system measurements?

The intended benchmark contribution is therefore **performance-portable
autonomous systems engineering under distribution shift and multi-fidelity
evaluation**.

This is an inference from the closest work reviewed below, not a claim that no
unindexed concurrent work exists. Before submission, rerun the literature
search and narrow any priority language.

## Closest benchmarks

| Work | What it already evaluates | Boundary that remains useful to MLSysBench |
|---|---|---|
| [InferenceBench](https://inferencebench.ai/assets/paper.pdf) | Open-ended inference-server optimization with held-out seeds and matched Random/TPE/SMAC baselines | Main study uses one H100 and fixed scenarios; hardware ablations rerun agents rather than requiring one submitted artifact to transfer. It does not make multi-fidelity probe allocation the task. |
| [ISO-Bench](https://arxiv.org/abs/2602.19594) | 54 optimization tasks mined from merged vLLM and SGLang performance PRs | Strong precedent for upstream-derived inference tasks; real-PR provenance alone is therefore not our novelty. |
| [SWE-efficiency](https://arxiv.org/abs/2511.06090) | 498 real performance-PR tasks across nine repositories with workloads, tests, expert patches and repeated validation | Provides the most reusable task-intake methodology; it is broader than inference and does not center deployment transfer or multi-fidelity experiments. |
| [SWE-Perf](https://arxiv.org/abs/2507.12415), [GSO](https://arxiv.org/abs/2505.23671), [SWE-Pro](https://arxiv.org/abs/2606.25530) | Repository-level runtime/memory optimization from real software changes | Establish that repository performance engineering is already a crowded benchmark category. |
| [KernelBench](https://arxiv.org/abs/2502.10517) | Correct and fast GPU kernels for 250 PyTorch workloads on real GPUs | Kernel generation is useful as one layer, not a sufficient full-benchmark thesis. |
| [FlashInfer-Bench](https://arxiv.org/abs/2601.00227) | Trace-driven kernel correctness/performance and framework substitution | A generic kernel or trace track would overlap; MLSysBench should emphasize system-level transfer and experimental decisions. |
| [FML-Bench](https://arxiv.org/abs/2605.17373) | Agent strategy and process-level performance-engineering metrics | Motivates measuring unique hypotheses, duplicate trials, best-state retention and invalid experiments, not only final score. |
| [RE-Bench](https://arxiv.org/abs/2411.15114) | AI agents versus human experts under matched time budgets | Motivates matched expert baselines and open trajectories. |
| [PaperBench](https://arxiv.org/abs/2504.01848) | Research reproduction with author-developed rubrics and PhD baselines | Motivates upstream-maintainer validation and a blind contributed-task track. |

## Systems evidence for the task dimensions

- [Vidur](https://arxiv.org/abs/2405.05465) reports a calibrated serving
  simulator and shows that configurations depend on the model–trace pair; a
  configuration transferred between pairs can be substantially suboptimal.
  This supports explicit simulator-to-hardware decision calibration and hidden
  transfer matrices.
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  establishes chunked prefill and stall-free scheduling. A task that directly
  hints at fixed chunk-size tuning is therefore a protocol fixture rather than
  a novel optimization problem.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin),
  [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao), and
  [VTC](https://www.usenix.org/conference/osdi24/presentation/sheng) motivate
  disaggregated placement, dynamic scheduling/migration, and multi-tenant
  fairness task families.
- [Agentix](https://www.usenix.org/conference/nsdi26/presentation/luo),
  [Libra](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra), and
  [JITServe](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)
  motivate program-aware agent serving, nonstationary goodput, and scheduling
  under imperfect information.

## External workload provenance

- [BurstGPT](https://github.com/HPMLL/BurstGPT) publishes a large Azure OpenAI
  workload trace including request timing and token-length information.
- [TraceLab](https://arxiv.org/abs/2606.30560) publishes coding-agent sessions,
  LLM steps, and tool-call behavior, exposing long loops, long-context/short-
  output patterns, and prefix-reuse opportunities.
- FlashInfer-Bench supplies another serving-trace schema and integration path.

Synthetic traces may reproduce mechanisms for unit tests, but must declare
`source_type: hand_authored_fixture`. A paper task may use `real_trace` only
when the exact source URL, revision, license, transformation, split, and
validator are recorded.

## Task-acquisition strategy

Publication tasks should come from four sources:

1. **Historical upstream changes:** reconstruct the parent commit of a vLLM,
   SGLang, FlashInfer, LMCache, AIBrix, TensorRT-LLM, or related performance PR.
2. **Paper artifacts:** turn one measured system intervention into a task, with
   the published method as an expert baseline rather than a hidden answer.
3. **Licensed traces:** create time- or domain-separated public/final replay
   slices without exposing final records to the agent.
4. **Blind maintainer contributions:** seal fresh issues, tests, and reference
   patches until the evaluation cutoff.

The expert patch is a baseline, not the scoring oracle. Hidden conditions must
include counterexamples where blindly applying the historical patch or public
optimum is neutral or harmful. This makes memorization insufficient while
keeping the underlying problem externally grounded.

## Evaluation questions

- **RQ1:** Can agents beat framework defaults, expert patches, and matched
  search on the public environment?
- **RQ2:** What fraction of the gain survives hidden workload/model/hardware
  shifts, and what is the worst-profile regression rate?
- **RQ3:** Can agents allocate simulator and real-system probes better than
  multi-fidelity Bayesian optimization under the same budget?
- **RQ4:** Which trajectory behaviors predict transfer—variable isolation,
  hypothesis diversity, rollback, or calibration probes?
- **RQ5:** When does systems knowledge help beyond black-box search, especially
  on sparse or counterexample-rich response surfaces?

## Review-risk controls

| Review risk | Required control |
|---|---|
| Author-invented tasks | External PR/paper/trace revision plus independent or maintainer validation |
| Arbitrary simulator equation | Identical configuration pairs on hardware; ranking, top-k overlap, and decision-agreement report |
| Memorized patch | Time split, blind tasks, hidden environment shifts, and counterexamples |
| Toy one-file API | Keep as protocol fixture; headline tasks modify real repositories or deploy online policies in real frameworks |
| Weak baseline | Naïve, framework default, expert recipe/patch, matched Random/TPE/SMAC, multi-fidelity BO, and human subset |
| Noisy speed measurement | Repeats, variance and confidence intervals; reject unstable task candidates |
| Metric gaming | Deterministic correctness, quality, integrity and fresh-replay gates before performance |
| Scaffold confound | Hold scaffold fixed in the primary comparison and report scaffold ablations separately |

## Current implementation boundary

The schema-v3 fixtures under `tasks/patch_transfer`, `tasks/policy_transfer`,
and `tasks/multifidelity` implement the protocol shape. Their provenance marks
them `fixture` and `proxy_only`; they cannot enter a paper leaderboard. The
next evidence milestone is to import externally sourced tasks and replace
`hardware_proxy` with a runner that accounts for actual GPU measurements.
