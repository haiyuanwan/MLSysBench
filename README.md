# MLSysBench

A benchmark for evaluating LLMs and AI agents on **LLM inference optimization** tasks — spanning kernel programming, algorithm design, and system-level optimization.

## Motivation

Large language models and AI coding agents are increasingly used to write and optimize code. But can they optimize *themselves* — i.e., can an AI agent profile an LLM inference workload, identify bottlenecks, and implement effective optimizations?

Existing benchmarks like [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluate kernel generation, and [PIE](https://pie4perf.com/) evaluates code optimization edits, but **no benchmark evaluates the full-stack LLM inference optimization capability** — from analysis and decision-making to implementation and verification.

MLSysBench aims to fill this gap.

## Benchmark Design

### Three-Level Task Hierarchy

| Level | Focus | # Tasks | What it tests |
|-------|-------|---------|---------------|
| **L1: Analysis & Decision** | Profiling interpretation, bottleneck identification, strategy selection | 10-15 | Does the agent *understand* inference optimization? |
| **L2: Implementation** | Kernel writing, quantization, system components | 10-15 | Can the agent *implement* known optimization techniques? |
| **L3: End-to-End Optimization** | Full optimization pipeline on real models | 5-8 | Can the agent *independently optimize* a real inference workload? |

### Evaluation Metrics

| Dimension | Metric | Description |
|-----------|--------|-------------|
| Correctness | pass@k | Generated code matches reference output |
| Performance | Speedup ratio | Acceleration over baseline |
| Efficiency | Interaction rounds | How many rounds the agent needs |
| Decision quality | Strategy accuracy | For L1 analysis tasks |

## Documentation

- [Survey of Existing Benchmarks](docs/existing-benchmarks.md)
- [Survey of Inference Optimization Competitions](docs/competitions.md)
- [Inference Optimization Task Taxonomy](docs/task-taxonomy.md)
- [Benchmark Design Proposal](docs/design-proposal.md)

## Coverage

```
Kernel Level                    Algorithm Level              System Level
├── CUDA (GEMM, Attention)      ├── Quantization             ├── Operator Fusion
├── Triton (GQA, PagedAttn)     ├── Pruning & Sparsity       ├── Memory Management
├── NKI (AWS Trainium)          ├── Speculative Decoding      ├── Scheduling (Batching)
└── Custom Op Wrapping          ├── KV Cache Optimization     ├── Parallelism Config
                                └── MoE Optimization          └── Compilation
```

## Related Work

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Stanford, ICML 2025) — GPU kernel generation benchmark
- [PIE](https://pie4perf.com/) (ICLR 2024) — Performance-improving edits for C++ code
- [ECCO](https://ecco-code-eff.github.io/) (CMU, EMNLP 2024) — Code computational optimality
- [ParEval](https://github.com/parallelcodefoundry/ParEval) (UMD, HPDC 2024) — Parallel code generation
- [MLSys 2026 FlashInfer Full-Agent Track](https://mlsys26.flashinfer.ai/) — First competition with a pure-AI kernel generation track

## License

MIT
