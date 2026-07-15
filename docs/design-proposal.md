# MLSysBench Design Proposal

> Historical broad-scope proposal. It is retained for research context and is
> not the implementation plan. The current normative design is the
> [benchmark protocol](benchmark-protocol.md); implemented scope and ordered
> milestones are recorded in [status and roadmap](status-and-roadmap.md).

## 1. Positioning & Differentiation

MLSysBench evaluates LLMs and AI agents on **full-stack LLM inference optimization** — a capability that no existing benchmark covers end-to-end.

|  | KernelBench | PIE / ECCO | MLSysBench |
|--|-------------|------------|------------|
| Task domain | GPU kernel generation | General code optimization | LLM inference optimization (full stack) |
| Evaluation target | Single-shot generation | Single-shot edit | Multi-round agent interaction |
| Tech stack coverage | Kernel layer only | Algorithm layer | Kernel + Algorithm + System |
| Real inference workloads | No | No | Yes |
| Tests analysis/decision ability | No | No | Yes |
| Agent-native evaluation | No | Partially (ECCO) | Yes |

## 2. Task Hierarchy

### Level 2 tasks subsume analysis ability

In an earlier design, we had a separate "Level 1: Analysis & Decision" layer with multiple-choice questions about profiling and strategy selection. We removed it because:

1. **Analysis ability is already tested by L2 and L3** — an agent that can't interpret profiling data won't achieve speedup in L2/L3 tasks.
2. **Results-driven evaluation is more meaningful** — we don't need to separately test "does the agent know what memory-bound means" if we can directly measure "does the agent optimize memory-bound kernels effectively."
3. **Avoids the ground-truth problem** — analysis questions require curated correct answers, which adds maintenance burden and potential for ambiguity.

### Level 2: Implementation (10-15 tasks)

Tests whether the agent can *implement* known optimization techniques.

**Example tasks**:
1. Write a Triton kernel for Grouped Query Attention (GQA)
2. Implement AWQ salient channel identification algorithm
3. Implement a PagedAttention block memory allocator
4. Write a fused SwiGLU activation CUDA kernel
5. Implement basic speculative decoding (draft-verify loop + rejection sampling)
6. Implement continuous batching scheduler with dynamic request insertion/removal
7. Implement W8A8 quantized inference pipeline (calibration + quantized forward)
8. Configure vLLM/SGLang for optimal throughput on a given model + GPU
9. Implement attention-score-based KV cache eviction policy
10. Write a fused RMSNorm + Residual Add kernel
11. Implement KV cache FP8 quantization with calibration and quality verification
12. Optimize kernel launch overhead in a Transformer block inference pipeline

**Evaluation**: Correctness (pass@k on randomized inputs) + Performance (speedup ratio vs baseline).

### Level 3: End-to-End Optimization (5-8 tasks)

Tests whether the agent can *independently optimize* a real inference workload through the full pipeline: profile → analyze → decide → implement → verify.

**Example tasks**:
1. **Decode throughput optimization**: Given Llama-3-8B baseline inference code on single A100, maximize decode throughput while maintaining output quality (agent must profile, identify bottlenecks, and implement optimizations)
2. **MoE serving optimization**: Given Mixtral-8x7B served with vLLM, optimize expert scheduling and kernel fusion to minimize P99 latency
3. **Long-context optimization**: Given a 128K-context inference workload, optimize attention + KV cache to reduce memory and latency
4. **Quantization pipeline**: Given a 70B model and 2×A100, design and implement a quantization + serving strategy that fits in memory with < 1% accuracy degradation
5. **Multi-GPU parallelism**: Given a model and heterogeneous GPU cluster, design optimal parallelism strategy and implement necessary changes
6. **Prefill-decode co-optimization**: Optimize a serving system handling mixed prefill and decode requests to meet SLO targets

**Evaluation**: Correctness + Speedup ratio + Number of interaction rounds + Quality of intermediate analysis.

## 3. Evaluation Philosophy

### Results-Driven, Not Code-Matching

MLSysBench evaluates agents by **measured performance outcomes**, not by comparing generated code against reference solutions. This is a deliberate design choice:

- **No ground truth code**: We provide baselines (unoptimized starting points), not reference solutions. The agent's job is to make the code faster, not to reproduce a known implementation.
- **Correctness = numerical equivalence**: The optimized code must produce outputs that match the baseline's outputs (within floating-point tolerance), not match any specific implementation.
- **Performance = measured speedup**: The only thing that matters is actual latency/throughput improvement on real hardware.

This design naturally **eliminates data contamination concerns** — even if the agent has memorized vLLM source code, what matters is whether its optimization actually achieves measurable speedup. Memorization that leads to better performance is a feature, not a bug.

```
What we provide:          What we measure:
├── Baseline code         ├── Correctness (output matches baseline numerically)
├── Hardware access       ├── Speedup (baseline_time / optimized_time)
├── Profiling tools       ├── Throughput (optimized_rps / baseline_rps)
├── Model weights         ├── Quality gate (MMLU-Pro accuracy ≥ 0.95× baseline)
└── Time budget           └── Efficiency (rounds needed to achieve target)
```

### Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **Correctness** | L2, L3 | Optimized output matches baseline output on randomized inputs |
| **Speedup ratio** | L2, L3 | `baseline_latency / optimized_latency` |
| **Throughput gain** | L3 | `optimized_throughput / baseline_throughput` |
| **Quality preservation** | L3 | Output quality (perplexity, accuracy) remains within tolerance |
| **Interaction efficiency** | L2, L3 | Number of agent rounds to achieve target speedup |

### Composite Score

```
MLSysBench Score = w1 × L2_score + w2 × L3_score

where:
  L2_score = Σ (correct_i × speedup_i) / N
  L3_score = Σ (correct_i × speedup_i × quality_gate_i × efficiency_bonus_i) / M
  efficiency_bonus = 1.0 + max(0, 1.0 - rounds / max_rounds)
```

Suggested weights: w1=0.4, w2=0.6 (E2E optimization matters most).

## 4. Agent Interface

### Standardized Tool API

The benchmark provides agents with a standardized set of tools:

```python
class MLSysBenchEnvironment:
    def read_file(self, path: str) -> str:
        """Read source code or configuration files."""

    def write_file(self, path: str, content: str) -> None:
        """Write or modify source code."""

    def run_inference(self, config: dict) -> InferenceResult:
        """Run inference with given configuration, return latency/throughput."""

    def profile(self, config: dict) -> ProfileResult:
        """Profile inference, return kernel timeline, memory usage, etc."""

    def run_tests(self) -> TestResult:
        """Run correctness tests against reference implementation."""

    def get_hardware_info(self) -> HardwareInfo:
        """Get GPU specs, memory, compute capability, etc."""

    def execute_shell(self, command: str) -> str:
        """Run arbitrary shell commands in sandboxed environment."""
```

### Evaluation Pipeline

```
┌─────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│ Task Loader │───▶│ Agent API │───▶│ Sandbox    │───▶│ Evaluator│
│ (tasks.json)│    │ (tools)   │    │ (Docker+GPU)│   │ (metrics)│
└─────────────┘    └───────────┘    └────────────┘    └──────────┘
```

## 5. Sandbox & Execution Environment

### Requirements
- Docker container with GPU passthrough (NVIDIA Container Toolkit)
- Pre-installed frameworks: PyTorch, Triton, vLLM, SGLang, TensorRT-LLM
- Pre-downloaded model weights (Llama-3-8B, Mixtral-8x7B, etc.)
- Profiling tools: nsight systems, torch.profiler, py-spy
- Deterministic timing: multiple trial runs with statistical reporting

### Hardware Tiers
- **Tier 1 (Minimal)**: Single NVIDIA A100 80GB
- **Tier 2 (Standard)**: 2× or 4× A100
- **Tier 3 (Multi-node)**: 8× A100 / H100 with NVLink

## 6. Dataset Construction

### What Each Task Provides

Every task ships a **baseline** (the unoptimized starting point) and a **correctness oracle** (a way to verify numerical equivalence). No ground truth implementation is needed.

```
task/
├── baseline.py          # Unoptimized PyTorch implementation (the starting point)
├── correctness_check.py # Compares optimized output against baseline output
├── benchmark.py         # Measures latency/throughput with proper timing
├── config.yaml          # Model, hardware requirements, time budget, quality gate
└── README.md            # Task description, constraints, what "optimization" means here
```

### Source of Baselines
1. **PyTorch eager execution**: The simplest correct implementation (e.g., standard attention, naive GEMM)
2. **Default framework configs**: vLLM/SGLang with default settings (for L3 serving tasks)
3. **Simplified competition starting points**: Adapted from MLSys/ASPLOS competition baselines
4. **Expert-designed unoptimized code**: Custom baselines for novel task scenarios

### Quality Assurance
- Each baseline is verified to produce correct outputs
- Baseline performance is measured on target hardware as the denominator for speedup
- Correctness checks use randomized inputs with appropriate tolerances per dtype
- Tasks cover varying sequence lengths, batch sizes, and dtypes to prevent shape-hardcoding

## 7. Roadmap

### Phase 1: Foundation (L1 + L2 basics)
- [ ] Define L1 analysis tasks (10 tasks)
- [ ] Implement L2 kernel writing tasks (5 tasks)
- [ ] Build sandbox environment with single-GPU support
- [ ] Implement evaluation pipeline and metrics
- [ ] Baseline evaluation on GPT-4, Claude, Gemini, DeepSeek

### Phase 2: Full L2 + L3
- [ ] Complete L2 tasks (10-15 tasks total)
- [ ] Implement L3 end-to-end tasks (5 tasks)
- [ ] Add multi-GPU support
- [ ] Build agent interaction logging and analysis
- [ ] Public leaderboard

### Phase 3: Community & Expansion
- [ ] Open task contribution pipeline
- [ ] Add AMD GPU and AWS Trainium support
- [ ] Cross-hardware generalization evaluation
- [ ] Integration with popular agent frameworks (Claude Code, Cursor, Devin, OpenHands)
- [ ] Annual benchmark refresh with new tasks
