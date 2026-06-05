# MLSysBench Design Proposal

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

### Level 1: Analysis & Decision (10-15 tasks)

Tests whether the agent *understands* inference optimization.

**Example tasks**:
1. Given profiling data (nsight trace), identify whether a kernel is memory-bound or compute-bound
2. Given model parameters and hardware specs, estimate VRAM usage and theoretical throughput ceiling
3. Given a scenario (model + hardware + SLO), select the optimal quantization strategy
4. Given two serving configurations, predict which achieves lower P99 latency and explain why
5. Analyze a Roofline plot and recommend which optimization class to pursue
6. Given a MoE model's expert activation statistics, recommend expert parallelism strategy
7. Estimate KV cache memory at different batch sizes and sequence lengths
8. Given a multi-GPU setup, recommend TP vs PP vs TP+PP configuration with justification
9. Identify optimization opportunities from a Transformer block's kernel timeline
10. Given an inference throughput regression, diagnose the root cause from logs

**Evaluation**: Multiple-choice or structured response; graded by correctness of analysis and strategy selection.

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

## 3. Evaluation Framework

### Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **Correctness** (pass@k) | L2, L3 | Output matches reference implementation on randomized inputs |
| **Speedup ratio** | L2, L3 | `baseline_latency / optimized_latency` |
| **Throughput gain** | L3 | `optimized_throughput / baseline_throughput` |
| **Decision accuracy** | L1 | Fraction of analysis/strategy questions answered correctly |
| **Interaction efficiency** | L2, L3 | Number of agent rounds to achieve target speedup |
| **Quality preservation** | L3 | Output quality (perplexity, accuracy) remains within tolerance |

### Composite Score

```
MLSysBench Score = w1 × L1_accuracy + w2 × L2_score + w3 × L3_score

where:
  L1_accuracy = correct_analyses / total_analyses
  L2_score = Σ (correctness_i × min(speedup_i / target_speedup_i, 1.0)) / N
  L3_score = Σ (correctness_i × speedup_i × efficiency_bonus_i) / M
  efficiency_bonus = 1.0 + max(0, 1.0 - rounds / max_rounds)  # bonus for fewer rounds
```

Suggested weights: w1=0.2, w2=0.4, w3=0.4 (implementation and E2E matter most).

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

### Source of Tasks
1. **Adapted from competitions**: Simplified versions of MLSys/ASPLOS competition tasks
2. **Extracted from frameworks**: Real optimization PRs from vLLM, SGLang, TensorRT-LLM repos
3. **Expert-designed**: Custom tasks designed by systems researchers
4. **Profiling-based**: Real profiling traces from production workloads

### Quality Assurance
- Each task has a verified reference solution with known speedup
- Multiple difficulty ratings validated by expert panel
- Baseline implementations provided as starting points
- Test cases cover edge cases (varying sequence lengths, batch sizes, dtypes)

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
