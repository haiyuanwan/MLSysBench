# Data Sources & Task Construction

How to build the MLSysBench dataset: where tasks come from, how baselines are constructed, and why the task selection is representative.

## 0. Design Principles

### Results-Driven, Not Code-Matching

MLSysBench evaluates agents by **measured performance outcomes**. Each task provides:
- A **baseline** (unoptimized starting point) — written by us
- A **correctness oracle** (numerical equivalence check) — automated
- A **performance measurement** (timing infrastructure) — standardized

No ground truth implementation is needed. Agents are free to optimize however they want.

### Data-Driven Task Selection

Task credibility comes from three pillars, not from the baseline code itself:

```
Pillar 1: Profiling      Pillar 2: Literature       Pillar 3: Community
real workload profiling → published optimization  →  competition problems
identifies bottlenecks    papers validate the gap     confirm practical relevance
```

### Task Construction Pipeline

```
Step 1                   Step 2                  Step 3                Step 4
Profile real models  →   Identify bottlenecks →  Write naive baseline → Validate optimization
on target hardware       by time percentage      (3-5 lines PyTorch)    space exists (cite lit)
```

Each task ships:
```yaml
task: fused_rmsnorm_residual
source:
  profiling: "Llama-3-8B decode, A100, batch=32 — RMSNorm+Add takes 12%"
  literature: "Liger Kernel (BSD-2) fused impl achieves 2.1x"
  competition: "GPU MODE PMPP series has similar task"
baseline:
  code: baseline.py    # naive PyTorch, written by us
  perf: 0.42ms         # measured on target hardware
known_ceiling:
  best_oss: 0.20ms     # best known open-source (Liger Kernel)
  theoretical: 0.15ms  # memory bandwidth bound
```

### How Competitions Build Their Baselines (Reference)

| Competition | Baseline | Who Wrote It | Difficulty |
|-------------|----------|-------------|------------|
| AWS MoE (MLSys 2026) | Neuron SDK default compilation of PyTorch model | AWS Neuron team | Must beat compiler auto-optimization with hand-written NKI |
| FlashInfer (MLSys 2026) | FlashInfer's own production kernels | FlashInfer team | Must beat already highly-optimized SOTA |
| GPU MODE | 1-line PyTorch or naive Python loop | Community contributors | Maximum optimization headroom |
| ASPLOS 2025 NKI | Neuron SDK default Llama 3.2 1B | AWS Neuron team | Same as AWS MoE but simpler model |

MLSysBench follows a mix of GPU MODE (naive baselines for L2) and AWS NKI (framework defaults for L3).

---

## 1. License Overview

All major sources use permissive licenses — code can be legally reused with attribution.

| Source | License | Reusability |
|--------|---------|-------------|
| vLLM | Apache-2.0 | Free to reuse with attribution |
| SGLang | Apache-2.0 | Free to reuse |
| FlashAttention | BSD-3-Clause | Free to reuse |
| KernelBench | MIT | Free to reuse |
| TritonBench | Apache-2.0 | Free to reuse |
| KernelBenchX | Apache-2.0 | Free to reuse |
| SOL-ExecBench | Apache-2.0 | Free to reuse |
| InferenceBench | Apache-2.0 | Free to reuse |
| TensorRT-LLM | Apache-2.0 | Free to reuse (check NVIDIA addendum) |
| GPTQ | Apache-2.0 | Free to reuse |
| AWQ | MIT | Free to reuse |
| GPU MODE reference-kernels | MIT | Free to reuse |
| Liger Kernel | BSD-2-Clause | Free to reuse |
| MLPerf Inference | Apache-2.0 | Free to reuse |

## 2. Profiling-Driven Task Selection (Primary Methodology)

The most credible way to select tasks: **profile real inference workloads, identify actual bottlenecks, and turn each bottleneck into an optimization task.**

### 2.1 Profiling Setup

Profile 3 representative models × multiple configs on target hardware:

| Model | Architecture | Why |
|-------|-------------|-----|
| Qwen2.5-7B (Apache-2.0) | Dense, GQA 28Q/4KV, SwiGLU, RMSNorm | Most complete dense Transformer |
| Mixtral-8x7B (Apache-2.0) | MoE, 8 experts, GQA | MoE-specific bottlenecks |
| Mistral-7B-v0.3 (Apache-2.0) | Dense, GQA, SWA | Serving optimization (InferenceBench validated) |

**Tools:**
```bash
# Kernel timeline
nsys profile --trace=cuda,nvtx -o profile python run_inference.py
nsys export --type=json profile.nsys-rep

# Detailed kernel metrics
ncu --set full --export ncu_output python run_kernel.py

# Python-level profiling
python -c "
import torch
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, profile_memory=True
) as prof:
    model.generate(inputs, max_new_tokens=256)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
"
```

### 2.2 Expected Bottleneck → Task Mapping

Based on published profiling studies and known optimization gaps:

| Bottleneck | Time% (decode) | Optimization Gap | Literature Support | → Task |
|-----------|---------------|-----------------|-------------------|--------|
| Attention (GQA/MHA) | ~35% | FlashAttention achieves 2-4x | Dao 2022, arXiv:2205.14135 | Optimize attention kernel |
| FFN GEMM | ~28% | W4A8 quantization 1.5-3x | AWQ, Lin 2024, MLSys Best Paper | Quantized matmul |
| RMSNorm + Residual | ~12% | Fusion achieves 2x | Liger Kernel (LinkedIn, BSD-2) | Fused normalization kernel |
| RoPE | ~5% | Fusion with attention 1.5x | Known optimization in vLLM/SGLang | Fused RoPE kernel |
| KV Cache ops | ~8% | PagedAttention reduces 50% mem | Kwon, SOSP 2023 | KV cache management |
| MoE routing + dispatch | ~15% (MoE) | Expert fusion 2-4x | Tsinghua NKI-MoE, MLSys 2026 1st | MoE kernel optimization |
| Softmax | ~3% | Online softmax saves one pass | Milakov & Gimelshein 2018 | Fused online softmax |
| Serving overhead | N/A | Continuous batching 2-5x | Sarathi-Serve, Agrawal 2024 | E2E serving optimization |

**The profiling data itself is a paper contribution** — it answers "what matters most in inference optimization" with empirical evidence.

### 2.3 Cross-Validation Sources

Each task selection is validated against multiple independent sources:

| Source | What it validates | Link |
|--------|------------------|------|
| **Competition problems** | Academic community agrees this is a meaningful challenge | MLSys/ASPLOS/GPU MODE |
| **Framework changelogs** | Industry prioritizes this optimization | vLLM/SGLang release notes |
| **Published papers** | Optimization gap is documented | See literature column above |
| **MLPerf submissions** | Real-world competitive dimension | mlcommons.org |
| **TRT-LLM tech blogs** | NVIDIA engineers optimize this in production | 17+ blogs in TRT-LLM repo |
| **GPU MODE lectures** | Teaching community considers this essential | 106+ lectures, cuda-mode/lectures |

---

## 3. Baseline Construction

### 3.1 Baseline Philosophy

Baselines are **naive correct implementations we write ourselves**. They are NOT extracted from external sources, eliminating data contamination concerns entirely.

Following competition conventions:
- **L2 tasks**: PyTorch eager (like GPU MODE) — maximum optimization headroom
- **L3 tasks**: Framework default config (like AWS NKI competitions) — realistic starting point

### 3.2 L2 Baselines: Naive PyTorch

Each L2 baseline is 3-10 lines of obvious PyTorch code:

```python
# Attention baseline — correct but unoptimized
def attention_baseline(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# RMSNorm + Residual — two separate operations
def rmsnorm_residual_baseline(x, residual, weight, eps=1e-6):
    x = x + residual
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * norm * weight

# MoE — serial expert execution
def moe_baseline(x, router_logits, experts):
    weights = torch.softmax(router_logits, dim=-1)
    top_k_weights, top_k_indices = weights.topk(2, dim=-1)
    output = torch.zeros_like(x)
    for i, expert in enumerate(experts):
        mask = (top_k_indices == i).any(dim=-1)
        if mask.any():
            output[mask] += expert(x[mask]) * top_k_weights[mask, (top_k_indices[mask] == i).nonzero()[:, 1]]
    return output
```

### 3.3 L3 Baselines: Framework Defaults

```bash
# L3 serving baseline — vLLM with zero tuning
vllm serve Qwen/Qwen2.5-7B --dtype float16
# No quantization, no custom kernels, no config optimization

# L3 inference baseline — naive PyTorch generate
python -c "
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', torch_dtype=torch.float16)
output = model.generate(input_ids, max_new_tokens=256)
"
```

### 3.4 Optimization Ceiling Reference

Each task documents a known performance ceiling from literature (NOT as ground truth, but to confirm optimization space exists and calibrate difficulty):

| Difficulty | Baseline | Known Ceiling | Headroom | Purpose |
|-----------|----------|---------------|----------|---------|
| Easy | Naive Python loop | PyTorch vectorized | 10-100x | Verify basic competence |
| Medium | PyTorch eager (cuBLAS) | Published optimized kernel | 1.5-5x | Main discriminator |
| Hard | Default framework config | Tuned production setup | 1.2-2x | Separate top agents |
| Expert | Optimized kernel | Hardware theoretical peak | <1.5x | Ceiling challenge |

---

## 4. Existing Resources (for reference, not as ground truth)

These resources inform task design and validate optimization potential. They are NOT used as ground truth solutions.

### 4.1 Known Optimization Implementations (Ceiling References)

| Technique | Source | License | Role in MLSysBench |
|-----------|--------|---------|-------------------|
| FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | Ceiling reference for attention tasks |
| Liger Kernel | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | BSD-2 | Ceiling for fused normalization/activation |
| AWQ | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | MIT | Ceiling for quantization tasks |
| PagedAttention | vLLM | Apache-2.0 | Ceiling for KV cache tasks |

### 4.2 Competition Problems (Task Design Reference)

| Competition | Open-Source Code | Informs Task Design |
|-------------|-----------------|-------------------|
| MLSys 2026 AWS MoE | [thustorage/NKI-MOE](https://github.com/thustorage/NKI-MOE) | MoE optimization task scope |
| GPU MODE | [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) | Task format (task.yml + reference.py) |
| FlashInfer Contest | [flashinfer-ai/flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) | Evaluation infrastructure |

### 4.3 Educational Resources (Domain Knowledge)

| Resource | Link | Useful For |
|----------|------|-----------|
| GPU MODE Lectures (106+) | [cuda-mode/lectures](https://github.com/cuda-mode/lectures) | Understanding optimization techniques |
| TRT-LLM Tech Blogs (17+) | TensorRT-LLM `docs/source/blogs/` | Real optimization case studies |
| MIT 6.5940 EfficientML | hanlab.mit.edu | Quantization / pruning techniques |
| Stanford CS149 Asst5 | [stanford-cs149/asst5-kernels](https://github.com/stanford-cs149/asst5-kernels) | Kernel optimization assignments |

---

## 5. Model Weights

| Model | Size | License | Architecture | Best For |
|-------|------|---------|-------------|----------|
| **Qwen2.5-7B** | 7.6B | Apache-2.0 | GQA (28Q/4KV), RoPE, SwiGLU, RMSNorm | L2 kernel tasks (most complete structure) |
| **Mistral-7B-v0.3** | 7B | Apache-2.0 | GQA, SWA | L3 serving optimization |
| **Mixtral-8x7B-v0.1** | 46.7B (12.9B active) | Apache-2.0 | MoE (8 experts), GQA | MoE optimization tasks |
| Llama-3.1-8B | 8B | Llama Community License | GQA, RoPE, SwiGLU, 128K | General dense model |
| DeepSeek-R1-Distill-7B | 7B | MIT | Simplified R1 | MLA concept tasks |

**Recommendation:** Prefer Apache-2.0 models (Qwen2.5-7B, Mistral-7B, Mixtral-8x7B) over Llama for cleaner licensing.

---

## 6. Evaluation Methodology

### L2: Correctness + Speedup

```python
# Correctness: optimized output matches baseline on random inputs
for _ in range(10):
    inputs = generate_random_inputs(task.config)
    baseline_out = baseline_fn(*inputs)
    optimized_out = optimized_fn(*inputs)
    assert torch.allclose(baseline_out, optimized_out, rtol=1e-3, atol=1e-3)

# Performance: speedup over baseline
baseline_time = benchmark(baseline_fn, warmup=10, trials=100)
optimized_time = benchmark(optimized_fn, warmup=10, trials=100)
speedup = baseline_time / optimized_time
```

### L3: Throughput/Latency + Quality Gate

- **Performance**: measured throughput (tok/s) or latency (ms) improvement
- **Quality gate**: MMLU-Pro accuracy ≥ 0.95× baseline (following InferenceBench)
- **No ground truth needed**: we only measure whether the agent made things faster while keeping them correct

---

## 7. Construction Roadmap

### Phase 1 (2-3 weeks)
1. **Profile** Qwen2.5-7B / Mixtral-8x7B on A100, identify top-10 bottleneck kernels
2. **Write baselines** for each bottleneck (naive PyTorch, 3-10 lines each)
3. **Validate optimization space** by citing published speedups from literature
4. **Build evaluation infrastructure** (timing, correctness checks, sandboxing)
5. **Pilot evaluation** on 2-3 frontier agents (Claude, GPT, Gemini)

### Phase 2 (3-4 weeks)
6. **Expand to 15-20 L2 tasks** covering kernel, algorithm, and system optimization
7. **Build 5-8 L3 tasks** (model + GPU + time budget → maximize throughput)
8. **Multi-hardware profiling** to ensure tasks are representative across GPUs
9. **Difficulty calibration** with expert panel

### Phase 3 (ongoing)
10. **Public leaderboard** with standardized evaluation
11. **Annual task refresh** with new profiling on latest models/hardware
12. **Community contribution pipeline** for new tasks
