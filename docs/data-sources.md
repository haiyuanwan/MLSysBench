# Data Sources & Task Construction

How to build the MLSysBench dataset: where tasks come from, how reference solutions are validated, and licensing considerations.

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

## 2. L1 Sources: Analysis & Decision Tasks

### 2.1 Real Profiling Traces from Inference Frameworks

**Most valuable source.** Run vLLM/SGLang on standard models, collect profiling data, and create analysis questions.

**vLLM built-in benchmarks:**
- `vllm bench serve`, `vllm bench throughput`, `vllm bench latency`
- Run with different configs (batch size, seq_len, TP degree) and collect `torch.profiler` traces
- Export Chrome trace JSON → create "identify the bottleneck" questions

**torch.profiler integration:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, profile_memory=True
) as prof:
    model(inputs)
# Text output suitable for agent consumption
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Nsight Systems/Compute:**
```bash
nsys profile --trace=cuda,nvtx --output=profile python run_inference.py
nsys export --type=json profile.nsys-rep  # parseable output
```

**Example L1 tasks from profiling:**
- Given Llama-3-8B torch.profiler trace on A100: Which kernel dominates? Is decode memory-bound or compute-bound?
- Given vLLM throughput curves at different `max_num_seqs`: Why does throughput plateau?
- Given TP4 vs TP2+PP2 performance data: Which config is better and why?

### 2.2 TensorRT-LLM Technical Blog Series

NVIDIA's TensorRT-LLM repo contains 17+ deep-dive blogs at `docs/source/blogs/tech_blog/`. Each is a complete optimization case study with profiling data, Roofline analysis, and strategy decisions.

| Blog Topic | Extractable L1 Tasks |
|------------|---------------------|
| Optimizing DeepSeek R1 on Blackwell | MLA weight absorb decision, DP vs TP, FP4 vs FP8 |
| Scaling Expert Parallelism (3-part) | MoE parallelism strategy analysis |
| Sparse Attention in TRT-LLM | Sparse attention pattern identification |
| Disaggregated Serving | Prefill-decode separation decision |
| N-Gram Speculative Decoding | Speculation strategy selection |
| Tuning CUDA Graph Batch Sizes | Batch size tuning analysis |

### 2.3 University Course Assignments

**Stanford CS149 Assignment 5** (with GPU MODE):
- https://github.com/stanford-cs149/asst5-kernels
- Tasks: FlashAttention, SwiGLU kernels on H100
- Students write work logs documenting profiling and optimization decisions — exactly L1 analysis

**MIT 6.5940 EfficientML** (Han Song):
- Labs on quantization (AWQ/GPTQ), pruning, distillation, efficient inference
- AWQ (MIT) and GPTQ (Apache-2.0) code directly usable

**GPU MODE Lecture Series** (106+ lectures):
- https://github.com/cuda-mode/lectures
- Key lectures for L1: #1 Profiling, #7 Quantization, #8 CUDA Perf Checklist, #12 Flash Attention, #16 Hands-on Profiling, #22 Speculative Decoding in vLLM, #35 SGLang Optimization

### 2.4 MLPerf Submission Reports

- 1,800+ peer-reviewed results in MLPerf Inference v5.0/v6.0
- Repo: https://github.com/mlcommons/inference (Apache-2.0)
- Task examples: Compare two submissions' configs and explain performance difference; predict Offline vs Server scenario performance

### 2.5 SGLang / AMD / AWS Performance Blogs

- SGLang: "7x Faster DeepSeek MLA", "25x on GB300 NVL72", "DeepSeek PD Disaggregation on 96 H100s"
- AMD: "Supercharge DeepSeek-R1 on MI300X" via ROCm blog
- AWS: Tsinghua's winning NKI-MoE code includes Report.pdf with detailed profiling

---

## 3. L2 Sources: Implementation Tasks

### 3.1 Real PRs from vLLM/SGLang (Primary Source)

Extract self-contained, single-optimization PRs. Use the PR's baseline as the task starting point and the final code as reference solution.

**vLLM Speculative Decoding PRs:**

| PR | Title | Extractable Task |
|----|-------|-----------------|
| #41745 | Gemma4 MTP speculative decoding | Implement MTP speculation |
| #37512 | MiniMax-M2: Eagle3 speculative decoding | Implement EAGLE3 |
| #39487 | Custom callable proposer backend | Implement custom proposer |

**vLLM Quantization PRs:**

| PR | Title | Extractable Task |
|----|-------|-----------------|
| #42566 | W4A16 NVFP4 fused MoE + mixed-precision | Implement FP4 MoE quantization |
| #42952 | FP8 block-scaled quantization on XPU | Implement FP8 quantization |
| #42124 | LM head quantization for ModelOpt | Implement LM head quantization |

**vLLM Attention PRs:**

| PR | Title | Extractable Task |
|----|-------|-----------------|
| #19152 | Split-KV Unified Triton Attention | Implement Split-KV attention |
| #11277 | NKI flash-attention with paged KV | Implement NKI flash attention |

**SGLang Kernel PRs:**

| PR | Title | Extractable Task |
|----|-------|-----------------|
| #26894 | Fuse norm+rope+hadamard Triton kernel | Implement fused Triton kernel |
| #24930 | Triton sparse MLA kernel | Implement sparse MLA kernel |
| #24897 | Port fused SiLU+clamp+FP8 quant | Implement fused activation+quant |

**Extraction method:**
1. `gh search prs --repo vllm-project/vllm "<keyword>" --merged`
2. Select self-contained PRs with single optimization goal
3. Extract baseline code as task input, final code as reference
4. Simplify complex PRs to keep only core algorithm

### 3.2 Existing Kernel Benchmarks (Curated Selection)

Select inference-relevant tasks from existing benchmarks:

| Source | Available Tasks | What to Select |
|--------|----------------|----------------|
| **KernelBench** (HF: ScalingIntelligence/KernelBench) | 250 PyTorch→CUDA | GEMM, LayerNorm, Softmax, attention-related |
| **TritonBench** (HF: LiShangZ/tritonbench) | 184 Triton operators + 8K RAG data | Attention, normalization, activation operators |
| **KernelBenchX** (HF: BonnieWang/KernelBenchX) | 176 Triton tasks, 15 categories | Quantization tasks (currently 0/30 solved!) |
| **SOL-ExecBench** (HF: nvidia/SOL-ExecBench) | 235 DL kernel problems | Most inference-relevant subsets |
| **GPU MODE reference-kernels** | PrefixSum, GEMM, etc. | GEMM and inference-relevant primitives |

### 3.3 Competition Problems (Adapted)

| Competition | Open-Source Code | Adaptable Task |
|-------------|-----------------|----------------|
| MLSys 2026 AWS MoE | [thustorage/NKI-MOE](https://github.com/thustorage/NKI-MOE) | Simplified MoE megakernel on GPU |
| ASPLOS 2025 NKI | [thustorage/nki-llama-contest](https://github.com/thustorage/nki-llama-contest) | GEMM/GEMV tiling optimization |
| GPU MODE AMD $100K | [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) | Direct task.yml + reference.py |
| Stanford CS149 Asst5 | [stanford-cs149/asst5-kernels](https://github.com/stanford-cs149/asst5-kernels) | FlashAttention, SwiGLU kernels |

### 3.4 Textbook Reference Implementations

| Technique | Source | License | Core Files |
|-----------|--------|---------|------------|
| FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | Forward/backward CUDA kernels |
| PagedAttention | vLLM `vllm/attention/`, `csrc/attention/` | Apache-2.0 | Block allocator + paged attention kernel |
| GPTQ | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | Apache-2.0 | `gptq.py` + `quant_cuda_kernel.cu` |
| AWQ | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | MIT | AWQ search + CUDA kernels |
| Liger Kernel | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | BSD-2 | Triton: RMSNorm, RoPE, SwiGLU, CrossEntropy |
| Continuous Batching | vLLM `vllm/core/scheduler.py` | Apache-2.0 | Scheduler with dynamic insert/remove |
| Prefix Caching | SGLang RadixAttention | Apache-2.0 | Radix tree for prefix sharing |

---

## 4. L3 Sources: End-to-End Optimization

### 4.1 Simplified Competition Tasks

| Original Task | Simplified Version |
|---------------|-------------------|
| MLSys 2026: Qwen3-30B-A3B on Trainium2/3 | Mixtral-8x7B MoE optimization on single A100 |
| ASPLOS 2025: Llama 3.2 1B on Trainium1 | Llama-3-8B optimization on A100 |
| GPU MODE Leaderboard | Start from basic kernels → full inference pipeline |

### 4.2 InferenceBench Methodology (Adapted)

Borrow InferenceBench's setup (model + GPU + time budget + scenario), but add:
- Kernel writing requirements (not just config tuning)
- Multi-model support (not just Mistral-7B)
- Intermediate process evaluation (profiling interpretation, strategy explanation)
- Analysis log requirements (not just final throughput number)

### 4.3 TensorRT-LLM Blog Case Studies

Each blog is a complete E2E optimization case. Example — "Optimizing DeepSeek R1 on Blackwell":
1. Analysis: MLA/MoE architecture characteristics
2. Precision: FP8 KV cache vs BF16, FP4 vs FP8 weights
3. Parallelism: DP for Attention + EP for MoE
4. Kernels: MLA absorb optimization, FP8 attention
5. Runtime: CUDA Graph, batch tuning
6. Result: 2000 → 4600 TPS/GPU

→ Task: Given DeepSeek-R1-Distill-7B baseline, maximize throughput on single A100 within 2 hours.

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

## 6. Reference Solution Validation

### L1: Analysis Tasks
- Multiple inference optimization experts independently answer → consensus
- Profiling analysis has deterministic answers (kernel X takes Y%, model is memory/compute-bound)
- Configuration decisions verified by actually running both configs

### L2: Implementation Tasks
- **Kernel correctness**: Random inputs, compare against PyTorch reference (rtol=1e-3, atol=1e-3 for FP16)
- **Kernel performance**: KernelBench's `fast_p` (correct + speedup ≥ p) or SOL-ExecBench's SOL-Score
- **Algorithm quality**: Paper-reported perplexity and speedup as reference (e.g., AWQ on WikiText-2)

### L3: End-to-End Tasks
- Upper bound: competition winning solutions
- Baseline: default framework config (vLLM/SGLang defaults)
- Grading: speedup tiers (e.g., 1.5x = pass, 2x = good, 3x+ = excellent)
- Quality gate: MMLU-Pro accuracy ≥ 0.95× baseline (following InferenceBench)

---

## 7. Construction Roadmap

### Phase 1 (1-2 weeks, minimal effort)
1. Select 15-20 inference-relevant kernel tasks from KernelBench/TritonBench/KernelBenchX → L2
2. Extract 5-8 analysis questions from TensorRT-LLM blogs → L1
3. Adapt InferenceBench framework for 2-3 simplified L3 tasks

### Phase 2 (3-4 weeks, engineering effort)
4. Extract 5-8 algorithm implementation tasks from vLLM/SGLang merged PRs → L2
5. Run profiling on target hardware, collect real traces → L1
6. Simplify MLSys/ASPLOS competition tasks → L3

### Phase 3 (ongoing, expert involvement)
7. Expert-design unique L1 tasks (Roofline analysis, KV cache estimation)
8. Expert-design unique L2 tasks (PagedAttention allocator, continuous batching scheduler)
9. Expert validation of all reference solutions and difficulty calibration
