# Survey of Existing Benchmarks

Benchmarks that evaluate LLMs/AI agents on code optimization, kernel writing, and performance engineering.

## 1. Agent-Level Inference Optimization Benchmarks

### InferenceBench (aisa-group, Agents Workshop 2026)

- **Paper**: Agents Workshop 2026
- **Repo**: https://github.com/aisa-group/InferenceBench
- **Website**: https://inferencebench.ai
- **Task**: Evaluate AI coding agents' ability to deploy and optimize LLM inference serving. Given an agent a base model (Mistral-7B), a single H100, and a 2-hour time budget, the agent must deploy an OpenAI-compatible inference server and maximize performance.
- **4 Scenarios**: (A) Prefill latency/TTFT, (B) Decode latency/TPOT, (C) Throughput/req/s, (D) Composite/geomean
- **Quality gate**: MMLU-Pro accuracy threshold + anti-cheating integrity audit
- **Scale**: Evaluated 15 frontier agents — Claude Sonnet 4.6 (#1), GLM-5 (#2), Gemini 3.1 Pro (#3), GPT-5.3 Codex (#4)
- **Key findings**: Non-agent search (Random/SMAC3/TPE) beats all agents within 2 hours; 93.9% of agent runs use vLLM; agents do "shallow search" and fail to retain best configurations
- **Limitations**:
  - Focuses on serving deployment and hyperparameter tuning — does not test kernel writing, quantization algorithm implementation, or system component development
  - Single model (Mistral-7B) only
  - Does not evaluate analysis/decision-making ability (profiling interpretation, bottleneck identification)
  - No algorithm-level tasks (speculative decoding, KV cache optimization, etc.)

### PerfCodeBench (2026)

- **Paper**: [arXiv:2605.15222](https://arxiv.org/abs/2605.15222)
- **Task**: Benchmark LLMs for system-level high-performance code optimization, including parallelization and GPU operations
- **Key findings**: Measures gap between LLM-generated code and human expert optimizations
- **Limitations**:
  - General system-level code optimization, not specific to LLM inference
  - No agent multi-round interaction
  - Does not cover inference-specific techniques (speculative decoding, KV cache, quantization)

---

## 2. GPU Kernel Generation & Optimization

### KernelBench (Stanford, ICML 2025)

- **Paper**: [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)
- **Repo**: https://github.com/ScalingIntelligence/KernelBench
- **Task**: Evaluate LLMs generating correct and fast GPU kernels (CUDA/Triton/HIP) for PyTorch ML workloads
- **Structure**:
  - Level 1 (100 tasks): Single kernel operators (Conv, GEMM, LayerNorm, etc.)
  - Level 2 (100 tasks): Simple fusion patterns (Conv+Bias+ReLU, Matmul+Scale+Sigmoid)
  - Level 3 (50 tasks): Full model architectures (MobileNet, VGG, MiniGPT, Mamba)
  - Level 4 (20 tasks): HuggingFace models (experimental)
- **Metrics**: `fast_p` — fraction of tasks that are both correct and achieve speedup ≥ p over PyTorch baseline
- **Key findings**: Frontier reasoning models (e.g., o1) perform best but still match PyTorch baseline in < 20% of cases
- **Hardware**: NVIDIA L40S, A100, H100; AMD gfx942, gfx950
- **Limitations**:
  - Fixed input shapes for correctness testing
  - No train/test split
  - Performance depends on specific hardware
  - Does not cover system-level optimization

### AgentKernelArena (AMD / GPU MODE, 2026)

- **Paper**: [arXiv:2605.16819](https://arxiv.org/abs/2605.16819)
- **Task**: Evaluate complete AI agent workflows on GPU kernel optimization with generalization testing
- **Dataset**: 196 tasks covering HIP-to-HIP, Triton-to-Triton, PyTorch-to-HIP transformations
- **Metrics**: Compilation, correctness, performance grading; unseen-configuration generalization tests
- **Scale**: Tested Cursor Agent, Claude Code, Codex Agent and other real agents
- **Limitations**: Kernel-layer only; no algorithm-level or system-level tasks; no analysis/decision evaluation

### FastKernels (Snowflake / CMU, 2026)

- **Paper**: [arXiv:2605.23215](https://arxiv.org/abs/2605.23215)
- **Task**: Production-aligned GPU kernel generation benchmark covering 46 representative architectures (96.2% of HuggingFace Transformers)
- **Key findings**: Best kernel agent achieves only 0.94x aggregate speedup (slower than production baselines), exposing benchmark-to-production gap
- **Limitations**: Kernel-layer only; focuses on the benchmark-production alignment problem

### KernelBenchX (Tsinghua THUNLP, 2026)

- **Paper**: [arXiv:2605.04956](https://arxiv.org/abs/2605.04956)
- **Repo**: https://github.com/BonnieW05/KernelBenchX
- **Task**: 176 Triton kernel tasks across 15 categories, evaluating buildability, correctness, and hardware efficiency
- **Key findings**: 46.6% of correct kernels are slower than PyTorch eager; quantization tasks completely unsolved (0/30)
- **Limitations**: Triton kernel generation only; not agent evaluation

### MultiKernelBench (Nanjing University, 2025)

- **Repo**: https://github.com/wzzll123/MultiKernelBench
- **Task**: Cross-platform kernel generation across CUDA, Triton, AscendC, TileLang, Pallas, SYCL — covering NVIDIA GPU, Huawei Ascend NPU, Google TPU, Intel GPU
- **Limitations**: Kernel generation only; broadest hardware coverage but no system-level optimization

### TritonBench (Tsinghua THUNLP, ACL 2025)

- **Paper**: [arXiv:2502.14752](https://arxiv.org/abs/2502.14752)
- **Repo**: https://github.com/thunlp/TritonBench
- **Task**: 184 real Triton operators, two evaluation tracks (TritonBench-G and TritonBench-T)
- **Limitations**: Triton operator generation only

### SOL-ExecBench (NVIDIA, 2026)

- **Paper**: [arXiv:2603.19173](https://arxiv.org/abs/2603.19173)
- **Repo**: https://github.com/NVIDIA/SOL-ExecBench
- **Task**: 235 real DL kernel problems on NVIDIA B200, scored by SOL-Score (percentage of theoretical roofline)
- **Limitations**: Evaluates kernel solutions, not agents; no system-level optimization

---

## 3. Source Code Performance Optimization

### PIE — Performance-Improving Edits (ICLR 2024 Spotlight)

- **Paper**: [arXiv:2302.07867](https://arxiv.org/abs/2302.07867)
- **Website**: https://pie4perf.com/
- **Task**: LLM optimizes existing C++ competitive programming code for speed
- **Dataset**: 77,000+ submission pairs (slow → fast), 4,085 fine-tuning pairs
- **Metrics**: %Optimized, Aggregate Speedup, %Correct
- **Evaluation**: Deterministic benchmarking via gem5 full-system simulator
- **Key findings**: Fine-tuned GPT-3.5 + self-play achieves 6.86x avg speedup at Best@8, exceeding human average (3.66x)
- **Limitations**: C++ competitive programming only; gem5 vs real hardware gap

### ECCO — Evaluating Code for Computational Optimality (CMU, EMNLP 2024)

- **Paper**: [arXiv:2407.14044](https://arxiv.org/abs/2407.14044)
- **Website**: https://ecco-code-eff.github.io/
- **Task**: Two paradigms — history-based code editing and NL-based code generation
- **Metrics**: Speedup, Memory Reduction, pass@1
- **Key findings**: NL feedback best for efficiency gains; execution feedback best for correctness; multi-round iteration consistently degrades pass@1
- **Limitations**: Primarily Python

### EffiBench (NeurIPS 2024)

- **Paper**: [arXiv:2402.02037](https://arxiv.org/abs/2402.02037)
- **Repo**: https://github.com/huangd1999/EffiBench
- **Leaderboard**: https://huggingface.co/spaces/EffiBench/effibench-leaderboard
- **Task**: LLM generates efficient code, compared against human SOTA solutions
- **Dataset**: 1,000 LeetCode efficiency-critical problems
- **Scale**: Evaluated 42 LLMs (35 open-source + 7 closed-source)
- **Key findings**: GPT-4 generated code averages 3.12x slower than human optimal; worst case 13.89x (time) and 43.92x (memory)
- **Limitations**: Python LeetCode only; only tests generation, not optimization of existing code

### EvalPerf — Differential Performance Evaluation (2024)

- **Paper**: [arXiv:2408.06450](https://arxiv.org/abs/2408.06450)
- **Task**: LLM code performance on efficiency-critical programming tasks
- **Dataset**: 121 performance-challenging tasks filtered from existing benchmarks
- **Method**: Profile solutions against reference set with known efficiency levels
- **Key findings**: Model scaling laws do not apply to code efficiency; general instruction tuning improves both correctness and efficiency

### Supersonic (IEEE TSE 2024)

- **Paper**: [arXiv:2309.14846](https://arxiv.org/abs/2309.14846)
- **Task**: Seq2seq models for minimal C/C++ source-level optimization edits
- **Key findings**: Models 600x-3700x smaller than GPT-3.5/GPT-4 outperform them on code optimization

### Code-Optimise (Huawei, NAACL 2025 Findings)

- **Paper**: [arXiv:2406.12502](https://arxiv.org/abs/2406.12502)
- **Repo**: https://github.com/huawei-noah/HEBO/tree/Code_Optimise
- **Task**: Optimize code runtime while preserving correctness
- **Key findings**: 6% in-domain, 3% out-of-domain runtime reduction; 23-48% code length reduction

---

## 4. Parallel Computing & HPC

### ParEval (UMD, HPDC 2024)

- **Paper**: [arXiv:2401.12554](https://arxiv.org/abs/2401.12554)
- **Repo**: https://github.com/parallelcodefoundry/ParEval
- **Task**: LLM writes parallel code
- **Dataset**: 420 tasks across 12 problem types × 6 parallel programming models (OpenMP, MPI, CUDA, HIP, Kokkos, C++17 threads)
- **Limitations**: Requires GPU + MPI cluster; classic parallel computing problems, not inference optimization

### HPC-Coder (UMD, ISC 2024)

- **Paper**: [arXiv:2306.17281](https://arxiv.org/abs/2306.17281)
- **Task**: LLM for HPC code completion, OpenMP pragma generation, performance prediction

---

## 5. Compiler Optimization

### CompilerGym (Meta, CGO 2022 Distinguished Paper)

- **Paper**: [arXiv:2109.08267](https://arxiv.org/abs/2109.08267)
- **Repo**: https://github.com/facebookresearch/CompilerGym
- **Task**: RL environment for compiler optimization pass selection (LLVM)
- **Limitations**: RL agents, not LLMs; maintenance stalled since 2022

### Meta LLM Compiler (2024)

- **Paper**: [arXiv:2407.02524](https://arxiv.org/abs/2407.02524)
- **Task**: 7B/13B models pre-trained on 546B tokens of LLVM-IR and assembly for code size optimization and disassembly
- **Key findings**: Achieves 77% of autotuning optimization potential

---

## 6. Summary & Differentiation

### Landscape Overview

| Benchmark | Year | Evaluates Agent? | Kernel | Algorithm | System | Inference-Specific |
|-----------|------|-----------------|--------|-----------|--------|--------------------|
| **InferenceBench** | 2026 | Yes | No | No | Config tuning | Yes (serving) |
| **AgentKernelArena** | 2026 | Yes | Yes | No | No | No |
| **PerfCodeBench** | 2026 | No | Partial | No | Partial | No |
| KernelBench | 2025 | No | Yes | No | No | No |
| KernelBenchX | 2026 | No | Yes (Triton) | No | No | No |
| FastKernels | 2026 | Partial | Yes | No | No | Partial |
| MultiKernelBench | 2025 | No | Yes (6 platforms) | No | No | No |
| SOL-ExecBench | 2026 | No | Yes | No | No | No |
| PIE | 2024 | No | No | No | No | No |
| ECCO | 2024 | Partial | No | No | No | No |
| ParEval | 2024 | No | Partial (CUDA) | No | No | No |
| **MLSysBench (ours)** | **2026** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### Research Gaps (Our Opportunity)

| Dimension | Best Existing Coverage | Gap that MLSysBench Fills |
|-----------|----------------------|--------------------------|
| Kernel generation | KernelBench ecosystem (extensive) | We include kernel tasks but go beyond |
| Serving config tuning | InferenceBench | We add kernel writing + algorithm implementation |
| Agent kernel optimization | AgentKernelArena | We add algorithm + system layers |
| Analysis & decision-making | **None** | Profiling interpretation, strategy selection |
| Algorithm-level tasks | **None** | Quantization, speculative decoding, KV cache |
| System-level tasks | **None** | Scheduling, parallelism config, memory management |
| End-to-end optimization | InferenceBench (config only) | Full pipeline: profile → analyze → implement → verify |
| Full-stack coverage | **None** | Kernel × Algorithm × System in one benchmark |
