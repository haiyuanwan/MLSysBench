# Survey of Existing Benchmarks

Benchmarks that evaluate LLMs/AI agents on code optimization, kernel writing, and performance engineering.

## 1. GPU Kernel Generation

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

---

## 2. Source Code Performance Optimization

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

## 3. Parallel Computing & HPC

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

## 4. Compiler Optimization

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

## 5. Summary & Research Gaps

| Dimension | Coverage | Gap |
|-----------|----------|-----|
| Kernel generation | KernelBench ✓ | No Triton-specific or NKI benchmark |
| Source-level optimization | PIE, ECCO, EffiBench ✓ | Only competitive programming / LeetCode |
| Parallel code | ParEval ✓ | Not inference-specific |
| Compiler optimization | CompilerGym, LLM Compiler ✓ | LLVM IR only, not runtime |
| **System-level optimization** | **None** | No benchmark for scheduling, parallelism config, memory management |
| **Agent multi-round optimization** | **ECCO partially** | No benchmark for profile → analyze → implement → verify loops |
| **End-to-end inference optimization** | **None** | No benchmark combining kernel + algorithm + system |
