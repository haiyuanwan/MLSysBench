# Survey of Inference Optimization Competitions

Competitions and challenges related to LLM/ML inference optimization at academic conferences and in industry.

## 1. MLSys 2026 Competition Track

MLSys 2026 introduced its first Competition Track with three challenges, held May 17-22, 2026 in Bellevue, WA.

### 1.1 AWS Trainium MoE Kernel Challenge

- **Organizer**: AWS Neuron team
- **Repo**: https://github.com/aws-neuron/nki-moe
- **Model**: Qwen3-30B-A3B (MoE, 128 experts, 8 activated per token)
- **Hardware**: AWS Trainium2/3, single chip
- **Programming**: Neuron Kernel Interface (NKI)
- **Scoring**: `Accuracy × Reduced_Latency × Increased_Throughput × (1 + Normalized_NKI_FLOPS)` (85% performance + 15% innovation)
- **Prize**: $25K / $10K / $5K + guaranteed AWS internship interview
- **Scale**: Max 50 teams, 1-4 members each, two rounds (Trn2 → top 15 on Trn3)

**Winners**:
1. **1st Place — Tsinghua University (thustorage)**: Ruwen Fan, Shiwei Gao, Tingxu Ren, Yibin Luo
   - Code: https://github.com/thustorage/NKI-MOE
   - Key techniques: MoE Megakernel (fusing routing + RMSNorm + expert selection + MLP), fused gate/up GEMV, decode-time selective expert execution
   - Result: ~4.2x latency reduction, end-to-end 14.91s → 3.56s
2. **2nd Place — UC Berkeley**: Charles Hong
3. **3rd Place — Tsinghua/BUAA (omnimind-ai)**: Latency from 13,596ms → 5,073ms (2.72x)

### 1.2 NVIDIA FlashInfer AI Kernel Generation Contest

- **Organizer**: NVIDIA + FlashInfer + Modal
- **Website**: https://mlsys26.flashinfer.ai/
- **Hardware**: NVIDIA Blackwell B200 GPU
- **Three tracks**:
  - Track A: Fused MoE (FP8)
  - Track B: Sparse Attention (DSA)
  - Track C: Gated Delta Net (from Qwen3-Next)
- **Two submission types**:
  - **Agent-Assisted**: Human + AI collaboration
  - **Full-Agent**: Purely AI-generated kernels ← *first competition track for pure-AI kernel optimization*
- **Languages**: CuTe DSL, CUDA, Tilelang, Triton, cuTile, etc.
- **Prize**: DGX Spark / RTX 5090 / RTX 5080

**Notable winners**:
- Track A Agent-Assisted: Team Wombat; Full-Agent: HAN Lab Kernel Mafia
- Track B both: Dogacel
- Track C Agent-Assisted: Kachua; Full-Agent: UW SyFI

### 1.3 Google Graph Scheduling Competition

- **Organizer**: Google
- **Repo**: https://github.com/yarongmu-google/MLSys
- **Task**: Graph scheduling optimization — assign optimal scheduling strategies to computation graph nodes
- **Two tracks**:
  - Track A: Systems Engineering (human-written solver)
  - Track B: Agent Reasoning (must use Google Gemini API)
- **24 benchmarks**: 4 nodes/6 edges → 152 nodes/280 edges
- **Prize**: $10K / $7.5K / $5K per track + $2.5K innovation award

---

## 2. ASPLOS/EuroSys 2025 Contest Track

First ASPLOS/EuroSys Contest Track, March 30 - April 3, 2025, Rotterdam, Netherlands.

### 2.1 NKI-Optimized Llama 3.2 1B Inference

- **Sponsor**: AWS
- **Repo**: https://github.com/asplos-contest/2025
- **Task**: Fastest Llama 3.2 1B inference on single AWS Trainium1 chip using NKI
- **Scoring**: Same formula as MLSys 2026 (Accuracy × Latency × Throughput × NKI ratio)

**Winners**:
1. **1st ($25K) — Tsinghua (thustorage)**: Gao Shiwei, Fan Ruwen, et al.
   - Code: https://github.com/thustorage/nki-llama-contest
   - Key: GEMM/GEMV tiling, instruction fusion, operator fusion (~1.1x improvement)
2. **2nd ($10K)**: UC Merced / UW-Madison
3. **3rd ($5K)**: Yonsei University (Korea)

### 2.2 Intra-Operator Parallelism for Distributed DL (IOPDDL)

- **Sponsor**: Google
- **Task**: Combinatorial optimization — assign optimal parallelism strategies to operators in distributed DL
- Tsinghua also won Runner-Up ($2K) in this track

---

## 3. NeurIPS 2024 — Edge-Device LLM Competition

- **Website**: https://edge-llms-challenge.github.io/
- **Task**: Deploy LLMs on resource-constrained edge devices (6-8GB DRAM)
- **Two tracks**: Track 1 — Model compression; Track 2 — Train small models from scratch
- **Winners**: Tinytron (both tracks); NICSEffalg (Tsinghua Ning Xuefei's team, co-runner-up)

---

## 4. GPU MODE Community Competitions

GPU MODE is a 20,000+ member Discord community for GPU programming with continuous competitions.

- **Repo**: https://github.com/gpu-mode/reference-kernels

### AMD $100K Kernel Competition
- **Hardware**: AMD MI300X
- **Task**: LLM inference kernels in HIP/ROCm

### AMD $1.1M Competition (Feb 2026)
- **Prize pool**: $1,100,000

### NVIDIA Blackwell NVFP4 Competition
- **Sponsors**: NVIDIA + GPU MODE + Dell + Sesterce
- **Task**: NVFP4 GEMM and low-precision kernel optimization on B200

### AMD × GPU MODE: E2E Model Speedrun (2026)
- **Hardware**: AMD Instinct MI355X
- **Task**: End-to-end LLM inference speed challenge

### GPU MODE Leaderboard (Continuously Open)
- **Tasks**: PrefixSum, VectorAdd, Histogram, Sort, Grayscale, GEMM, etc.
- **GPUs**: B200, H100, A100, L4

---

## 5. Industry Benchmarks

### NVIDIA SOL-ExecBench
- **Website**: https://research.nvidia.com/benchmarks/sol-execbench
- **Task**: Benchmark custom GPU kernels on NVIDIA B200 real hardware
- **Format**: Submit optimized CUDA/PyTorch code, get SOL Score, compete on global leaderboard
- **Status**: Continuously open

### MLCommons MLPerf Inference
- **Website**: https://mlcommons.org/benchmarks/inference/
- **Latest**: v5.0 (April 2025), v6.0 (2026)
- **Tracks**: Datacenter / Edge / Mobile / Tiny; Closed / Open division
- **Participants**: NVIDIA, AMD, Intel, Google, Qualcomm, etc. — 1,800+ peer-reviewed results

---

## 6. Summary

| Competition | Hardware | Model/Task | Agent Track? | Tsinghua Wins |
|-------------|----------|------------|--------------|---------------|
| MLSys 2026 AWS MoE | Trainium2/3 | Qwen3-30B-A3B | No | 1st |
| MLSys 2026 FlashInfer | B200 | MoE/Attention/GDN | **Yes (Full-Agent)** | — |
| MLSys 2026 Google Graph | CPU | Graph scheduling | Yes (Gemini) | — |
| ASPLOS/EuroSys 2025 | Trainium1 | Llama 3.2 1B | No | 1st |
| NeurIPS 2024 Edge-LLM | Edge devices | Model compression | No | Co-runner-up |
| GPU MODE AMD | MI300X/MI355X | Inference kernels | No | — |
| GPU MODE NVIDIA | B200 | NVFP4 GEMM | No | — |
| MLPerf Inference | Multi-platform | Industry standard | No | — |

**Key observation**: The MLSys 2026 FlashInfer contest is the first to include a **Full-Agent track** requiring purely AI-generated kernels — the closest existing format to evaluating AI agents on inference optimization.
