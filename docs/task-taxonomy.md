# LLM Inference Optimization Task Taxonomy

A comprehensive classification of inference optimization tasks, organized by abstraction level. Each category includes concrete benchmark task examples with difficulty ratings.

## 1. Kernel-Level Optimization

### 1.1 CUDA Kernel Writing & Optimization

Core areas: GEMM optimization (tiling, shared memory, Tensor Core), FlashAttention, fused attention, quantized GEMM kernels.

| ID | Task | Difficulty |
|----|------|------------|
| T1.1 | Write a basic CUDA GEMM kernel with shared memory tiling | Medium |
| T1.2 | Implement FlashAttention forward pass (tiling + online softmax) | Hard |
| T1.3 | Convert FP16 GEMM to use Tensor Core (WMMA API) | Hard |
| T1.4 | Write a fused RMSNorm + Residual Add kernel | Easy-Medium |
| T1.5 | Implement W4A8 quantized GEMM with efficient dequantization | Hard |
| T1.6 | Optimize Softmax kernel to reduce global memory access (online softmax) | Medium |
| T1.7 | Implement fused Rotary Position Embedding (RoPE) kernel | Easy-Medium |

### 1.2 Triton Kernel Development

| ID | Task | Difficulty |
|----|------|------------|
| T2.1 | Write fused softmax in Triton | Easy |
| T2.2 | Implement PagedAttention decode kernel in Triton | Medium-Hard |
| T2.3 | Write Grouped Query Attention (GQA) kernel in Triton | Medium |
| T2.4 | Port an existing CUDA attention kernel to Triton | Medium |
| T2.5 | Implement fused SiLU + gating in Triton | Easy |
| T2.6 | Write FP8 matmul kernel in Triton | Medium-Hard |

### 1.3 NKI (AWS Neuron Kernel Interface) Programming

| ID | Task | Difficulty |
|----|------|------------|
| T3.1 | Write matmul kernel using NKI Tensor Engine | Medium |
| T3.2 | Implement FlashAttention variant in NKI | Hard |
| T3.3 | Port a Triton kernel to NKI | Hard |
| T3.4 | Write LayerNorm kernel using NKI Vector Engine | Medium |

### 1.4 Custom Operator Development

| ID | Task | Difficulty |
|----|------|------------|
| T4.1 | Write PyTorch C++ extension wrapping a custom CUDA kernel | Easy-Medium |
| T4.2 | Register custom attention op for torch.compile | Medium |
| T4.3 | Write TensorRT plugin for non-standard activation function | Medium-Hard |

---

## 2. Algorithm-Level Optimization

### 2.1 Quantization

Key techniques: GPTQ ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)), AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978), MLSys 2024 Best Paper), SmoothQuant, QServe/QoQ ([arXiv:2405.04532](https://arxiv.org/abs/2405.04532)), BitNet b1.58 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)).

| ID | Task | Difficulty |
|----|------|------------|
| T5.1 | Apply GPTQ 4-bit quantization to Llama-3-8B and evaluate perplexity | Easy |
| T5.2 | Implement AWQ salient channel identification and equivalent transformation | Medium |
| T5.3 | Implement SmoothQuant per-channel scaling factor search | Medium |
| T5.4 | Design efficient dequantize kernel for W4A8 mixed quantization | Hard |
| T5.5 | Compare GPTQ vs AWQ vs SmoothQuant accuracy-speed tradeoffs | Medium |
| T5.6 | Implement KV Cache FP8/INT4 quantization with quality evaluation | Medium-Hard |
| T5.7 | Design adaptive quantization strategy with per-layer bit-widths | Hard |

### 2.2 Pruning & Sparsity

Key techniques: SparseGPT, Wanda, 2:4 structured sparsity, MInference ([arXiv:2407.02490](https://arxiv.org/abs/2407.02490), NeurIPS 2024 Spotlight).

| ID | Task | Difficulty |
|----|------|------------|
| T6.1 | Apply 2:4 structured sparsity and accelerate on Ampere GPU | Medium |
| T6.2 | Implement SparseGPT one-shot weight pruning | Medium-Hard |
| T6.3 | Identify and implement sparse attention patterns for long context (A-shape/Block-Sparse) | Hard |
| T6.4 | Analyze per-layer redundancy and design layerwise pruning strategy | Medium |

### 2.3 Speculative Decoding

Key techniques: Classic speculative decoding ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192), ICML 2023), Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)), EAGLE/EAGLE-2, self-speculative decoding.

| ID | Task | Difficulty |
|----|------|------------|
| T7.1 | Implement basic speculative decoding: draft-verify loop + rejection sampling | Medium |
| T7.2 | Select/train draft model and tune speculation length for a target model | Medium |
| T7.3 | Implement Medusa-style multi-head parallel decoding + tree verification | Hard |
| T7.4 | Implement self-speculative decoding (early exit) | Hard |
| T7.5 | Analyze acceptance rate vs speedup relationship and find optimal config | Medium |

### 2.4 KV Cache Optimization

Key techniques: PagedAttention ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180), SOSP 2023), TOVA ([arXiv:2401.06104](https://arxiv.org/abs/2401.06104)), CacheGen ([arXiv:2310.07240](https://arxiv.org/abs/2310.07240), SIGCOMM 2024), RadixAttention (SGLang).

| ID | Task | Difficulty |
|----|------|------------|
| T8.1 | Implement PagedAttention block allocator | Medium-Hard |
| T8.2 | Implement attention-score-based KV cache eviction | Medium |
| T8.3 | Implement Prefix Caching with shared prefix detection and KV reuse | Medium |
| T8.4 | Implement KV cache FP8 quantization with calibration | Medium |
| T8.5 | Design cross-request KV cache sharing (beam search / parallel sampling) | Medium-Hard |
| T8.6 | Implement CacheGen-style KV cache serialization compression | Hard |

### 2.5 MoE Optimization

Key techniques: DeepSeek-V2/V3/R1 MLA, expert offloading ([arXiv:2312.17238](https://arxiv.org/abs/2312.17238)), expert parallelism.

| ID | Task | Difficulty |
|----|------|------------|
| T9.1 | Implement expert offloading inference for Mixtral-8x7B (GPU + CPU) | Medium-Hard |
| T9.2 | Configure Expert Parallelism for MoE model | Medium |
| T9.3 | Analyze MoE router activation patterns and optimize prefetching | Hard |
| T9.4 | Implement DeepSeek-MoE fine-grained expert grouping | Hard |

---

## 3. System-Level Optimization

### 3.1 Operator Fusion

| ID | Task | Difficulty |
|----|------|------------|
| T10.1 | Fuse RMSNorm + Residual Add into single CUDA kernel | Easy-Medium |
| T10.2 | Implement fused SwiGLU (SiLU + Gate + Mul) kernel | Medium |
| T10.3 | Analyze Transformer block kernel launch overhead and propose fusion plan | Medium |
| T10.4 | Implement custom fusion pass with torch.compile | Medium-Hard |

### 3.2 Memory Management

Key techniques: Memory pooling, CPU/NVMe offloading ([arXiv:2312.11514](https://arxiv.org/abs/2312.11514), "LLM in a Flash"), activation checkpointing.

| ID | Task | Difficulty |
|----|------|------------|
| T11.1 | Implement block-level memory pool for KV cache | Medium |
| T11.2 | Implement weight CPU-GPU offloading with prefetch pipeline | Medium-Hard |
| T11.3 | Analyze memory bottlenecks for given model on specific GPU | Medium |
| T11.4 | Estimate peak memory usage for model + batch size configuration | Easy-Medium |

### 3.3 Scheduling & Batching

Key techniques: Continuous Batching (Orca), Chunked Prefill / Sarathi-Serve ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310)), Disaggregated Prefill-Decode, Llumnix ([arXiv:2406.03243](https://arxiv.org/abs/2406.03243), OSDI 2024).

| ID | Task | Difficulty |
|----|------|------------|
| T12.1 | Implement continuous batching scheduler (dynamic insert/remove) | Medium |
| T12.2 | Implement chunked prefill and analyze decode latency impact | Medium-Hard |
| T12.3 | Design prefill-decode disaggregated scheduling | Hard |
| T12.4 | Configure optimal batch size given SLO constraints and arrival patterns | Medium |
| T12.5 | Implement request priority queue with preemption | Medium |

### 3.4 Parallelism Configuration

| ID | Task | Difficulty |
|----|------|------------|
| T13.1 | Configure TP+PP strategy for 70B model on 8 GPUs | Medium |
| T13.2 | Configure TP+EP hybrid parallelism for Mixtral-8x7B | Medium-Hard |
| T13.3 | Analyze TP communication overhead and determine optimal TP degree | Medium |
| T13.4 | Implement Ring Attention for ultra-long sequence inference | Hard |
| T13.5 | Design asymmetric parallelism for heterogeneous GPU cluster | Hard |

### 3.5 Compilation Optimization

| ID | Task | Difficulty |
|----|------|------------|
| T14.1 | Build and optimize LLM inference engine with TensorRT-LLM | Easy-Medium |
| T14.2 | Optimize inference with torch.compile and analyze generated Triton kernels | Medium |
| T14.3 | Write TVM schedule to optimize matmul | Medium-Hard |
| T14.4 | Compare TensorRT vs torch.compile latency and throughput | Medium |

---

## 4. Difficulty Analysis for LLM Agents

### Easy (LLM agents can likely handle)
- Framework configuration and deployment (vLLM, SGLang, TensorRT-LLM)
- Quantization tool usage (AutoGPTQ, AutoAWQ)
- Basic profiling analysis and memory estimation
- Simple fused kernels (RMSNorm+Add, RoPE)
- Basic Triton kernels (softmax, elementwise)

### Medium (requires deeper understanding)
- Quantization algorithm implementation (AWQ channel identification, SmoothQuant scaling)
- KV cache strategies (PagedAttention, prefix caching)
- Scheduling algorithms (continuous batching, chunked prefill)
- Multi-GPU parallelism analysis and configuration
- Triton GQA/MQA attention kernels
- Performance modeling (Roofline analysis)

### Hard (requires deep expert knowledge)
- FlashAttention-level CUDA kernels (GPU memory hierarchy, warp scheduling, bank conflict avoidance)
- Novel kernel algorithm design (sparse attention patterns + efficient kernels)
- Cross-hardware kernel porting (CUDA → NKI/ROCm)
- Speculative decoding innovation (tree verification, optimal acceptance)
- Compiler pass writing (TVM/MLIR transformations)
- End-to-end system optimization (request migration, disaggregated serving)
- Mixed-precision kernel optimization (FP8/FP4 GEMM with Tensor Core)

---

## 5. Frameworks & Hardware Ecosystem

### Inference Frameworks

| Framework | Core Features |
|-----------|---------------|
| **vLLM** | PagedAttention, continuous batching, most active open-source community |
| **SGLang** | RadixAttention, frontend DSL, structured output optimization |
| **TensorRT-LLM** | NVIDIA official, extreme performance, FP8/INT4, custom plugins |
| **llama.cpp** | Pure C/C++, CPU-friendly, GGUF quantization formats |
| **DeepSpeed-FastGen** | Dynamic SplitFuse |
| **Llumnix** (Alibaba PAI) | Cross-instance scheduling, request migration, SLO guarantees |
| **MLC LLM** | TVM-based cross-platform compilation |

### Hardware Platforms

| Platform | Key Features |
|----------|-------------|
| NVIDIA GPU (H100/A100/L40S) | Tensor Core, NVLink, HBM3, FP8 |
| NVIDIA Blackwell (B200) | FP4, larger HBM |
| AMD GPU (MI300X) | 192GB HBM, ROCm, Triton compatible |
| AWS Trainium/Inferentia2 | NeuronCore, NKI programming |
| Google TPU (v5e/v6) | MXU, HBM, ICI interconnect |
| Apple Silicon (M series) | Unified memory, Metal/MLX |
| CPU (x86/ARM) | AVX-512, AMX, NEON, llama.cpp |

---

## 6. Key References

| # | Paper | Category |
|---|-------|----------|
| 1 | [arXiv:2404.14294](https://arxiv.org/abs/2404.14294) (Zhou et al., 2024) | Survey: data-model-system taxonomy |
| 2 | [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) (Yuan et al., 2024) | Survey: Roofline model + LLM-Viewer |
| 3 | [arXiv:2407.12391](https://arxiv.org/abs/2407.12391) (Li et al., 2024) | Survey: LLM serving systems |
| 4 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (Dao, 2022) | FlashAttention |
| 5 | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) (Kwon et al., SOSP 2023) | vLLM / PagedAttention |
| 6 | [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) (Zheng et al., 2024) | SGLang / RadixAttention |
| 7 | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) (Leviathan et al., ICML 2023) | Speculative Decoding |
| 8 | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) (Cai et al., 2024) | Medusa |
| 9 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) (Frantar et al., ICLR 2023) | GPTQ |
| 10 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) (Lin et al., MLSys 2024 Best Paper) | AWQ |
| 11 | [arXiv:2405.04532](https://arxiv.org/abs/2405.04532) (Lin et al., 2024) | QServe W4A8KV4 |
| 12 | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) | BitNet b1.58 |
| 13 | [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) (Agrawal et al., 2024) | Sarathi-Serve / Chunked Prefill |
| 14 | [arXiv:2406.03243](https://arxiv.org/abs/2406.03243) (Zhao et al., OSDI 2024) | Llumnix |
| 15 | [arXiv:2407.02490](https://arxiv.org/abs/2407.02490) (Jiang et al., NeurIPS 2024) | MInference |
| 16 | [arXiv:2312.11514](https://arxiv.org/abs/2312.11514) (Apple, ACL 2024) | LLM in a Flash |
| 17 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025) | DeepSeek-R1 |
