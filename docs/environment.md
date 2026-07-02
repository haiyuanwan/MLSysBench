# Evaluation Environment

How to set up the execution environment for MLSysBench: cloud platforms, containers, timing methodology, sandboxing, and cost estimates.

## 1. Cloud GPU Platform Comparison

### Recommendation: Modal (Primary) + RunPod (Alternative)

| Platform | A100 80GB $/hr | H100 $/hr | B200 $/hr | Billing | Root Access | Used By |
|----------|---------------|-----------|-----------|---------|-------------|---------|
| **Modal** | **$2.50** | **$3.95** | $6.25 | Per-second | No | KernelBench, FlashInfer Contest |
| **RunPod** | $1.49 | $3.29 | $5.89 | Per-second | Yes (Pod) | GPU MODE |
| **GCP** | $3.67 ($1.80 Spot) | $11.06 | N/A | Per-second | Yes | — |
| **Vast.ai** | ~$1.20 | ~$2.50 | — | Per-second | Yes | — |
| **Lambda Labs** | ~$1.10 | ~$2.49 | — | Per-hour | Yes | — |
| **AWS** | $40.97 (8×A100) | $98.32 (8×H100) | — | Per-second | Yes | ASPLOS/MLSys contests |

### Why Modal First

1. **KernelBench already has full Modal integration** — we can reuse their `eval_modal.py` pattern
2. Per-second billing — no GPU cost when agent is "thinking" (waiting for LLM response)
3. Built-in `Sandbox` mode for executing untrusted code
4. Academic credits up to $10K available
5. `H100!` syntax prevents auto-upgrade to H200 (important for benchmark consistency)

### When to Use RunPod Instead

- Need root access for GPU clock locking (`nvidia-smi -lgc`)
- Long-running L3 tasks (2-hour agent sessions)
- Need guaranteed hardware consistency

---

## 2. Container Setup

### Option A: Modal (Serverless, for L2 kernel tasks)

```python
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", 
        add_python="3.10"
    )
    .apt_install("git", "gcc", "g++", "cmake", "ninja-build")
    .pip_install(
        "torch==2.8.0", 
        index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("triton", "vllm", "sglang[all]", "transformers", "accelerate")
    .pip_install("lm-eval")  # for quality gate
)

# Model weights as persistent volume
vol = modal.Volume.from_name("model-weights", create_if_missing=True)

@app.function(gpu="A100", image=image, volumes={"/models": vol}, timeout=3600)
def evaluate_task(task_id: str, agent_code: str):
    ...
```

### Option B: Docker (Self-hosted, for L3 E2E tasks)

Reference: [SOL-ExecBench Dockerfile](https://github.com/NVIDIA/SOL-ExecBench/tree/main/docker)

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git gcc g++ cmake ninja-build python3.10 python3-pip

# ML frameworks
RUN pip install torch==2.8.0 triton vllm sglang[all] \
    transformers accelerate flash-attn lm-eval

# Profiling tools
RUN apt-get install -y nsight-systems nsight-compute

# Non-root user with nvidia-smi sudo
ARG HOST_USER=benchuser
RUN useradd -m ${HOST_USER} && \
    echo "${HOST_USER} ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" >> /etc/sudoers
```

**Launch command:**
```bash
docker run --rm \
    --gpus '"device=0"' \
    --ipc=host \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --memory=80g \
    --cpus=16 \
    --pids-limit=4096 \
    --network=none \
    -v /data/models:/models:ro \
    -v $(pwd)/workspace:/workspace \
    mlsysbench:latest
```

### Option C: Bare-Metal CUDA Host for SimAI/Vidur/AICB

Use this path when running the real SimAI/Vidur/AICB benchmark stack directly
from this repository, not the mock runner. AICB imports and JIT-compiles CUDA
extensions from PyTorch, vLLM, DeepGEMM, FlashMLA, FlashInfer, grouped_gemm, and
Triton, so the host must expose NVIDIA devices and a working CUDA toolkit.

Verified host profile:

```text
GPU: 2 x NVIDIA RTX5880-Ada-48Q
Driver: 570.172.18
CUDA runtime reported by driver: 12.8
CUDA toolkit used for builds: /usr/local/cuda-12.9
Python: CPython 3.11
PyTorch: 2.8.0+cu128
vLLM: 0.11.0
```

Minimum setup outline:

```bash
# Install a modern toolkit. DeepGEMM/FlashMLA builds need newer CUDA than the
# Ubuntu 24.04 nvidia-cuda-toolkit package provides.
sudo apt-get install -y cuda-toolkit-12-9

# Create Python 3.11 if the system Python is too new for CUDA packages.
python3 -m pip install uv
uv venv --python 3.11 .venv311

.venv311/bin/python -m pip install -U pip setuptools wheel packaging ninja cmake
.venv311/bin/python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

CUDA_HOME=/usr/local/cuda-12.9 \
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
FLASH_MLA_DISABLE_SM100=1 \
PATH=/home/haiyuan/MLSysBench/.venv311/bin:/usr/local/cuda-12.9/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
  .venv311/bin/python -m pip install --no-build-isolation \
  -r third_party/SimAI/aicb/requirements.txt
```

When invoking real AICB or Vidur+AICB commands, keep the venv and CUDA toolkit
at the front of the environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/home/haiyuan/MLSysBench/.venv311/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/haiyuan/MLSysBench/.venv311/lib/python3.11/site-packages/torch/lib:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=8.9
export DG_JIT_NVCC_COMPILER=/home/haiyuan/MLSysBench/scripts/deepgemm_nvcc_sm89_wrapper.sh
```

`DG_JIT_NVCC_COMPILER` is only needed on RTX 5880/Ada sm89 hosts. Current
DeepGEMM may emit `sm_89a`, which CUDA 12.9 `nvcc` rejects. The wrapper rewrites
that arch to `sm_89`. DeepGEMM FP8 GEMM itself still targets sm90/sm100; the
local AICB helpers therefore use real CUDA bf16 matmul timing on sm89 instead of
mock/default timings.

### Option D: Apptainer (HPC Clusters)

Reference: [InferenceBench containers](https://github.com/aisa-group/InferenceBench/tree/main/containers)

For university HPC clusters that don't allow Docker:
```
Bootstrap: docker
From: nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
%post
    pip install torch triton vllm sglang transformers
```

### Model Weight Pre-loading

| Method | Pros | Cons |
|--------|------|------|
| Docker bind mount `-v /data/models:/models:ro` | Flexible, shared across containers | Requires host-side download |
| Modal Volume | Persistent, pay-once download | Modal-specific |
| Baked into image | Fast startup | Large image, hard to update |

**Recommended:** Bind mount for Docker, Modal Volume for serverless.

---

## 3. Performance Measurement

### Kernel-Level Tasks (L2)

Reference: [KernelBench timing.py](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/timing.py), [SOL-ExecBench clock_lock.py](https://github.com/NVIDIA/SOL-ExecBench/blob/main/src/sol_execbench/core/bench/clock_lock.py)

**Step 1: Lock GPU Clocks** (eliminates 10-30% variance from boost clocks)
```bash
# A100: lock to base clocks
nvidia-smi -lgc 1410        # GPU clock
nvidia-smi -lmc 1215        # Memory clock
# Set exclusive mode
nvidia-smi -c EXCLUSIVE_PROCESS
```

**Step 2: Timing with CUDA Events**
```python
def benchmark_kernel(kernel_fn, args, warmup=10, trials=100):
    # Warmup
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(trials):
        # Clear L2 cache (256MB dummy tensor)
        torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, 
                     device='cuda').zero_()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        "median": np.median(times),   # prefer median over mean
        "mean": np.mean(times),
        "std": np.std(times),
        "cv": np.std(times) / np.mean(times),  # flag if >5%
    }
```

**Step 3: Report median** (more robust to outliers than mean)

### End-to-End Tasks (L3)

```python
def benchmark_e2e(serve_fn, requests, warmup=3, trials=5):
    for _ in range(warmup):
        serve_fn(requests)
    
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        results = serve_fn(requests)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {"mean": np.mean(times), "std": np.std(times)}
```

### Timing Method Comparison

| Method | Use Case | Overhead | Precision |
|--------|----------|----------|-----------|
| `cuda_event` | Kernel timing | Lowest | GPU-only time |
| `triton.do_bench` | Triton kernels | Low | Auto-adapts trial count |
| `time.perf_counter` | E2E / serving | Higher | Wall-clock (includes CPU) |
| `nsight` profiling | Analysis tasks | Highest | Most detailed |

---

## 4. Sandboxing & Anti-Cheating

### Resource Limits

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Memory | 80GB | Prevent OOM affecting host |
| CPUs | 16 | Fair resource allocation |
| PIDs | 4096 | Prevent fork bombs |
| Network | Disabled | L2/L3 tasks don't need internet |
| Disk | 50GB | Prevent disk exhaustion |
| Compilation timeout | 120s | Per SOL-ExecBench |
| Execution timeout | 600s (L2) / 7200s (L3) | L3 allows 2hr agent session |

### Anti-Cheating (Reference: KernelBench, SOL-ExecBench)

| Attack | Detection |
|--------|-----------|
| Cache reuse | Different random seeds per correctness check |
| Input mutation | Copy inputs before/after, compare |
| Empty kernel | Check output is not all-zeros or same as input |
| Pre-computed output | Test with multiple random inputs, outputs should differ |
| Suspicious speedup | Flag >10x for manual review |
| Non-default CUDA stream | Check execution stream (KernelBench `test_eval_adversarial.py`) |

### Quality Gate (L3 tasks)

Following InferenceBench:
- MMLU-Pro 500-question accuracy ≥ 0.95× baseline
- Integrity audit: judge agent reviews transcript for reward hacking

---

## 5. Profiling Infrastructure (for L1 Tasks)

### Generating Profiling Data

```bash
# Nsight Systems: kernel timeline
nsys profile --trace=cuda,nvtx -o profile python run_inference.py
nsys export --type=json profile.nsys-rep

# Nsight Compute: detailed kernel metrics
ncu --set full --export ncu_output python run_kernel.py
ncu --import ncu_output.ncu-rep --csv > metrics.csv

# torch.profiler: Python-level integration
# See code example in data-sources.md
```

### Presenting Profiling Data to Agents

Convert raw profiling data to structured text format:

```
=== GPU Kernel Profile Summary ===
Model: Llama-3-8B | Hardware: NVIDIA A100 80GB | Batch: 32

Top-10 Kernels by GPU Time:
# | Kernel Name                    | Time(ms) | Calls | Bound
1 | flash_fwd_kernel               | 12.34    | 32    | Compute
2 | volta_sgemm_128x128_nn         | 8.56     | 96    | Compute
3 | rmsnorm_kernel                 | 3.21     | 64    | Memory
...

Memory: Peak 45.2/80 GB | KV Cache 12.8 GB | Weights 16.0 GB
Roofline: 187/312 TFLOPS (60%) | 1.8/2.0 TB/s BW (90%)
```

---

## 6. Cost Estimates

### Single Evaluation Run (one agent, ~35 tasks)

| Component | GPU Hours | Modal A100 | Modal H100 | RunPod A100 |
|-----------|-----------|------------|------------|-------------|
| L1 tasks (10-15) | ~1h | $2.50 | $3.95 | $1.49 |
| L2 tasks (10-15) | ~2.5h | $6.25 | $9.88 | $3.73 |
| L3 tasks (5-8) | ~5h | $12.50 | $19.75 | $7.45 |
| Agent wait overhead | ~5h | $0* | $0* | $7.45 |
| **Total** | **~14h** | **~$21** | **~$34** | **$20** |

*Modal per-second billing means no charge during agent "thinking" time.

### Leaderboard (10 agents × 3 repeats)

| Platform | GPU | 420h cost | With L3 2hr sessions (+360h) |
|----------|-----|-----------|------------------------------|
| Modal | A100 | $1,050 | $1,950 |
| Modal | H100 | $1,659 | $3,081 |
| RunPod | A100 | $626 | $1,162 |
| Vast.ai | A100 | $504 | $936 |

### Reference: InferenceBench Cost

- 15 agents × 4 scenarios × 3 seeds = 180 runs
- 360 GPU hours on H100 → ~$1,422 on Modal
- Total including baselines: ~$2,000

### Budget Recommendations

| Phase | Budget | Platform |
|-------|--------|----------|
| Phase 1 (MVP) | $200-500 | Modal (apply for academic credits) |
| Phase 2 (Paper) | $2,000-5,000 | Modal + RunPod |
| Phase 3 (Leaderboard) | $5,000+/year | RunPod Reserved or self-hosted |

---

## 7. Reference Implementations from Existing Benchmarks

| Component | Reference | Link |
|-----------|-----------|------|
| Modal integration | KernelBench `eval_modal.py` | [KernelBench/scripts/](https://github.com/ScalingIntelligence/KernelBench) |
| Docker setup | SOL-ExecBench `docker/` | [SOL-ExecBench/docker/](https://github.com/NVIDIA/SOL-ExecBench) |
| Clock locking | SOL-ExecBench `clock_lock.py` | [SOL-ExecBench/core/bench/](https://github.com/NVIDIA/SOL-ExecBench) |
| Timing methods | KernelBench `timing.py` | [KernelBench/src/kernelbench/](https://github.com/ScalingIntelligence/KernelBench) |
| Anti-cheating | KernelBench `kernel_static_checker.py` | [KernelBench/src/kernelbench/](https://github.com/ScalingIntelligence/KernelBench) |
| Apptainer containers | InferenceBench `containers/` | [InferenceBench/containers/](https://github.com/aisa-group/InferenceBench) |
| Agent harness | InferenceBench evaluation pipeline | [InferenceBench](https://github.com/aisa-group/InferenceBench) |
| Quality gate | InferenceBench MMLU-Pro | [InferenceBench](https://github.com/aisa-group/InferenceBench) |
