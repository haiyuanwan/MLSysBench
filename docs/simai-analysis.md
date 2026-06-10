# SimAI 代码库深度分析

> 基于 https://github.com/aliyun/SimAI 源码的逐文件阅读分析，用于指导 MLSysBench 的设计。

## 1. 项目概览

SimAI（Simulator for AI）是阿里云开发的全栈 AI 训练/推理仿真工具包，发表于 NSDI'25。它能在 **无需真实 GPU 集群** 的情况下仿真大规模 LLM 训练和推理的端到端性能。

### 1.1 五大组件

```
SimAI/
├── aicb/                          [子模块] AI Communication Benchmark — 通信 workload 生成
├── SimCCL/                        [子模块] 集合通信分解库
├── ns-3-alibabacloud/             [子模块] 定制 NS-3 网络仿真器
├── astra-sim-alibabacloud/        核心仿真引擎（基于 astra-sim 1.0 扩展）
└── vidur-alibabacloud/            多请求推理仿真器（基于 Microsoft Vidur 扩展）
```

### 1.2 三种仿真模式

| 模式 | 后端 | 精度 | 速度 | 适用场景 |
|------|------|------|------|---------|
| **SimAI-Analytical** | 分析模型（busbw） | 中 | 极快（秒级） | 并行参数优化、带宽选型、快速探索 |
| **SimAI-Simulation** | NS-3 包级仿真 | 高 | 慢（分钟-小时） | 集合通信算法研究、网络协议评估、架构设计 |
| **SimAI-Physical** | 真实物理网络 | 最高 | 取决于硬件 | TODO，尚未完整开源 |

### 1.3 许可证

Apache-2.0，可自由使用和修改。

---

## 2. AICB（AI Communication Benchmark）详细分析

### 2.1 功能定位

AICB 负责 **生成通信 workload 文件**，描述 LLM 训练/推理过程中每一层的计算时间和通信操作。它有两个运行模式：
1. **Workload 生成模式**：不需要 GPU，根据模型参数计算通信量，输出 `.txt` 或 `.csv` 文件
2. **物理执行模式**：需要 GPU 集群，实际执行通信操作并测量时间

### 2.2 核心类层次

```
MockedModel (基类)
├── MockedParam(shape, elem_size)  — 模拟参数张量
├── parameters() → List[MockedParam]
└── child_modules() → List[MockedModel]

训练模型:
├── MegatronModel(args) — Megatron 风格 (GPT/Llama)
│   ├── MegatronEmbedding
│   ├── MegatronAttention → MegatronColumnLinear + MegatronRowLinear
│   ├── MegatronMlp → MegatronColumnLinear + MegatronRowLinear
│   └── MOEMLP (MoE 版本)
├── DeepspeedForCausalLM(args) — DeepSpeed ZeRO
└── DeepSeekV3Model(args) — DeepSeek-V3
    ├── DeepSeekMLA (Multi-Latent Attention)
    └── DeepSeekMoE

推理模型 (新增，更精细):
├── MockedDeepSeek.DeepSeekModel
│   ├── DeepSeekAttention (DeepSeekMLA 用 kv_lora_rank)
│   ├── DeepSeekMLP (dense FFN)
│   └── DeepSeekMOE (routing + experts)
├── MockedQwen3Moe.Qwen3MoeModel
│   ├── Qwen3MoeRMSNorm
│   ├── Qwen3MoeAttention (GQA)
│   ├── Qwen3MoeRoute
│   └── Qwen3MoeExpert
└── MockedQwen3Next.Qwen3NextModel
    ├── Qwen3NextAttention (全注意力)
    ├── Qwen3NextGatedDeltaNet (线性注意力)
    ├── Qwen3NextRoute
    └── Qwen3NextExpert
```

### 2.3 支持的模型架构常量

#### DeepSeek-V3 671B (`deepseek_default.json`)
- `num_layers: 61`, `dense_layer: 3`（前 3 层 dense，后 58 层 MoE）
- `hidden_size: 7168`, `num_attention_heads: 128`
- MLA: `d_kv_c: 512`（KV 压缩维度）, `d_q_c: 1536`（Q 压缩维度）, `d_r: 64`（rope）, `d_q: 128`, `d_kv: 128`
- `router_expert: 256`, `duped_expert: 32` → `num_experts: 288`
- `shared_experts: 1`, `moe_router_topk: 8`, `expert_dim: 2048`
- `FP8_FACTOR = (1 + 4/128) / 2 ≈ 0.515625`（FP8 dispatch 额外 scale 开销）

#### Qwen3-MoE-235B (`qwen3_moe_default.json`)
- `num_hidden_layers: 94`, `hidden_size: 4096`, `head_dim: 128`
- `num_attention_heads: 64`, `num_key_value_heads: 4`（GQA 16:1 ratio）
- `intermediate_size: 12288`, `moe_intermediate_size: 1536`
- `num_experts: 128`, `num_experts_per_tok: 8`
- `vocab_size: 151936`, `rope_theta: 1000000.0`

#### Qwen3-Next-80B (`qwen3_next_default.json`)
- `num_hidden_layers: 48`, `hidden_size: 2048`, `head_dim: 256`
- `num_attention_heads: 16`, `num_key_value_heads: 2`
- `full_attention_interval: 4`（每 4 层一层全注意力，其余用 GDN 线性注意力）
- GDN 参数: `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`, `linear_num_key_heads: 16`, `linear_num_value_heads: 32`
- `num_experts: 512`, `num_experts_per_tok: 10`, `moe_intermediate_size: 512`
- `max_position_embeddings: 262144`, `rope_theta: 10000000`

### 2.4 训练模型通信模式

#### MegatronRowLinear
- Forward: 计算 → ReduceScatter(TP) if SP, AllReduce(TP) if no SP
- Backward: AllGather(TP) if SP → 计算(grad_input) → 计算(grad_weight)
- 通信量: `2 × seq × batch × output_size` bytes

#### MegatronColumnLinear
- Forward: AllGather(TP) if SP → 计算
- Backward: AllGather → 计算 → ReduceScatter(TP) if SP → 计算
- 通信量: `2 × seq × batch × input_size` bytes

#### DeepSeek MoE Dispatch/Combine (FP8)
- Forward dispatch: AllToAll(EP), size = `seq × h × batch × topk / tp × 2 × FP8_FACTOR`
- Forward combine: AllToAll(EP), size = `seq × h × batch × topk / tp × 2`（BF16）
- Backward: dispatch 和 combine 反向

#### DeepSpeed ZeRO-3
- Forward: 每个参数 AllGather(DP)，使用预取 bucket
- Backward: bucket 化 ReduceScatter(DP) 用于梯度同步
- Step: AllGather(DP) 重新收集更新后的参数

### 2.5 AIOB（AI Operation Benchmark）

AIOB 用于在真实 GPU 上 **profiling 计算核心的时间**。对于每种模型，有对应的 Aiob 类（如 `AiobMegatron`、`AiobDeepSeek`），它实际执行前向/反向传播中的 GEMM、attention 等操作，记录每个操作的执行时间到文件。

这些计算时间被注入到 workload 文件的 `forward_compute_time` 和 `backward_compute_time` 字段中。

AIOB 模型是真实的 `torch.nn.Module`，在 GPU 上执行计算并用 CUDA events 计时：

| 组件 | 训练 AIOB | 推理 AIOB |
|------|-----------|-----------|
| Attention | FlashAttention (`flash_attn_unpadded_func`) | FlashMLA (`flash_mla_with_kvcache`) |
| MLP (Dense) | 标准 GEMM | FP8 GEMM (DeepGEMM) |
| MoE Expert | GroupedMLP (CUTLASS grouped_gemm) | DeepGEMM masked/contiguous grouped FP8 |
| Normalization | APEX FastLayerNorm | vLLM `fused_add_rms_norm` |
| 输出 | 标准 Linear | FP8 量化 + FP8 GEMM |

Qwen3-Next 的 AIOB 还包括:
- Gated Delta Net (GDN) 线性注意力 profiling
- Causal Conv1D (`causal_conv1d_update`)
- 自定义 Triton kernel (`fused_gdn_gating_kernel`)
- Recurrent core (`fused_recurrent_gated_delta_rule`)

### 2.6 入口点

**训练 workload 生成**: `aicb/workload_generator/SimAI_training_workload_generator.py`
```bash
# 无需 GPU，使用默认计算时间
python SimAI_training_workload_generator.py --model_name llama-65B --world_size 128 \
    --tensor_model_parallel_size 8 --pipeline_model_parallel 4 --seq_length 4096

# 使用 AIOB 获取精确计算时间（需要 GPU）
python SimAI_training_workload_generator.py --model_name llama-65B --world_size 128 \
    --tensor_model_parallel_size 8 --aiob_enable
```

**推理 workload 生成**: `aicb/workload_generator/SimAI_inference_workload_generator.py`
```bash
python SimAI_inference_workload_generator.py DeepSeek-671B \
    scripts/inference_configs/deepseek_default.json \
    --world_size 256 --tensor_model_parallel_size 4 --expert_model_parallel_size 64 \
    --phase decode --micro_batch 128
```

**Vidur 专用 workload 生成**: `aicb/workload_generator/Vidur_workload_generator.py`
- 生成聚合后的 per-layer CSV，包含 `layer_id, layer_type(attention/mlp/moe), comp_time, comm_size`
- vidur 的 execution_time_predictor 使用这些数据

**物理执行模式**: `aicb/aicb.py`（需要 `torchrun`，实际在 GPU 集群上执行通信操作）

### 2.7 Workload 文件格式

#### 训练/仿真用 `.txt` 格式

```
# 第1行: header
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: <TP> ep: <EP> pp: <PP> vpp: <VPP> ga: <GA> all_gpus: <总GPU数> checkpoints: 0 checkpoint_initiates: 0 pp_comm: <PP通信量>

# 第2行: 操作总数
<N>

# 第3-N+2行: 每行一个操作（11个 TAB 分隔字段）
<name> <placeholder> <fwd_comp_time> <fwd_comm_type> <fwd_comm_size> <bwd_comp_time> <bwd_comm_type> <bwd_comm_size> <dp_comp_time> <dp_comm_type> <dp_comm_size> <process_time>
```

**字段说明**:
- `name`: 层名称（如 `attention_column`, `mlp_moelayer`, `embedding_layer`）
- `placeholder`: 固定 -1
- `fwd_comp_time`: 前向计算时间（纳秒），AIOB 未启用时为 1
- `fwd_comm_type`: 前向通信类型（ALLREDUCE, ALLGATHER, REDUCESCATTER, ALLTOALL, ALLTOALL_EP, NONE）
- `fwd_comm_size`: 前向通信字节数
- `bwd_*`: 反向传播对应字段
- `dp_*`: 数据并行对应字段（grad sync）
- `process_time`: 额外处理时间，默认 100

**通信类型与并行组映射**:
- `ALLREDUCE/ALLGATHER/REDUCESCATTER` 在 `fwd_comm` → TP 组
- `ALLTOALL_EP` → EP 组
- `dp_comm` 中的操作 → DP 组
- `ALLGATHER_DP_EP / REDUCESCATTER_DP_EP` → EP+DP 联合组

#### Vidur 用 `.csv` 格式

```
layer_id	layer_name	comp_time	comm_size
0	attention	12345	1048576
0	moe	67890	2097152
1	attention	12345	1048576
...
```

### 2.8 通信量计算公式

| 通信操作 | 计算公式 |
|---------|---------|
| TP 通信量 | `2 × micro_batch × seq_length × hidden_size` (bytes) |
| DP AllGather (grad) | `2 × total_params` (bytes) |
| DP ReduceScatter | `4 × total_params` (bytes) |
| EP Dispatch (FP8) | `tp_comm_size × topk / tp × FP8_FACTOR` |
| EP Combine | `tp_comm_size × topk / tp` |
| PP 通信量 | `2 × micro_batch × seq_length × hidden_size` |

其中 `FP8_FACTOR ≈ 0.516`（DeepEP 使用 FP8 减半通信量加上 scale overhead）。

### 2.9 WorkloadApplyer — 物理执行

`workload_applyer.py` 在真实 GPU 集群上执行 workload：
- 使用 Megatron 的 rank 分组逻辑（`tp-cp-ep-dp-pp` 排列）创建 TP/DP/PP/EP 进程组
- 为每种通信类型分配执行函数（`_apply_all_reduce`, `_apply_all_gather` 等）
- 支持 P2P 通信（`isend/irecv`）用于 PP
- 计算操作支持两种模式：AIOB（用 `time.sleep` 模拟）和 GEMM（实际执行矩阵乘法）

---

## 3. astra-sim-alibabacloud 详细分析

### 3.1 功能定位

astra-sim 是核心仿真引擎，扩展自 Georgia Tech 的 astra-sim 1.0。它将 workload、集合通信和网络仿真三层串联。

### 3.2 架构层次

```
┌──────────────┐
│  Workload    │  解析 .txt 文件，驱动逐层执行
│  (Layer.cc)  │
├──────────────┤
│  System      │  GPU 级事件调度、计算-通信 overlap
│  (Sys.cc)    │
├──────────────┤
│  Collective  │  集合通信算法 → 点对点通信分解
│  (Ring/DBT)  │
├──────────────┤
│  MockNccl    │  模拟 NCCL 的 channel/group 管理
│  (SimCCL)    │
├──────────────┤
│  Network     │  网络后端（analytical / ns3 / phynet）
│  Frontend    │
└──────────────┘
```

### 3.3 核心枚举与数据结构

```cpp
enum GPUType { A100, A800, H100, H800, H20, NONE };
enum ComType { None, Reduce_Scatter, All_Gather, All_Reduce, All_to_All, ... };
enum CollectiveImplementationType {
    Ring, OneRing, Direct, OneDirect, AllToAll,
    DoubleBinaryTreeLocalAllToAll, LocalRingNodeA2AGlobalDBT,
    HierarchicalRing, DoubleBinaryTree, HalvingDoubling,
    OneHalvingDoubling, NcclFlowModel, NcclTreeFlowModel
};
enum ParallelStrategy { TP, DP, PP, EP, DP_EP, NONE };
```

常量: `CLOCK_PERIOD = 1`, `FREQ = 1000.0`（1 GHz 仿真时钟）

### 3.4 Sys 类（per-GPU 仿真节点）

每个 GPU 对应一个 `Sys` 实例，核心成员：
- `AstraNetworkAPI* NI` — 可插拔网络后端
- `Workload* workload` — 已解析的 workload
- `MockNccl::MockNcclComm*` per ParallelStrategy — NCCL 模拟
- `event_queue: map<Tick, list<(Callable*, EventType, CallData*)>>` — 离散事件队列

仿真循环：`iterate()` → `call_events()` → 处理当前 tick 所有事件 → 产生新事件

### 3.5 Workload 解析（Workload.cc / Layer.cc）

`Workload.cc` 解析 header 行提取并行参数，然后逐行解析 `Work_Item` 创建 `Layer` 对象。

每个 `Layer` 包含：
- `forward_pass_compute_time` — 前向计算时间
- `forward_pass_comm_type` — 前向通信类型（映射到 CollectiveBarrier 枚举）
- `forward_pass_comm_size` — 前向通信字节数
- 类似的反向和 DP 字段

Analytical 模式中 `Layer::compute_time()` 使用 `cal_busbw()` 分析计算通信延迟。

#### 训练仿真循环（HYBRID_TRANSFORMER_FWD_IN_BCKWD）

```
for pass in 0..TOTAL_PASS:
  Forward: for layer 0..N-1:
    wait(wg_comm[layer])  // 等待上一轮权重梯度通信完成
    delay(fwd_compute_time)
    issue(fwd_comm) → blocking
  Input Gradient: for layer N-1..0:
    delay(ig_compute_time)
    issue(ig_comm) → blocking
    if checkpoint: re-compute
  Weight Gradient:
    delay(wg_compute_time)
    issue(wg_comm) → non-blocking  // 与下一层 forward overlap
  PP bubble: pre_bubble_time × (PP-1) / (GA × VPP)
```

### 3.6 集合通信算法

`astra-sim/system/collective/` 目录下实现了以下算法：

| 算法 | 文件 | 描述 |
|------|------|------|
| **Ring** | `Ring.cc` | Ring AllReduce/AllGather/ReduceScatter |
| **DoubleBinaryTree** | `DoubleBinaryTreeAllReduce.cc` | 双二叉树 AllReduce |
| **HalvingDoubling** | `HalvingDoubling.cc` | 递归减半加倍算法 |
| **NcclTreeFlowModel** | `NcclTreeFlowModel.cc` | NVLS 树流量模型 |
| **AllToAll** | `AllToAll.cc` | 全对全通信 |

每个算法继承 `Algorithm` 基类，实现 `run()` 方法将集合操作分解为点对点 send/recv。

#### Ring Algorithm 细节
- AllReduce: `stream_count = 2*(N-1)`, `msg_size = data/N`
- AllGather: `stream_count = N-1`, `msg_size = data`
- ReduceScatter: `stream_count = N-1`, `msg_size = data/N`
- 在 `run()` 中: StreamInit 发起初始包 → PacketReceived 触发后续 → 完成时 `exit()`

#### NcclTreeFlowModel（最复杂）
- 使用 `MockNccl::FlowModels` 管理 flow DAG
- 每个 flow: `(channel_id, flow_id, src, dest, flow_size, parent_flow_ids, child_flow_ids)`
- 多 channel 并发通信，入度跟踪依赖
- 支持 Tree 和 NVLS 两种拓扑
- QP (Queue Pair) 流水线管理

### 3.7 MockNccl（SimCCL 集成）

`MockNccl*.cc/h` 实现了模拟 NCCL 的核心逻辑：
- **MockNcclGroup**: 管理一组 GPU 的通信组，决定使用哪种集合通信算法。构造时接受 TP/DP/PP/EP 尺寸，自动生成 ring channels、tree channels、NVLS channels
- **MockNcclChannel**: 管理通信通道，处理消息分块和流水线。定义 `SingleFlow` 数据结构（flow_id, src, dest, flow_size, chunk info, connection type）
- **MockNcclLog**: 记录通信日志用于分析
- **MockNcclQps**: 管理 QP (Queue Pair) 连接状态，跟踪 busy/idle per (channel, src→dest)

### 3.8 拓扑系统

```
LogicalTopology (abstract)
├── BasicLogicalTopology
│   ├── RingTopology (id, nodes, index, offset)
│   └── BinaryTree
└── ComplexLogicalTopology
    ├── GeneralComplexTopology (vector of BasicLogicalTopology per dimension)
    ├── DoubleBinaryTreeTopology (DBMAX + DBMIN)
    ├── LocalRingGlobalBinaryTree
    ├── LocalRingNodeA2AGlobalDBT
    └── Torus3D
```

`GeneralComplexTopology` 为每个维度创建 `RingTopology`（用于 Ring/Direct/HalvingDoubling/NcclFlowModel）或 `DoubleBinaryTreeTopology`。

### 3.9 网络后端

#### Analytical 后端
- `AnaSim.cc` + `AnalyticalNetwork.cc`
- `AnaSim`: 静态 FIFO 任务队列，`Run()` 按序处理，推进 tick
- `sim_send/recv()` 立即返回 0（不做网络仿真）
- 通信时间由 `Layer::compute_time()` 使用 `cal_busbw()` 分析计算
- 支持 overlap ratio 参数

#### NS-3 后端
- `AstraSimNetwork.cc` — 接入 ns-3-alibabacloud
- `sim_send()` 调用 `SendFlow()` 将流量注入 NS-3
- `sim_recv()` 使用 hash 表匹配 send/recv 对
- 支持 NS3_MTP（多线程）和 NS3_MPI（分布式）模式
- 提供包级别的 RDMA/RoCEv2 仿真，支持 QCN、PFC、DCQCN 拥塞控制

#### Phynet 后端
- `PhySimAi.cc` + `SimAiPhyNetwork.cc`
- 接入真实物理网络（TODO 状态），使用 MPI barriers 同步

### 3.10 Bus Bandwidth 计算引擎

`calbusbw.cc` 硬编码的带宽常量：

| 硬件 | NVLink BW | NIC BW | NVLS BW | PCIe BW |
|------|-----------|--------|---------|---------|
| H100 | 370.8 GB/s | CX7: 48.5 | 475 | 51.2 |
| H800 | 164.8 | CX7: 48.5 | 215 | 51.2 |
| A100 | 240 | CX6: 23.5 | — | 25.6 |
| A800 | 160 | CX6: 23.5 | — | 25.6 |

`cal_busbw()` 函数根据 GPU 类型、节点数、操作类型返回有效 busbw：
- Ring: `busbw = min(nvlink_bw, nic_bw × nics)`
- Tree: 受 min(NVLink, NIC × nics) 限制
- NVLS: 需要 H 系列 + 8 GPU/node
- AllToAll: 单节点 NVLink，多节点 `nic_bw × nics / gpus_per_node × (N-1) / ((nodes-1) × gpus)`

Ratio 修正：从 CSV 查找表（`nic_ratio.csv`, `nvlink_ratio.csv`, `ata_ratio.csv`）按消息大小和节点数插值，校正小消息开销。

### 3.11 拓扑生成

`inputs/topo/gen_Topo_Template.py` 生成 NS-3 可读的拓扑文件，支持 5 种模板：

| 模板 | GPU 数 | 架构特点 |
|------|--------|---------|
| Spectrum-X | 4096 | NVIDIA rail-optimized, 400Gbps |
| AlibabaHPN (单平面) | 15360 | 阿里 HPN, dual-ToR, 200Gbps |
| AlibabaHPN (双平面) | 15360 | 阿里 HPN, dual-plane |
| DCN+ (单ToR) | 512 | A100, 400Gbps |
| DCN+ (双ToR) | 512 | dual-ToR, 200Gbps |

拓扑文件格式：
```
total_nodes gpus_per_server nvswitch_num switch_num link_count gpu_type
switch_node_ids ...
src dst bandwidth latency error_rate
src dst bandwidth latency error_rate
...
```

### 3.12 SimAI.conf 配置

NS-3 仿真的网络参数配置文件，关键参数：

```
ENABLE_QCN 1                  # 启用量化拥塞通知
USE_DYNAMIC_PFC_THRESHOLD 1   # 动态 PFC 阈值
PACKET_PAYLOAD_SIZE 9000      # MTU 9000 bytes (jumbo frame)
CC_MODE 1                     # 拥塞控制模式 (DCQCN)
BUFFER_SIZE 32                # 交换机缓冲区大小 (MB)
U_TARGET 0.95                 # 目标链路利用率
KMAX_MAP / KMIN_MAP / PMAX_MAP  # ECN 标记阈值（按带宽分档）
```

### 3.13 CLI 接口

```bash
# Analytical 模式
./bin/SimAI_analytical \
    -w <workload.txt> \
    -g <总GPU数> \
    -g_p_s <每服务器GPU数> \
    -r <输出前缀> \
    -busbw <busbw.yaml> \
    [-v]                      # 生成可视化
    [-dp_o 0.5]               # DP overlap ratio
    [-tp_o 0.0]               # TP overlap ratio

# NS-3 仿真模式
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 \
./bin/SimAI_simulator \
    -t <线程数> \
    -w <workload.txt> \
    -n <拓扑文件路径> \
    -c <SimAI.conf路径>
```

### 3.14 输出格式

Analytical 模式输出 CSV，包含：
- 汇总行: 每个通信组的 exposure time、计算时间占比、端到端迭代时间
- 逐层行: 每层的计算时间、各通信组通信时间、overlap 时间

---

## 4. vidur-alibabacloud 详细分析

### 4.1 功能定位

vidur 是基于离散事件仿真（DES）的 **多请求推理仿真器**，扩展自 Microsoft Vidur。它模拟完整的推理服务流程：请求到达 → 调度 → prefill → decode → 完成。

### 4.2 核心架构

```
┌──────────────────┐
│  SimulationConfig │  统一配置系统（CLI→dataclass）
├──────────────────┤
│  RequestGenerator │  请求生成（Synthetic/Trace）
├──────────────────┤
│  GlobalScheduler  │  全局调度（LOR/RoundRobin/Splitwise）
├──────────────────┤
│  ReplicaScheduler │  副本调度（vLLM/Sarathi/ORCA/...）
├──────────────────┤
│  ExecutionTime    │  执行时间预测（Sklearn/AICB/SimAI）
│  Predictor        │
├──────────────────┤
│  MemoryPlanner    │  GPU 内存规划 + KV cache 预算
├──────────────────┤
│  MetricsStore     │  指标收集（TTFT/TBT/E2E/CDF）
└──────────────────┘
```

### 4.3 离散事件仿真循环

`simulator.py` 实现了经典的 DES 循环：

```python
class Simulator:
    def __init__(self, config):
        self._event_queue = []  # 最小堆
        self._cluster = Cluster(config)
        self._scheduler = GlobalSchedulerRegistry.get(...)
        self._request_generator = RequestGeneratorRegistry.get(...)
        self._init_event_queue()  # 将所有 RequestArrivalEvent 入队

    def run(self):
        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)
```

### 4.4 事件系统

#### 事件类型与完整事件链

```
RequestArrivalEvent
  → scheduler.add_request(request)
  → emit GlobalScheduleEvent

GlobalScheduleEvent
  → scheduler.schedule() → [(replica_id, request)]
  → 添加到 per-replica scheduler
  → emit ReplicaScheduleEvent (per affected replica)

ReplicaScheduleEvent
  → replica_scheduler.on_schedule()
  → 填充 PP pipeline (max num_stages 个 batch)
  → emit BatchStageArrivalEvent (stage=0, per batch)

BatchStageArrivalEvent
  → 加入 PP stage 队列
  → emit ReplicaStageScheduleEvent

ReplicaStageScheduleEvent
  → stage_scheduler.on_schedule()
  → execution_time = predictor.get_execution_time(batch, stage)
  → emit BatchStageEndEvent(time + execution_time)

BatchStageEndEvent
  → if 最后 PP stage: emit BatchEndEvent
  → else: emit BatchStageArrivalEvent(next_stage)
  → emit ReplicaStageScheduleEvent (继续当前 stage)

BatchEndEvent
  → batch.on_batch_end() → 更新每个 request 的 processed_tokens
  → if PD 分离 && prefill 完成:
      计算 KV cache transfer time
      调度 decode 到 D replica
  → emit ReplicaScheduleEvent (继续处理)
```

#### 事件优先级（同一时刻）

```python
BATCH_STAGE_ARRIVAL = 1  # 最高优先级
REQUEST_ARRIVAL = 2
BATCH_STAGE_END = 3
BATCH_END = 4
GLOBAL_SCHEDULE = 5
REPLICA_SCHEDULE = 6
REPLICA_STAGE_SCHEDULE = 7  # 最低优先级
```

### 4.5 请求生成器

| 类型 | 描述 |
|------|------|
| SyntheticRequestGenerator | 合成请求（可配 QPS、输入/输出长度分布） |
| TraceReplayRequestGenerator | 重放真实 trace |

输入/输出长度分布支持：Fixed, Uniform, Zipf。
请求到达间隔支持：Poisson, Gamma, Static, Trace。

### 4.6 调度器

#### 全局调度器

| 调度器 | 策略 |
|--------|------|
| RoundRobinGlobalScheduler | 轮询分配到副本 |
| LORGlobalScheduler | Least Outstanding Requests — 路由到待处理请求最少的副本 |
| RandomGlobalScheduler | 随机分配 |
| SplitwiseGlobalScheduler | Prefill/Decode 分离到不同副本组，P/D 各自 round-robin，构建 task DAG |

#### 副本调度器

| 调度器 | 策略 |
|--------|------|
| VLLMReplicaScheduler | Paged attention blocks，`max_tokens_in_batch` 限制，preemption 从后淘汰 |
| SarathiReplicaScheduler | Chunked prefill with `chunk_size`，混合 decode + partial prefill 在同一 batch |
| ORCAReplicaScheduler | ORCA iteration-level scheduling |
| SplitwiseReplicaScheduler | P 副本只 batch prefill 请求，D 副本只 batch decode 请求 |
| FasterTransformerReplicaScheduler | FasterTransformer 风格 |
| LightLLMReplicaScheduler | LightLLM 风格 |

所有副本调度器继承 `BaseReplicaScheduler`，提供 `_allocation_map`（request_id → num_blocks）、block 化 `allocate()/free()`、`on_schedule()` pipeline 填充循环。

### 4.7 执行时间预测

vidur 支持 4 种执行时间预测后端：

| 后端 | 计算时间来源 | 通信时间来源 | 精度 | 速度 |
|------|------------|------------|------|------|
| **Sklearn (vidur native)** | profiling CSV 训练 RandomForest | profiling CSV 训练模型 | 依赖 profiling 数据质量 | 快（查表） |
| **AICB** | AICB CSV per-layer comp_time | 设为 0（不模拟 TP comm） | 高（GPU profiled） | 中（可能触发 AICB 子进程） |
| **SimAI Analytical** | AICB CSV | busbw 分析模型 | 中 | 中 |
| **SimAI Simulation** | AICB CSV | NS-3 包级仿真 | 高 | 慢 |

AICB 后端使用全局缓存 `AICBGlobalCache`：精确匹配 → 相邻 seq 值线性插值 → 缓存未命中时运行 AICB 子进程。

`SimAIWorkload.py` 定义了 vidur 内部使用的 workload 格式，与 AICB 输出格式一致，用于将推理请求的计算/通信需求传递给 astra-sim。

### 4.8 GPU 内存规划

`MemoryPlanner` 精确计算每个设备的内存预算：

**1. 参数内存**: `ParamCounter` 计算每个设备上的参数量
   - 支持 TP 切分：attention heads / FFN 列切分
   - 支持 MoE：每个设备只存 `num_experts / EP` 个专家
   - 支持 FP8 量化：内存减半
   - 支持 MLA（DeepSeek）: 使用 `kv_lora_rank + qk_rope_head_dim` 代替完整 KV
   - PD 分离时，P 和 D 副本各自使用不同的 world_size 计算 EP 分配

**2. KV Cache 预算**:

| 模型类型 | KV Cache per token per layer | 公式 |
|---------|------------------------------|------|
| MHA/GQA | `2 × num_kv_heads × head_dim × 2` (BF16) | 标准 K+V |
| MLA (DeepSeek) | `(kv_lora_rank + qk_rope_head_dim) × 2` (BF16) | 压缩潜在表示 |
| FP8 | 以上 ÷ 2 | 半精度存储 |

DeepSeek-671B MLA: `(512 + 64) × 2 = 1152 bytes/token/layer`（BF16）
标准 128 KV heads × 128 dim: `2 × 128 × 128 × 2 = 65536 bytes/token/layer`
→ MLA 压缩比约 **57×**

**3. 最大 batch size**: `available_kv_cache_memory / (max_tokens × kv_per_token)`，带 OOM 检测和错误提示。

### 4.9 Prefill-Decode 分离（PD Separation）

vidur-alibabacloud 的核心创新之一，完整流程：

1. **Cluster 初始化**: `pd_node_ratio`（0-1）或 `num_prefill_replicas` 划分 P/D 副本。`pd_node_ratio=1` 为 MIXED 模式（无分离）。分别计算 `prefill_world_size` 和 `decode_world_size`，以及各自的 EP。
2. **SplitwiseGlobalScheduler.schedule()**: 请求分配到 P 副本（round-robin），同时预分配 D 副本存入 `request.decode_replica_id`
3. **SplitwiseReplicaScheduler**: P 副本只 batch prefill 请求（`is_prefill_complete == False`），D 副本只 batch decode 请求（`is_prefill_complete == True`）
4. **BatchEndEvent**: prefill 完成时计算 KV cache 传输:
   - `comm_size = estimate_kv_cache_size(num_processed_tokens)`
   - `transfer_time = comm_size / bandwidth`（默认 800 Gbps）
   - `decode_arrived_at = prefill_completed_at + transfer_time`
5. **请求转移到 D 副本**: `ReplicaScheduleEvent` 在 `decode_arrived_at` 时触发，D 副本开始 decode

### 4.10 指标系统

`MetricsStore` 收集以下指标：

| 指标 | 定义 |
|------|------|
| TTFT | Time To First Token = `prefill_completed_at - arrived_at` |
| TBT/TPOT | Time Between Tokens = `decode_time / num_decode_tokens` |
| E2E Latency | `completed_at - arrived_at` |
| Scheduling Delay | `scheduled_at - arrived_at` |
| Preemption Time | 被抢占的时间 |
| PD 专属 | `pd_p2p_comm_size/time/bandwidth`, `prefill/decode_replica_id` |

使用 `DDSketch`（`relative_accuracy=0.001`）高效存储百分位数据，支持 P25/P50/P75/P95/P99/P99.9 报告。

MFU 计算: `MFUCalculator` 对每个 batch_stage 计算 MLP FLOPs + Attention FLOPs，除以执行时间和设备 FP16 TFLOPS。

### 4.11 Config Optimizer

vidur 包含自动配置优化器，可搜索 Pareto 最优配置。

**搜索空间**: TP, PP, batch_size, scheduler_type, chunk_size 等配置参数的笛卡尔积。

**CapacitySearch**: 二分搜索最大 QPS under SLO:
```
while not converged:
    if delay << SLO: qps *= 4
    elif delay < SLO/4: qps *= 2
    elif delay > SLO: qps /= 2
    run_simulation(qps) → measure P{quantile} scheduling_delay
```

**BottleneckAnalyzer**: 诊断瓶颈类型:
- TTFT 违规 → batch 大小不足 / GPU 内存不足 / prefill 吞吐低 / pipeline bubble
- TBT 违规 → 模型执行延迟高 / 尾延迟（调度器选择 / chunk_size）

Streamlit dashboard 可视化 Pareto 曲线。

### 4.12 支持的模型

```
vidur-alibabacloud/data/hf_configs/
├── deepseek_v3_config.json        # DeepSeek-V3 671B (MLA + MoE)
├── deepseek_R1_0528_config.json   # DeepSeek-R1 (MLA + MoE)
├── qwen3-235B-A22B_config.json    # Qwen3-MoE 235B (GQA + MoE)
├── qwen3-235B-A22B_FP8_config.json
├── qwen3-30B-A3B_config.json      # Qwen3-MoE 30B
├── qwen3-8B_config.json           # Qwen3 8B (Dense)
├── qwen3-next-80B-A3B_config.json # Qwen3-Next 80B (混合注意力 + MoE)
└── qwen3-next-80B-A3B_Instruct_FP8_config.json
```

### 4.13 GPU 设备规格

| 设备 | FP16 TFLOPS | FP8 TFLOPS | HBM (GB) | Nodes |
|------|-------------|------------|----------|-------|
| A40 | 150 | — | 45 | 4-GPU pairwise NVLink |
| A100 | 312 | — | 80 | 8-GPU DGX / pairwise |
| H20 | 148 | 296 | 141 | 8-GPU DGX |
| H100 | 1000 | — | 80 | 8-GPU DGX / pairwise |
| H800 | 989 | 1979 | 80 | 8-GPU DGX |

---

## 5. SimCCL 详细分析

### 5.1 功能

SimCCL 是集合通信分解库，将高层集合操作（AllReduce, AllGather 等）分解为点对点 send/recv 操作。它以 C++ 库的形式嵌入到 astra-sim 中，通过 `MockNccl*` 文件整合。

### 5.2 支持的算法

- Ring AllReduce (ReduceScatter + AllGather)
- Double Binary Tree AllReduce
- NVLS Tree Flow Model
- Halving-Doubling
- AllToAll (full mesh)

算法选择取决于 GPU 拓扑和消息大小。NVLS 在大消息 + NVLink 场景下表现最优（论文 benchmark: 512MB 时 NVLS 218 GB/s vs Ring 167 GB/s）。

---

## 6. 端到端工作流

### 6.1 训练仿真（Analytical）

```
Step 1: 生成 Workload
$ python aicb/workload_generator/SimAI_training_workload_generator.py \
    --model_name DeepSeek-671B --world_size 9216 \
    --tensor_model_parallel_size 2 --expert_model_parallel_size 16 \
    --pipeline_model_parallel 12 --seq_length 4096 --aiob_enable
                                    ↓
Step 2: 准备 busbw.yaml（手动配置各并行组带宽）
                                    ↓
Step 3: 运行仿真
$ ./bin/SimAI_analytical -w workload.txt -g 9216 -g_p_s 8 \
    -r result- -busbw busbw.yaml -v
                                    ↓
Step 4: 分析输出 CSV（迭代时间、通信占比、瓶颈分析）
```

### 6.2 训练仿真（NS-3）

```
Step 1: 生成 Workload（同上）
                                    ↓
Step 2: 生成网络拓扑
$ python astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo Spectrum-X -g 32 -bw 400Gbps -gt H100
                                    ↓
Step 3: 运行 NS-3 仿真
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 \
  ./bin/SimAI_simulator -t 16 -w workload.txt \
    -n ./Spectrum-X_32g_8gps_400Gbps_H100 \
    -c astra-sim-alibabacloud/inputs/config/SimAI.conf
                                    ↓
Step 4: 分析输出（实际带宽、拥塞情况、per-layer timing）
```

### 6.3 推理仿真（Vidur）

```
Step 1: 生成推理 Workload
$ python aicb/workload_generator/Vidur_workload_generator.py \
    DeepSeek-671B scripts/inference_configs/deepseek_default.json \
    --world_size 256 --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 64 --phase decode
                                    ↓
Step 2: 运行 Vidur 仿真
$ python -m vidur.main \
    --replica_config_model_name deepseek-671B \
    --replica_config_tensor_parallel_size 4 \
    --replica_config_num_pipeline_stages 1 \
    --cluster_config_num_replicas 4 \
    --request_generator_config_type synthetic \
    --replica_scheduler_config_type vllm \
    --global_scheduler_config_type lor
                                    ↓
Step 3: 分析输出（TTFT, TBT, E2E latency, batch_size timeline, chrome trace）
```

---

## 7. 关键数据结构与接口

### 7.1 Work_Item（AICB ↔ astra-sim 的桥梁）

```python
@dataclasses.dataclass
class Work_Item:
    name: str                    # 层名称
    placeholder: int = -1        # 占位符
    forward_compute_time: int    # 前向计算时间 (ns)
    forward_comm: str            # 前向通信类型
    forward_comm_size: int       # 前向通信字节数
    backward_compute_time: int   # 反向计算时间 (ns)
    backward_comm: str           # 反向通信类型
    backward_comm_size: int      # 反向通信字节数
    dp_compute_time: int         # DP 计算时间 (ns)
    dp_comm: str                 # DP 通信类型
    dp_comm_size: int            # DP 通信字节数
    process_time: int = 100      # 额外处理时间
```

### 7.2 busbw.yaml 格式

```yaml
<名称>
TP:
  allreduce,: <GB/s>
  allgather,: <GB/s>
  reducescatter,: <GB/s>
  alltoall,: <GB/s>
DP:
  allreduce,: <GB/s or null>
  allgather,: <GB/s>
  reducescatter,: <GB/s>
  alltoall,: <GB/s or null>
EP:
  allreduce,: <GB/s or null>
  allgather,: <GB/s>
  reducescatter,: <GB/s>
  alltoall,: <GB/s>
PP:
  busbw: <GB/s>
```

---

## 8. 编译与依赖

### 8.1 编译

```bash
# Analytical 模式（仅需 C++ 编译器）
./scripts/build.sh -c analytical

# NS-3 模式（需要完整 NS-3 编译环境）
apt remove ninja-build && pip uninstall ninja  # NS-3 编译与 ninja 冲突
./scripts/build.sh -c ns3
```

### 8.2 依赖

- **AICB**: Python, PyTorch（仅 workload 执行模式需要）
- **astra-sim**: C++17, CMake, yaml-cpp, protobuf
- **NS-3**: NS-3 3.x, C++17
- **vidur**: Python 3.10+, scipy, numpy, pandas, scikit-learn, wandb, ray (可选)

### 8.3 Docker

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.07-py3
# 预装 PyTorch, CUDA, cuDNN
# 需要 apt remove ninja-build 后编译 NS-3
```

---

## 9. 与 MLSysBench 的对接点

### 9.1 可直接使用的能力

| SimAI 能力 | MLSysBench 应用 | 具体接口 |
|-----------|-----------------|---------|
| vidur 推理仿真 | L3 系统级优化 task 的评估环境 | `vidur.main` CLI |
| 多种调度器 | 调度优化 task 的 baseline/oracle | `ReplicaSchedulerRegistry` |
| PD 分离 | Prefill-decode 优化 task | `SplitwiseGlobalScheduler` |
| 配置优化器 | 自动化配置搜索 task | `config_optimizer/` |
| AICB workload 生成 | 为仿真提供标准化输入 | `SimAI_inference_workload_generator.py` |
| busbw + Analytical | 快速并行策略评估 | `SimAI_analytical` |
| 内存规划 | GPU 内存优化 task | `MemoryPlanner` |

### 9.2 需要扩展的能力

| 需求 | 现状 | 扩展方向 |
|------|------|---------|
| Kernel 级仿真 | SimAI 以 kernel time 为输入 | 不适用，仍需真实 GPU |
| 量化效果仿真 | ParamCounter 支持 FP8 | 需添加 W4A8/INT4 支持 |
| 新调度算法评估 | 固定的 6 种调度器 | 可注册新调度器到 registry |
| 新模型支持 | 已支持 7 种模型 | 通过 JSON config 添加 |
| Agent 接口 | 无 | 需要包装 vidur CLI 为 tool API |

### 9.3 不适用的场景

- L2 kernel 编写 task（CUDA/Triton kernel）— SimAI 不仿真 kernel 内部
- 注意力算法优化（FlashAttention 等）— 需要真实计算验证
- 数值精度验证（量化 accuracy）— SimAI 不处理数值

---

## 10. 关键设计决策总结

1. **计算与通信分离**: SimAI 将模型执行抽象为 `(compute_time, comm_type, comm_size)` 三元组，计算时间来自 AIOB profiling 或默认值，通信时间由仿真器计算。

2. **可插拔网络后端**: analytical（快速估算）/ NS-3（精确仿真）/ phynet（真实网络），同一 workload 可在不同后端上运行。

3. **MockedModel 模式**: 不实际创建 PyTorch 模型，只构建参数形状信息，无需 GPU 即可生成 workload。

4. **事件驱动仿真**: vidur 使用最小堆事件队列，事件处理可触发新事件，自然支持异步调度和并发请求。

5. **Registry 模式**: 调度器、请求生成器、执行时间预测器都使用 registry 模式，方便扩展。

6. **NCCL 仿真保真度**: SimCCL 精确复现 NCCL 的 channel 分配、消息分块、流水线行为，而非简单的带宽除法。
