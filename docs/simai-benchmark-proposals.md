# 基于 SimAI 的推理系统优化 Benchmark 设计方案

> 目标：利用 SimAI/vidur-alibabacloud 构建一个递进式 benchmark，评测 LLM/Agent 在推理系统配置优化、诊断修复、规模外推上的能力。

## 0. 核心洞察：为什么现有 Benchmark 不够

| Benchmark | 测的是什么 | 缺什么 |
|-----------|-----------|--------|
| KernelBench/FastKernels | 能不能写出快的 kernel | 不测系统决策（并行策略、调度、内存规划） |
| InferenceBench | 能不能配好 vLLM | 只测配置搜索，不测新方案设计；单模型单 GPU；非 agent 搜索反而更好 |
| AgentKernelArena | Agent 能不能迭代优化 kernel | 仅 kernel 层，不涉及服务系统 |
| PIE/ECCO | 能不能让代码跑更快 | 竞赛题，非推理系统 |
| PerfCodeBench | 系统级代码优化 | 通用 HPC，非推理场景 |

**空白地带**: 没有 benchmark 评测 agent 在 **推理系统层面的设计与优化能力** —— 给定一个大模型和硬件约束，agent 能否做出正确的并行策略选择、调度算法设计、内存规划、PD 分离决策，并最终实现高吞吐低延迟的推理服务？

**SimAI 带来的独特机会**: 仿真环境让我们可以：
1. 在 CPU 上评估涉及 9216+ GPU 的优化决策（真实评估不可能）
2. 获得确定性的仿真结果（消除硬件噪声，benchmark 可复现）
3. 快速迭代（秒级仿真 vs 小时级真实运行）
4. 评估 **决策能力**（不只是 coding 能力）

### 代码核验后的第一版边界

当前 `third_party/SimAI/vidur-alibabacloud` 已经支持多请求离散事件推理仿真、trace/synthetic workload、TP/PP/replica/scheduler/batch/chunk/PD 配置、MemoryPlanner OOM 检查、TTFT/TBT/E2E/throughput 等指标输出。

但第一版不能把以下能力作为正式评分 action：

- **手动 EP 搜索**：`expert_model_parallel_size` 会被自动设为 `world_size`，用户手动指定不一致会报错。
- **EP/AllToAll 网络瓶颈**：vidur 接入 SimAI 的通信预测路径当前主要建模 TP AllReduce。
- **拓扑级 PD/网络竞争**：PD KV 传输当前是 `kv_size / pd_p2p_comm_bandwidth` 的参数化估算，不是 topology-aware congestion simulation。
- **异构 GPU 调度**：当前 cluster/config 假设较强，缺少非均匀设备调度模型。
- **Prefix overlap routing**：request 对象尚无语义级 prefix overlap 字段。

因此，第一版 benchmark 应先覆盖：

```text
TP / PP / replica 数 / scheduler / batch cap / Sarathi chunk size /
PD on-off / P:D ratio / PD bandwidth sensitivity / QPS under SLO
```

EP、AllToAll、topology-aware networking、heterogeneous scheduling 和 prefix-aware routing 作为后续扩展。

---

## 1. 方案一：SimAI-Gym —— 推理系统优化的交互式环境

### 设计灵感

借鉴 **CompilerGym** 的 "编译器即环境" 思路，将 SimAI 包装为 OpenAI Gym 风格的交互式环境。Agent 通过 observation-action-reward 循环与仿真器交互，而非一次性生成解决方案。

### 核心设计

```
┌─────────────────────────────────────────────────┐
│                SimAI-Gym Environment             │
│                                                  │
│  Observation:                                    │
│  ├── 模型架构描述 (JSON)                          │
│  ├── 硬件配置 (GPU 型号/数量/互联拓扑)             │
│  ├── 当前配置的仿真结果 (TTFT/TBT/throughput)      │
│  ├── 当前配置的瓶颈分析 (TTFT/TBT/memory/util)     │
│  └── 历史 action-reward 对                        │
│                                                  │
│  Action Space:                                   │
│  ├── 并行策略: set_tp(n), set_pp(n), set_replicas(n)│
│  ├── 调度器: set_scheduler(vllm|sarathi|orca)     │
│  ├── PD 分离: set_pd_ratio(r), set_pd_bandwidth() │
│  ├── Batching: set_batch_size(n), set_chunk_size()│
│  ├── 容量: set_qps(q), set_max_tokens(n)          │
│  └── 内存: set_pd_transfer_dtype(fp16|fp8)        │
│                                                  │
│  Reward: Δ(target_metric)                        │
│  ├── 吞吐场景: Δ(throughput)                      │
│  ├── 延迟场景: -Δ(P99_latency)                    │
│  └── 成本场景: Δ(throughput / gpu_cost)            │
└─────────────────────────────────────────────────┘
```

### Task 设计

**Level 1: 单维度优化（10 tasks）**

Agent 只需调整一个维度，测试基本理解：
- T1.1: 给定 Llama-70B on 8×A100，找最优 TP 值（1/2/4/8）
- T1.2: 给定 Llama-70B on 8×A100，找最优 PP 值（1/2/4）
- T1.3: 给定高 QPS 场景，选择最优调度器（vLLM vs Sarathi vs ORCA）
- T1.4: 给定 prefill-heavy workload，决定是否启用 PD 分离
- T1.5: 给定内存紧张场景，调整 TP 或 batch cap 避免 OOM

**Level 2: 多维度联合优化（10 tasks）**

Agent 需要同时调整多个维度：
- T2.1: Llama-70B on 64×H100，联合优化 TP+PP+scheduler
- T2.2: Qwen3-MoE-235B on H20/A100 集群，优化 TP+PD ratio+batch_size
- T2.3: Qwen3-Next-80B on 32×H100，考虑混合注意力的特殊性
- T2.4: 给定 SLO 约束（TTFT P99 < 500ms, TBT P99 < 50ms），最大化 QPS
- T2.5: 给定 GPU 预算约束，最小化 P99 延迟

**Level 3: 开放优化（5 tasks）**

Agent 可以修改 vidur 代码，实现新的调度策略或优化：
- T3.1: 实现一个自适应 batch size 调度器（根据队列深度动态调整）
- T3.2: 实现 workload-aware chunked prefill 策略
- T3.3: 设计混合 PD 策略（部分副本 mixed，部分 disaggregated）
- T3.4: 实现 memory-aware / SLO-aware 请求路由
- T3.5: 实现 admission control，在高 QPS 下保护 tail latency

### 评估指标

```python
score = correctness_gate × (
    w1 × normalized_metric_improvement +   # 优化效果
    w2 × efficiency_bonus +                 # 步骤效率 (fewer steps = higher)
    w3 × generalization_score              # 跨场景泛化
)
```

- **correctness_gate**: 仿真必须成功运行（配置合法、不 OOM）
- **normalized_metric_improvement**: 相对 baseline 的 TTFT/TBT/E2E/throughput/goodput 改进；best-known config 只用于难度校准，不作为唯一正确答案
- **efficiency_bonus**: `max(0, 1 - steps / max_steps)`
- **generalization_score**: 在 unseen 的 model+hardware 组合上测试

### 与现有 Benchmark 的关键区分

| 维度 | InferenceBench | SimAI-Gym |
|------|---------------|-----------|
| 评测层次 | 框架选择 + 参数调整 | 并行策略 + 调度设计 + 系统架构 |
| 硬件需求 | 真实 H100 | CPU 即可（仿真） |
| 模型规模 | 单 GPU（Mistral-7B） | 最大 9216 GPU（DeepSeek-671B） |
| 评估速度 | 2 小时/run | 秒级/run |
| 能力维度 | 配置搜索 | 系统设计与推理 |
| 开放程度 | 选择已有框架配置 | 可实现新算法 |

---

## 2. 方案二：InfraAgent-Bench —— 诊断-修复范式的推理系统 Benchmark

### 设计灵感

借鉴 **SWE-bench** 的 "真实问题 → 修复" 范式。不同于 SWE-bench 使用 GitHub issue，我们构造 **有性能问题的推理系统配置**，agent 需要诊断问题并修复。

### 核心理念

> "给 agent 一个跑得慢的推理系统，看它能不能找到瓶颈并修好。"

这比从零配置更接近真实工作场景——系统工程师的日常就是诊断和修复性能问题。

### Task 构造方法

每个 task 包含：
```
task/
├── scenario.yaml         # 模型、硬件、workload 描述
├── baseline_config.yaml  # 有性能问题但合法可运行的初始配置
├── symptoms.txt          # 性能症状（如 "TTFT P99 = 2.3s, 目标 < 500ms"）
├── public_report.json    # 公开指标摘要，如 TTFT/TBT、memory、utilization、batch stats
├── hidden/
│   ├── eval_workload.yaml
│   └── baseline_metrics.json
└── evaluator.py          # 运行 vidur 仿真，检查 correctness + metric
```

可以保留人工 diagnosis label 作为论文分析材料，但第一版正式分数不依赖 root-cause 文本匹配。正式评分看修复后的配置在 hidden workload 上是否满足 SLO，以及相对 baseline 的实测改进。

### 瓶颈类型分类（任务来源）

| 瓶颈类型 | 症状 | 正确修复 | 难度 |
|---------|------|---------|------|
| **TP 过大** | TTFT 高，GPU 利用率低 | 减小 TP，增大 batch | Easy |
| **TP 过小** | 模型/缓存内存压力高，batch 容量不足 | 增大 TP 或降低 batch cap | Easy |
| **调度器错配** | 长 prefill 阻塞 decode | 从 vLLM 换到 Sarathi chunked prefill | Medium |
| **chunk size 不合适** | TTFT 或 TBT tail latency 变差 | 调整 Sarathi chunk size | Medium |
| **batch cap 不合适** | GPU 利用率低或 P99 爆炸 | 调整 batch cap / max tokens in batch | Medium |
| **内存 OOM** | batch size 只能设 1 或仿真失败 | 减小 max_tokens / batch cap / 增大 TP | Medium |
| **PD 不分离** | prefill 和 decode 互相干扰 | 启用 PD 分离 + 调整 ratio | Medium |
| **PP bubble** | GPU idle time > 30% | 减少 PP stages / 增大 micro-batch / VPP | Hard |
| **多因素叠加** | 多个症状同时出现 | 需要逐步排查和修复 | Expert |

### 评估维度

1. **诊断合理性**: agent 是否给出 coherent 的根因分析？用于人工分析或附加报告，不作为主分
2. **修复有效性**: 修复后的配置性能提升多少？
3. **修复效率**: 用了多少步？
4. **SLO 满足率**: hidden workload 上 TTFT/TBT/E2E 是否满足任务约束

### 与现有 Benchmark 的关键区分

这个范式独特在于测试 **诊断推理能力**，而非生成能力。InferenceBench 测 "你能配好系统吗"，我们测 "系统出了问题你能找到原因并修好吗"。这更接近高级系统工程师的核心技能。

---

## 3. 方案三：ScaleUp Challenge —— 跨规模外推 Benchmark

### 设计灵感

借鉴 AgentKernelArena 的 **unseen-configuration generalization**，但用在系统规模维度。测试 agent 的配置决策能否从小规模 **外推** 到大规模。

### 核心理念

> "在 8 GPU 上找到好的配置不难；难的是从 8 GPU 的经验推断出 1024 GPU 上该怎么配。"

这是真实场景中极其重要但完全未被评测的能力。公司从 prototype 到 production 的 scale-up 过程中，系统工程师需要预判大规模下的行为变化。

### Task 设计

每个 task 分两阶段：

**Phase 1: 小规模探索（开放）**
- Agent 获得完整的 SimAI 仿真环境（analytical + vidur）
- 可以在小规模（8-32 GPU）上自由探索、运行仿真
- 时间预算: 无限次仿真

**Phase 2: 大规模预测（闭卷）**
- Agent 必须在 **不运行仿真** 的情况下，预测大规模配置的最优策略
- 或者：给有限次仿真机会（如 5 次），在大规模上快速收敛

| Task | Phase 1 (小规模) | Phase 2 (大规模) | 测试的外推能力 |
|------|-----------------|-----------------|---------------|
| S1 | Llama-70B, 8 GPU | 同模型, 128 GPU | TP→TP+PP+replica 的规模效应 |
| S2 | Llama-3-8B, 4 replicas | 同模型, 64 replicas | 多副本调度和 QPS capacity 外推 |
| S3 | Qwen3-MoE-235B, 16 GPU | 同模型, 128 GPU | MoE 模型的内存/TP/replica 外推，不评分 EP 通信 |
| S4 | prefill-heavy workload, 8 GPU | 同 workload, 128 GPU | PD ratio 和 PD bandwidth sensitivity 外推 |
| S5 | 低 QPS, 4 replicas | 高 QPS, 64 replicas | SLO goodput 随负载和副本数变化 |

### 评估指标

```python
prediction_score = 1 - |predicted_optimal_time - actual_optimal_time| / actual_optimal_time
efficiency_score = max(0, 1 - num_large_scale_sims / budget)
final_score = prediction_score × 0.7 + efficiency_score × 0.3
```

---

## 4. 方案对比与推荐

| 方案 | 创新度 | 实现复杂度 | GPU 需求 | 与现有 benchmark 区分度 | 学术贡献 |
|------|--------|-----------|---------|----------------------|---------|
| **SimAI-Gym** | ★★★★ | ★★★ | 无 | ★★★★★ | 首个推理系统优化的 Gym 环境 |
| **InfraAgent-Bench** | ★★★★★ | ★★ | 无 | ★★★★★ | 首个诊断-修复范式的系统 benchmark |
| **ScaleUp Challenge** | ★★★★★ | ★★★ | 无 | ★★★★★ | 首个评测规模外推能力 |

### 推荐组合

**最小可行产品（MVP）**: 方案一 SimAI-Gym 的 Level 1-2（纯仿真，~12-20 tasks）
- 实现最快，GPU 零成本，已有足够区分度
- 直接利用 vidur 的 CLI、metrics CSV、MemoryPlanner 和 config explorer
- action space 限定为 TP/PP/replica/scheduler/batch/chunk/PD/QPS

**差异化最大**: 方案二 InfraAgent-Bench 或方案三 ScaleUp Challenge
- 完全没有同类 benchmark
- 测试的能力维度（诊断推理 / 规模外推）是现有 benchmark 完全空白的

**完整版**: 方案一 + 方案二 + 方案三递进
- Level 1-2 用 Gym 式交互优化
- Level 3 用诊断-修复范式
- Level 4 用 ScaleUp Challenge 测小规模到大规模的决策迁移
- 覆盖 "调配置"、"修复已有系统"、"规模外推" 三种真实场景

---

## 5. 技术可行性分析

### SimAI 作为 Backend 的改造量

| 需要做的 | 复杂度 | 涉及的 SimAI 组件 |
|---------|--------|------------------|
| Python API wrapper | 低 | vidur CLI → Python API |
| 配置合法性校验 | 低 | MemoryPlanner OOM 检测 |
| 标准化 observation 输出 | 中 | MetricsStore + BottleneckAnalyzer |
| 批量仿真加速 | 中 | vidur 的 config_explorer |
| best-known 配置搜索 | 中 | CapacitySearch + grid/random search，用于难度校准 |
| 新模型支持 | 低 | 添加 JSON config + AICB 模型 |
| Agent 接口标准化 | 中 | OpenAI function calling format |

### 关键风险

1. **仿真保真度**: SimAI 的仿真结果与真实系统有多大差距？如果差距过大，benchmark 结论可能不可靠
   - 缓解: NSDI'25 论文验证了训练仿真的精度；推理仿真（vidur）的验证是开放问题
   - 可以在论文中明确声明 benchmark 评测的是 "在仿真环境中的优化能力"

2. **Action space 设计**: 如果 action space 太小，benchmark 退化为穷举搜索问题
   - 缓解: Level 3 的开放任务要求实现新算法，不只是调参

3. **数据泄露**: Agent 可能在训练数据中见过 SimAI 的最优配置
   - 缓解: 使用不公开的模型配置（如修改 hidden_size/num_experts）；引入随机化的硬件约束

4. **代码能力边界**: 当前 vidur 不支持手动 EP 搜索、EP AllToAll 网络评分、拓扑级 PD 拥塞和异构调度
   - 缓解: 第一版任务只覆盖已核验 action space；EP/AllToAll/topology-aware/heterogeneous/prefix-aware 任务进入后续扩展
