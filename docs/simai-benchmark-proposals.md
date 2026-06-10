# 基于 SimAI 的推理系统优化 Benchmark 设计方案

> 头脑风暴：如何利用 SimAI 仿真平台构建一个评测 LLM/Agent 端到端推理系统优化能力的 benchmark。

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
│  ├── 当前配置的瓶颈分析 (compute/memory/comm)      │
│  └── 历史 action-reward 对                        │
│                                                  │
│  Action Space:                                   │
│  ├── 并行策略: set_tp(n), set_pp(n), set_ep(n)   │
│  ├── 调度器: set_scheduler(vllm|sarathi|orca)     │
│  ├── PD 分离: set_pd_ratio(r), set_pd_bandwidth() │
│  ├── Batching: set_batch_size(n), set_chunk_size()│
│  ├── 内存: set_kv_cache_dtype(fp16|fp8)           │
│  └── 网络: set_topology(), set_busbw()            │
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
- T1.2: 给定 DeepSeek-671B on 256×H100，找最优 EP 值
- T1.3: 给定高 QPS 场景，选择最优调度器（vLLM vs Sarathi vs ORCA）
- T1.4: 给定 prefill-heavy workload，决定是否启用 PD 分离
- T1.5: 给定内存紧张场景，决定 KV cache 精度（FP16 vs FP8）

**Level 2: 多维度联合优化（10 tasks）**

Agent 需要同时调整多个维度：
- T2.1: DeepSeek-671B on 64×H100，联合优化 TP+EP+scheduler
- T2.2: Qwen3-MoE-235B on 128×H100，优化 TP+EP+PD ratio+batch_size
- T2.3: Qwen3-Next-80B on 32×H100，考虑混合注意力的特殊性
- T2.4: 给定 SLO 约束（TTFT P99 < 500ms, TBT P99 < 50ms），最大化 QPS
- T2.5: 给定 GPU 预算约束，最小化 P99 延迟

**Level 3: 开放优化（5 tasks）**

Agent 可以修改 vidur 代码，实现新的调度策略或优化：
- T3.1: 实现一个自适应 batch size 调度器（根据队列深度动态调整）
- T3.2: 实现分层 EP 策略（热门 expert 本地，冷门 expert 远程）
- T3.3: 设计混合 PD 策略（部分副本 mixed，部分 disaggregated）
- T3.4: 实现 KV cache 感知的请求路由（基于 prefix 重叠度）
- T3.5: 给定异构硬件（A100+H100 混合集群），设计非均匀并行策略

### 评估指标

```python
score = correctness_gate × (
    w1 × normalized_metric_improvement +   # 优化效果
    w2 × efficiency_bonus +                 # 步骤效率 (fewer steps = higher)
    w3 × generalization_score              # 跨场景泛化
)
```

- **correctness_gate**: 仿真必须成功运行（配置合法、不 OOM）
- **normalized_metric_improvement**: `(agent_metric - baseline_metric) / (oracle_metric - baseline_metric)`，oracle 由穷举搜索获得
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
├── broken_config.py      # 有性能问题的初始配置
├── symptoms.txt          # 性能症状（如 "TTFT P99 = 2.3s, 目标 < 500ms"）
├── oracle_diagnosis.json # 人工标注的根因（hidden, 不给 agent 看）
├── oracle_config.py      # 已知最优配置（hidden, 用于计算 normalized score）
└── evaluator.py          # 运行 vidur 仿真，检查 correctness + metric
```

### 瓶颈类型分类（任务来源）

| 瓶颈类型 | 症状 | 正确修复 | 难度 |
|---------|------|---------|------|
| **TP 过大** | TTFT 高，GPU 利用率低 | 减小 TP，增大 batch | Easy |
| **EP 不匹配** | MoE 通信占比 > 50% | 调整 EP 使通信量最小化 | Easy |
| **调度器错配** | 长 prefill 阻塞 decode | 从 vLLM 换到 Sarathi chunked prefill | Medium |
| **内存 OOM** | batch size 只能设 1 | FP8 KV cache / 减小 max_tokens / 增大 TP | Medium |
| **PD 不分离** | prefill 和 decode 互相干扰 | 启用 PD 分离 + 调整 ratio | Medium |
| **PP bubble** | GPU idle time > 30% | 减少 PP stages / 增大 micro-batch / VPP | Hard |
| **网络瓶颈** | AllToAll 延迟 > 计算时间 | 调整网络拓扑 / 减小 EP | Hard |
| **多因素叠加** | 多个症状同时出现 | 需要逐步排查和修复 | Expert |

### 评估维度

1. **诊断准确度**: agent 是否正确识别了根因？（与 oracle_diagnosis 比较）
2. **修复有效性**: 修复后的配置性能提升多少？
3. **修复效率**: 用了多少步？
4. **解释质量**: agent 的分析过程是否 coherent？（可选的 LLM-as-judge）

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
| S1 | Llama-70B, 8 GPU | 同模型, 128 GPU | TP→TP+PP+DP 的规模效应 |
| S2 | Mixtral-8x7B, 16 GPU | 同模型, 512 GPU | EP 通信随规模增长的非线性效应 |
| S3 | DeepSeek-671B, 64 GPU | 同模型, 2048 GPU | MoE 大规模下的 EP+DP 交互 |
| S4 | 固定 workload, Spectrum-X | 同 workload, AlibabaHPN | 拓扑变化对性能的影响 |
| S5 | 低 QPS, 4 replicas | 高 QPS, 64 replicas | 多副本调度策略的规模效应 |

### 评估指标

```python
prediction_score = 1 - |predicted_optimal_time - actual_optimal_time| / actual_optimal_time
efficiency_score = max(0, 1 - num_large_scale_sims / budget)
final_score = prediction_score × 0.7 + efficiency_score × 0.3
```

---

## 4. 方案四：ConfigArena —— 推理配置的对抗式 Benchmark

### 设计灵感

借鉴 **Chatbot Arena** 的 ELO 评分 + **KernelBench** 的 speedup ratio，创建 agent 之间的直接对抗。

### 核心理念

> "两个 agent 面对同一个推理优化挑战，谁配出来的系统更快？"

### 运行流程

```
1. 随机采样一个 (model, hardware, workload, SLO) 四元组
2. 两个 agent 各自独立与 SimAI 交互，限定相同步数
3. 取各自最终配置的 vidur 仿真结果
4. 更高 throughput / 更低 latency 的一方获胜
5. 更新 ELO 评分
```

### 场景池设计

为确保评测全面，场景池覆盖多个维度的组合：

| 维度 | 变化范围 |
|------|---------|
| 模型 | Dense (Llama-70B), MoE (Mixtral, Qwen3-MoE-235B), MLA+MoE (DeepSeek-671B), Hybrid (Qwen3-Next-80B) |
| GPU | A100-80GB, H100-80GB, H20-141GB |
| 规模 | 8, 32, 128, 512, 2048 GPU |
| Workload | Prefill-heavy, Decode-heavy, Mixed, Bursty |
| SLO | Latency-first, Throughput-first, Cost-first |

### 对抗维度的独特价值

- **避免 ground truth 问题**: 不需要知道最优解，只需要比较两个 agent
- **自动难度校准**: ELO 系统天然区分 easy/hard 场景
- **持续评估**: 新 agent/model 加入时只需与现有选手对战
- **防止 overfitting**: 从大场景池随机采样，agent 无法提前准备

---

## 5. 方案五：FullStack-Opt —— 从分析到实现的全栈评测

### 设计灵感

融合 InferenceBench 的 "真实执行" + SimAI 的 "仿真验证"，创建两阶段评测。

### 核心理念

> "先在仿真中验证设计决策，再在真实硬件上验证实现。"

这直接模拟了工业界的推理系统优化工作流：工程师先用仿真工具探索方案，确认可行后再在实际集群上部署。

### 两阶段设计

**Stage A: 仿真阶段（SimAI，无 GPU）**

Agent 使用 SimAI 环境探索优化方案：
1. 分析模型架构和硬件约束
2. 使用 analytical 模式快速筛选并行策略
3. 使用 vidur 仿真验证调度和 PD 分离策略
4. 输出优化方案文档 + 配置文件

评估: 方案的仿真指标 vs oracle

**Stage B: 实现阶段（真实 GPU）**

Agent 将 Stage A 的方案落地到真实推理框架：
1. 配置 vLLM/SGLang 按照 Stage A 的策略
2. 如果方案需要代码修改（如新调度器），实现并集成
3. 在真实 GPU 上运行性能测试

评估: 实际性能 vs 仿真预测 + 实际性能 vs baseline

### 评分公式

```python
# Stage A: 仿真方案质量 (60%)
sim_score = normalized_improvement(sim_metric, baseline, oracle)

# Stage B: 实现能力 (30%)  
impl_score = normalized_improvement(real_metric, real_baseline, real_oracle)

# Sim-Real Gap (10%): 仿真预测与实际的一致性
gap_score = 1 - |sim_metric - real_metric| / real_metric

total = 0.6 * sim_score + 0.3 * impl_score + 0.1 * gap_score
```

### 独特价值

这是唯一同时评测 **设计能力**（仿真阶段）和 **实现能力**（真实阶段）的 benchmark。还额外评测了 agent 对 **仿真保真度的理解**——好的系统工程师知道仿真结果和真实结果之间的差距在哪里。

---

## 6. 方案对比与推荐

| 方案 | 创新度 | 实现复杂度 | GPU 需求 | 与现有 benchmark 区分度 | 学术贡献 |
|------|--------|-----------|---------|----------------------|---------|
| **SimAI-Gym** | ★★★★ | ★★★ | 无 | ★★★★★ | 首个推理系统优化的 Gym 环境 |
| **InfraAgent-Bench** | ★★★★★ | ★★ | 无 | ★★★★★ | 首个诊断-修复范式的系统 benchmark |
| **ScaleUp Challenge** | ★★★★★ | ★★★ | 无 | ★★★★★ | 首个评测规模外推能力 |
| **ConfigArena** | ★★★ | ★★★★ | 无 | ★★★★ | 对抗式评测 |
| **FullStack-Opt** | ★★★★ | ★★★★★ | 需要 | ★★★★ | 仿真+真实的两阶段评测 |

### 推荐组合

**最小可行产品（MVP）**: 方案一 SimAI-Gym 的 Level 1-2（纯仿真，~20 tasks）
- 实现最快，GPU 零成本，已有足够区分度
- 直接利用 vidur 的 CLI + analytical 模式

**差异化最大**: 方案二 InfraAgent-Bench 或方案三 ScaleUp Challenge
- 完全没有同类 benchmark
- 测试的能力维度（诊断推理 / 规模外推）是现有 benchmark 完全空白的

**完整版**: 方案一 + 方案二融合
- Level 1-2 用 Gym 式交互优化
- Level 3 用诊断-修复范式
- 覆盖 "从零配置" 和 "修复已有系统" 两种真实场景

---

## 7. 技术可行性分析

### SimAI 作为 Backend 的改造量

| 需要做的 | 复杂度 | 涉及的 SimAI 组件 |
|---------|--------|------------------|
| Python API wrapper | 低 | vidur CLI → Python API |
| 配置合法性校验 | 低 | MemoryPlanner OOM 检测 |
| 标准化 observation 输出 | 中 | MetricsStore + BottleneckAnalyzer |
| 批量仿真加速 | 中 | vidur 的 config_explorer |
| Oracle 配置穷举 | 中 | CapacitySearch + grid search |
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
