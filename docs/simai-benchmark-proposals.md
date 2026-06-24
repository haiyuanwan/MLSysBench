# 基于 SimAI 的推理系统优化 Benchmark 设计方案

> 目标：利用 SimAI/vidur-alibabacloud 构建一个可复现、可自动评分、能评测 agent 推理系统优化能力的 benchmark。

本文已按当前 `third_party/SimAI` 代码和已有 benchmark 调研重新校准。核心方法是：

> **不需要 hidden oracle config，也不需要 root-cause label 作为评分依据。第一版采用 baseline-relative measured-outcome evaluation：给定 baseline，agent 提交配置 diff，evaluator 在 hidden workload 上实测指标，并按相对 baseline 的改进打分。**

这与 KernelBench、MLPerf Inference、CompilerGym、PIE/ECCO 等性能 benchmark 的共同思路一致：需要 correctness/validity oracle，但不需要最优解标签或诊断标签。

---

## 0. 代码核验结论

### 0.1 当前可直接依赖的能力

| 能力 | 代码依据 | Benchmark 用法 |
|------|----------|----------------|
| 多请求离散事件推理仿真 | `vidur-alibabacloud/vidur/simulator.py`、事件系统 | 运行固定 workload，输出 TTFT/TBT/E2E/throughput |
| 调度器选择 | `ReplicaSchedulerRegistry`: vLLM, Sarathi, ORCA, LightLLM, FasterTransformer, Splitwise | 构造 scheduler/config 优化任务 |
| PD 分离 | `SplitwiseGlobalScheduler` + `BatchEndEvent` | 搜索 P/D replica 划分、PD P2P bandwidth、scheduler 组合 |
| 内存规划 | `MemoryPlanner`, `ParamCounter`, `Replica` KV cache tracking | 构造 OOM、batch size、KV cache 容量任务 |
| TP/PP/batch 搜索 | `config_optimizer/config_explorer` | 可用作 baseline 生成和离线 sanity check，不作为评分 oracle |
| AICB/SimAI 后端 | `BaseExecutionTimePredictor`, `TPTimePredictor` | 可选地让 TP 通信通过 SimAI analytical/simulation 估算 |

### 0.2 当前不能直接作为第一版核心 action 的能力

| 设想能力 | 当前代码状态 | 处理方式 |
|----------|--------------|----------|
| 手动搜索 EP | `expert_model_parallel_size` 自动设为 `world_size`，手动不一致会报错 | 第一版不设计 EP action |
| EP AllToAll 网络瓶颈 | vidur 的 SimAI backend 当前主要建模 TP allreduce | 第一版只把 MoE 作为模型类型，不评分 EP 通信优化 |
| 拓扑级 PD/网络竞争 | PD P2P 时间是 `kv_size / bandwidth` 参数化估算 | 第一版可调 bandwidth，但不声称拓扑拥塞保真 |
| 异构 GPU 非均匀并行 | config 和 cluster 假设较强，缺少异构调度模型 | 后续扩展 |
| Prefix overlap 路由 | 请求对象没有语义级 prefix overlap 信息 | 后续扩展，需要新增 workload 字段 |
| 全栈真实推理实现 | SimAI 只模拟性能，不验证数值/真实 serving 行为 | 可作为第二阶段，不放入 MVP |

### 0.3 方法学定位

第一版 benchmark 评测的是：

> Agent 在 SimAI/vidur 抽象下，能否通过受控配置修改，让推理服务在 hidden workload 上取得比 baseline 更好的实测性能，同时保持配置合法并满足 SLO。

它不是：
- GPU kernel 编写 benchmark
- 真实 vLLM/SGLang 部署 benchmark
- 网络协议/拓扑仿真 benchmark
- EP/DeepEP/AllToAll 算法 benchmark
- root-cause classification benchmark

---

## 1. 为什么不需要 hidden oracle label

### 1.1 现有 benchmark 的共同做法

已有性能 benchmark 大多采用“结果驱动”而不是“标签匹配”：

| Benchmark | 核心验证 | 是否需要 hidden root-cause label |
|-----------|----------|----------------------------------|
| KernelBench | 生成 kernel 的数值正确性 + 相对 PyTorch baseline 的 speedup | 不需要 |
| KernelBenchX / AgentKernelArena | build/correctness/performance/generalization | 不需要 |
| MLPerf Inference | accuracy/quality gate + latency/throughput/goodput | 不需要 |
| CompilerGym | 环境 action 的 reward/metric，如 code size/runtime | 不需要 |
| PIE / ECCO | correctness/pass + runtime/memory improvement | 不需要 |
| Inference serving tuning 类工作 | SLO 约束下的 latency/throughput/goodput | 不需要 |

参考链接：
- KernelBench: https://github.com/ScalingIntelligence/KernelBench
- KernelBenchX: https://github.com/BonnieW05/KernelBenchX
- AgentKernelArena: https://arxiv.org/abs/2605.16819
- MLPerf Inference: https://mlcommons.org/benchmarks/inference/
- CompilerGym: https://github.com/facebookresearch/CompilerGym
- PIE: https://pie4perf.com/
- ECCO: https://ecco-code-eff.github.io/
- SCOOT: https://arxiv.org/abs/2408.04323
- SLO-Guard: https://arxiv.org/abs/2604.17627

这说明我们需要的是：
- **validity/correctness oracle**：配置是否合法、仿真是否成功、是否 OOM、是否满足 SLO。
- **baseline metric**：用于计算相对提升。
- **hidden eval workload**：避免 agent 只针对 public workload 过拟合。

不需要的是：
- hidden oracle config
- hidden diagnosis/root-cause label
- 理论最优 performance label

### 1.2 为什么 oracle config 反而有问题

1. **最优解不唯一**：TP/batch/scheduler/PD ratio 可能有多个近似等价解。
2. **搜索空间会持续变化**：一旦后续开放新 scheduler 或新 backend，旧 oracle config 会过期。
3. **会弱化 benchmark 哲学**：MLSysBench 原本强调 measured performance，不比较 reference solution。
4. **维护成本高**：每次 SimAI 或 vidur 更新，都要重跑 oracle 搜索。
5. **容易误导评分**：agent 找到比 oracle 更好的配置时，评分公式反而需要特殊处理。

因此第一版应把 oracle config 降级为内部调试工具，而不是正式评分依赖。

### 1.3 diagnosis label 的位置

诊断文本可以保留，但只用于 qualitative analysis：

```yaml
diagnosis:
  - "baseline over-batches long-prefill requests, increasing TTFT tail latency"
```

Evaluator 不对诊断文本打分。正式分数只看：
- 是否提交合法配置 diff
- 仿真是否成功
- SLO 是否满足
- hidden workload 上的 measured improvement
- 运行预算是否遵守

---

## 2. 推荐方案：SimAI-ConfigBench

### 2.1 核心范式

每个 task 给 agent 一个 baseline 推理服务配置，agent 需要在限定 action space 内提交一个配置 diff，使 hidden eval workload 上的性能优于 baseline。

这个范式更准确地说是：

> **configuration optimization under measured simulation feedback**

而不是“诊断标签分类”。agent 可以在过程里做诊断，但 benchmark 不需要知道它诊断得对不对；如果诊断真的有用，它会体现在最终性能上。

### 2.2 Task 目录结构

```
tasks/simai_config/<task_id>/
├── scenario.yaml              # 模型、硬件、SLO、公开 workload 描述
├── baseline_config.yaml       # 初始配置
├── allowed_actions.yaml       # agent 可改字段白名单和取值范围
├── objective.yaml             # 主指标、方向、SLO、run budget
├── public_workload.yaml       # agent 可见/可试跑 workload
├── README.md                  # 任务说明
├── evaluator.py               # 调 vidur.main 并解析指标
└── hidden/
    ├── eval_workload.yaml     # hidden trace slice / seed / request mix
    ├── baseline_metrics.json  # baseline 在 hidden workload 上的指标
    └── audit.yaml             # 运行预算、超时、反作弊规则
```

不再包含：

```
hidden/oracle_config.yaml
hidden/diagnosis.json
hidden/score_bounds.json
```

### 2.3 Agent 输出格式

Agent 只提交配置 diff，不提交完整配置，不提交代码：

```yaml
changes:
  replica_config_tensor_parallel_size: 4
  replica_scheduler_config_type: sarathi
  sarathi_scheduler_config_chunk_size: 512
  vllm_scheduler_config_max_tokens_in_batch: 2048
notes: |
  Optional explanation for human analysis. Not used for scoring.
```

Evaluator 只读取 `changes`。`notes` 可记录到日志，但不参与分数。

### 2.4 可调 action space

第一版只开放当前代码可验证的配置：

| Action | 字段 | 说明 |
|--------|------|------|
| TP | `replica_config_tensor_parallel_size` | 影响单 replica 并行、通信和内存 |
| PP | `replica_config_num_pipeline_stages` | 影响 pipeline stage 和 bubble |
| Replica 数 | `cluster_config_num_replicas` | 影响总服务副本和 per-replica world size |
| Scheduler | `replica_scheduler_config_type` | `vllm`, `sarathi`, `orca`, `split_wise` 等 |
| Chunk size | `sarathi_scheduler_config_chunk_size` | 只在 Sarathi 下有效 |
| Batch cap | `*_scheduler_config_batch_size_cap`, `vllm_scheduler_config_max_tokens_in_batch` | 控制吞吐/延迟/内存 |
| PD ratio | `replica_config_pd_node_ratio` 或 `replica_config_num_prefill_replicas` | 只在 splitwise/PD 任务开放 |
| PD bandwidth/dtype | `replica_config_pd_p2p_comm_bandwidth`, `replica_config_pd_p2p_comm_dtype` | 用于 bandwidth sensitivity，不等同拓扑仿真 |
| Workload QPS | `poisson_request_interval_generator_config_qps` | 用于容量搜索或 stress test |

第一版明确禁止：
- `replica_config_expert_model_parallel_size`
- 网络拓扑文件修改
- SimAI/vidur 源码修改
- 模型结构 JSON 任意修改
- evaluator 或 hidden workload 读取

---

## 3. Task 来源与 Baseline 构造

### 3.1 总体原则

SimAI-ConfigBench 的 task 来源不是“最优答案”，而是“有真实优化压力的推理服务场景”。baseline 也不是 oracle，而是一个可信、可复现、可被改进的起点。

这与已有性能 benchmark 的方法一致：

| Benchmark 类型 | Task 来源 | Baseline 来源 | 对 SimAI-ConfigBench 的启发 |
|----------------|-----------|---------------|-----------------------------|
| KernelBench / KernelBenchX | PyTorch workload、常见算子、模型子图 | PyTorch eager 或原始实现 | baseline 可以很朴素，只要正确且可测 |
| InferenceBench | 真实 serving 场景，如 TTFT、TPOT、throughput、composite | 默认 vLLM / agent 自己部署出的默认服务 | 系统级任务适合使用 framework default 或默认配置 |
| PIE / ECCO | 慢代码到快代码的历史编辑对 | 慢版本代码 | 可以从“低性能但合理”的配置或实现出发 |
| CompilerGym | 编译优化环境中的程序和 action space | 默认编译选项 | 不需要最优标签，只需要环境 reward |
| MLPerf Inference | 标准模型、标准场景、标准质量门槛 | 统一度量协议和质量约束 | measurement protocol 和 validity gate 比 oracle solution 更重要 |

因此第一版 benchmark 应遵循：

1. task 必须来自真实或可解释的 serving 优化压力；
2. baseline 必须合法、可运行、可复现；
3. baseline 不能是明显无意义的坏配置；
4. 每个 task 必须离线验证存在优化空间；
5. best-known config 只用于内部 sanity check 和难度校准，不进入正式评分。

### 3.2 Task 来源一：SimAI / vidur 已支持能力

第一版最稳的来源是当前代码已经能直接模拟和验证的配置空间。具体包括：

| 来源 | 代码/机制 | 可形成的 task |
|------|-----------|---------------|
| Scheduler registry | `vllm`, `sarathi`, `orca`, `split_wise`, `lightllm`, `faster_transformer` | scheduler 选择、scheduler-specific 参数调优 |
| Replica / parallel config | TP、PP、replica 数 | TP/PP/replica tradeoff、cost-normalized goodput |
| Batch 控制 | batch size cap、max tokens in batch、chunk size | latency-throughput tradeoff、SLO-constrained goodput |
| Splitwise / PD | prefill replica、decode replica、PD transfer | P/D ratio、PD on/off、prefill-heavy workload |
| Memory planner | KV cache capacity、model memory、batch memory | OOM 避免、memory pressure、max batch/tokens tuning |
| Workload generator | QPS、request length、arrival process | capacity search、burst/mixed workload stress |

这些 task 的优点是边界清楚：agent 只改配置，evaluator 可以自动检查字段白名单、运行仿真、解析指标。

### 3.3 Task 来源二：Serving 论文中的优化压力

Serving 论文不应被当作“参考答案库”，而应被当作 task 设计的来源。我们可以把论文中的系统 tradeoff 转成 SimAI 场景：

| 论文/系统方向 | 可抽象出的优化压力 | SimAI task 设计 |
|---------------|-------------------|-----------------|
| Orca / continuous batching | 动态请求到达下提升吞吐，同时控制 decode latency | mixed workload 下选择 scheduler 和 batch cap |
| Sarathi / chunked prefill | 长 prompt prefill 会阻塞 decode，chunk size 影响 TTFT/TBT | long-prefill workload 下调 chunk size 和 batch cap |
| Splitwise / disaggregated serving | prefill 和 decode 资源需求不同，P/D ratio 影响尾延迟 | prefill-heavy 或 mixed workload 下搜索 P/D split |
| vLLM / PagedAttention | KV cache memory pressure 影响并发和 OOM | max tokens/batch、replica、TP 的 memory-aware tuning |
| Llumnix / SLO serving | SLO goodput 比裸 throughput 更符合服务质量 | TTFT/TBT SLO 下最大化 goodput |
| MLPerf-style serving | 标准模型、标准场景、标准指标 | 固定模型/硬件/workload，统一 measurement protocol |

这种来源适合写在每个 task 的 `scenario.yaml` 里，作为设计依据，而不是作为 hidden label：

```yaml
source:
  type: literature_tradeoff
  references:
    - "Sarathi-Serve: chunked prefill improves serving latency-throughput tradeoff"
  benchmark_role: "motivates a long-prefill chunk-size tuning task"
```

### 3.4 Task 来源三：Workload Stressor Matrix

第一版不必依赖真实 production trace。更可控的方法是定义 workload stressor matrix，并用不同组合系统化生成任务：

| 维度 | 候选取值 | 主要暴露的问题 |
|------|----------|----------------|
| Arrival pattern | low QPS, near saturation, bursty | capacity、排队、tail latency |
| Prompt length | short, medium, long, heavy-tail | prefill cost、chunked prefill、TTFT |
| Decode length | short answer, long generation, mixed | TPOT、decode batch、scheduler |
| SLO | TTFT-sensitive, TPOT-sensitive, E2E-sensitive, goodput-first | objective selection |
| Model | dense 7B, large dense, MoE-as-memory-stress | memory、parallelism、scheduler sensitivity |
| Hardware budget | 1 node 4 GPU, 1 node 8 GPU, multi-replica | TP/PP/replica/cost tradeoff |

每个 task 可以由一个 scenario template 生成：

```yaml
scenario:
  model: llama-3-8b
  hardware:
    num_gpus: 8
  workload:
    arrival: near_saturation
    prompt_length: heavy_tail_long
    decode_length: mixed
  slo:
    ttft_p99_ms: 800
    tbt_p99_ms: 80
  objective:
    primary: goodput_under_slo
```

Public workload 和 hidden workload 应该同分布但不同 seed 或不同 trace slice。这样 agent 可以试跑 public workload，但无法硬编码 hidden 请求序列。

### 3.5 Task 来源四：Config Search Mining

Config search mining 是最适合 SimAI-ConfigBench 的自动化 task 生成方法。它不需要 oracle config，也不需要 root-cause label。

离线流程：

```text
define scenario template
  -> define legal action space
  -> generate candidate configs
  -> run SimAI/vidur on public and hidden workloads
  -> filter invalid / OOM / timeout configs
  -> rank valid configs by target metric
  -> select a credible low-percentile config as baseline
  -> keep best-known configs only for sanity check
```

Candidate configs 可以来自：

1. framework default / vidur example config；
2. default 周围的小扰动；
3. grid search；
4. random search；
5. config optimizer 的输出；
6. 手工注入的常见误配置。

Baseline 选择标准：

```text
baseline must:
- pass schema validation
- pass simulator execution
- not OOM
- not timeout
- satisfy minimum service validity
- be worse than sampled best by a meaningful margin
- be near a default or plausible human configuration
- be stable across seeds
```

一个实用的选择规则是：

```text
valid_configs = filter(all_candidates)
ranked = sort_by_metric(valid_configs)
baseline_pool = percentile_range(ranked, 25%, 40%)
baseline = choose_most_default_like(baseline_pool)
```

这样得到的 baseline 不是最差配置，而是“低性能但合理”的起点。agent 需要真正理解 workload 和 action space，才能稳定改进。

### 3.6 Task 来源五：Failure / Regression Pattern

为了测试 agent 的诊断能力，可以从一个合理配置出发，注入常见系统误配置：

| Failure pattern | Baseline 表现 | 期望 agent 行为 |
|-----------------|---------------|-----------------|
| TP 过大 | 通信开销过高，单请求 latency 变差 | 降低 TP 或增加 replica |
| TP 过小 | 单卡内存压力过高，batch 容量不足 | 增大 TP 或降低 batch cap |
| batch cap 过小 | GPU 利用率低，throughput 低 | 增大 batch cap/max tokens |
| batch cap 过大 | queueing 或 TTFT/P99 爆炸 | 降低 batch cap 或换 scheduler |
| long prompt 不 chunk | prefill 阻塞 decode，TTFT/TBT tail 变差 | 使用 Sarathi/chunked prefill |
| decode-heavy 误用 prefill-friendly 配置 | TPOT 或 goodput 下降 | 改 scheduler/batch policy |
| P/D ratio 错误 | prefill 或 decode 一侧成为瓶颈 | 调整 prefill/decode replica |
| QPS 接近过载 | 大量请求违反 SLO | 调整容量、batch、replica 或 QPS |

这些 failure pattern 可以写入 task 的 human-readable README，但不要作为评分 label。正式评分仍然只看 hidden workload 上的 measured improvement。

### 3.7 Baseline 类型

第一版建议同时保留三类 baseline，覆盖 realism、controllability 和 scalability：

| Baseline 类型 | 构造方式 | 适用场景 | 风险 |
|---------------|----------|----------|------|
| Natural baseline | SimAI/vidur example config、framework default、保守默认值 | 最真实，适合论文叙述 | 可能优化空间不足 |
| Perturbed baseline | 从合理配置出发注入一个或两个常见误配置 | 适合诊断能力、单因素/多因素任务 | 需要避免过于刻意 |
| Sampled-percentile baseline | 在合法 config pool 中选择低分位但稳定的配置 | 适合大规模自动生成 | 需要额外离线仿真成本 |

不推荐的 baseline：

- 明显非法配置；
- 必然 OOM 的配置；
- 明显不可能由人写出的随机配置；
- 极端差到任何改动都会提升的配置；
- 需要知道 hidden workload 才能解释的配置。

好的 baseline 应该满足：

> 能跑、合理、现实中可能出现、但存在明确优化空间。

### 3.8 Hidden Workload 与 Baseline Metrics

每个 task 至少包含两套 workload：

```text
public workload:
  agent 可见，可用于调试和有限次数试跑

hidden workload:
  agent 不可见，只用于最终评分
```

Hidden workload 不是 oracle，它只是防止过拟合 public trace。baseline hidden metrics 是评分所需的参照值：

```json
{
  "baseline_config_id": "default_vllm_batchcap_2048",
  "hidden_workload_id": "seed_9_rows_512_1023",
  "metrics": {
    "goodput_under_slo": 42.0,
    "p99_ttft_ms": 910.5,
    "p99_tbt_ms": 76.2,
    "throughput_req_per_s": 51.3
  }
}
```

Agent 不需要知道 hidden metric 的详细分布；evaluator 只用它计算相对提升。

### 3.9 难度校准

难度不应由是否存在 oracle config 决定，而应由 action space、headroom、workload shift 和 objective 复杂度决定。

| 难度 | Action space | Baseline headroom | Workload | Objective |
|------|--------------|-------------------|----------|-----------|
| Easy | 单个 knob | baseline 比 sampled best 差 20%-50% | public/hidden 基本同分布 | 单一 latency 或 throughput |
| Medium | 2-3 个 knob | baseline 有 1.5x-2x 改进空间 | hidden 有不同 seed / trace slice | SLO goodput |
| Hard | 多个互相耦合 knob | headroom 约 1.1x-1.5x | hidden 有轻微 distribution shift | cost-normalized goodput 或多 SLO |

离线校准时可以保留 `sanity_configs.json`，用于确认任务不是无解或过于简单：

```json
{
  "baseline_ratio_to_best_known": 0.67,
  "num_valid_candidates": 128,
  "best_known_metric": 62.8,
  "median_metric": 45.1,
  "baseline_metric": 42.0
}
```

这些数据不进入正式任务包的 public 部分。

### 3.10 推荐首批 Task Families

第一版推荐优先落地 6 类任务：

| Family | Task 来源 | Baseline | 主要 action | 主指标 |
|--------|-----------|----------|-------------|--------|
| Scheduler Selection | Orca/Sarathi/vLLM/Splitwise tradeoff | 默认 vLLM 或低分位 scheduler config | scheduler, batch cap | goodput under SLO |
| Chunked Prefill Tuning | Sarathi / long-prefill pressure | 不开 chunk 或 chunk size 不合适 | scheduler, chunk size, batch cap | P99 TTFT / SLO goodput |
| Batch Cap Tuning | InferenceBench-style serving tuning | 默认 batch cap 或保守 batch cap | max tokens, batch size cap | goodput under SLO |
| TP vs Replica Tradeoff | parallelism config | TP 过大/过小或默认 TP | TP, replicas | cost-normalized goodput |
| PD Ratio Tuning | Splitwise / disaggregated serving | 不开 PD 或错误 P/D ratio | PD ratio, prefill replicas, decode replicas | P99 E2E / goodput |
| Memory Pressure Tuning | KV cache capacity / memory planner | max tokens 或 batch 配置不合理 | TP, batch cap, max tokens | valid goodput |

这些 family 都能在当前 SimAI/vidur 代码能力范围内实现，并且不依赖 EP/AllToAll/topology fidelity。

### 3.11 与通用 MLSysBench Task 来源的关系

`docs/data-sources.md` 中的通用方法仍然适用于 MLSysBench 的 L2/L3 代码优化任务：

- L2 kernel/operator tasks 使用 naive PyTorch baseline；
- L3 real serving tasks 使用 framework default baseline；
- task 来源来自 profiling、literature、community 三类证据。

SimAI-ConfigBench 是这个体系下的配置优化特例：

- baseline 不是一段 naive code，而是一份 baseline config；
- correctness oracle 不是数值等价，而是配置合法性、仿真成功、SLO/validity gate；
- performance measurement 来自 SimAI/vidur 仿真指标；
- hidden oracle solution 不需要存在。

---

## 4. 评分：Baseline-Relative Measured Outcome

### 4.1 Validity gate

先做硬门槛：

```python
validity_gate = (
    schema_valid
    and whitelist_valid
    and simulator_success
    and not oom
    and not timeout
    and hard_slo_satisfied
)
```

如果 `validity_gate == False`，该 task 得分为 0。

### 4.2 Latency objective

lower-is-better 指标，例如 P99 E2E、P99 TTFT、P99 TBT：

```python
ratio = baseline_latency / agent_latency
score = log(ratio)
score = clip(score, 0.0, cap)
```

也可以报告原始 speedup：

```python
latency_speedup = baseline_latency / agent_latency
```

### 4.3 Throughput / goodput objective

higher-is-better 指标，例如 goodput under SLO：

```python
ratio = agent_goodput / baseline_goodput
score = log(ratio)
score = clip(score, 0.0, cap)
```

推荐主指标：

```python
goodput = number_of_requests_meeting_ttft_and_tbt_slo / simulation_time
score = log(agent_goodput / baseline_goodput)
```

这比单纯 throughput 更稳，因为它同时惩罚 SLO 违反。

### 4.4 Cost-normalized objective

如果 task 允许改变 GPU 数或 replica 数，可以使用：

```python
cost_normalized_goodput = goodput / num_gpus
score = log(agent_cost_normalized_goodput / baseline_cost_normalized_goodput)
```

这样避免 agent 仅靠增加资源取胜。

### 4.5 多 task 聚合

每个 task 输出：

```json
{
  "valid": true,
  "primary_metric": "goodput_under_slo",
  "baseline_metric": 42.0,
  "agent_metric": 68.0,
  "ratio": 1.619,
  "score": 0.482,
  "num_runs": 7
}
```

总体分数：

```python
final_score = mean(task_score_i)
```

排行榜同时展示：
- mean score
- median ratio
- valid task count
- average runs used
- per-category score

---

## 5. 验证流程

### 5.1 Evaluator 执行流程

```
load scenario.yaml
load baseline_config.yaml
load allowed_actions.yaml
load hidden/eval_workload.yaml
load hidden/baseline_metrics.json
load submission.yaml
validate schema
validate whitelist and value ranges
merge baseline_config + allowed changes
render vidur CLI args
run python -m vidur.main on hidden workload
parse metrics csv
check validity gate and SLO
compute baseline-relative score
write result.json
```

### 5.2 Schema validation

示例：

```yaml
replica_config_tensor_parallel_size:
  type: int
  choices: [1, 2, 4, 8]
replica_scheduler_config_type:
  type: str
  choices: [vllm, sarathi, orca, split_wise]
replica_config_pd_node_ratio:
  type: float
  min: 0.0
  max: 1.0
  exclusive_min: true
```

### 5.3 Whitelist validation

只允许修改 `allowed_actions.yaml` 里出现的字段。任何额外字段直接 invalid：

```yaml
forbidden:
  - replica_config_expert_model_parallel_size
  - random_forrest_execution_time_predictor_config_simai_simulation_topo
  - model_config_path
```

### 5.4 Hidden workload validation

Public workload 用于 agent 调试，hidden workload 用于最终评分。二者应同分布但不同 seed/trace slice：

```text
public:  128 requests, seed=1, trace rows 0-127
hidden:  512 requests, seed=9, trace rows 512-1023
```

这样可以减少硬编码某个请求序列的风险。

### 5.5 Budget validation

每个 task 限制最大仿真次数：

```yaml
budget:
  max_public_runs: 20
  max_final_submissions: 1
  timeout_seconds_per_run: 300
```

`num_runs` 不需要进入核心评分，但应该记录。后续如果要评估 agent efficiency，可以单独报告，不混入主性能分数。

---

## 6. MVP 任务集

建议第一版做 12 个任务，按 workload 和 objective 覆盖不同系统 tradeoff。

### Level 1: 单因素配置优化

| ID | 场景 | 主要考点 | 允许 action | 主指标 |
|----|------|----------|-------------|--------|
| L1-01 | Llama-3-8B, short prompts, high QPS | batch cap | batch cap | goodput under SLO |
| L1-02 | Llama-3-8B, long prompts | chunk/batch tradeoff | batch cap, chunk | P99 TTFT |
| L1-03 | Llama-3-70B, 8 GPU | TP degree | TP | P99 E2E |
| L1-04 | Llama-3-8B, 8 GPU | TP vs replicas | TP, replicas | cost-normalized goodput |
| L1-05 | mixed workload | scheduler choice | scheduler | goodput under SLO |
| L1-06 | prefill-heavy workload | PD on/off | PD ratio, scheduler | P99 TTFT |

### Level 2: 多因素配置优化

| ID | 场景 | 主要考点 | 允许 action | 主指标 |
|----|------|----------|-------------|--------|
| L2-01 | Llama-3-70B, latency SLO | TP + batch + scheduler | TP, batch, scheduler | goodput under SLO |
| L2-02 | Qwen3-MoE-235B, H20 | memory pressure | TP, batch, max tokens | valid goodput |
| L2-03 | DeepSeek-671B, PD | P/D split and transfer cost | num_prefill_replicas, bandwidth | P99 E2E |
| L2-04 | Qwen3-Next-80B | scheduler + chunk | scheduler, chunk, batch | P99 TBT |
| L2-05 | capacity search | max QPS under SLO | QPS, TP, batch | max goodput |
| L2-06 | PP stress case | PP bubble vs batch | PP, batch | cost-normalized goodput |

### Task 生成原则

每个 task 离线生成：
1. public workload 和 hidden workload
2. baseline hidden metrics
3. baseline public metrics
4. 若干 sanity-check configs 的指标，用于确认任务确实有优化空间
5. runtime profile，用于设置 timeout 和 run budget

这些 sanity-check configs 可以来自 grid/random search，但它们不是 oracle，不进入正式评分。

---

## 7. 与原五个方案的关系

### 7.1 SimAI-Gym

保留为第二阶段。第一版先做 batch-style evaluator，因为它更容易复现和排行。后续可以把同一 evaluator 包装成：

```python
obs = env.reset(task_id)
obs, reward, done, info = env.step(action)
```

### 7.2 InfraAgent-Bench

诊断-修复范式仍然有价值，但诊断不进入分数。更准确的名称是 `SimAI-ConfigBench` 或 `SimAI-ServingBench`。

### 7.3 ScaleUp Challenge

放到第二阶段。可做 baseline-relative scale-up score，但第一版先限制为 TP/PP/replica/batch，不包含 EP/拓扑外推。

### 7.4 ConfigArena

暂不做。对抗式评分需要大量对战和稳定场景池，不如 baseline-relative score 清晰。

### 7.5 FullStack-Opt

作为长期目标。它需要真实 GPU、vLLM/SGLang 配置映射、accuracy gate 和 sim-real calibration。

---

## 8. 后续扩展路线

### Phase A: 当前 MVP

- 配置优化任务
- vidur/AICB 后端
- hidden workload
- baseline-relative measured score
- TP/PP/batch/scheduler/PD ratio

### Phase B: SimAI backend 加强

需要改代码：
- 将 `TPTimePredictor` 扩展成 `CollectiveTimePredictor`
- 支持 AllToAll/AllGather/ReduceScatter 等 collective
- 将 EP、PP、PD flow 显式转成 SimAI workload
- 让 PD P2P bandwidth 可从 topology 或 flow model 推导

完成后才能把 EP/AllToAll/topology tasks 放进正式评分。

### Phase C: 开放式代码修改任务

在有足够测试后开放：
- 新 replica scheduler
- 新 global scheduler
- PD routing policy
- prefix-aware routing，需要新增 workload 字段

### Phase D: 仿真到真实系统

将最佳仿真配置映射到 vLLM/SGLang，在少量真实 GPU 场景验证 sim-real gap。这个阶段才适合加入真实吞吐、准确率和成本指标。

---

## 9. 最终推荐

当前最优方法是：

> **SimAI-ConfigBench: 一个基于 SimAI/vidur 的 baseline-relative 配置优化 benchmark。**

第一版成功标准：
- 12 个任务全部可自动运行
- 每个任务有 public/hidden workload 和 baseline hidden metrics
- evaluator 能稳定验证配置合法性、运行仿真、计算 baseline-relative score
- 不依赖 hidden oracle config 或 diagnosis label
- 文档明确说明评测的是 SimAI/vidur 抽象下的配置优化能力

这条路线与 MLSysBench 原本的 measured-outcome 哲学一致：不问 agent 是否说对了根因，也不问它是否命中了参考解，只问它是否在受控约束下让系统变快、变稳、满足 SLO。
