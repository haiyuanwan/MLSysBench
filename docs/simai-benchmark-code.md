# SimAI Benchmark Code Scaffold

The first code scaffold implements the benchmark protocol described in
`docs/simai-benchmark-proposals.md`:

- task specification loading
- allowed action validation
- baseline config + submission diff merging
- runner abstraction (`mock` and `vidur`)
- Vidur metrics parsing
- SLO gates
- baseline-relative scoring
- CLI evaluation

Run the bundled smoke test task:

```bash
python -m mlsysbench.simai_bench evaluate \
  --task tasks/simai_gym/l1_scheduler_choice \
  --submission submissions/examples/sarathi_scheduler.json
```

Run the model-agent path without network access:

```bash
python3 -m mlsysbench.simai_bench run-agent \
  --task tasks/simai_gym/l1_scheduler_choice \
  --provider dry-run \
  --output-dir runs/dry_run_l1_scheduler
```

Run an OpenAI-compatible model endpoint:

```bash
cp .env.example .env
# Fill MODEL_API_KEY in .env.

python3 -m mlsysbench.simai_bench run-agent \
  --task tasks/simai_gym/l1_scheduler_choice \
  --provider openai-compatible \
  --api-key-env MODEL_API_KEY \
  --output-dir runs/model_l1_scheduler
```

The agent runner writes:

```text
runs/<run_name>/
├── prompt_context.json   # public task context shown to the model
├── submission.json       # model-produced changes
└── result.json           # evaluator output
```

`prompt_context.json` contains only public task data:

- task metadata
- README / symptoms / public report if present
- `baseline_config`
- `allowed_actions`
- objective and SLO

The model API never receives files under `hidden/`. Hidden workload and hidden
baseline metrics are read only by the evaluator.

Real SimAI/Vidur tasks should use `runner.type = "vidur"` in `task.json` and
provide a valid `vidur_root`, timeout, and output directory. The evaluator will
run `python -m vidur.main` with the merged flat config.

## Real SimAI/Vidur Setup

The scaffold has two execution layers:

- `mock`: dependency-free harness tests and task protocol validation.
- `vidur`: real `python -m vidur.main` execution against the SimAI/Vidur tree.

For an RTX 5880 or similar CUDA host:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r third_party/SimAI/vidur-alibabacloud/requirements.txt
python -m pip install -r third_party/SimAI/aicb/requirements.txt
```

Build the SimAI analytical backend:

```bash
cd third_party/SimAI/astra-sim-alibabacloud/build/simai_analytical
./build.sh -c
cd ../../../..
mkdir -p third_party/SimAI/bin
ln -sf ../astra-sim-alibabacloud/build/simai_analytical/build/simai_analytical/SimAI_analytical \
  third_party/SimAI/bin/SimAI_analytical
```

For `backend=aicb`, install the CUDA packages required by the AICB mocked
models. The local code imports CUDA PyTorch, DeepGEMM, FlashInfer/FlashMLA,
Triton, and vLLM utilities for DeepSeek/Qwen timing generation. Without those,
Vidur can still run the request-level event simulation but AICB layer CSV
generation will fall back or fail.

Useful direct checks on the CUDA host:

```bash
cd third_party/SimAI
./bin/SimAI_analytical -w example/workload_analytical.txt -g 2 -g_p_s 1 -r smoke_

cd aicb
python -m workload_generator.Vidur_workload_generator \
  Qwen3-Next-80B ./scripts/inference_configs/qwen3_next_default.json \
  --seq_length 16 --micro_batch 1 --world_size 1 \
  --tensor_model_parallel_size 1 --expert_model_parallel_size 1 \
  --aiob_enable --phase decode
```

The first supported action space is intentionally limited to the code-verified
surface:

```text
TP / PP / replica count / scheduler / batch cap / Sarathi chunk size /
PD on-off / P:D ratio / PD bandwidth sensitivity / QPS under SLO
```

EP search, EP/AllToAll network scoring, topology-aware PD congestion,
heterogeneous scheduling, and prefix-aware routing are future extensions.
