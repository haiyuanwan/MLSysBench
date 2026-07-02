#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv311/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
CUDA_BIN="$CUDA_HOME/bin"
CUDA_LIB="$CUDA_HOME/lib64"
TORCH_LIB="$ROOT_DIR/.venv311/lib/python3.11/site-packages/torch/lib"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/real_bench/aicb_qwen3_next_smoke}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-Smoke benchmark}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-100}"
DECODE_TOKENS="${DECODE_TOKENS:-8}"
WORLD_SIZE="${WORLD_SIZE:-32}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-32}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-$ROOT_DIR/scripts/deepgemm_nvcc_sm89_wrapper.sh}"
CHECK_ENV_ONLY="${CHECK_ENV_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export DG_JIT_NVCC_COMPILER
export PATH="$ROOT_DIR/.venv311/bin:$CUDA_BIN:$PATH"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"

AICB_DIR="$ROOT_DIR/third_party/SimAI/aicb"
VIDUR_DIR="$ROOT_DIR/third_party/SimAI/vidur-alibabacloud"
MODEL_NAME="Qwen3-Next-80B"
MODEL_CONFIG="./scripts/inference_configs/qwen3_next_default.json"
AICB_CSV="$AICB_DIR/results/workload/vidur-${MODEL_NAME}-world_size${WORLD_SIZE}-tp${TP_SIZE}-pp1-ep${EP_SIZE}-bs1-seq${PREFILL_TOKENS}-decode.csv"
RUN_DIR="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$LOG_DIR"

echo "== Environment =="
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "DG_JIT_NVCC_COMPILER=$DG_JIT_NVCC_COMPILER"
echo "RUN_DIR=$RUN_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$CUDA_BIN/nvcc" ]]; then
  echo "nvcc not found: $CUDA_BIN/nvcc" >&2
  exit 1
fi

AICB_CMD=(
  "$PYTHON_BIN" -m workload_generator.Vidur_workload_generator
  "$MODEL_NAME" "$MODEL_CONFIG"
  --seq_length "$PREFILL_TOKENS"
  --micro_batch 1
  --world_size "$WORLD_SIZE"
  --tensor_model_parallel_size "$TP_SIZE"
  --expert_model_parallel_size "$EP_SIZE"
  --aiob_enable
  --phase decode
)

VIDUR_CMD=(
  "$PYTHON_BIN" -m vidur.main
  --replica_config_pd_p2p_comm_bandwidth 800
  --replica_config_nvlink_bandwidth 1600
  --replica_config_rdma_bandwidth 800
  --replica_config_pd_p2p_comm_dtype fp8
  --replica_config_network_device h20_dgx
  --replica_config_device h20
  --request_generator_config_type synthetic
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps 100
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv
  --random_forrest_execution_time_predictor_config_backend aicb
  --random_forrest_execution_time_predictor_config_aicb_force_bs1
  --cluster_config_num_replicas "$WORLD_SIZE"
  --replica_config_pd_node_ratio 1
  --global_scheduler_config_type lor
  --replica_scheduler_config_type sarathi
  --replica_config_model_name qwen3-next-80B
  --replica_config_tensor_parallel_size "$TP_SIZE"
  --replica_config_num_pipeline_stages 1
  --metrics_config_output_dir "$RUN_DIR/vidur_metrics"
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "== Dry run commands =="
  printf 'cd %q && ' "$AICB_DIR"
  printf '%q ' "${AICB_CMD[@]}"
  printf '\n'
  printf 'cd %q && ' "$VIDUR_DIR"
  printf '%q ' "${VIDUR_CMD[@]}"
  printf '\n'
  exit 0
fi

echo "== CUDA and Python import check =="
nvidia-smi -L | tee "$LOG_DIR/nvidia-smi.txt"
"$PYTHON_BIN" -c "import torch, vllm, grouped_gemm, deep_gemm, flash_mla, flashinfer; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)); print('vllm', vllm.__version__)" \
  | tee "$LOG_DIR/import_check.txt"

if [[ "$CHECK_ENV_ONLY" == "1" ]]; then
  echo "Environment check completed."
  exit 0
fi

echo "== Direct AICB workload generation =="
(
  cd "$AICB_DIR"
  "${AICB_CMD[@]}"
) 2>&1 | tee "$LOG_DIR/aicb_direct.log"

if [[ ! -s "$AICB_CSV" ]]; then
  echo "AICB CSV was not generated: $AICB_CSV" >&2
  exit 1
fi

if ! head -1 "$AICB_CSV" | grep -q $'layer_id\tlayer_name\tcomp_time\tcomm_size'; then
  echo "AICB CSV header is unexpected: $AICB_CSV" >&2
  exit 1
fi

echo "AICB CSV: $AICB_CSV"

echo "== Vidur + AICB smoke benchmark =="
(
  cd "$VIDUR_DIR"
  "${VIDUR_CMD[@]}"
) 2>&1 | tee "$LOG_DIR/vidur_aicb.log"

if grep -Eq "AICB data is empty|using default .*execution time|Expected CSV file was NOT created|AICB command failed|无法找到任何AICB CSV" "$LOG_DIR/vidur_aicb.log"; then
  echo "Vidur used an AICB failure/default fallback path; see $LOG_DIR/vidur_aicb.log" >&2
  exit 1
fi

METRICS_FILE="$(find "$RUN_DIR/vidur_metrics" -name request_metrics.csv -print -quit)"
if [[ -z "$METRICS_FILE" || ! -s "$METRICS_FILE" ]]; then
  echo "request_metrics.csv was not written under $RUN_DIR/vidur_metrics" >&2
  exit 1
fi

echo "Metrics: $METRICS_FILE"
"$PYTHON_BIN" - "$METRICS_FILE" "$RUN_DIR/summary.json" "$AICB_CSV" <<'PY'
import csv
import json
import sys

path = sys.argv[1]
summary_path = sys.argv[2]
aicb_csv = sys.argv[3]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"no rows in {path}")
e2e = [float(row["request_e2e_time"]) for row in rows]
tokens = [int(row["request_num_tokens"]) for row in rows]
prefill = [float(row["prefill_e2e_time"]) for row in rows if row.get("prefill_e2e_time")]
decode = [float(row["decode_time"]) for row in rows if row.get("decode_time")]
decode_tokens = [int(row["request_num_decode_tokens"]) for row in rows if row.get("request_num_decode_tokens")]
summary = {
    "metrics_file": path,
    "aicb_csv": aicb_csv,
    "num_requests": len(rows),
    "avg_request_e2e_time": sum(e2e) / len(e2e),
    "total_tokens": sum(tokens),
}
if prefill:
    summary["avg_prefill_e2e_time"] = sum(prefill) / len(prefill)
if decode:
    summary["avg_decode_time"] = sum(decode) / len(decode)
if decode and decode_tokens:
    summary["avg_tbt_time"] = sum(d / max(t, 1) for d, t in zip(decode, decode_tokens)) / len(decode)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(f"requests={len(rows)} avg_request_e2e_time={sum(e2e) / len(e2e):.12f} total_tokens={sum(tokens)}")
print(f"summary={summary_path}")
PY

echo "$BENCHMARK_LABEL completed: $RUN_DIR"
