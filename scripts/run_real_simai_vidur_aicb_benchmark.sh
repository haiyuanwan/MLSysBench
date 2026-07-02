#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export BENCHMARK_LABEL="${BENCHMARK_LABEL:-Qwen3-Next AICB benchmark}"
export NUM_REQUESTS="${NUM_REQUESTS:-32}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/real_bench/aicb_qwen3_next_benchmark}"

exec "$SCRIPT_DIR/run_real_simai_vidur_aicb_smoke.sh"
