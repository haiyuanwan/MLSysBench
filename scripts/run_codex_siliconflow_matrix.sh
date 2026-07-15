#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK="${TASK:-$ROOT_DIR/tasks/scale_up/mock_scale_transfer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/experiments/codex_matrix_$(date +%Y%m%d_%H%M%S)}"
MAX_QUERIES="${MAX_QUERIES:-5}"
WALL_TIME_SECONDS="${WALL_TIME_SECONDS:-900}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-1048576}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-131072}"
THINKING_BUDGET="${THINKING_BUDGET:-32768}"

models=(
  "meituan-longcat/LongCat-2.0"
  "zai-org/GLM-5.2"
  "moonshotai/Kimi-K2.7-Code"
  "deepseek-ai/DeepSeek-V4-Pro"
)

mkdir -p "$OUTPUT_ROOT"

for model in "${models[@]}"; do
  slug="${model//\//__}"
  "$PYTHON_BIN" -m mlsysbench.simai_bench run-cli-agent \
    --task "$TASK" \
    --agent-profile codex \
    --model "$model" \
    --context-window "$CONTEXT_WINDOW" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --thinking-budget "$THINKING_BUDGET" \
    --max-queries "$MAX_QUERIES" \
    --wall-time-seconds "$WALL_TIME_SECONDS" \
    --isolation landlock \
    --output-dir "$OUTPUT_ROOT/$slug"
done

printf 'Codex SiliconFlow matrix artifacts: %s\n' "$OUTPUT_ROOT"
