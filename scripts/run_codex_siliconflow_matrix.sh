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
ISOLATION="${ISOLATION:-bwrap}"

mkdir -p "$OUTPUT_ROOT"

# The benchmark scaffold is pinned and verified before any model run. CC Switch
# is a protocol bridge for SiliconFlow; InferenceBench itself invokes Codex
# directly because its evaluated GPT models use native Codex authentication.
"$PYTHON_BIN" -m mlsysbench.simai_bench prepare-codex-runtime \
  > "$OUTPUT_ROOT/runtime-assets.json"

models=(
  "meituan-longcat/LongCat-2.0"
  "zai-org/GLM-5.2"
  "moonshotai/Kimi-K2.7-Code"
  "deepseek-ai/DeepSeek-V4-Pro"
)

for model in "${models[@]}"; do
  slug="${model//\//__}"
  "$PYTHON_BIN" -m mlsysbench.simai_bench run-cli-agent \
    --task "$TASK" \
    --agent-mode benchmark \
    --agent-profile codex \
    --model "$model" \
    --context-window "$CONTEXT_WINDOW" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --thinking-budget "$THINKING_BUDGET" \
    --max-queries "$MAX_QUERIES" \
    --wall-time-seconds "$WALL_TIME_SECONDS" \
    --isolation "$ISOLATION" \
    --output-dir "$OUTPUT_ROOT/$slug"
done

printf 'Codex SiliconFlow matrix artifacts: %s\n' "$OUTPUT_ROOT"
