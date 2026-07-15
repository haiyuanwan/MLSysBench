#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/experiments/protocol_$(date +%Y%m%d_%H%M%S)}"
TASK="$ROOT_DIR/tasks/scale_up/mock_scale_transfer"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -m mlsysbench.simai_bench run-agent-loop \
  --task "$TASK" \
  --provider dry-run \
  --output-dir "$OUTPUT_ROOT/agent"

"$PYTHON_BIN" -m mlsysbench.simai_bench search \
  --task "$TASK" \
  --method grid \
  --budget 18 \
  --output-dir "$OUTPUT_ROOT/grid"

for seed in 1 2 3; do
  "$PYTHON_BIN" -m mlsysbench.simai_bench search \
    --task "$TASK" \
    --method random \
    --budget 6 \
    --seed "$seed" \
    --output-dir "$OUTPUT_ROOT/random_seed_$seed"
done

printf 'Protocol experiment artifacts: %s\n' "$OUTPUT_ROOT"
