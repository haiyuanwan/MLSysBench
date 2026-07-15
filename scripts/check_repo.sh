#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m compileall -q mlsysbench tests
python3 -m unittest discover -s tests -v

python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scale_up/mock_scale_transfer >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scenarios/mock_prefill_heavy >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scenarios/mock_decode_heavy >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/scenarios/mock_balanced >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/code_scheduler/workload_aware_chunked_prefill >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/patch_transfer/adaptive_chunk_patch >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/policy_transfer/nonstationary_fair_scheduler >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/multifidelity/scheduler_probe_allocation >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/simai_gym/qwen3_next_aicb_smoke >/dev/null
python3 -m mlsysbench.simai_bench validate-task \
  --task tasks/simai_gym/qwen3_next_aicb_benchmark >/dev/null

echo "Known exclusion: tasks/simai_gym/l1_scheduler_choice is a legacy invalid fixture."
echo "Repository checks passed."
