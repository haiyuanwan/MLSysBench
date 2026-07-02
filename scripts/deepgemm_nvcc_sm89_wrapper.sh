#!/usr/bin/env bash
set -euo pipefail

REAL_NVCC="${REAL_NVCC:-/usr/local/cuda-12.9/bin/nvcc}"

args=()
for arg in "$@"; do
  case "$arg" in
    --gpu-architecture=sm_89a)
      args+=(--gpu-architecture=sm_89)
      ;;
    --gpu-architecture=compute_89a)
      args+=(--gpu-architecture=compute_89)
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

exec "$REAL_NVCC" "${args[@]}"
