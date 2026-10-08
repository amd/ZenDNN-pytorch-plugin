#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# usage: validate_hybrid.sh <correctness|load> [label suffix]
#
# Runs correctness_check.sh or load_check.sh for the six Qwen3.5/Qwen3.6
# hybrid checkpoints with the LMCache hybrid recipe, then prints a summary.
# Exits 0 only if every selected model passes.
#
# Env: VLLM_IMAGE (required; the harness uses ${VLLM_IMAGE}_lmcache, building
# it from LMCACHE_SRC if missing), ONLY (comma-separated keys, e.g.
# q35bf,q36w4), PREFILL_BUDGET (n | 2n-1, default n), plus the env of the two
# check scripts.
set -uo pipefail
mode=$1 sfx=${2:-}
: "${VLLM_IMAGE:?set VLLM_IMAGE to the base vLLM image}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $mode in
    correctness) check=$here/correctness_check.sh ;;
    load) check=$here/load_check.sh ;;
    *) echo "usage: $0 <correctness|load> [label suffix]" >&2; exit 2 ;;
esac

# key | N (vLLM's unified block size on CPU) | extra env | checkpoint
models=(
    "q35bf|640||Qwen/Qwen3.5-9B"
    "q35w8|640||RedHatAI/Qwen3.5-9B-quantized.w8a8"
    "q35w4|640|export VLLM_CPU_INT4_W4A8=1 &&|RedHatAI/Qwen3.5-9B-quantized.w4a16"
    "q36bf|1152||RedHatAI/Qwen3.6-35B-A3B"
    "q36w8|1152||amd/Qwen3.6-35B-A3B-w8a8-llmcompressor"
    "q36w4|1152|export VLLM_CPU_INT4_W4A8=1 &&|amd/Qwen3.6-35B-A3B-w4a16-llmcompressor"
)
# Prefill budget N makes the cold run compute the same chunks as a restore, so
# outputs match token for token; 2N-1 is ~5% faster but the cold run can drift
# after the first sentence (GDN is not bit-exact across chunk sizes).
# With the VFS storage driver every container is a full image copy, and a
# full store makes podman-compose hang or torch.compile fail mid-warmup.
if [[ $(podman info --format '{{.Store.GraphDriverName}}') == vfs ]]; then
    containers=$([[ $mode == load ]] && echo $((${N:-5} + 2)) || echo 4)
    need=$(( $(podman image inspect --format '{{.Size}}' "${VLLM_IMAGE}_lmcache") * containers ))
    free=$(( $(df --output=avail -B1 "$(podman info --format '{{.Store.GraphRoot}}')" | awk 'NR == 2') ))
    if (( free < need )); then
        echo "Need ~$((need >> 30)) GiB free in the podman store for $containers containers," \
            "have $((free >> 30)) GiB" >&2
        exit 1
    fi
fi

budget=${PREFILL_BUDGET:-n}
if [[ $budget == n ]]; then export STRICT=1; else export STRICT=0; fi

summary=()
failed=0
for m in "${models[@]}"; do
    IFS='|' read -r key n extra model <<< "$m"
    [[ -n ${ONLY:-} && ",$ONLY," != *",$key,"* ]] && continue
    tokens=$n
    [[ $budget == 2n-1 ]] && tokens=$((2 * n - 1))
    env ENABLE_PREFIX_CACHING=true LMCACHE_CHUNK_SIZE="$n" \
        EXTRA_VLLM_ARGS="--max-num-batched-tokens $tokens" \
        SET_ENV_VAR_BEFORE_VLLM_SERVE="${extra:+$extra }export VLLM_ZENTORCH_WEIGHT_PREPACK=0 &&" \
        TAG="$key" "$check" "$key$sfx" "$model"
    rc=$?
    summary+=("$key$sfx $model: $([[ $rc == 0 ]] && echo PASS || echo FAIL)")
    [[ $rc == 0 ]] || failed=1
done
echo "=== $mode summary (image $VLLM_IMAGE, prefill budget $budget) ==="
printf '  %s\n' "${summary[@]}"
exit $failed
