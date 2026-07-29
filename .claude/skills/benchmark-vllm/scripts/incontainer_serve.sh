#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# incontainer_serve.sh -- runs INSIDE the container. Sets the perf LD_PRELOAD
# via incontainer_env.sh, then execs `vllm serve "$@"`. Used as the entrypoint
# for the online single-instance quadrant so serving gets the same tuned
# runtime as the offline launcher.
#
#   entrypoint: bash /bench/incontainer_serve.sh --model ... --port 8000 ...
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/incontainer_env.sh"

echo "[incontainer_serve] VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-<unset>}"
echo "[incontainer_serve] launching: vllm serve $*"
exec vllm serve "$@"
