#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# incontainer_serve.sh -- runs INSIDE the container. Sets the perf LD_PRELOAD
# via incontainer_env.sh, prints a provenance banner, then execs
# `vllm serve "$@"`. The entrypoint for every vLLM instance in the compose
# stack, so all N get identical tuning and all N record what they ran.
#
#   entrypoint: bash /bench/incontainer_serve.sh --model ... --port 8000 ...
#
# The banner is not decoration. A results directory is worthless if you cannot
# say which zentorch/vLLM/torch build produced it and with which env, and the
# image tag alone does not pin that (images get rebuilt, and --native strips
# zentorch out of an otherwise identical image). Everything below lands in the
# container log, which `podman compose logs` captures next to the numbers.
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/incontainer_env.sh"

banner() { echo "=== [incontainer_serve] $* ==="; }

banner "instance ${HOSTNAME:-unknown}"

# --- exported env ----------------------------------------------------------
# Only the vars that change results: core binding, KV cache, the preload, and
# every ZenDNN/zentorch knob present. Printed as NAME=VALUE so a later run can
# be diffed against this one with plain `diff`.
banner "env"
for _v in VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_KVCACHE_SPACE \
          LD_PRELOAD OMP_NUM_THREADS HF_HUB_OFFLINE HF_HOME; do
  echo "  ${_v}=${!_v:-<unset>}"
done
# The zentorch/ZenDNN knob space is open-ended (new ZENDNNL_* vars appear per
# release), so match by prefix rather than an allowlist that silently goes
# stale. HF_TOKEN is deliberately never printed.
while IFS='=' read -r _name _val; do
  case "$_name" in
    HF_TOKEN|*TOKEN|*SECRET|*PASSWORD) continue ;;
  esac
  echo "  ${_name}=${_val}"
done < <(env | grep -E '^(ZENDNNL_|ZENTORCH_|USE_ZENDNN|THP_MODE|ZEN_)' | sort)

# --- installed packages ----------------------------------------------------
# Call out the three that decide the numbers first, so they are greppable
# without scrolling the full list, then dump everything for exact provenance.
banner "key packages"
if command -v pip >/dev/null 2>&1; then
  PIP=(pip)
elif python -m pip --version >/dev/null 2>&1; then
  PIP=(python -m pip)
else
  PIP=()
fi
if ((${#PIP[@]})); then
  "${PIP[@]}" list 2>/dev/null | grep -iE '^(zentorch|zentorch-weekly|vllm|torch|torchao|transformers|zendnn)[[:space:]]' \
    || echo "  (zentorch not installed -- native variant?)"
  banner "pip list"
  "${PIP[@]}" list 2>/dev/null
else
  echo "  (pip unavailable)"
fi

banner "launching: vllm serve $*"
exec vllm serve "$@"
