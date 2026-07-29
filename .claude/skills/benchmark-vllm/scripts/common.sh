#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# common.sh -- shared helpers for the benchmark-vllm skill.
#
# Source this from the host-side runner scripts:
#   source "$(dirname "$0")/common.sh"
#
# Provides:
#   - container runtime detection ($CRUNTIME / $CCOMPOSE)
#   - the default amdih/zendnn_zentorch image + pull-if-missing
#   - the perf env vars every quadrant passes into the container
#   - a shared container name prefix used by mem_poll.sh
#
# It intentionally does NOT run anything on source; it only defines vars
# and functions.

# The container name prefix. mem_poll.sh filters `podman stats` on this, so
# every container this skill starts (single, multi, nginx) must be named
# "${BENCH_PREFIX}-...".
BENCH_PREFIX="${BENCH_PREFIX:-bench-vllm}"

# Newest amdih/zendnn_zentorch tag at authoring time. The skill ALWAYS asks
# the user for the image; this is only the suggested default. Browse tags at
# https://hub.docker.com/r/amdih/zendnn_zentorch/tags
DEFAULT_IMAGE="${DEFAULT_IMAGE:-docker.io/amdih/zendnn_zentorch:vllm_v0.24.0_zentorch_v2.11.0.3_ubuntu22.04_2026_ww28}"

# ---------------------------------------------------------------------------
# Container runtime detection
# ---------------------------------------------------------------------------
# The skill standardises on podman (+ podman-compose) but falls back to docker
# so the scripts still work on a docker-only host. Sets:
#   CRUNTIME  -> "podman" | "docker"
#   CCOMPOSE  -> "podman-compose" | "podman compose" | "docker compose" | "docker-compose"
detect_runtime() {
  if [[ -n "${CRUNTIME:-}" ]]; then
    return 0
  fi
  if command -v podman >/dev/null 2>&1; then
    CRUNTIME="podman"
  elif command -v docker >/dev/null 2>&1; then
    CRUNTIME="docker"
  else
    echo "ERROR: neither podman nor docker found on PATH." >&2
    return 1
  fi

  if [[ "$CRUNTIME" == "podman" ]]; then
    if command -v podman-compose >/dev/null 2>&1; then
      CCOMPOSE="podman-compose"
    elif podman compose version >/dev/null 2>&1; then
      CCOMPOSE="podman compose"
    else
      CCOMPOSE=""
    fi
  else
    if docker compose version >/dev/null 2>&1; then
      CCOMPOSE="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
      CCOMPOSE="docker-compose"
    else
      CCOMPOSE=""
    fi
  fi
  export CRUNTIME CCOMPOSE
}

# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
# image_present <image> -> 0 if the image exists locally, 1 otherwise.
image_present() {
  local img="$1"
  "$CRUNTIME" image exists "$img" 2>/dev/null && return 0
  "$CRUNTIME" image inspect "$img" >/dev/null 2>&1
}

# pull_image_if_missing <image>
pull_image_if_missing() {
  local img="$1"
  detect_runtime || return 1
  if image_present "$img"; then
    echo "Image already present locally: $img"
    return 0
  fi
  echo "Image not found locally, pulling: $img"
  "$CRUNTIME" pull "$img"
}

# ---------------------------------------------------------------------------
# Perf env
# ---------------------------------------------------------------------------
# The tuning env vars that every quadrant forwards into the container. Values
# can be overridden by exporting them before calling the runner. LD_PRELOAD is
# NOT set here -- it is discovered inside the container by incontainer_env.sh
# (the tcmalloc / libiomp5 paths are image-specific).
#
# Prints one `-e KEY=VALUE` per var, suitable for `podman run` arg expansion.
perf_env_docker_args() {
  local kv=90 freeze=1
  kv="${VLLM_CPU_KVCACHE_SPACE:-90}"
  freeze="${TORCHINDUCTOR_FREEZING:-1}"
  printf -- '-e VLLM_CPU_KVCACHE_SPACE=%s -e TORCHINDUCTOR_FREEZING=%s' "$kv" "$freeze"
  # Optional zentorch knobs -- only forwarded when the user set them.
  [[ -n "${ZENDNNL_MATMUL_ALGO:-}" ]]     && printf -- ' -e ZENDNNL_MATMUL_ALGO=%s' "$ZENDNNL_MATMUL_ALGO"
  [[ -n "${USE_ZENDNN_MATMUL_DIRECT:-}" ]] && printf -- ' -e USE_ZENDNN_MATMUL_DIRECT=%s' "$USE_ZENDNN_MATMUL_DIRECT"
  [[ -n "${ZENTORCH_FP16_OPS:-}" ]]        && printf -- ' -e ZENTORCH_FP16_OPS=%s' "$ZENTORCH_FP16_OPS"
  [[ -n "${THP_MODE:-}" ]]                 && printf -- ' -e THP_MODE=%s' "$THP_MODE"
}

# short_model_name meta-llama/Llama-3.1-8B-Instruct -> Llama-3.1-8B-Instruct
# Also handles local paths with trailing slashes (/models/foo/ -> foo).
short_model_name() {
  local m="$1"
  while [[ "$m" == */ ]]; do m="${m%/}"; done   # strip trailing slashes
  echo "${m##*/}"
}

# timestamp for run tags / result dirs
bench_timestamp() {
  date '+%Y%m%d_%H%M%S' 2>/dev/null || echo "run"
}
