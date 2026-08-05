#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# common.sh -- shared helpers for the benchmark-vllm skill.
#
# Sourced by scripts/bench.sh:
#   source "$(dirname "$0")/common.sh"
#
# Deliberately small. Everything to do with the container runtime, image pulls
# and the native-image build lives in harness/start.sh, which owns the stack;
# duplicating it here only creates two implementations that can disagree.
#
# It does NOT run anything on source; it only defines vars and functions.

# Newest amdih/zendnn_zentorch tag at authoring time. The skill ALWAYS asks
# the user for the image; this is only the suggested default. Browse tags at
# https://hub.docker.com/r/amdih/zendnn_zentorch/tags
DEFAULT_IMAGE="${DEFAULT_IMAGE:-docker.io/amdih/zendnn_zentorch:vllm_v0.24.0_zentorch_v2.11.0.3_ubuntu22.04_2026_ww28}"

# short_model_name meta-llama/Llama-3.1-8B-Instruct -> Llama-3.1-8B-Instruct
# Also handles local paths with trailing slashes (/models/foo/ -> foo).
short_model_name() {
  local m="$1"
  while [[ "$m" == */ ]]; do m="${m%/}"; done   # strip trailing slashes
  echo "${m##*/}"
}
