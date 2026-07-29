#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# run_offline_single.sh -- OFFLINE single-instance benchmark (host side).
#
# Thin wrapper over run_offline_multi.sh with N=1: one container, one
# `vllm bench throughput` pinned to CPI cores. All other flags pass through.
#
# Example:
#   ./run_offline_single.sh --model meta-llama/Llama-3.1-8B-Instruct --cpi 64
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SELF_DIR}/run_offline_multi.sh" --mode single --n 1 "$@"
