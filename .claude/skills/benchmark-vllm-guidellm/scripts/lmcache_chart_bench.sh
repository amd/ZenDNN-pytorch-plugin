#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Chart-style variant sweep driven by the REAL pipeline (ansible + guidellm +
# nginx + podman), adapted to this single-socket box. For one model it runs:
#
#   native          - vLLM native build (zentorch removed), no KV reuse
#   zentorch        - vLLM zentorch build, no KV reuse
#   zentorch+lmcache- vLLM zentorch + shared LMCache server (cross-instance reuse)
#
# Workload defaults to chat_shared (a shared cacheable prefix) so the LMCache
# variant can actually show a benefit under guidellm (plain chat = random
# prompts = no reuse). For the compute-only comparison (chart's DA8W8 etc.) use
# BASE_WORKLOAD=chat.
#
# Each variant goes through run_sweep.sh -> start.sh -> ansible guidellm. Scores
# are pulled from guidellm.log (server-aggregate, authoritative).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# ---- knobs ----
VLLM_IMAGE="${VLLM_IMAGE:?set VLLM_IMAGE to the base vLLM image}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="${TAG:-qwen05b}"
N="${N:-3}"
CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
VLLM_START_CORE="${VLLM_START_CORE:-32}"
BASE_WORKLOAD="${BASE_WORKLOAD:-chat_shared}"
GUIDELLM_RATES="${GUIDELLM_RATES:-[64,128]}"
GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-120}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
HF_TOKEN="${HF_TOKEN:-}"
PARSER="$SCRIPT_DIR/extract_guidellm_metrics.py"

export VLLM_IMAGE MODEL N CORES_PER_INSTANCE VLLM_START_CORE BASE_WORKLOAD \
       GUIDELLM_RATES GUIDELLM_MAX_SECONDS HEALTH_TIMEOUT HF_TOKEN \
       NUM_INSTANCES="$N" ANSIBLE_SSH_USER=root

OUT="${OUT:-$SCRIPT_DIR/chart-bench-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
RESULTS_ROOT="$SCRIPT_DIR/vllm-cpu-perf-eval/results/llm/${MODEL//\//__}"

run_variant() {
    # $1 = label, $2 = test_name_prefix, $3 = prefix_caching(true|false|""), $4.. flags
    # "" leaves the choice to start.sh: off, or on for a hybrid model under --lmcache.
    local label="$1" prefix="$2" pc="$3"; shift 3
    echo ""
    echo "############################################################"
    echo "#  VARIANT: $label   (model=$MODEL, N=$N, workload=$BASE_WORKLOAD, prefix_caching=${pc:-default})"
    echo "############################################################"
    ENABLE_PREFIX_CACHING="$pc" TEST_NAME_PREFIX="$prefix" bash "$SCRIPT_DIR/run_sweep.sh" \
        -N "$N" --no-limits "$@" -m "$MODEL | $TAG" \
        > "$OUT/run_${label}.log" 2>&1
    echo "  run_sweep rc=$? (log: $OUT/run_${label}.log)"
}

# Fair caching baseline: native & zentorch use vLLM's own (per-instance) prefix
# caching ON; LMCache uses its shared cross-instance pool (vLLM prefix caching
# OFF, except for hybrid models, which LMCache serves only with it on; start.sh
# turns it on for them). Prefixes must not be substrings of one another
# (result-dir glob).
run_variant native   NAT true  --native
run_variant zentorch ZEN true  --zentorch
run_variant lmcache  LMC ""    --zentorch --lmcache

echo ""
echo "############################################################"
echo "#  SCORES (from guidellm.log)"
echo "############################################################"
declare -A PREFIX=( [native]=NAT [zentorch]=ZEN [lmcache]=LMC )
declare -A VAR=( [native]=native [zentorch]=zentorch [lmcache]=zentorch )
for label in native zentorch lmcache; do
    tn="${PREFIX[$label]}-${VAR[$label]}-${TAG}"
    # shellcheck disable=SC2012
    rd=$(ls -1dt "$RESULTS_ROOT"/"${BASE_WORKLOAD}"-*"${tn}" 2>/dev/null | head -1)
    echo ""
    echo "=== $label  ($tn) ==="
    if [ -n "$rd" ] && [ -f "$rd/external-endpoint/guidellm.log" ]; then
        python3 "$PARSER" "$rd/external-endpoint/guidellm.log" 2>/dev/null \
            || echo "  (parse failed)"
    else
        echo "  no result dir found (run may have failed; see $OUT/run_${label}.log)"
    fi
done
echo ""
echo "Done. Logs under $OUT ; results under $RESULTS_ROOT"
