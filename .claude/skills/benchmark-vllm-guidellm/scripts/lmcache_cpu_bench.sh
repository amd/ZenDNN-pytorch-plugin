#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Small single-instance CPU benchmark for the LMCache CPU connector
# (lmcache.v1.gpu_connector.cpu_connector.VLLMCPUConnector).
#
# Runs the same `lmcache bench engine` long-doc-qa workload against two setups
# and prints both summaries so the LMCache offload win is directly comparable:
#
#   A (baseline) : vLLM-CPU alone, no KV connector
#   B (lmcache)  : vLLM-CPU + standalone lmcache server via LMCacheMPConnector
#
# The shared bench config is captured once (auto-detecting tokens-per-GB from
# the lmcache server) and replayed against both, so the only variable is the
# connector.
#
# Everything runs on the host venv -- no containers. Tear-down is automatic.

set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
VENV="${VENV:-$HOME/lmcache-cpu-venv}"
MODEL="${MODEL:-facebook/opt-125m}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
CPUS="${CPUS:-0-7}"                       # OMP thread bind for vLLM
KVCACHE_SPACE_GB="${KVCACHE_SPACE_GB:-4}" # VLLM_CPU_KVCACHE_SPACE
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
L1_SIZE_GB="${L1_SIZE_GB:-4}"             # lmcache CPU pool

# bench workload (kept small for a smoke-scale run)
KV_CACHE_VOLUME="${KV_CACHE_VOLUME:-0.3}"
LDQA_DOC_LEN="${LDQA_DOC_LEN:-1024}"
LDQA_QPD="${LDQA_QPD:-1}"
LDQA_INFLIGHT="${LDQA_INFLIGHT:-4}"
LDQA_MAX_OUT="${LDQA_MAX_OUT:-32}"

WORKDIR="${WORKDIR:-/tmp/lmcache_cpu_bench_$$}"
mkdir -p "$WORKDIR"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export VLLM_DEVICE=cpu VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_KVCACHE_SPACE="$KVCACHE_SPACE_GB"
export VLLM_CPU_OMP_THREADS_BIND="$CPUS"
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo
export LMCACHE_DISABLE_BANNER=1

LMCACHE_PID="" VLLM_PID=""
cleanup() {
    set +e
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null
    [ -n "$LMCACHE_PID" ] && kill "$LMCACHE_PID" 2>/dev/null && wait "$LMCACHE_PID" 2>/dev/null
    VLLM_PID="" LMCACHE_PID=""
    set -e
}
trap 'cleanup; rm -rf "$WORKDIR"' EXIT

wait_ready() {
    local url="$1" needle="$2" timeout="${3:-180}"
    for _ in $(seq 1 "$timeout"); do
        if curl -fsS "$url" 2>/dev/null | grep -q "$needle"; then return 0; fi
        sleep 1
    done
    echo "TIMEOUT waiting for $url ($needle)" >&2
    return 1
}

start_lmcache() {
    echo "--- starting lmcache server (zmq=$LMCACHE_ZMQ_PORT http=$LMCACHE_HTTP_PORT) ---"
    lmcache server --port "$LMCACHE_ZMQ_PORT" --http-port "$LMCACHE_HTTP_PORT" \
        --l1-size-gb "$L1_SIZE_GB" --eviction-policy LRU --chunk-size 128 \
        >"$WORKDIR/lmcache.log" 2>&1 &
    LMCACHE_PID=$!
    wait_ready "http://localhost:$LMCACHE_HTTP_PORT/healthcheck" "" 30
    echo "    lmcache healthy (pid=$LMCACHE_PID)"
}

start_vllm() {
    # $1 = "lmcache" | "baseline"
    local mode="$1" kv_args=()
    if [ "$mode" = "lmcache" ]; then
        export LMCACHE_MP_TRANSFER_MODE=engine_driven
        local cfg
        cfg=$(python3 -c "import json;print(json.dumps({'kv_connector':'LMCacheMPConnector','kv_role':'kv_both','kv_connector_module_path':'lmcache.integration.vllm.lmcache_mp_connector','kv_connector_extra_config':{'lmcache.mp.host':'tcp://localhost','lmcache.mp.port':$LMCACHE_ZMQ_PORT}}))")
        kv_args=(--kv-transfer-config "$cfg")
    fi
    echo "--- starting vllm serve ($mode) ---"
    vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
        --disable-hybrid-kv-cache-manager --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
        --enforce-eager "${kv_args[@]}" \
        >"$WORKDIR/vllm_${mode}.log" 2>&1 &
    VLLM_PID=$!
    wait_ready "http://localhost:$VLLM_PORT/v1/models" "$MODEL" 240
    echo "    vllm ready ($mode, pid=$VLLM_PID)"
}

run_bench() {
    # $1 = label, $2 = extra args
    local label="$1"; shift
    echo ""
    echo "============================================================"
    echo "  BENCH: $label"
    echo "============================================================"
    lmcache bench engine \
        --engine-url "http://localhost:$VLLM_PORT" \
        --model "$MODEL" \
        --no-interactive --json --output-dir "$WORKDIR/$label" \
        "$@"
}

# tokens-per-GiB of KV cache for the model. opt-125m: 12 layers, 12 kv-heads,
# head_dim 64, bf16 => 2*12*(12*64)*2 = 36864 bytes/token => ~29127 tok/GiB.
# Override TOKENS_PER_GB for other models.
TOKENS_PER_GB="${TOKENS_PER_GB:-29127}"

BENCH_COMMON=(
    --workload long-doc-qa
    --kv-cache-volume "$KV_CACHE_VOLUME"
    --tokens-per-gb-kvcache "$TOKENS_PER_GB"
    --ldqa-document-length "$LDQA_DOC_LEN"
    --ldqa-query-per-document "$LDQA_QPD"
    --ldqa-num-inflight-requests "$LDQA_INFLIGHT"
    --ldqa-max-output-length "$LDQA_MAX_OUT"
    --ignore-eos
)

# === Setup B: vLLM + LMCache ================================================
start_lmcache
start_vllm lmcache
run_bench setup_b_lmcache "${BENCH_COMMON[@]}"
cleanup

# === Setup A: vLLM alone (identical workload) ===============================
start_vllm baseline
run_bench setup_a_baseline "${BENCH_COMMON[@]}"
cleanup

echo ""
echo "Done. Per-run JSON/CSV under $WORKDIR (removed on exit; copy if needed)."
