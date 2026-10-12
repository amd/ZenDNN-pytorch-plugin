#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# A/B benchmark of the multi-instance vLLM cluster (N instances behind nginx),
# WITHOUT vs WITH the shared LMCache server, driven by `lmcache bench engine`
# through the nginx load balancer.
#
#   A (baseline) : N vLLM instances + nginx, no KV connector
#   B (lmcache)  : N vLLM instances + nginx + one shared lmcache server
#                  (LMCacheMPConnector -> cross-instance KV reuse)
#
# Same image (B uses the derived _lmcache image, identical vLLM), same model,
# same workload -> the only variable is the connector. The long-doc-qa workload
# primes a prefix then re-queries it, so with prefix caching disabled the win
# comes entirely from LMCache restoring the prefill KV.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- knobs (override via env) ----------------------------------------------
VENV="${VENV:-$HOME/lmcache-cpu-venv}"
VLLM_IMAGE="${VLLM_IMAGE:?set VLLM_IMAGE to the base vLLM image}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
N="${N:-2}"
NGINX_PORT="${NGINX_PORT:-8080}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-9090}"
TOKENS_PER_GB="${TOKENS_PER_GB:-43690}"   # Qwen2.5-0.5B; override per model
READY_TIMEOUT="${READY_TIMEOUT:-600}"

# bench workload (small smoke-scale defaults)
KV_CACHE_VOLUME="${KV_CACHE_VOLUME:-0.3}"
LDQA_DOC_LEN="${LDQA_DOC_LEN:-1024}"
LDQA_QPD="${LDQA_QPD:-2}"
LDQA_INFLIGHT="${LDQA_INFLIGHT:-4}"
LDQA_MAX_OUT="${LDQA_MAX_OUT:-32}"

export PATH="$HOME/.local/bin:$PATH"          # podman-compose
# shellcheck disable=SC1091
source "$VENV/bin/activate"                    # lmcache bench engine
export LMCACHE_DISABLE_BANNER=1

OUT="${OUT:-$SCRIPT_DIR/sweep-logs/lmcache-ab-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

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

wait_cluster_ready() {
    local timeout="$1" t=0 ready hc
    while [ "$t" -lt "$timeout" ]; do
        ready=0
        for i in $(seq 1 "$N"); do
            # vLLM logs to stderr -> must capture 2>&1, not discard it.
            if podman logs "vllm-instance-$i" 2>&1 \
                 | grep -qiE "Application startup complete|Uvicorn running on http"; then
                ready=$((ready + 1))
            fi
        done
        hc=$(curl -fsS "http://localhost:$NGINX_PORT/health" -o /dev/null -w "%{http_code}" 2>/dev/null || echo 000)
        echo "    [${t}s] instances_ready=${ready}/${N} nginx=${hc}"
        [ "$ready" -ge "$N" ] && [ "$hc" = "200" ] && return 0
        sleep 15; t=$((t + 15))
    done
    return 1
}

run_setup() {
    # $1 = label (baseline|lmcache), $2.. = extra start.sh flags
    local label="$1"; shift
    echo ""
    echo "############################################################"
    echo "#  SETUP: $label   (N=$N, model=$MODEL)"
    echo "############################################################"

    bash "$SCRIPT_DIR/stop.sh" --clean --no-logs >/dev/null 2>&1 || true
    podman ps -aq --filter name=vllm- | xargs -r podman rm -f >/dev/null 2>&1 || true

    # An empty PREFIX_CACHE leaves the choice to start.sh (see below).
    echo "--- starting cluster ($label, prefix_caching=${PREFIX_CACHE-false}) ---"
    if ! ENABLE_PREFIX_CACHING="${PREFIX_CACHE-false}" VLLM_IMAGE="$VLLM_IMAGE" \
            bash "$SCRIPT_DIR/start.sh" \
            --regenerate --no-wait --no-limits -n "$N" -m "$MODEL" "$@" \
            > "$OUT/start_${label}.log" 2>&1; then
        echo "  ERROR: start.sh failed; see $OUT/start_${label}.log"; return 1
    fi

    echo "--- waiting for $N instances + nginx ---"
    if ! wait_cluster_ready "$READY_TIMEOUT"; then
        echo "  ERROR: cluster not ready within ${READY_TIMEOUT}s"; return 1
    fi

    # Reset LMCache metrics (setup B only) so the read/write deltas are clean.
    if [ "$label" = "lmcache" ]; then
        podman exec vllm-lmcache-server curl -fsS -X POST \
            "http://localhost:$LMCACHE_HTTP_PORT/metrics/reset" >/dev/null 2>&1 || true
    fi

    echo "--- running lmcache bench engine via nginx :$NGINX_PORT ---"
    lmcache bench engine \
        --engine-url "http://localhost:$NGINX_PORT" \
        --model "$MODEL" \
        --no-interactive --json --output-dir "$OUT/$label" \
        "${BENCH_COMMON[@]}" 2>&1 | tee "$OUT/bench_${label}.log" \
        | grep -vE "LMCache INFO|LMCache WARNING|Skipping backend|torch_dev|multi_layer" || true

    if [ "$label" = "lmcache" ]; then
        echo "--- LMCache server chunk metrics (store vs cross-instance read) ---"
        podman exec vllm-lmcache-server curl -fsS \
            "http://localhost:$LMCACHE_HTTP_PORT/metrics" 2>/dev/null \
            | grep -iE "lmcache_mp_l1_(write|read)_chunks_total" | grep -v '#' \
            | awk '{s[$1]+=$2} END{for(k in s) printf "    %s %d\n", k, s[k]}'
        echo "--- request distribution across instances ---"
        for i in $(seq 1 "$N"); do
            n=$(podman logs "vllm-instance-$i" 2>&1 | grep -cE "POST /v1/(chat/)?completions" || true)
            echo "    vllm-instance-$i served: $n"
        done
    fi
}

# Three honest configurations:
#   nocache     - no reuse at all (prefix caching off, no LMCache)
#   prefixcache - vLLM native prefix caching ON (per-instance reuse only)
#   lmcache     - LMCache shared server (cross-instance reuse), prefix off
#                 (start.sh turns it on for hybrid models, which LMCache serves
#                 only with prefix caching on)
export PREFIX_CACHE=false; run_setup nocache
export PREFIX_CACHE=true;  run_setup prefixcache
export PREFIX_CACHE=;      run_setup lmcache --lmcache

echo ""
echo "############################################################"
echo "#  TEARDOWN"
echo "############################################################"
bash "$SCRIPT_DIR/stop.sh" --clean --no-logs >/dev/null 2>&1 || true
podman ps -aq --filter name=vllm- | xargs -r podman rm -f >/dev/null 2>&1 || true

echo ""
echo "Done. Per-setup JSON/CSV + logs under: $OUT"
echo "Compare the two 'Engine Benchmark Result (long-doc-qa)' blocks above"
echo "(Mean TTFT / duration / throughput): baseline vs lmcache."
