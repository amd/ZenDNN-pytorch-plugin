#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# run_online_single.sh -- ONLINE single-instance benchmark (host side).
#
# Starts one detached vLLM server container (image default entrypoint replaced
# by incontainer_serve.sh so LD_PRELOAD/tuning is applied), waits for /health,
# then drives it with GuideLLM from the host. Peak memory is sampled by
# mem_poll.sh; the GuideLLM log is parsed by the parse-guidellm skill.
#
# Example:
#   ./run_online_single.sh --model meta-llama/Llama-3.1-8B-Instruct \
#       --cpi 64 --rates 1,2,4,8 --max-seconds 120
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/common.sh"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CPI=""
IMAGE="$DEFAULT_IMAGE"
PORT=8000
INPUT_LEN=128; OUTPUT_LEN=128
RATES="1,2,4,8"
MAX_SECONDS=300
RESULTS_DIR=""
CPUSET=""
# Host GuideLLM client core pinning. By default reserve the LAST N physical
# cores (GUIDELLM_TAIL) so the client does not fight the vLLM server for cores.
# An explicit --guidellm-cores overrides the tail reservation.
GUIDELLM_CORES="${GUIDELLM_CORES:-}"
GUIDELLM_TAIL="${GUIDELLM_TAIL:-3}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_HUB_OFFLINE_OPT="${HF_HUB_OFFLINE:-}"   # forwarded into the container when set
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
# Locate the GuideLLM client. NOTE: do not name this GUIDELLM_* -- GuideLLM
# treats every GUIDELLM_* env var as config and warns about unknown ones.
BENCH_GUIDELLM_BIN="${BENCH_GUIDELLM_BIN:-guidellm}"
DRY_RUN=0

usage() {
  cat >&2 <<EOF
Usage: $0 --model M --cpi C [options]
  --model NAME        HF repo or local path (default: $MODEL)
  --cpi C             cores for the server (default: recommended)
  --image IMG         container image (default: newest amdih tag)
  --port PORT         host/container port (default: $PORT)
  --input N           GuideLLM prompt tokens  (default: $INPUT_LEN)
  --output N          GuideLLM output tokens  (default: $OUTPUT_LEN)
  --rates LIST        concurrency levels, comma list (default: $RATES)
  --max-seconds N     seconds per rate (default: $MAX_SECONDS)
  --results-dir DIR   host results dir (default: ./bench_results/online_single_<ts>)
  --cpuset SET        explicit cpuset (skip auto-detect)
  --guidellm-cores R  pin the host GuideLLM client to this cpuset (taskset);
                      also excluded from the vLLM server cores. Overrides tail.
  --guidellm-tail N   reserve the last N physical cores for the host GuideLLM
                      client when --guidellm-cores is not given (default: $GUIDELLM_TAIL)
  --hf-cache-dir DIR  host HF cache mount (default: $HF_CACHE_DIR)
  --hf-offline        run server with HF_HUB_OFFLINE=1 (pre-cached / air-gapped)
  --serve-args "..."  extra args appended to 'vllm serve'
  --dry-run           print commands, don't run

Env: BENCH_GUIDELLM_BIN (path to guidellm), HF_TOKEN (gated models).
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --cpi) CPI="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --input) INPUT_LEN="$2"; shift 2 ;;
    --output) OUTPUT_LEN="$2"; shift 2 ;;
    --rates) RATES="$2"; shift 2 ;;
    --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --cpuset) CPUSET="$2"; shift 2 ;;
    --guidellm-cores) GUIDELLM_CORES="$2"; shift 2 ;;
    --guidellm-tail) GUIDELLM_TAIL="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --hf-offline) HF_HUB_OFFLINE_OPT=1; shift ;;
    --serve-args) EXTRA_SERVE_ARGS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

detect_runtime || exit 1

# Reserve host GuideLLM cores off the vLLM pool: an explicit --guidellm-cores is
# excluded; otherwise the last --guidellm-tail physical cores are reserved and
# reported back as TAIL_CPUSET (then used to pin the host client).
HW_RESERVE_ARGS=()
if [[ -n "$GUIDELLM_CORES" ]]; then
  HW_RESERVE_ARGS+=( --exclude "$GUIDELLM_CORES" )
elif [[ "$GUIDELLM_TAIL" =~ ^[0-9]+$ ]] && (( GUIDELLM_TAIL > 0 )); then
  HW_RESERVE_ARGS+=( --reserve-tail "$GUIDELLM_TAIL" )
fi

if [[ -z "$CPUSET" ]]; then
  if ! HW_OUT="$("${SELF_DIR}/check_hardware.sh" --mode single ${CPI:+--cpi "$CPI"} "${HW_RESERVE_ARGS[@]}" --quiet)"; then
    echo "ERROR: requested cores do not fit after reserving GuideLLM cores; see recommendation:" >&2
    "${SELF_DIR}/check_hardware.sh" --mode single ${CPI:+--cpi "$CPI"} "${HW_RESERVE_ARGS[@]}" >/dev/null
    exit 3
  fi
  eval "$(echo "$HW_OUT" | grep -E '^(REQ_CPI|CPUSET|TAIL_CPUSET)=')"
  CPI="${REQ_CPI:-$CPI}"
  # If we reserved a tail (no explicit cores given), use it for the client.
  [[ -z "$GUIDELLM_CORES" && -n "${TAIL_CPUSET:-}" ]] && GUIDELLM_CORES="$TAIL_CPUSET"
fi
[[ -z "$CPUSET" ]] && { echo "ERROR: could not resolve cpuset." >&2; exit 3; }

# Pin the host GuideLLM client to its reserved cores when possible.
GUIDELLM_PREFIX=()
if [[ -n "$GUIDELLM_CORES" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    GUIDELLM_PREFIX=( taskset -c "$GUIDELLM_CORES" )
  else
    echo "WARNING: taskset not found; host GuideLLM will not be pinned to $GUIDELLM_CORES." >&2
  fi
fi

MODEL_SHORT="$(short_model_name "$MODEL")"
TS="$(bench_timestamp)"
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$(pwd)/bench_results/online_single_${TS}"
mkdir -p "$RESULTS_DIR"
mkdir -p "$HF_CACHE_DIR"

CNAME="${BENCH_PREFIX}-instance-online-${TS}"
MEM_CSV="${RESULTS_DIR}/mem_${MODEL_SHORT}.csv"

# --- assemble the server run command ---
SERVE_CMD=( "$CRUNTIME" run -d --name "$CNAME"
  --cpuset-cpus "$CPUSET"
  --shm-size "${SHM_SIZE:-16g}"
  -p "${PORT}:${PORT}"
  -v "${SELF_DIR}:/bench:ro"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "VLLM_CPU_OMP_THREADS_BIND=${CPUSET}"
)
# shellcheck disable=SC2206
SERVE_CMD+=( $(perf_env_docker_args) )
[[ -n "${HF_TOKEN:-}" ]] && SERVE_CMD+=( -e "HF_TOKEN=${HF_TOKEN}" )
[[ -n "$HF_HUB_OFFLINE_OPT" ]] && SERVE_CMD+=( -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_OPT}" )
SERVE_CMD+=( --entrypoint bash "$IMAGE"
  /bench/incontainer_serve.sh
    --model "$MODEL" --host 0.0.0.0 --port "$PORT" --trust-remote-code
)
# shellcheck disable=SC2206
[[ -n "$EXTRA_SERVE_ARGS" ]] && SERVE_CMD+=( $EXTRA_SERVE_ARGS )

TARGET="http://localhost:${PORT}"
OUT_JSON="${RESULTS_DIR}/benchmarks.json"
GUIDELLM_LOG="${RESULTS_DIR}/guidellm.log"
# Use the explicit `benchmark run` subcommand: works on GuideLLM 0.6 and 0.7+
# (0.7 dropped the implicit default subcommand).
GUIDELLM_CMD=( "${GUIDELLM_PREFIX[@]}" "$BENCH_GUIDELLM_BIN" benchmark run
  --target "$TARGET"
  --model "$MODEL"
  --rate-type concurrent
  --rate "$RATES"
  --max-seconds "$MAX_SECONDS"
  --data "prompt_tokens=${INPUT_LEN},output_tokens=${OUTPUT_LEN}"
  --output-path "$OUT_JSON"
)

echo "=== ONLINE single benchmark ==="
echo "  runtime:   $CRUNTIME"
echo "  image:     $IMAGE"
echo "  model:     $MODEL"
echo "  cores:     cpuset=$CPUSET (CPI=$CPI)"
echo "  guidellm:  client cores=${GUIDELLM_CORES:-<host default, unpinned>}"
echo "  target:    $TARGET"
echo "  guidellm:  rates=$RATES max-seconds=$MAX_SECONDS in=$INPUT_LEN out=$OUTPUT_LEN"
echo "  results:   $RESULTS_DIR"
echo
echo "Server command:"; printf '  %q ' "${SERVE_CMD[@]}"; echo
echo "Client command:"; printf '  %q ' "${GUIDELLM_CMD[@]}"; echo
if [[ "$DRY_RUN" == "1" ]]; then echo "(dry-run: not executing)"; exit 0; fi

if ! command -v "${BENCH_GUIDELLM_BIN%% *}" >/dev/null 2>&1; then
  echo "ERROR: '$BENCH_GUIDELLM_BIN' not found on host. Install it (pip install guidellm) or set BENCH_GUIDELLM_BIN." >&2
  exit 4
fi

pull_image_if_missing "$IMAGE"

# --- start server ---
echo "Starting server container $CNAME ..."
"${SERVE_CMD[@]}"

teardown() {
  echo "Tearing down $CNAME ..."
  "$CRUNTIME" logs "$CNAME" > "${RESULTS_DIR}/server.log" 2>&1 || true
  "$CRUNTIME" rm -f "$CNAME" >/dev/null 2>&1 || true
  rm -f "${MEM_CSV}.run" 2>/dev/null || true
}
trap teardown EXIT

# --- health wait ---
echo "Waiting for $TARGET/health (timeout ${HEALTH_TIMEOUT}s) ..."
deadline=$(( SECONDS + HEALTH_TIMEOUT ))
until curl -fsS "${TARGET}/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: server did not become healthy in ${HEALTH_TIMEOUT}s." >&2
    "$CRUNTIME" logs --tail 40 "$CNAME" >&2 || true
    exit 5
  fi
  sleep 5
done
echo "Server healthy."

# --- peak memory sampler ---
"${SELF_DIR}/mem_poll.sh" "online_single_${MODEL_SHORT}" "$MEM_CSV" 2 "$BENCH_PREFIX" &
MEM_PID=$!

# --- run GuideLLM ---
echo "Running GuideLLM ..."
set +e
"${GUIDELLM_CMD[@]}" 2>&1 | tee "$GUIDELLM_LOG"
RC=${PIPESTATUS[0]}
set -e 2>/dev/null || true

rm -f "${MEM_CSV}.run" 2>/dev/null || true
wait "$MEM_PID" 2>/dev/null || true

echo
echo "=== parsing results ==="
PARSER="$HOME/.claude/skills/parse-guidellm/scripts/parse_guidellm_log.py"
if [[ -f "$PARSER" ]]; then
  python3 "$PARSER" "$GUIDELLM_LOG" || true
else
  echo "parse-guidellm skill not found at $PARSER; raw log at $GUIDELLM_LOG"
fi
grep -h '^PEAK' "$MEM_CSV" 2>/dev/null || true

exit "$RC"
