#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# run_sweep.sh -- end-to-end ONLINE multi-instance benchmark:
#   derive physical-core-aware slices -> generate-config -> start (N vLLM +
#   NGINX) -> GuideLLM against the LB -> stop -> parse (parse-guidellm skill)
#   + peak memory.
#
# Per-instance cpusets are computed by check_hardware.sh from the physical-core
# list (first thread per core), AFTER excluding the nginx cores, so instances
# never land on SMT sibling threads and the nginx reservation is validated
# (fixes review H3 + M1).
#
# Example:
#   ./run_sweep.sh --model meta-llama/Llama-3.1-8B-Instruct \
#       --n 4 --cpi 32 --nginx-cores 0-7 --rates 32,64 --max-seconds 300
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/../scripts/common.sh"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
N=4; CPI=32
IMAGE="$DEFAULT_IMAGE"
# Default 3-band CPU layout (mirrors ZenDNN_tools/vllm_multiinstance):
#   nginx cores 0-15, host GuideLLM client cores 16-31, vLLM instances 32+.
# Both nginx and guidellm cores are reserved from the vLLM physical-core pool.
NGINX_CORES="0-15"
GUIDELLM_CORES="16-31"
NGINX_PORT=8080
RATES="32,64"
MAX_SECONDS=300
INPUT_LEN=128; OUTPUT_LEN=128
RESULTS_DIR=""
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# Locate the GuideLLM client. NOTE: do not name this GUIDELLM_* -- GuideLLM
# treats every GUIDELLM_* env var as config and warns about unknown ones.
BENCH_GUIDELLM_BIN="${BENCH_GUIDELLM_BIN:-guidellm}"
HF_HUB_OFFLINE_OPT=""          # set to 1 by --hf-offline; forwarded to instances
NO_LIMITS="${NO_LIMITS:-false}"
NO_MEM_LIMIT="${NO_MEM_LIMIT:-false}"
DRY_RUN=0

usage() {
  cat >&2 <<EOF
Usage: $0 --model M --n N --cpi C [options]
  --model NAME       (default $MODEL)
  --n N              instances (default $N)
  --cpi C            cores per instance (default $CPI)
  --image IMG        (default newest amdih tag)
  --nginx-cores R    nginx cpuset, reserved from the vLLM pool (default $NGINX_CORES)
  --guidellm-cores R host GuideLLM client cpuset; reserved from the vLLM pool and
                     used to taskset-pin the client (default $GUIDELLM_CORES)
  --nginx-port P     LB host port (default $NGINX_PORT)
  --rates LIST       GuideLLM concurrency levels (default $RATES)
  --max-seconds N    seconds per rate (default $MAX_SECONDS)
  --input N          prompt tokens (default $INPUT_LEN)
  --output N         output tokens (default $OUTPUT_LEN)
  --results-dir DIR  (default ./bench_results/online_multi_<ts>)
  --hf-cache-dir DIR (default $HF_CACHE_DIR)
  --hf-offline       run instances with HF_HUB_OFFLINE=1 (pre-cached / air-gapped)
  --no-limits        pass through to generate-config (rootless/LSF): drop ALL
                     cgroup limits (cpuset + mem_limit + shm + caps)
  --no-mem-limit     drop only the per-instance mem_limit cgroup cap while keeping
                     cpuset/shm/caps -- use when a container is OOM-killed by the
                     memory cgroup despite ample free host RAM
  --dry-run          derive slices + generate config preview, print commands, no run

Env: BENCH_GUIDELLM_BIN (path to guidellm), HF_TOKEN (gated models).
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --cpi) CPI="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --nginx-cores) NGINX_CORES="$2"; shift 2 ;;
    --guidellm-cores) GUIDELLM_CORES="$2"; shift 2 ;;
    --nginx-port) NGINX_PORT="$2"; shift 2 ;;
    --rates) RATES="$2"; shift 2 ;;
    --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
    --input) INPUT_LEN="$2"; shift 2 ;;
    --output) OUTPUT_LEN="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --hf-offline) HF_HUB_OFFLINE_OPT=1; shift ;;
    --no-limits) NO_LIMITS=true; shift ;;
    --no-mem-limit) NO_MEM_LIMIT=true; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

detect_runtime || exit 1

# --- derive physical-core-aware per-instance cpusets ---
# Reserve BOTH the nginx cores and the host-GuideLLM client cores from the vLLM
# physical-core pool (3-band layout: nginx | guidellm | vLLM).
RESERVED_CORES="$NGINX_CORES"
[[ -n "$GUIDELLM_CORES" ]] && RESERVED_CORES="${RESERVED_CORES},${GUIDELLM_CORES}"
if ! HW_OUT="$("${SELF_DIR}/../scripts/check_hardware.sh" --mode multi -n "$N" -c "$CPI" --exclude "$RESERVED_CORES" --slices --quiet)"; then
  echo "ERROR: N x CPI ($N x $CPI) does not fit the physical cores left after reserving nginx ($NGINX_CORES) + guidellm ($GUIDELLM_CORES) cores." >&2
  "${SELF_DIR}/../scripts/check_hardware.sh" --mode multi -n "$N" -c "$CPI" --exclude "$RESERVED_CORES" >/dev/null
  exit 3
fi
INSTANCE_CPUSETS="$(echo "$HW_OUT" | awk -F= '/^SLICE_/{print $2}' | paste -sd';' -)"
[[ -z "$INSTANCE_CPUSETS" ]] && { echo "ERROR: failed to derive per-instance cpusets." >&2; exit 3; }

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
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$(pwd)/bench_results/online_multi_${TS}"
mkdir -p "$RESULTS_DIR"

# Forward HF offline mode into the generated compose when requested.
[[ -n "$HF_HUB_OFFLINE_OPT" ]] && export HF_HUB_OFFLINE="$HF_HUB_OFFLINE_OPT"

GEN_ARGS=( --instance-cpusets "$INSTANCE_CPUSETS" --nginx-cores "$NGINX_CORES"
  -m "$MODEL" --image "$IMAGE" --nginx-port "$NGINX_PORT" --max-model-len "$MAX_MODEL_LEN"
  --hf-cache-dir "$HF_CACHE_DIR" )
$NO_LIMITS && GEN_ARGS+=( --no-limits )
$NO_MEM_LIMIT && GEN_ARGS+=( --no-mem-limit )
[[ "$DRY_RUN" == "1" ]] && GEN_ARGS+=( --dry-run )

echo "=== ONLINE multi benchmark ==="
echo "  image:      $IMAGE"
echo "  model:      $MODEL"
echo "  N x CPI:    $N x $CPI (nginx: $NGINX_CORES  guidellm: ${GUIDELLM_CORES:-<unpinned>})"
echo "  instances:  $INSTANCE_CPUSETS"
echo "  LB target:  http://localhost:${NGINX_PORT}"
echo "  guidellm:   rates=$RATES max-seconds=$MAX_SECONDS in=$INPUT_LEN out=$OUTPUT_LEN"
echo "  mem-limit:  $([[ "$NO_MEM_LIMIT" == "true" ]] && echo 'dropped (--no-mem-limit)' || echo 'default')"
echo "  results:    $RESULTS_DIR"
if $NO_LIMITS; then
  echo
  echo "  !! WARNING: --no-limits drops ALL cgroup limits INCLUDING cpuset pinning." >&2
  echo "  !!          vLLM instances will NOT be core-pinned to $INSTANCE_CPUSETS," >&2
  echo "  !!          so results are unreliable and instances may fight for cores." >&2
  echo "  !!          Use only when cgroups are unavailable (rootless/LSF). To keep" >&2
  echo "  !!          pinning but avoid mem OOM, use --no-mem-limit instead." >&2
fi
echo

echo ">> generate-config.sh ${GEN_ARGS[*]}"
"${SELF_DIR}/generate-config.sh" "${GEN_ARGS[@]}"

TARGET="http://localhost:${NGINX_PORT}"
OUT_JSON="${RESULTS_DIR}/benchmarks.json"
GUIDELLM_LOG="${RESULTS_DIR}/guidellm.log"
# Use the explicit `benchmark run` subcommand: works on GuideLLM 0.6 and 0.7+
# (0.7 dropped the implicit default subcommand).
GUIDELLM_CMD=( "${GUIDELLM_PREFIX[@]}" "$BENCH_GUIDELLM_BIN" benchmark run --target "$TARGET" --model "$MODEL"
  --rate-type concurrent --rate "$RATES" --max-seconds "$MAX_SECONDS"
  --data "prompt_tokens=${INPUT_LEN},output_tokens=${OUTPUT_LEN}" --output-path "$OUT_JSON" )

echo ">> GuideLLM: ${GUIDELLM_CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then echo "(dry-run: no files written, stack not started)"; exit 0; fi

if ! command -v "${BENCH_GUIDELLM_BIN%% *}" >/dev/null 2>&1; then
  echo "ERROR: '$BENCH_GUIDELLM_BIN' not found on host (pip install guidellm) or set BENCH_GUIDELLM_BIN." >&2
  exit 4
fi

pull_image_if_missing "$IMAGE"

MEM_CSV="${RESULTS_DIR}/mem_${MODEL_SHORT}.csv"
cleanup() {
  rm -f "${MEM_CSV}.run" 2>/dev/null || true
  "${SELF_DIR}/stop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Gate on ALL N instances being healthy (not just the LB, which passes as soon
# as one backend is up), then run the client.
"${SELF_DIR}/start.sh" --nginx-port "$NGINX_PORT" --instances "$N"

"${SELF_DIR}/../scripts/mem_poll.sh" "online_multi_${MODEL_SHORT}_N${N}_C${CPI}" "$MEM_CSV" 2 "$BENCH_PREFIX" &
MEM_PID=$!

echo ">> Running GuideLLM ..."
set +e
"${GUIDELLM_CMD[@]}" 2>&1 | tee "$GUIDELLM_LOG"
RC=${PIPESTATUS[0]}
set -e 2>/dev/null || true

rm -f "${MEM_CSV}.run" 2>/dev/null || true
wait "$MEM_PID" 2>/dev/null || true

"${SELF_DIR}/stop.sh" || true
trap - EXIT

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
