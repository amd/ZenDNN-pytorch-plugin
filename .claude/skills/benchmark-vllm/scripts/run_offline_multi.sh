#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# run_offline_multi.sh -- OFFLINE multi-instance benchmark (host side).
#
# Starts ONE container over the validated cpuset, overrides the entrypoint to
# the bundled offline_launcher.sh, which fans out N taskset-pinned
# `vllm bench throughput` processes. Peak memory is sampled by mem_poll.sh and
# results are summarised by parse_offline.py.
#
# This same script backs offline SINGLE (via run_offline_single.sh, which just
# passes --n 1).
#
# Example:
#   ./run_offline_multi.sh --model meta-llama/Llama-3.1-8B-Instruct \
#       --n 4 --cpi 32 --image docker.io/amdih/zendnn_zentorch:...ww28
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/common.sh"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
N=""; CPI=""
IMAGE="$DEFAULT_IMAGE"
INPUT_LEN=128; OUTPUT_LEN=128; PROMPTS=128; MAX_SEQS=128
RESULTS_DIR=""
CPUSET=""
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_HUB_OFFLINE_OPT="${HF_HUB_OFFLINE:-}"   # forwarded into the container when set
DRY_RUN=0
MODE="multi"

usage() {
  cat >&2 <<EOF
Usage: $0 --model M --n N --cpi C [options]
  --model NAME        HF repo or local path (default: $MODEL)
  --n N               instances (default: recommended)
  --cpi C             cores per instance (default: recommended)
  --image IMG         container image (default: newest amdih tag)
  --input N           random input len   (default: $INPUT_LEN)
  --output N          random output len  (default: $OUTPUT_LEN)
  --prompts N         num prompts        (default: $PROMPTS)
  --max-seqs N        max num seqs       (default: $MAX_SEQS)
  --results-dir DIR   host results dir   (default: ./bench_results/offline_<ts>)
  --cpuset SET        explicit cpuset (with --n/--cpi, or alone for 1 instance)
  --hf-cache-dir DIR  host HF cache mount (default: $HF_CACHE_DIR)
  --hf-offline        run with HF_HUB_OFFLINE=1 (use pre-cached model, no network)
  --dry-run           print the podman command, don't run
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --cpi) CPI="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --input) INPUT_LEN="$2"; shift 2 ;;
    --output) OUTPUT_LEN="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --max-seqs) MAX_SEQS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --cpuset) CPUSET="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --hf-offline) HF_HUB_OFFLINE_OPT=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

detect_runtime || exit 1

# --- resolve N/CPI + cpuset via check_hardware.sh unless cpuset given ---
if [[ -z "$CPUSET" ]]; then
  HW_ARGS=(--mode "$MODE" --quiet)
  [[ -n "$N"   ]] && HW_ARGS+=(--instances "$N")
  [[ -n "$CPI" ]] && HW_ARGS+=(--cpi "$CPI")
  if ! HW_OUT="$("${SELF_DIR}/check_hardware.sh" "${HW_ARGS[@]}")"; then
    echo "ERROR: requested layout does not fit this machine. Re-run check_hardware.sh for a recommendation:" >&2
    "${SELF_DIR}/check_hardware.sh" --mode "$MODE" ${N:+--instances "$N"} ${CPI:+--cpi "$CPI"} >/dev/null
    exit 3
  fi
  # shellcheck disable=SC2046
  eval "$(echo "$HW_OUT" | grep -E '^(REQ_N|REQ_CPI|CPUSET)=')"
  N="${REQ_N:-$N}"; CPI="${REQ_CPI:-$CPI}"
else
  # H2: an explicit --cpuset must still produce N and CPI for the launcher.
  # Default to a single instance spanning the whole cpuset; if --n was given,
  # derive CPI by dividing the cpuset evenly.
  cpuset_core_count() {
    local list="$1" part lo hi c=0
    IFS=',' read -ra _p <<< "$list"
    for part in "${_p[@]}"; do
      if [[ "$part" == *-* ]]; then lo="${part%-*}"; hi="${part#*-}"; c=$((c + hi - lo + 1)); else c=$((c+1)); fi
    done
    echo "$c"
  }
  TOTAL_IN_SET="$(cpuset_core_count "$CPUSET")"
  [[ -z "$N" ]] && N=1
  if [[ -z "$CPI" ]]; then
    CPI=$(( TOTAL_IN_SET / N ))
    [[ "$CPI" -le 0 ]] && CPI=1
    echo "Note: --cpuset given without --cpi; using N=$N, CPI=$CPI over $TOTAL_IN_SET cores in '$CPUSET'."
  fi
fi
[[ -z "$N" || -z "$CPI" || -z "$CPUSET" ]] && { echo "ERROR: could not resolve N/CPI/cpuset." >&2; exit 3; }

MODEL_SHORT="$(short_model_name "$MODEL")"
TS="$(bench_timestamp)"
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$(pwd)/bench_results/offline_${TS}"
mkdir -p "$RESULTS_DIR/$MODEL_SHORT"
mkdir -p "$HF_CACHE_DIR"

CNAME="${BENCH_PREFIX}-instance-offline-${TS}"

# Assemble the podman run command.
RUN_CMD=( "$CRUNTIME" run --rm --name "$CNAME"
  --cpuset-cpus "$CPUSET"
  --shm-size "${SHM_SIZE:-16g}"
  -v "${SELF_DIR}:/bench:ro"
  -v "${RESULTS_DIR}:/results"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -e "HF_HOME=/root/.cache/huggingface"
)
# perf + hf env
# shellcheck disable=SC2206
RUN_CMD+=( $(perf_env_docker_args) )
[[ -n "${HF_TOKEN:-}" ]] && RUN_CMD+=( -e "HF_TOKEN=${HF_TOKEN}" )
[[ -n "$HF_HUB_OFFLINE_OPT" ]] && RUN_CMD+=( -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_OPT}" )
RUN_CMD+=( --entrypoint bash "$IMAGE"
  /bench/offline_launcher.sh
    --model "$MODEL" --n "$N" --cpi "$CPI"
    --input "$INPUT_LEN" --output "$OUTPUT_LEN"
    --prompts "$PROMPTS" --max-seqs "$MAX_SEQS"
    --log-dir "/results/${MODEL_SHORT}"
)

echo "=== OFFLINE ${MODE} benchmark ==="
echo "  runtime:   $CRUNTIME"
echo "  image:     $IMAGE"
echo "  model:     $MODEL"
echo "  N x CPI:   $N x $CPI   cpuset=$CPUSET"
echo "  workload:  in=$INPUT_LEN out=$OUTPUT_LEN prompts=$PROMPTS max-seqs=$MAX_SEQS"
echo "  results:   $RESULTS_DIR"
echo
echo "Command:"
printf '  %q ' "${RUN_CMD[@]}"; echo
if [[ "$DRY_RUN" == "1" ]]; then echo "(dry-run: not executing)"; exit 0; fi

pull_image_if_missing "$IMAGE"

# Start peak-memory sampler in the background.
MEM_CSV="${RESULTS_DIR}/mem_${MODEL_SHORT}.csv"
"${SELF_DIR}/mem_poll.sh" "offline_${MODEL_SHORT}_N${N}_C${CPI}" "$MEM_CSV" 2 "$BENCH_PREFIX" &
MEM_PID=$!
cleanup() { rm -f "${MEM_CSV}.run" 2>/dev/null || true; wait "$MEM_PID" 2>/dev/null || true; }
trap cleanup EXIT

set +e
"${RUN_CMD[@]}"
RC=$?
set -e 2>/dev/null || true
cleanup; trap - EXIT

echo
echo "=== parsing results ==="
python3 "${SELF_DIR}/parse_offline.py" "$RESULTS_DIR" || true
echo "Peak memory trace: $MEM_CSV"
grep -h '^PEAK' "$MEM_CSV" 2>/dev/null || true

# Warn on a silent-zero result: benchmark "succeeded" but produced no throughput
# line (a common symptom is vLLM failing to reach the HF Hub on an air-gapped
# host even though the model is cached -- rerun with --hf-offline).
if ! grep -rhq 'Throughput:' "${RESULTS_DIR}/${MODEL_SHORT}" 2>/dev/null; then
  echo "WARNING: no 'Throughput:' line found in the results -- the run may have failed" >&2
  echo "         (e.g. HF Hub unreachable with a cached model). Try re-running with --hf-offline," >&2
  echo "         and check ${RESULTS_DIR}/${MODEL_SHORT}/result_inst*.txt." >&2
fi

exit "$RC"
