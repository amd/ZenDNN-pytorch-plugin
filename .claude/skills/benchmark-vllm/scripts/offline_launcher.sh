#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# offline_launcher.sh -- runs INSIDE the container. Launches N parallel
# `vllm bench throughput` instances, each pinned to a disjoint CPI-core slice
# of the container's cpuset. Used by BOTH offline quadrants:
#   - offline single -> --n 1
#   - offline multi  -> --n N
#
# Adapted from ZenDNN_tools/.../offline/multi_offline_launcher.sh, but reads
# the cpuset from /proc/self/status (which reflects `podman run --cpuset-cpus`)
# instead of an LSF cgroup, and sources incontainer_env.sh for LD_PRELOAD.
#
# Invocation (inside container):
#   bash /bench/offline_launcher.sh \
#     --model M --n N --cpi C --log-dir /results \
#     [--input 128] [--output 128] [--prompts 128] [--max-seqs 128] \
#     [--run-tag TAG]
#
# Exit code: 0 if every instance returned 0, else 1.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/incontainer_env.sh"

MODEL=""; N=0; CPI=0
INPUT_LEN=128; OUTPUT_LEN=128; PROMPTS=128; MAX_SEQS=128
LOG_DIR=""; RUN_TAG=""

usage() {
  cat >&2 <<EOF
Usage: $0 --model M --n N --cpi C --log-dir DIR [options]
  --input NUM     random input length   (default 128)
  --output NUM    random output length  (default 128)
  --prompts NUM   num prompts           (default 128)
  --max-seqs NUM  max num sequences     (default 128)
  --run-tag STR   result filename suffix
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2"; shift 2 ;;
    --n)        N="$2"; shift 2 ;;
    --cpi)      CPI="$2"; shift 2 ;;
    --input)    INPUT_LEN="$2"; shift 2 ;;
    --output)   OUTPUT_LEN="$2"; shift 2 ;;
    --prompts)  PROMPTS="$2"; shift 2 ;;
    --max-seqs) MAX_SEQS="$2"; shift 2 ;;
    --log-dir)  LOG_DIR="$2"; shift 2 ;;
    --run-tag)  RUN_TAG="$2"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -z "$MODEL" || -z "$LOG_DIR" \
      || ! "$N"   =~ ^[0-9]+$ || "$N"   -le 0 \
      || ! "$CPI" =~ ^[0-9]+$ || "$CPI" -le 0 ]]; then
  echo "ERROR: missing/invalid required args." >&2
  usage
fi

ALGO_RAW="${ZENDNNL_MATMUL_ALGO:-none}"
ALGO_TAG="${ALGO_RAW//[^A-Za-z0-9._-]/_}"
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="N${N}_C${CPI}_bs${MAX_SEQS}_I${INPUT_LEN}_O${OUTPUT_LEN}_p${PROMPTS}_algo${ALGO_TAG}"
fi

mkdir -p "$LOG_DIR"

# --- read the cpuset this container was granted ---
ALLOWED=$(awk -F: '/^Cpus_allowed_list:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/self/status)
if [[ -z "$ALLOWED" ]]; then
  echo "ERROR: could not read Cpus_allowed_list from /proc/self/status" >&2
  exit 3
fi

expand_cpuset() {
  local list="$1" part lo hi i
  local -a parts result=()
  IFS=',' read -ra parts <<< "$list"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      lo="${part%-*}"; hi="${part#*-}"
      for ((i=lo; i<=hi; i++)); do result+=("$i"); done
    else
      result+=("$part")
    fi
  done
  printf '%s\n' "${result[@]}"
}

mapfile -t CORES < <(expand_cpuset "$ALLOWED")
TOTAL_CORES="${#CORES[@]}"

if (( N * CPI > TOTAL_CORES )); then
  echo "ERROR: N*CPI ($N*$CPI=$((N*CPI))) > container cores ($TOTAL_CORES). Cpus_allowed=$ALLOWED" >&2
  exit 4
fi

slice_to_string() {
  local -a slice=("$@")
  local lo="${slice[0]}" hi="${slice[$((${#slice[@]} - 1))]}"
  local expected=$((hi - lo + 1))
  if (( expected == ${#slice[@]} )); then
    printf '%s-%s' "$lo" "$hi"
  else
    local IFS=,; printf '%s' "${slice[*]}"
  fi
}

BINDS=()
for ((i=0; i<N; i++)); do
  start=$((i * CPI))
  SLICE=( "${CORES[@]:$start:$CPI}" )
  BINDS+=( "$(slice_to_string "${SLICE[@]}")" )
done

START_TS="$(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || echo now)"
echo "=============================================="
echo "offline_launcher.sh"
echo "  model:              $MODEL"
echo "  N x CPI:            $N x $CPI  (using $((N*CPI)) of $TOTAL_CORES cores)"
echo "  cpus_allowed_list:  $ALLOWED"
echo "  input/output:       $INPUT_LEN / $OUTPUT_LEN"
echo "  prompts/max-seqs:   $PROMPTS / $MAX_SEQS"
echo "  run tag:            $RUN_TAG"
echo "  log dir:            $LOG_DIR"
echo "  started:            $START_TS"
echo "=============================================="

PIDS=(); LOGS=()
for ((i=0; i<N; i++)); do
  INST_ID=$((i + 1))
  BIND="${BINDS[$i]}"
  LOG_FILE="${LOG_DIR}/result_inst${INST_ID}_${RUN_TAG}.txt"
  LOGS+=( "$LOG_FILE" )
  {
    echo "===================== RUN CONFIG ====================="
    echo "Model:                  $MODEL"
    echo "Instance:               $INST_ID / $N"
    echo "Cores (taskset -c):     $BIND  ($CPI cores)"
    echo "Cpus_allowed_list:      $ALLOWED  ($TOTAL_CORES cores)"
    echo "Batch size (max-seqs):  $MAX_SEQS"
    echo "Input tokens:           $INPUT_LEN"
    echo "Output tokens:          $OUTPUT_LEN"
    echo "Num prompts:            $PROMPTS"
    echo "ZENDNNL_MATMUL_ALGO:    ${ZENDNNL_MATMUL_ALGO:-<not set>}"
    echo "VLLM_CPU_KVCACHE_SPACE: ${VLLM_CPU_KVCACHE_SPACE:-<not set>}"
    echo "LD_PRELOAD:             ${LD_PRELOAD:-<not set>}"
    echo "Started at:             $START_TS"
    echo "======================================================"
    echo
  } > "$LOG_FILE"

  echo "  spawn inst $INST_ID -> taskset -c $BIND -> $LOG_FILE"
  VLLM_CPU_OMP_THREADS_BIND="$BIND" \
  taskset -c "$BIND" \
    vllm bench throughput \
      --model "$MODEL" \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$PROMPTS" \
      --max-num-seqs "$MAX_SEQS" \
      --trust-remote-code \
      >> "$LOG_FILE" 2>&1 &
  PIDS+=( "$!" )
done

echo
echo "All $N instance(s) spawned. Waiting..."
FAIL=0
for ((i=0; i<N; i++)); do
  INST_ID=$((i + 1))
  pid="${PIDS[$i]}"
  if wait "$pid"; then rc=0; else rc=$?; FAIL=1; fi
  echo "  inst $INST_ID (pid $pid) rc=$rc -> ${LOGS[$i]}"
done

echo "offline_launcher.sh finished (fail=$FAIL)"
exit "$FAIL"
