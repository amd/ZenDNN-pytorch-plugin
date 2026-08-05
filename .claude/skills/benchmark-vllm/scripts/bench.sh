#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# bench.sh -- THE entry point of the benchmark-vllm skill.
#
# Translates a flag-style request into the environment the vendored harness
# expects, then hands off to run_combo.sh:
#
#   bench.sh --model X --n 4 --cpi 32 --rates 32,64 -w chat -w rag
#     -> LABEL/MODEL/NUM_INSTANCES/CORES_PER_INSTANCE/GUIDELLM_RATES/
#        BASE_WORKLOAD/MAX_MODEL_LEN/INSTANCE_CPUSETS ... run_combo.sh
#        -> harness/run_sweep.sh -> start.sh -> ansible/GuideLLM -> stop.sh
#
# Everything under harness/ and run_combo.sh comes from amd/skills PR #81 and
# is kept close to upstream so re-syncs stay cheap. This wrapper adds three
# adjustments the upstream harness does not make for a zentorch-on-EPYC run:
#
#   1. --max-model-len is derived from the workload (workload_info.py) instead
#      of hardcoded to 4096, so a workload whose isl+osl exceeds the context
#      window (rag needs 8320, context_scaling_8k 16384) is not sent to an
#      undersized server.
#   2. Instance cpusets come from check_hardware.sh --slices (physical cores,
#      SMT siblings kept together) instead of contiguous core arithmetic, which
#      straddles sibling threads on SMT-enabled EPYC.
#   3. The model tag that ends up in ansible's test_name is derived and
#      length-checked here, so a long HF repo id can't fail pre-flight.
#
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SELF_DIR/.." && pwd)"
HARNESS="${HARNESS:-$SKILL_DIR/harness}"
EVAL_DIR="$HARNESS/vllm-cpu-perf-eval"
INVENTORY="$EVAL_DIR/automation/test-execution/ansible/inventory/group_vars/all/test-workloads.yml"

# shellcheck source=/dev/null
source "$SELF_DIR/common.sh"

# --- defaults ---------------------------------------------------------------
MODEL=""
NUM_INSTANCES=""
CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
RATES=""
MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-300}"
WORKLOADS=()
# Backend under test. zentorch is the default; "native" is the SAME image with
# zentorch pip-uninstalled (harness/start.sh derives and commits it), which is
# the non-zentorch half of an A/B. Part of the run label and of ansible's
# test_name, so results on disk say which backend produced them.
VARIANT="zentorch"
IMAGE="${VLLM_IMAGE:-$DEFAULT_IMAGE}"
KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-}"
DTYPE="${DTYPE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
MAX_MODEL_LEN=""
MODELS_DIR="${MODELS_DIR:-}"
LABEL=""
RUN_TAG="${RUN_TAG:-}"
DRY_RUN=false
# Cores reserved for nginx and the containerized GuideLLM load generator, in
# that order. They are excluded from the pool check_hardware.sh slices for
# vLLM. Upstream's 3-band layout: nginx 1-15 | guidellm 16-31 | vLLM 32+.
NGINX_CORES="${NGINX_CORES:-1-15}"
GUIDELLM_CPUS="${GUIDELLM_CPUS:-16-31}"
EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:-}"

usage() {
    cat <<EOF
Usage: $0 --model MODEL --n N --cpi C --rates R[,R...] [-w WORKLOAD]...

Required:
  -m, --model MODEL      HF repo id, or a path under --models-dir. Accepts the
                         upstream "path | tag" form; otherwise the tag used in
                         ansible's test_name is derived from the basename.
  --n, -N N              Number of vLLM instances.

Load:
  --rates LIST           GuideLLM concurrency levels, comma-separated.
                         throughput profile: 32,64 (and up). latency: 1,2,4,8.
                         Absolute at the load balancer, NOT per instance.
  --max-seconds N        Seconds per rate (default: $MAX_SECONDS).
  -w, --workload NAME    Traffic shape, repeatable (default: chat). All of them
                         run against ONE warm stack. --list-workloads to see
                         the table.

Stack:
  -c, --cpi C            Cores per instance (default: $CORES_PER_INSTANCE).
  --image IMAGE          Container image (default: $DEFAULT_IMAGE).
  --zentorch             zentorch backend (default).
  --native               Native backend for an A/B against zentorch: derived
                         <image>_native with zentorch uninstalled, zentorch env
                         knobs off. Everything else identical.
  --kv-cache-space N     VLLM_CPU_KVCACHE_SPACE in GiB, per instance.
  --dtype D              vllm serve --dtype (default: image/vLLM default).
  --block-size N         vllm serve --block-size.
  --max-model-len N      Override the workload-derived context window. WARNS
                         if smaller than the workloads need.
  --models-dir DIR       Host dir of local models, bind-mounted into instances.
  --nginx-cores RANGE    cpuset for nginx (default: $NGINX_CORES).
  --guidellm-cpus RANGE  cpuset for the GuideLLM container (default: $GUIDELLM_CPUS).

Other:
  --label NAME           Run label used in results/{sweep,mem}_<label>.* (default:
                         derived from N, the backend and the model tag).
  --run-tag TAG          Suffix so repeat sweeps don't clobber each other.
  --dry-run              Print everything, run nothing.
  --list-workloads       Print the workload table (isl/osl/max-model-len).
  -h, --help             This.

Results (upstream layout, unchanged):
  $EVAL_DIR/results/llm/<model>/<workload>-<ts>-<test_name>/external-endpoint/
  ./results/{sweep,mem}_<label>.*
EOF
    exit 0
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -N|--n|--instances) NUM_INSTANCES="$2"; shift 2 ;;
        -c|--cpi|--cores-per-instance) CORES_PER_INSTANCE="$2"; shift 2 ;;
        --rates) RATES="$2"; shift 2 ;;
        --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
        -w|--workload) WORKLOADS+=("$2"); shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --native) VARIANT=native; shift ;;
        --zentorch) VARIANT=zentorch; shift ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --kv-cache-space) KV_CACHE_SPACE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --nginx-cores) NGINX_CORES="$2"; shift 2 ;;
        --guidellm-cpus) GUIDELLM_CPUS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --list-workloads)
            [[ -f "$INVENTORY" ]] || die "workload inventory missing -- run scripts/setup-harness.sh first"
            exec python3 "$SELF_DIR/workload_info.py" --inventory "$INVENTORY" --list ;;
        -h|--help) usage ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

case "$VARIANT" in
    zentorch|native) ;;
    *) die "invalid --variant '$VARIANT' (expected zentorch or native)" ;;
esac

[[ -n "$MODEL" ]] || die "--model is required"
[[ -n "$NUM_INSTANCES" ]] || die "--n is required (number of vLLM instances)"
[[ "$NUM_INSTANCES" =~ ^[0-9]+$ && "$NUM_INSTANCES" -ge 1 ]] || die "--n must be a positive integer"
[[ "$CORES_PER_INSTANCE" =~ ^[0-9]+$ && "$CORES_PER_INSTANCE" -ge 1 ]] || die "--cpi must be a positive integer"
[[ -n "$RATES" ]] || die "--rates is required (throughput: 32,64 -- latency: 1,2,4,8)"
[[ "$RATES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--rates must be a comma-separated list of integers"
[[ -f "$INVENTORY" ]] || die "harness not set up -- run scripts/setup-harness.sh first"

(( ${#WORKLOADS[@]} == 0 )) && WORKLOADS=(chat)
WORKLOAD_CSV=$(IFS=,; echo "${WORKLOADS[*]}")

# --- model spec -> "path | tag" ---------------------------------------------
# run_sweep.sh splits on "|". An explicit tag wins; otherwise derive one from
# the basename and let run_sweep.sh's make_test_name trim it to the 30-char
# test_name budget.
if [[ "$MODEL" == *"|"* ]]; then
    MODEL_SPEC="$MODEL"
    MODEL_TAG="${MODEL##*|}"; MODEL_TAG="${MODEL_TAG// /}"
    MODEL_PATH="${MODEL%%|*}"; MODEL_PATH="${MODEL_PATH%"${MODEL_PATH##*[![:space:]]}"}"
else
    MODEL_PATH="$MODEL"
    MODEL_TAG="$(short_model_name "$MODEL")"
    MODEL_TAG="$(echo "$MODEL_TAG" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | tr -s '-')"
    MODEL_TAG="${MODEL_TAG#-}"; MODEL_TAG="${MODEL_TAG%-}"
    [[ -n "$MODEL_TAG" ]] || die "could not derive a test_name tag from --model '$MODEL'"
    MODEL_SPEC="$MODEL_PATH | $MODEL_TAG"
fi

# --- context window ---------------------------------------------------------
DERIVED_LEN=$(python3 "$SELF_DIR/workload_info.py" --inventory "$INVENTORY" \
                  --max-model-len "$WORKLOAD_CSV") || exit 1
if [[ -n "$MAX_MODEL_LEN" ]]; then
    if (( MAX_MODEL_LEN < DERIVED_LEN )); then
        echo "WARNING: --max-model-len $MAX_MODEL_LEN is below what ${WORKLOAD_CSV} needs ($DERIVED_LEN)." >&2
        echo "         Requests that overflow the context window will be rejected by vLLM," >&2
        echo "         so the sweep may finish with no usable data. Check guidellm.log." >&2
    fi
else
    MAX_MODEL_LEN="$DERIVED_LEN"
fi

# --- cpusets ----------------------------------------------------------------
# check_hardware.sh --slices emits SLICE_<i>=<cpuset> on stdout, sized in
# PHYSICAL cores with SMT siblings kept inside one instance. The nginx and
# guidellm bands are excluded from the pool first so the load generator and the
# server never share a core.
EXCLUDE="${NGINX_CORES},${GUIDELLM_CPUS}"
HW_OUT=$("$SELF_DIR/check_hardware.sh" --mode multi -n "$NUM_INSTANCES" -c "$CORES_PER_INSTANCE" \
            --exclude "$EXCLUDE" --slices --quiet 2>/dev/null)
HW_RC=$?
INSTANCE_CPUSETS=""
if [[ $HW_RC -eq 0 ]]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^SLICE_[0-9]+=(.*)$ ]] || continue
        INSTANCE_CPUSETS="${INSTANCE_CPUSETS:+$INSTANCE_CPUSETS;}${BASH_REMATCH[1]}"
    done <<< "$HW_OUT"
fi
if [[ -z "$INSTANCE_CPUSETS" ]]; then
    # Not fatal: generate-config.sh falls back to contiguous arithmetic. Warn,
    # because that fallback can straddle SMT siblings.
    echo "WARNING: check_hardware.sh produced no slices (rc=$HW_RC) for" \
         "${NUM_INSTANCES}x${CORES_PER_INSTANCE} cores excluding $EXCLUDE." >&2
    echo "         Falling back to contiguous core arithmetic from VLLM_START_CORE." >&2
    echo "         Re-run scripts/check_hardware.sh --mode multi -n $NUM_INSTANCES -c $CORES_PER_INSTANCE --exclude $EXCLUDE" >&2
    echo "         to see why (usually: not enough physical cores left)." >&2
fi

# --- environment for run_combo.sh / run_sweep.sh / generate-config.sh -------
LABEL="${LABEL:-n${NUM_INSTANCES}-${VARIANT}-${MODEL_TAG}}"
export LABEL RUN_TAG
export VLLM_IMAGE="$IMAGE"
export MODEL="$MODEL_SPEC"
export NUM_INSTANCES CORES_PER_INSTANCE
export GUIDELLM_RATES="[${RATES}]"
export GUIDELLM_MAX_SECONDS="$MAX_SECONDS"
export BASE_WORKLOAD="$WORKLOAD_CSV"
export MAX_MODEL_LEN
export TEST_NAME_PREFIX="${TEST_NAME_PREFIX:-n${NUM_INSTANCES}}"
export NGINX_CORES GUIDELLM_CPUS
export BENCH_SCRIPTS_DIR="$SELF_DIR"
[[ -n "$INSTANCE_CPUSETS" ]] && export INSTANCE_CPUSETS
[[ -n "$KV_CACHE_SPACE" ]]   && export VLLM_KV_CACHE_SPACE="$KV_CACHE_SPACE"
[[ -n "$DTYPE" ]]            && export DTYPE
[[ -n "$BLOCK_SIZE" ]]       && export BLOCK_SIZE
[[ -n "$MODELS_DIR" ]]       && export MODELS_DIR
[[ "$VARIANT" == "native" ]] && export NATIVE=1
# GUIDELLM_MAX_CONCURRENCY must be >= the largest rate or the concurrent
# profile silently clamps it.
MAX_RATE=$(tr ',' '\n' <<< "$RATES" | sort -n | tail -1)
export GUIDELLM_MAX_CONCURRENCY="${GUIDELLM_MAX_CONCURRENCY:-$(( MAX_RATE > 1024 ? MAX_RATE : 1024 ))}"
[[ -n "$EXTRA_SWEEP_ARGS" ]] && export EXTRA_SWEEP_ARGS

cat <<EOF

=== benchmark-vllm ===
  model         : $MODEL_PATH   (test_name tag: $MODEL_TAG)
  backend       : $VARIANT
  image         : $IMAGE
  stack         : ${NUM_INSTANCES} instances x ${CORES_PER_INSTANCE} cores
  cpusets       : ${INSTANCE_CPUSETS:-<contiguous fallback>}
  nginx/guidellm: $NGINX_CORES / $GUIDELLM_CPUS
  workloads     : $WORKLOAD_CSV   (one warm stack, one GuideLLM run each)
  rates         : $GUIDELLM_RATES  (absolute at the load balancer)
  max-seconds   : $MAX_SECONDS per rate
  max-model-len : $MAX_MODEL_LEN (derived from $WORKLOAD_CSV)
  kv cache      : ${KV_CACHE_SPACE:-<harness default>} GiB/instance
  serve knobs   : dtype=${DTYPE:-<default>} block-size=${BLOCK_SIZE:-<default>}
  label         : ${LABEL}${RUN_TAG}
======================

EOF

if $DRY_RUN; then
    echo "[dry-run] would run: bash $SELF_DIR/run_combo.sh"
    echo "[dry-run] handing --dry-run to run_sweep.sh:"
    export EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:+$EXTRA_SWEEP_ARGS }--dry-run"
fi

exec bash "$SELF_DIR/run_combo.sh"
