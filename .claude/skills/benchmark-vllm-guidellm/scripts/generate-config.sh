#!/bin/bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

set -euo pipefail

# Configurable parameters (override via env vars or flags)
NUM_INSTANCES="${NUM_INSTANCES:-5}"
CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
NGINX_CORES="${NGINX_CORES:-1-15}"
# Dedicated cores for the shared lmcache server (default: share nginx's cores for
# backward-compat). Set separately to isolate the KV cache server from nginx.
LMCACHE_SERVER_CORES="${LMCACHE_SERVER_CORES:-$NGINX_CORES}"
VLLM_START_CORE="${VLLM_START_CORE:-32}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
MEM_LIMIT="${MEM_LIMIT:-100g}"
NGINX_MEM_LIMIT="${NGINX_MEM_LIMIT:-5g}"
SHM_SIZE="${SHM_SIZE:-16g}"
# Extra args appended verbatim to the vLLM command in each instance
# (e.g. EXTRA_VLLM_ARGS="--trust-remote-code --max-model-len 8192").
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# Shell snippet spliced verbatim into each instance's entrypoint, immediately
# before `vllm serve`. Use it to export test env vars without editing this
# script. Provide the full prefix INCLUDING the trailing "&&", e.g.:
#   SET_ENV_VAR_BEFORE_VLLM_SERVE="export ZENDNNL_MATMUL_WEIGHT_CACHE=2 &&"
#   SET_ENV_VAR_BEFORE_VLLM_SERVE="export FOO=1 && export BAR=2 &&"
# Empty by default (no extra exports). run_sweep.sh forwards this through.
SET_ENV_VAR_BEFORE_VLLM_SERVE="${SET_ENV_VAR_BEFORE_VLLM_SERVE:-}"
# vLLM's OpenMP thread binding, the same mode for every instance. Exported
# after SET_ENV_VAR_BEFORE_VLLM_SERVE, so set it here, not in that prefix.
#   unset  - each instance binds its OpenMP workers to its own core range
#            (minus VLLM_RESERVED_CPUS_PER_INSTANCE, see below).
#   auto   - vLLM picks cores from the instance CPU affinity established with
#            taskset. vLLM's auto-bind takes the first visible NUMA node for
#            every instance, so this script also exports
#            CPU_VISIBLE_MEMORY_NODES=<node of that instance's cores>; without
#            it, an instance whose cores are not on node 0 gets an empty core
#            list. As a safety policy, auto is rejected with --no-limits, and
#            each instance's cores must be on one node.
#   nobind - vLLM does not set affinity; inherited OpenMP variables decide the
#            thread count and any binding.
VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-}"
# Cores of each instance kept free of OpenMP workers, for vLLM's API server and
# engine core (the vLLM CPU docs recommend 1-2 for online serving). The
# container cpuset is unchanged. Defaults to 1 in auto mode and 0 otherwise, so
# existing jobs remain unchanged while `VLLM_CPU_OMP_THREADS_BIND=auto` is safe
# for online serving without a second setting.
#   per-instance binding: workers bind to <start>-<end - N>
#   auto:                 passed to vLLM as VLLM_CPU_NUM_OF_RESERVED_CPU=N
#   nobind:               ignored (the script prints a warning)
if [[ -z "${VLLM_RESERVED_CPUS_PER_INSTANCE+x}" ]]; then
    if [[ "$VLLM_CPU_OMP_THREADS_BIND" == auto ]]; then
        VLLM_RESERVED_CPUS_PER_INSTANCE=1
    else
        VLLM_RESERVED_CPUS_PER_INSTANCE=0
    fi
fi
# No default on purpose: pass --image or export VLLM_IMAGE explicitly.
VLLM_IMAGE="${VLLM_IMAGE:-}"
# NGINX load-balancer image. Defaults to Google's public Docker Hub mirror
# (mirror.gcr.io), which serves the identical nginx:alpine image but is not
# subject to Docker Hub's anonymous "toomanyrequests" pull rate limit.
# Override with a private/authenticated registry if preferred.
NGINX_IMAGE="${NGINX_IMAGE:-mirror.gcr.io/library/nginx:alpine}"
VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-63}"
# Context window passed to "vllm serve --max-model-len". Must cover the largest
# isl+osl of every workload benchmarked against this stack: vLLM rejects any
# request longer than this with HTTP 400. run_sweep.sh derives it from the
# workloads selected with -w/--workload and exports it; the 4096 fallback here
# only applies when generate-config.sh is invoked directly.
# Do NOT leave this unset/blank to let vLLM derive it from the model's
# max_position_embeddings -- that is 131072 for Llama-3.1-8B and 262144 for
# Qwen3.5-4B, both larger than VLLM_KV_CACHE_SPACE can hold, and vLLM then
# refuses to start ("max seq len ... larger than the maximum number of tokens
# that can be stored in KV cache").
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
NGINX_PORT="${NGINX_PORT:-8080}"
# Networking model:
#   bridge (default) -- each service on a private compose bridge with static IPs;
#                       nginx reaches instances by <subnet>.IP. Needs a working
#                       CNI/netavark bridge with inter-container forwarding.
#   host             -- every service shares the host network namespace; each
#                       vLLM instance binds a distinct host port (VLLM_BASE_PORT+i-1)
#                       and nginx proxies to 127.0.0.1:<port>. Bypasses the bridge
#                       entirely (no CNI/IPAM/DNS/firewall), which is the reliable
#                       choice on rootless LSF nodes where inter-container bridge
#                       traffic is dropped (nginx->vLLM 'upstream timed out').
VLLM_NETWORK_MODE="${VLLM_NETWORK_MODE:-bridge}"
# Base host port for vLLM instances in host-network mode (instance i binds
# VLLM_BASE_PORT + i - 1). Ignored in bridge mode (each instance uses 8000 in
# its own namespace).
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
# Dedicated /24 subnet for the compose bridge network. Each vLLM instance gets a
# static IP on this subnet so NGINX can reach instances by IP instead of by
# hostname. This is required because rootless podman's DNS (aardvark-dns) is not
# reachable on the bridge gateway in this environment -- container name/alias
# resolution fails with "connection refused", which made NGINX crash-loop with
# "host not found in upstream". Static IPs sidestep DNS entirely. Use a subnet
# outside netavark's default 10.89.0.0/16 auto-allocation pool to avoid clashes.
VLLM_SUBNET="${VLLM_SUBNET:-10.201.0.0/24}"
# Derive the /24 prefix (e.g. 10.201.0.0/24 -> 10.201.0). Instance i is assigned
# <prefix>.IP_BASE+i; NGINX takes <prefix>.IP_BASE.
_subnet_base="${VLLM_SUBNET%/*}"
IP_PREFIX="${_subnet_base%.*}"
IP_BASE="${IP_BASE:-10}"
# Run each vLLM instance with HF_HUB_OFFLINE so it loads the model straight from
# the shared cache instead of contacting huggingface.co. Defaults to 1 (offline)
# because start.sh pre-warms the cache on the host before any container starts,
# so the model is always present on disk -- and the compose bridge network has no
# reliable DNS to huggingface.co, which otherwise makes instances crash with
# "Temporary failure in name resolution". Set to 0 (or pass --hf-online) only
# when instances must download/refresh the model themselves.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
MODELS_DIR="${MODELS_DIR:-}"

# --- LMCache / KV-connector wiring ------------------------------------------
# KV_TRANSFER_CONFIG: raw JSON passed verbatim to vLLM's --kv-transfer-config.
#   Emitted as a single-quoted YAML scalar so the leading '{' is NOT parsed as
#   a YAML flow mapping. Works for any KV connector, not just LMCache.
# LMCACHE_CONFIG_FILE: host path to an LMCache yaml; bind-mounted read-only into
#   every container at the same path and exported as LMCACHE_CONFIG_FILE (for
#   the in-process LMCacheConnectorV1 path).
# LMCACHE_SERVER=true: emit one shared `lmcache server` service on vllm-network
#   and auto-generate the MP kv-transfer-config for every instance pointing at
#   it (N instances share one CPU KV pool -> cross-instance reuse).
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"
LMCACHE_CONFIG_FILE_HOST="${LMCACHE_CONFIG_FILE:-}"
LMCACHE_SERVER="${LMCACHE_SERVER:-false}"
LMCACHE_SERVER_NAME="${LMCACHE_SERVER_NAME:-vllm-lmcache-server}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-9090}"
# Bind the server's ZMQ socket on all interfaces so vLLM instances in *other*
# containers can reach it over the bridge net (the CLI default is localhost,
# which is only reachable inside the server's own container).
LMCACHE_BIND_HOST="${LMCACHE_BIND_HOST:-0.0.0.0}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-100}"
# The connector rejects a chunk size that is not a multiple of every KV-cache
# group's tokens_per_block, so models whose block size exceeds 128 (gemma-4 on
# the CPU backend uses 256) need this raised to match.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-engine_driven}"
# LMCACHE_SHM=true: zero-copy SHM transport with a SINGLE shared server (shared
# cross-instance L1). All instances + the server bind-mount one shared /dev/shm
# (/dev/shm/lmc-shared). This normally corrupts vLLM's own CPU-SHM segments
# across instances (they're named from VLLM_DIST_IDENT, which defaults to the
# distributed-init port and collides across identical containers) -> EngineCore
# dies on 'assert req_id in self.requests'. We avoid that by giving each instance
# a UNIQUE VLLM_DIST_IDENT (requires the vLLM cpu_worker.py setdefault patch,
# applied in the --lmcache image build). LMCACHE_SHM=false = shared server on the
# slower pickle transport (no shared memory needed; saturates at high conc).
LMCACHE_SHM="${LMCACHE_SHM:-false}"

# vLLM native prefix caching. Default OFF (historical harness behaviour, and
# what LMCache wants since it manages KV reuse itself). Set true to benchmark
# vLLM's own (per-instance) prefix caching as a fair baseline against LMCache.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
# Shared HuggingFace cache dir bind-mounted into every vLLM instance at
# /opt/app-root/src/.cache. Lets all instances reuse a single set of model
# downloads instead of each one filling its own named volume. Persists across
# `podman compose down` / `podman system prune` because it's a host path.
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/hf-shared}"

# Torch profiler. When PROFILE=true, each instance gets
# VLLM_TORCH_PROFILER_DIR set (which enables the /start_profile and
# /stop_profile HTTP endpoints) plus a dedicated host bind-mount so traces
# survive container teardown. PROFILER_DIR is the host dir; it is mounted at
# /profiler in every instance and each writes to /profiler/instance-<i>.
PROFILE="${PROFILE:-false}"
# Default resolved after SCRIPT_DIR is known (see below).
PROFILER_DIR="${PROFILER_DIR:-}"

# Backend selection: "zentorch" (default) or "native". Override via the
# BACKEND env var or the --native/--zentorch flags. Native mode skips the
# zentorch-specific env vars emitted into the compose file.
BACKEND="${BACKEND:-zentorch}"

# dtype passed to "vllm serve --dtype <value>". Defaults to bfloat16. Use
# "half" for fp16, or float32/auto as needed. Set DTYPE="" to omit the flag
# and let vLLM pick its own default.
DTYPE="${DTYPE-bfloat16}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Generate docker-compose.yml and nginx.conf for multi-instance vLLM.

Options:
  -n, --num-instances N       Number of vLLM instances (default: $NUM_INSTANCES)
  -c, --cores-per-instance N  Cores per instance (default: $CORES_PER_INSTANCE)
  --nginx-cores RANGE         CPU range for nginx (default: $NGINX_CORES)
  --start-core N              First core for vLLM instances (default: $VLLM_START_CORE)
  -m, --model NAME            Model name or path (default: $MODEL_NAME)
  --dtype TYPE                dtype for "vllm serve --dtype" (default: bfloat16;
                              use "half" for fp16, or float32/auto). Pass an
                              empty value to omit the flag entirely.
  --image IMAGE               vLLM container image (required; no default)
  --mem-limit SIZE            Memory limit per instance (default: $MEM_LIMIT)
  --kv-cache-space N          KV cache space in GB (default: $VLLM_KV_CACHE_SPACE)
  --max-model-len N           Context window for "vllm serve --max-model-len"
                              (default: $VLLM_MAX_MODEL_LEN). Must be >= the largest
                              isl+osl of the workloads being benchmarked, and
                              small enough to fit in --kv-cache-space.
  --nginx-port PORT           Host port for nginx (default: $NGINX_PORT)
  --hf-offline                Run instances with HF_HUB_OFFLINE=1 (default; load
                              from the pre-warmed shared cache, no network)
  --hf-online                 Run instances with HF_HUB_OFFLINE=0 (let each
                              instance contact huggingface.co)
  --models-dir DIR            Host dir bind-mounted into each container at
                              the same absolute path (for local model loading)
  --kv-transfer-config JSON   Raw JSON for vLLM --kv-transfer-config (emitted
                              as a quoted YAML scalar). Any KV connector.
  --lmcache-config-file PATH  Host LMCache yaml; bind-mounted ro into each
                              container + exported as LMCACHE_CONFIG_FILE
  --lmcache-server            Add a shared 'lmcache server' service on
                              vllm-network and auto-wire every instance to it
                              via the MP connector (shared CPU KV pool)
                              over the ZMQ/pickle transport
  --lmcache-shm               --lmcache-server over the zero-copy SHM transport
                              (shared /dev/shm, engine_driven; start.sh default)
  --lmcache-l1-size-gb N      lmcache server L1 CPU pool size (default: $LMCACHE_L1_SIZE_GB)
  --hf-cache-dir DIR          Host dir bind-mounted as the shared HuggingFace
                              cache in every instance (default: $HF_CACHE_DIR).
                              All instances reuse one cache so each model is
                              downloaded exactly once per sweep.
  --native                    Use native backend: skip zentorch-specific env
                              vars (same as BACKEND=native)
  --zentorch                  Use zentorch backend (default; BACKEND=zentorch)
  --no-limits                 Skip cpuset/mem_limit/shm_size (for rootless/LSF environments)
  --no-mem-limit              Drop mem_limit only; keep cpuset/shm_size/cap_add/security_opt
  --profile                   Enable torch profiler: set VLLM_TORCH_PROFILER_DIR
                              per instance (enables /start_profile,/stop_profile)
                              and bind-mount PROFILER_DIR at /profiler.
  --profiler-dir DIR          Host dir for profiler traces (default:
                              profiler-traces/ next to this script).
  -o, --output-dir DIR        Output directory (default: generated/)
  --dry-run                   Print config summary without writing files
  -h, --help                  Show this help

Environment variables:
  NUM_INSTANCES, CORES_PER_INSTANCE, NGINX_CORES, VLLM_START_CORE,
  MODEL_NAME, HF_TOKEN, MEM_LIMIT, SHM_SIZE, VLLM_IMAGE, NGINX_IMAGE,
  VLLM_KV_CACHE_SPACE, VLLM_MAX_MODEL_LEN, NGINX_PORT,
  HF_HUB_OFFLINE (0|1, default 1),
  BACKEND (zentorch|native, default zentorch),
  SET_ENV_VAR_BEFORE_VLLM_SERVE (shell prefix spliced before "vllm serve",
    include trailing "&&", e.g. "export FOO=1 && export BAR=2 &&")
  VLLM_CPU_OMP_THREADS_BIND (unset|auto|nobind; default: each instance's own
    core range)
  VLLM_RESERVED_CPUS_PER_INSTANCE (cores per instance left free of OpenMP
    workers for vLLM's API server and engine core; default 1 with auto,
    otherwise 0)

Example:
  # 3 instances on a 96-core CPU, starting at core 16
  $0 -n 3 -c 24 --start-core 16 --nginx-cores 0-15

  # Use a local model path
  $0 -m /models/Llama-3.1-8B-Instruct
EOF
    exit 0
}

# STACK_TAG namespaces a second, independent stack: its own generated dir /
# compose project ("generated-<tag>") and a "-<tag>" suffix on every container
# name, so it can run alongside the default stack (e.g. when the previous
# stack's containers are wedged and cannot be removed). Empty (default) = the
# normal "generated" stack. Pair with a distinct NGINX_PORT + VLLM_SUBNET to
# avoid host-port / subnet collisions with the other stack.
STACK_TAG="${STACK_TAG:-}"
OUTPUT_DIR="generated${STACK_TAG:+-$STACK_TAG}"
DRY_RUN=false
NO_LIMITS=false
NO_MEM_LIMIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-instances) NUM_INSTANCES="$2"; shift 2 ;;
        -c|--cores-per-instance) CORES_PER_INSTANCE="$2"; shift 2 ;;
        --nginx-cores) NGINX_CORES="$2"; shift 2 ;;
        --start-core) VLLM_START_CORE="$2"; shift 2 ;;
        -m|--model) MODEL_NAME="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --image) VLLM_IMAGE="$2"; shift 2 ;;
        --mem-limit) MEM_LIMIT="$2"; shift 2 ;;
        --kv-cache-space) VLLM_KV_CACHE_SPACE="$2"; shift 2 ;;
        --max-model-len) VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
        --nginx-port) NGINX_PORT="$2"; shift 2 ;;
        --hf-offline) HF_HUB_OFFLINE=1; shift ;;
        --hf-online) HF_HUB_OFFLINE=0; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --kv-transfer-config) KV_TRANSFER_CONFIG="$2"; shift 2 ;;
        --lmcache-config-file) LMCACHE_CONFIG_FILE_HOST="$2"; shift 2 ;;
        --lmcache-server) LMCACHE_SERVER=true; shift ;;
        --lmcache-shm) LMCACHE_SERVER=true; LMCACHE_SHM=true; shift ;;
        --lmcache-l1-size-gb) LMCACHE_L1_SIZE_GB="$2"; shift 2 ;;
        --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --native) BACKEND=native; shift ;;
        --zentorch) BACKEND=zentorch; shift ;;
        --no-limits) NO_LIMITS=true; shift ;;
        --no-mem-limit) NO_MEM_LIMIT=true; shift ;;
        --profile) PROFILE=true; shift ;;
        --profiler-dir) PROFILER_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

case "$BACKEND" in
    zentorch) NATIVE_MODE=false ;;
    native)   NATIVE_MODE=true ;;
    *) echo "ERROR: invalid BACKEND='$BACKEND' (expected 'zentorch' or 'native')." >&2; exit 1 ;;
esac

if [[ -z "$VLLM_IMAGE" ]] && ! $DRY_RUN; then
    echo "ERROR: no vLLM image set. Pass --image IMAGE or export VLLM_IMAGE." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"

# Resolve HF_CACHE_DIR to an absolute path and ensure it exists so podman
# doesn't auto-create it as root-owned on first mount.
mkdir -p "$HF_CACHE_DIR"
HF_CACHE_DIR="$(cd "$HF_CACHE_DIR" && pwd)"

# Resolve PROFILER_DIR (default next to this script) and ensure it exists so
# podman doesn't auto-create it root-owned on first mount. Only matters when
# PROFILE=true, but resolving unconditionally keeps the value absolute.
if $PROFILE; then
    PROFILER_DIR="${PROFILER_DIR:-$SCRIPT_DIR/profiler-traces}"
    mkdir -p "$PROFILER_DIR"
    PROFILER_DIR="$(cd "$PROFILER_DIR" && pwd)"
fi

# --- Rootless user-namespace handling --------------------------------------
# In rootless podman WITHOUT subuid/subgid ranges (single-id mapping), the host
# user maps only to container UID 0. Images that declare a non-root USER (the
# vLLM image bakes in UID 2000 / app-root) then fail to start with:
#   OCI runtime error: ... cannot setresuid to 2000: Invalid argument
# Force such services to run as container-root (UID 0, which == the invoking
# host user under rootless), needing no extra sub-id mapping. On hosts that DO
# have subuid ranges we leave the image's default user untouched.
#   FORCE_ROOT_USER=auto (default) | 1/true (always) | 0/false (never)
FORCE_ROOT_USER="${FORCE_ROOT_USER:-auto}"
RUN_AS_ROOT=false
case "$FORCE_ROOT_USER" in
    1|true|yes)  RUN_AS_ROOT=true ;;
    0|false|no)  RUN_AS_ROOT=false ;;
    auto|*)
        _uid_name="${USER:-$(id -un)}"
        if ! grep -q "^${_uid_name}:" /etc/subuid 2>/dev/null; then
            RUN_AS_ROOT=true
        fi
        ;;
esac

# Validate networking model.
case "$VLLM_NETWORK_MODE" in
    bridge|host) ;;
    *) echo "ERROR: invalid VLLM_NETWORK_MODE='$VLLM_NETWORK_MODE' (expected 'bridge' or 'host')." >&2; exit 1 ;;
esac

# Port a given instance's vLLM listens on. Host mode: distinct host ports so all
# instances can share the host netns. Bridge mode: always 8000 (each instance is
# isolated in its own namespace).
vllm_port_for() {
    local idx="$1"
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        echo $(( VLLM_BASE_PORT + idx - 1 ))
    else
        echo 8000
    fi
}

# Resolve LMCache config file to an absolute host path (mounted at same path).
if [[ -n "$LMCACHE_CONFIG_FILE_HOST" ]]; then
    if [[ ! -f "$LMCACHE_CONFIG_FILE_HOST" ]]; then
        echo "ERROR: --lmcache-config-file '$LMCACHE_CONFIG_FILE_HOST' not found." >&2
        exit 1
    fi
    LMCACHE_CONFIG_FILE_HOST="$(cd "$(dirname "$LMCACHE_CONFIG_FILE_HOST")" && pwd)/$(basename "$LMCACHE_CONFIG_FILE_HOST")"
fi

# Append each token in EXTRA_VLLM_ARGS as a YAML list item to the compose
# command block (word-split on whitespace; no quoting/escaping needed for
# typical flags like --trust-remote-code or --max-model-len 8192).
emit_extra_vllm_args() {
    [[ -z "$EXTRA_VLLM_ARGS" ]] && return 0
    local arg
    for arg in $EXTRA_VLLM_ARGS; do
        printf '      - %s\n' "$arg" >> "$COMPOSE"
    done
}

# --- Resolve LMCache wiring -------------------------------------------------
# One shared lmcache server; all instances point at it via the MP connector.
# Works for BOTH transports:
#   - pickle (LMCACHE_SHM=false): KV over ZMQ, no shared memory.
#   - shm    (LMCACHE_SHM=true) : zero-copy via a shared /dev/shm. Safe across
#     instances ONLY because we give each a unique VLLM_DIST_IDENT (below) so
#     vLLM's own CPU-SHM segments don't collide on the shared /dev/shm.
# The env-var forms imply the same topology as their flags.
$LMCACHE_SHM && LMCACHE_SERVER=true
LMCACHE_SHM_DIR="/dev/shm/lmc-shared${STACK_TAG:+-$STACK_TAG}"
if $LMCACHE_SERVER; then
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        echo "ERROR: --lmcache-server/--lmcache-shm require VLLM_NETWORK_MODE=bridge." >&2
        exit 1
    fi
    if (( IP_BASE < 3 )); then
        echo "ERROR: the LMCache server takes address IP_BASE-1; need IP_BASE >= 3 (got $IP_BASE)." >&2
        exit 1
    fi
    # Static address just below nginx (IP_BASE): instances reach the shared
    # server without container DNS, which is unreliable on rootless bridges.
    LMCACHE_SERVER_IP="${IP_PREFIX}.$(( IP_BASE - 1 ))"
    _mode_extra=""
    $LMCACHE_SHM && _mode_extra=',"lmcache.mp.mp_transfer_mode":"'"$LMCACHE_MP_TRANSFER_MODE"'"'
    KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_connector_extra_config":{"lmcache.mp.host":"tcp://'"$LMCACHE_SERVER_IP"'","lmcache.mp.port":'"$LMCACHE_ZMQ_PORT""$_mode_extra"'}}'
fi
LMCACHE_ENABLED=false
if $LMCACHE_SERVER || [[ -n "$KV_TRANSFER_CONFIG" || -n "$LMCACHE_CONFIG_FILE_HOST" ]]; then
    LMCACHE_ENABLED=true
fi

# Append `--kv-transfer-config '<json>'` as a single-quoted YAML scalar so the
# leading '{' is not parsed as a YAML flow mapping. Single quotes inside the
# JSON (there are none in the connector configs we emit) would be doubled.
emit_kv_transfer_config() {
    [[ -z "$KV_TRANSFER_CONFIG" ]] && return 0
    local esc=${KV_TRANSFER_CONFIG//\'/\'\'}
    printf '      - --kv-transfer-config\n' >> "$COMPOSE"
    printf "      - '%s'\n" "$esc" >> "$COMPOSE"
}

# Extra environment lines (under an existing `environment:` block) for LMCache.
emit_lmcache_env() {
    [[ "$LMCACHE_ENABLED" != "true" ]] && return 0
    printf '      - LMCACHE_MP_TRANSFER_MODE=%s\n' "$LMCACHE_MP_TRANSFER_MODE" >> "$COMPOSE"
    if [[ -n "$LMCACHE_CONFIG_FILE_HOST" ]]; then
        printf '      - LMCACHE_CONFIG_FILE=%s\n' "$LMCACHE_CONFIG_FILE_HOST" >> "$COMPOSE"
    fi
}

# Extra volume lines (under an existing `volumes:` block).
# In shm mode, bind-mount ONE shared host tmpfs dir ($LMCACHE_SHM_DIR) at
# /dev/shm into every instance AND the server -> the LMCache server's KV pool is
# visible to all workers for zero-copy cross-instance reuse. Safe only because
# each instance gets a unique VLLM_DIST_IDENT so vLLM's own CPU-SHM segments
# (named from that ident) don't collide on the shared /dev/shm.
emit_lmcache_volume() {
    if [[ "$LMCACHE_SHM" == "true" ]]; then
        printf '      - %s:/dev/shm\n' "$LMCACHE_SHM_DIR" >> "$COMPOSE"
    fi
    [[ -z "$LMCACHE_CONFIG_FILE_HOST" ]] && return 0
    printf '      - %s:%s:ro,z\n' "$LMCACHE_CONFIG_FILE_HOST" "$LMCACHE_CONFIG_FILE_HOST" >> "$COMPOSE"
}

# --- OpenMP binding and reserved cores -------------------------------------
if ! [[ "$VLLM_RESERVED_CPUS_PER_INSTANCE" =~ ^[0-9]+$ ]] \
        || (( VLLM_RESERVED_CPUS_PER_INSTANCE >= CORES_PER_INSTANCE )); then
    echo "ERROR: VLLM_RESERVED_CPUS_PER_INSTANCE='$VLLM_RESERVED_CPUS_PER_INSTANCE' must be an integer from 0 to $(( CORES_PER_INSTANCE - 1 ))." >&2
    exit 1
fi
case "$VLLM_CPU_OMP_THREADS_BIND" in
    ""|nobind) ;;
    auto)
        if $NO_LIMITS; then
            echo "ERROR: as a safety policy, VLLM_CPU_OMP_THREADS_BIND=auto cannot be used with --no-limits." >&2
            exit 1
        fi ;;
    *)  echo "ERROR: VLLM_CPU_OMP_THREADS_BIND='$VLLM_CPU_OMP_THREADS_BIND': use auto, nobind, or leave it unset (per-instance core ranges)." >&2
        exit 1 ;;
esac
if [[ "$VLLM_CPU_OMP_THREADS_BIND" == nobind ]] && (( VLLM_RESERVED_CPUS_PER_INSTANCE > 0 )); then
    echo "WARNING: VLLM_RESERVED_CPUS_PER_INSTANCE is ignored with VLLM_CPU_OMP_THREADS_BIND=nobind." >&2
fi

# NUMA node(s) of the host cores <first>..<last>, comma-separated.
numa_nodes_of() {
    local c d found nodes=()
    for (( c = $1; c <= $2; c++ )); do
        found=0
        for d in /sys/devices/system/cpu/cpu$c/node[0-9]*; do
            if [[ -e "$d" ]]; then
                nodes+=("${d##*node}")
                found=$(( found + 1 ))
            fi
        done
        if (( found != 1 )); then
            echo "ERROR: expected exactly one NUMA node for CPU $c, found $found under /sys/devices/system/cpu/cpu$c." >&2
            return 1
        fi
    done
    printf '%s\n' "${nodes[@]}" | sort -un | paste -sd, -
}

# Sets CPUSET_MEMS to the NUMA node(s) of host cores <start>..<end>, emitted as
# the instance's x-cpuset-mems (podman --cpuset-mems). podman-compose translates
# neither cpuset key, so start.sh writes both to the container's cgroup.
cpuset_mems_for() {
    if ! CPUSET_MEMS=$(numa_nodes_of "$1" "$2" 2>/dev/null); then
        echo "WARNING: NUMA node of cores $1-$2 unknown; that instance gets no cpuset mems." >&2
        CPUSET_MEMS=""
    fi
}

# Sets OMP_BIND_DESC (for the summary), OMP_BIND_EXPORTS (the shell prefix
# spliced into the instance entrypoint) and VLLM_SERVE_WRAPPER for the instance
# on cores <start>..<end>.
omp_bind_for() {
    local start=$1 end=$2 n=$VLLM_RESERVED_CPUS_PER_INSTANCE nodes
    VLLM_SERVE_WRAPPER=""
    case "$VLLM_CPU_OMP_THREADS_BIND" in
        "")
            OMP_BIND_DESC="$start-$(( end - n ))"
            (( n > 0 )) && OMP_BIND_DESC+=" ($n reserved)"
            OMP_BIND_EXPORTS="export VLLM_CPU_OMP_THREADS_BIND=$start-$(( end - n )) &&" ;;
        nobind)
            OMP_BIND_DESC="nobind"
            OMP_BIND_EXPORTS="export VLLM_CPU_OMP_THREADS_BIND=nobind &&" ;;
        auto)
            if ! nodes=$(numa_nodes_of "$start" "$end"); then
                exit 1
            fi
            if [[ "$nodes" == *,* ]]; then
                echo "ERROR: cores $start-$end span NUMA nodes $nodes. vLLM auto-bind uses one node per instance;" >&2
                echo "  use per-instance binding (unset VLLM_CPU_OMP_THREADS_BIND) or a core range on one node." >&2
                exit 1
            fi
            OMP_BIND_DESC="auto, NUMA node $nodes, $n reserved"
            OMP_BIND_EXPORTS="export VLLM_CPU_OMP_THREADS_BIND=auto && export VLLM_CPU_NUM_OF_RESERVED_CPU=$n && export CPU_VISIBLE_MEMORY_NODES=$nodes && echo \"=== VLLM_CPU_NUM_OF_RESERVED_CPU=$n CPU_VISIBLE_MEMORY_NODES=$nodes ===\" &&"
            # podman-compose 3.4 can ignore compose's cpuset until start.sh
            # applies a cgroup pin after the container has started. taskset
            # constrains vLLM before it imports the CPU backend and resolves
            # auto binding; every child process inherits the same affinity.
            VLLM_SERVE_WRAPPER="taskset -c $start-$end" ;;
    esac
}

last_core=$(( VLLM_START_CORE + NUM_INSTANCES * CORES_PER_INSTANCE - 1 ))

echo "=== Multi-Instance vLLM Configuration ==="
echo "  Instances:       $NUM_INSTANCES"
echo "  Cores/instance:  $CORES_PER_INSTANCE"
echo "  NGINX cores:     $NGINX_CORES"
echo "  vLLM cores:      $VLLM_START_CORE-$last_core"
echo "  Model:           $MODEL_NAME"
echo "  dtype:           ${DTYPE:-<vllm default>}"
echo "  Pre-serve env:   ${SET_ENV_VAR_BEFORE_VLLM_SERVE:-<none>}"
echo "  OMP bind:        ${VLLM_CPU_OMP_THREADS_BIND:-<per-instance core range>}"
echo "  Reserved/inst:   $VLLM_RESERVED_CPUS_PER_INSTANCE core(s)"
echo "  Backend:         $BACKEND"
echo "  Image:           ${VLLM_IMAGE:-<unset>}"
echo "  NGINX image:     $NGINX_IMAGE"
echo "  Memory limit:    $MEM_LIMIT per instance"
echo "  KV cache space:  ${VLLM_KV_CACHE_SPACE}GB"
echo "  Max model len:   $VLLM_MAX_MODEL_LEN"
echo "  NGINX port:      $NGINX_PORT"
if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
    echo "  Network mode:    host (vLLM ports ${VLLM_BASE_PORT}..$(( VLLM_BASE_PORT + NUM_INSTANCES - 1 )); nginx->127.0.0.1)"
else
    echo "  Network mode:    bridge"
    echo "  Network subnet:  $VLLM_SUBNET (nginx=${IP_PREFIX}.${IP_BASE}, vllm-i=${IP_PREFIX}.$((IP_BASE+1))..)"
fi
echo "  HF hub offline:  $HF_HUB_OFFLINE"
echo "  Models dir:      ${MODELS_DIR:-<unset>}"
echo "  HF cache dir:    $HF_CACHE_DIR (shared across all instances)"
if $PROFILE; then
    echo "  Profiler:        ON -> $PROFILER_DIR mounted at /profiler (per-instance subdirs)"
else
    echo "  Profiler:        OFF"
fi
echo "  Run as root:     $RUN_AS_ROOT (uid 0 workaround for rootless no-subuid)"
if $LMCACHE_ENABLED; then
echo "  LMCache:         ENABLED"
echo "    server:        $($LMCACHE_SERVER && echo "$LMCACHE_SERVER_NAME @ $LMCACHE_SERVER_IP (l1=${LMCACHE_L1_SIZE_GB}GB, zmq=$LMCACHE_ZMQ_PORT)" || echo "<external / in-process>")"
echo "    config file:   ${LMCACHE_CONFIG_FILE_HOST:-<none>}"
echo "    kv-transfer:   ${KV_TRANSFER_CONFIG:-<none>}"
fi
echo "  Output dir:      $OUTPUT_DIR"
echo "=========================================="

if $DRY_RUN; then
    echo ""
    echo "Core allocation:"
    for i in $(seq 1 "$NUM_INSTANCES"); do
        start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
        end=$(( start + CORES_PER_INSTANCE - 1 ))
        omp_bind_for "$start" "$end"
        cpuset_mems_for "$start" "$end"
        echo "  vllm-$i: cores $start-$end, mems ${CPUSET_MEMS:-<unset>}, OMP bind $OMP_BIND_DESC"
    done
    echo "  nginx:  cores $NGINX_CORES"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

# SHM mode: recreate the shared host tmpfs dir that every instance and the
# server bind-mount at /dev/shm (clears stale segments from a previous stack).
if $LMCACHE_SHM; then
    rm -rf "$LMCACHE_SHM_DIR" 2>/dev/null || true
    mkdir -p "$LMCACHE_SHM_DIR"
    echo "  Created shared SHM dir: $LMCACHE_SHM_DIR (all instances + server)"
fi

# --- Generate .env ---
cat > "$OUTPUT_DIR/.env" <<EOF
MODEL_NAME=$MODEL_NAME
HF_TOKEN=$HF_TOKEN
MEM_LIMIT=$MEM_LIMIT
HF_CACHE_DIR=$HF_CACHE_DIR
EOF
echo "  Written: $OUTPUT_DIR/.env"

# --- Generate profiler-patch.sh (only used when PROFILE) ---
# Runs INSIDE each vLLM container before `vllm serve`. Adds
# group_by_input_shape=True to the torch profiler's key_averages() call so the
# emitted trace table is broken down per input shape. The vLLM module files
# only exist inside the image, so the patch must run there, not on the host.
# torch_profiler_record_shapes is handled separately by --profiler-config.
# Idempotent: greps for the already-patched form before editing, and patches
# whichever of the native / zentorch files is present in the image.
if $PROFILE; then
    cat > "$OUTPUT_DIR/profiler-patch.sh" <<'PATCH_EOF'
#!/bin/bash
# Auto-generated by generate-config.sh. Adds group_by_input_shape=True to the
# torch profiler key_averages() table in whichever backend module is present.
set -u
SP="$(python -c 'import sys; print(next(p for p in sys.path if p.endswith("site-packages")))' 2>/dev/null)"
if [[ -z "$SP" ]]; then
    echo "[profiler-patch] WARNING: could not locate site-packages; skipping." >&2
    exit 0
fi

# Native: vllm/profiler/wrapper.py -- key_averages() -> key_averages(group_by_input_shape=True)
NATIVE="$(find "$SP" -path '*/vllm/profiler/wrapper.py' 2>/dev/null | head -1)"
if [[ -n "$NATIVE" ]]; then
    if grep -q 'key_averages(group_by_input_shape=True)' "$NATIVE"; then
        echo "[profiler-patch] native wrapper already patched: $NATIVE"
    elif grep -q 'key_averages()' "$NATIVE"; then
        sed -i 's/key_averages()/key_averages(group_by_input_shape=True)/g' "$NATIVE"
        echo "[profiler-patch] patched native wrapper: $NATIVE"
    fi
fi

# Zentorch: zentorch/vllm/_platform.py -- .key_averages().table( -> .key_averages(group_by_input_shape=True).table(
ZEN="$(find "$SP" -path '*/zentorch/vllm/_platform.py' 2>/dev/null | head -1)"
if [[ -n "$ZEN" ]]; then
    if grep -q 'key_averages(group_by_input_shape=True).table(' "$ZEN"; then
        echo "[profiler-patch] zentorch platform already patched: $ZEN"
    elif grep -q '\.key_averages()\.table(' "$ZEN"; then
        sed -i 's/\.key_averages()\.table(/.key_averages(group_by_input_shape=True).table(/g' "$ZEN"
        echo "[profiler-patch] patched zentorch platform: $ZEN"
    fi
fi
exit 0
PATCH_EOF
    chmod +x "$OUTPUT_DIR/profiler-patch.sh"
    echo "  Written: $OUTPUT_DIR/profiler-patch.sh"
fi

# --- Generate nginx.conf ---
cat > "$OUTPUT_DIR/nginx.conf" <<'NGINX_HEADER'
user root;
pid /run/nginx.pid;

events {
    worker_connections 4096;
}

http {
    upstream vllm_backend {
NGINX_HEADER

for i in $(seq 1 "$NUM_INSTANCES"); do
    # max_fails=0 disables passive failure marking. Connect failures during the
    # minutes-long model load would otherwise blacklist every upstream at once,
    # leaving nginx with "no live upstreams" just as the instances come up.
    # proxy_next_upstream still routes around a backend failing the current request.
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        echo "        server 127.0.0.1:$(vllm_port_for "$i") max_fails=0;" >> "$OUTPUT_DIR/nginx.conf"
    else
        echo "        server ${IP_PREFIX}.$(( IP_BASE + i )):8000 max_fails=0;" >> "$OUTPUT_DIR/nginx.conf"
    fi
done

cat >> "$OUTPUT_DIR/nginx.conf" <<'NGINX_BODY'
    }

    log_format upstream_log '$remote_addr - [$time_local] "$request" $status '
                            'upstream=$upstream_addr response_time=$upstream_response_time';

    access_log /var/log/nginx/access.log upstream_log;
    error_log /var/log/nginx/error.log warn;

    server {
        listen __NGINX_LISTEN__;
        server_name _;

        proxy_connect_timeout 2s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        send_timeout 300s;

        proxy_buffer_size 128k;
        proxy_buffers 8 256k;
        proxy_busy_buffers_size 512k;

        location /health {
            proxy_pass http://vllm_backend/health;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
        }

        location / {
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/completions {
            proxy_pass http://vllm_backend/v1/completions;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/chat/completions {
            proxy_pass http://vllm_backend/v1/chat/completions;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/models {
            proxy_pass http://vllm_backend/v1/models;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }

        location /nginx_status {
            stub_status on;
            access_log off;
        }
    }
}
NGINX_BODY

# Substitute nginx's listen port. In host mode there is no compose port mapping,
# so nginx must bind NGINX_PORT directly; in bridge mode it listens on 80 and the
# compose file maps NGINX_PORT:80.
if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
    NGINX_LISTEN="$NGINX_PORT"
else
    NGINX_LISTEN="80"
fi
sed -i "s/listen __NGINX_LISTEN__;/listen $NGINX_LISTEN;/" "$OUTPUT_DIR/nginx.conf"
echo "  Written: $OUTPUT_DIR/nginx.conf"

# --- Generate docker-compose.yml ---
COMPOSE="$OUTPUT_DIR/docker-compose.yml"

cat > "$COMPOSE" <<EOF
version: '3.8'

services:
EOF

DEPENDS_LIST=""
for i in $(seq 1 "$NUM_INSTANCES"); do
    start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
    end=$(( start + CORES_PER_INSTANCE - 1 ))
    omp_bind_for "$start" "$end"
    port=$(vllm_port_for "$i")
    DEPENDS_LIST="${DEPENDS_LIST}      - vllm-$i
"

    cat >> "$COMPOSE" <<EOF
  vllm-$i:
    image: $VLLM_IMAGE
    container_name: vllm-instance-$i${STACK_TAG:+-$STACK_TAG}
EOF

    # Host-network mode: share the host netns (bypass the CNI bridge entirely).
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        cat >> "$COMPOSE" <<EOF
    network_mode: host
EOF
    fi

    # Run as container-root under rootless-no-subuid so the image's baked-in
    # USER 2000 doesn't trip "cannot setresuid to 2000: Invalid argument".
    if $RUN_AS_ROOT; then
        cat >> "$COMPOSE" <<EOF
    user: "0:0"
EOF
    fi

    if ! $NO_LIMITS; then
        cpuset_mems_for "$start" "$end"
        cat >> "$COMPOSE" <<EOF
    cpuset: "$start-$end"
EOF
        if [[ -n "$CPUSET_MEMS" ]]; then
            printf '    x-cpuset-mems: "%s"\n' "$CPUSET_MEMS" >> "$COMPOSE"
        fi
        if ! $NO_MEM_LIMIT; then
            cat >> "$COMPOSE" <<EOF
    mem_limit: \${MEM_LIMIT:-$MEM_LIMIT}
EOF
        fi
        cat >> "$COMPOSE" <<EOF
    shm_size: $SHM_SIZE
    cap_add:
      - SYS_NICE
    security_opt:
      - seccomp=unconfined
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    environment:
      - HF_HUB_OFFLINE=$HF_HUB_OFFLINE
      - HF_HOME=/opt/app-root/src/.cache/huggingface
      - HF_TOKEN=\${HF_TOKEN}
      - VLLM_CPU_KVCACHE_SPACE=$VLLM_KV_CACHE_SPACE
EOF

    if $PROFILE; then
        cat >> "$COMPOSE" <<EOF
      - VLLM_TORCH_PROFILER_DIR=/profiler/instance-$i
EOF
    fi

    # When forced to container-root, pin HOME to the image's app-root dir so
    # vLLM/torch caches keep landing in the bind-mounted, writable location
    # instead of /root.
    if $RUN_AS_ROOT; then
        cat >> "$COMPOSE" <<EOF
      - HOME=/opt/app-root/src
EOF
    fi

    # gpt-oss uses the openai_harmony tokenizer (Rust tiktoken backend), which
    # downloads the o200k_base vocab at startup. start.sh pre-fetches it into
    # the shared HF cache; point both tiktoken cache vars at it so the instances
    # load it offline instead of raising openai_harmony.HarmonyError.
    if [[ "${MODEL_NAME,,}" == *gpt-oss* ]]; then
        cat >> "$COMPOSE" <<EOF
      - TIKTOKEN_CACHE_DIR=/opt/app-root/src/.cache/tiktoken_cache
      - TIKTOKEN_RS_CACHE_DIR=/opt/app-root/src/.cache/tiktoken_cache
EOF
    fi

    if $NATIVE_MODE; then
        cat >> "$COMPOSE" <<EOF
EOF
    else
        cat >> "$COMPOSE" <<EOF

EOF
    fi

    emit_lmcache_env
    # Unique per-instance distributed identity so vLLM's CPU-SHM segment names
    # (derived from VLLM_DIST_IDENT) don't collide on the shared /dev/shm.
    # Relies on the cpu_worker.py setdefault patch baked into the --lmcache image.
    if [[ "$LMCACHE_SHM" == "true" ]]; then
        printf '      - VLLM_DIST_IDENT=vllm-inst-%s\n' "$i" >> "$COMPOSE"
    fi

    cat >> "$COMPOSE" <<EOF
    volumes:
      - $HF_CACHE_DIR:/opt/app-root/src/.cache:z
EOF

    if $PROFILE; then
        cat >> "$COMPOSE" <<EOF
      - $PROFILER_DIR:/profiler:z
      - $OUTPUT_DIR/profiler-patch.sh:/profiler-patch.sh:ro,z
EOF
    fi

    if [[ -n "$MODELS_DIR" ]]; then
        cat >> "$COMPOSE" <<EOF
      - $MODELS_DIR:$MODELS_DIR:ro,z
EOF
    fi

    emit_lmcache_volume

    cat >> "$COMPOSE" <<EOF
    entrypoint:
      - bash
      - -c
      - '${SET_ENV_VAR_BEFORE_VLLM_SERVE} ${OMP_BIND_EXPORTS} echo "=== VLLM_CPU_OMP_THREADS_BIND=\$\$VLLM_CPU_OMP_THREADS_BIND ===" && { [ -f /profiler-patch.sh ] && bash /profiler-patch.sh || true; } && echo "=== pip list (\$\$HOSTNAME) ===" && { command -v pip >/dev/null 2>&1 && pip list || python -m pip list 2>/dev/null || echo "(pip unavailable)"; } && echo "=== launching: vllm serve \$\$@ ===" && exec ${VLLM_SERVE_WRAPPER:+$VLLM_SERVE_WRAPPER }vllm serve "\$\$@"'
      - --
    command:
      - --model
      - \${MODEL_NAME:-$MODEL_NAME}
      - --port
      - "$port"
      - --host
      - "0.0.0.0"
EOF

    if $ENABLE_PREFIX_CACHING; then
        printf '      - --enable-prefix-caching\n' >> "$COMPOSE"
    else
        printf '      - --no-enable-prefix-caching\n' >> "$COMPOSE"
    fi

    cat >> "$COMPOSE" <<EOF
      - --max-model-len
      - "$VLLM_MAX_MODEL_LEN"
EOF

    if [[ -n "$DTYPE" ]]; then
        cat >> "$COMPOSE" <<EOF
      - --dtype
      - $DTYPE
EOF
    fi

    # vLLM 0.22+ takes profiler settings via the --profiler-config JSON flag
    # (torch_profiler_dir replaces VLLM_TORCH_PROFILER_DIR). The env var above
    # is kept as a fallback for 0.21 and earlier; both point at the same dir.
    if $PROFILE; then
        cat >> "$COMPOSE" <<EOF
      - --profiler-config
      - '{"profiler":"torch","torch_profiler_dir":"/profiler/instance-$i","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":false}'
EOF
    fi

    emit_extra_vllm_args
    emit_kv_transfer_config

    cat >> "$COMPOSE" <<EOF
    ulimits:
      nproc: ${CONTAINER_NPROC:-4096}
      nofile: ${CONTAINER_NOFILE:-65536}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:$port/health"]
      interval: 30s
      timeout: 30s
      retries: 10
      start_period: 600s
EOF

    # Instances talk only to the shared server.
    if $LMCACHE_SERVER; then
        cat >> "$COMPOSE" <<EOF
    depends_on:
      - $LMCACHE_SERVER_NAME
EOF
    fi

    # Bridge mode: attach to the private compose network with a static IP.
    if [[ "$VLLM_NETWORK_MODE" != "host" ]]; then
        cat >> "$COMPOSE" <<EOF
    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.$(( IP_BASE + i ))
EOF
    fi
    printf '\n' >> "$COMPOSE"
done

# --- LMCache server(s) ---
lmc_mem=$(( LMCACHE_L1_SIZE_GB + 8 ))
if $LMCACHE_SERVER; then
    # Single shared server = one CPU KV pool shared by all instances
    # (cross-instance reuse). In shm mode it bind-mounts the same shared
    # /dev/shm as the instances for zero-copy transfer; in pickle mode it uses
    # its own /dev/shm (shm_size) and moves KV over ZMQ.
    cat >> "$COMPOSE" <<EOF
  $LMCACHE_SERVER_NAME:
    image: $VLLM_IMAGE
    container_name: $LMCACHE_SERVER_NAME${STACK_TAG:+-$STACK_TAG}
EOF
    if $RUN_AS_ROOT; then printf '    user: "0:0"\n' >> "$COMPOSE"; fi
    if [[ "$LMCACHE_SHM" == "true" ]]; then
        cat >> "$COMPOSE" <<EOF
    volumes:
      - $LMCACHE_SHM_DIR:/dev/shm
EOF
    fi
    if ! $NO_LIMITS; then
        cat >> "$COMPOSE" <<EOF
    cpuset: "$LMCACHE_SERVER_CORES"
EOF
        if ! $NO_MEM_LIMIT; then
            printf '    mem_limit: %sg\n' "${lmc_mem}" >> "$COMPOSE"
        fi
        [[ "$LMCACHE_SHM" != "true" ]] && printf '    shm_size: %s\n' "$SHM_SIZE" >> "$COMPOSE"
    fi
    cat >> "$COMPOSE" <<EOF
    entrypoint:
      - lmcache
      - server
      - --host
      - "$LMCACHE_BIND_HOST"
      - --port
      - "$LMCACHE_ZMQ_PORT"
      - --http-port
      - "$LMCACHE_HTTP_PORT"
      - --l1-size-gb
      - "$LMCACHE_L1_SIZE_GB"
      - --eviction-policy
      - LRU
      - --chunk-size
      - "$LMCACHE_CHUNK_SIZE"
EOF
    # CPU MP transfer is worker-side gather/scatter (engine_driven).
    if [[ "$LMCACHE_SHM" == "true" ]]; then
        printf '      - --supported-transfer-mode\n      - engine_driven\n' >> "$COMPOSE"
    fi
    cat >> "$COMPOSE" <<EOF
    networks:
      vllm-network:
        ipv4_address: $LMCACHE_SERVER_IP
    restart: unless-stopped
EOF
    cat >> "$COMPOSE" <<EOF
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:$LMCACHE_HTTP_PORT/healthcheck"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s

EOF
fi

cat >> "$COMPOSE" <<EOF
  nginx:
    image: $NGINX_IMAGE
    container_name: vllm-nginx-lb${STACK_TAG:+-$STACK_TAG}
EOF

    # Host-network mode: nginx shares the host netns and binds NGINX_PORT itself.
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        cat >> "$COMPOSE" <<EOF
    network_mode: host
EOF
    fi

    if ! $NO_LIMITS; then
        cat >> "$COMPOSE" <<EOF
    cpuset: "$NGINX_CORES"
    mem_limit: $NGINX_MEM_LIMIT
EOF
    fi

    # Bridge mode publishes NGINX_PORT:80; host mode binds NGINX_PORT directly
    # (no port mapping allowed with host networking).
    if [[ "$VLLM_NETWORK_MODE" != "host" ]]; then
        cat >> "$COMPOSE" <<EOF
    ports:
      - "$NGINX_PORT:80"
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro,Z
    depends_on:
${DEPENDS_LIST}
EOF

    if [[ "$VLLM_NETWORK_MODE" != "host" ]]; then
        cat >> "$COMPOSE" <<EOF
    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.${IP_BASE}
EOF
    fi

    # Probe stub_status, which nginx answers locally, rather than /health, which
    # proxies to the whole upstream pool and charges a failure to every backend
    # while vLLM is still loading.
    if [[ "$VLLM_NETWORK_MODE" == "host" ]]; then
        _nginx_health_url="http://localhost:${NGINX_PORT}/nginx_status"
    else
        _nginx_health_url="http://localhost/nginx_status"
    fi
    cat >> "$COMPOSE" <<EOF
    restart: unless-stopped
    ulimits:
      nproc: ${CONTAINER_NPROC:-4096}
      nofile: ${CONTAINER_NOFILE:-65536}
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "$_nginx_health_url"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 360s
EOF

# Only bridge mode declares a compose network; host mode uses no network at all.
if [[ "$VLLM_NETWORK_MODE" != "host" ]]; then
    cat >> "$COMPOSE" <<EOF

networks:
  vllm-network:
    driver: bridge
    ipam:
      config:
        - subnet: $VLLM_SUBNET
EOF
fi

echo "  Written: $OUTPUT_DIR/docker-compose.yml"
echo ""
echo "Configuration generated successfully in $OUTPUT_DIR/"
echo "Core allocation:"
for i in $(seq 1 "$NUM_INSTANCES"); do
    start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
    end=$(( start + CORES_PER_INSTANCE - 1 ))
    omp_bind_for "$start" "$end"
    echo "  vllm-$i: cores $start-$end, OMP bind $OMP_BIND_DESC"
done
echo "  nginx:  cores $NGINX_CORES"
