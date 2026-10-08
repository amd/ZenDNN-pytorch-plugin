#!/bin/bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STACK_TAG="${STACK_TAG:-}"
GENERATED_DIR="$SCRIPT_DIR/generated${STACK_TAG:+-$STACK_TAG}"
COMPOSE_FILE="$GENERATED_DIR/docker-compose.yml"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
HEALTH_INTERVAL=10

# Ensure podman uses an OCI runtime that actually exists on this host before any
# podman/podman-compose call (avoids 'default OCI runtime "runc" not found').
# shellcheck source=podman-runtime-setup.sh
# source "$SCRIPT_DIR/podman-runtime-setup.sh"
# ensure_oci_runtime

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Start the multi-instance vLLM + NGINX stack.

Options:
  --regenerate    Force regeneration of config files (passes all extra args to generate-config.sh)
  --native        Use the native backend: build an image without zentorch from
                  the base VLLM_IMAGE and use it (same as BACKEND=native)
  --zentorch      Use the zentorch backend (default; same as BACKEND=zentorch)
  --torchao       Build a derived image with torchao>=0.10.0 pre-installed
                  (required for models with quant_method=torchao)
  --lmcache       Build a derived image with the local LMCache checkout
                  installed (NO_GPU_EXT, CPU connector) and bring up a shared
                  lmcache server wired to every instance via the MP connector.
                  Zero-copy SHM transport; LMCACHE_PICKLE=1 selects ZMQ/pickle.
                  Source dir: \$LMCACHE_SRC (default: \$HOME/LMCache)
  --lmcache-mooncake
                  --lmcache plus a Mooncake Store L2 pool (master + metadata
                  services) behind the shared server. Builds the native
                  lmcache_mooncake extension against a local Mooncake build.
                  Source dir: \$MOONCAKE_SRC (default: \$HOME/Mooncake)
  --no-limits     Skip cpuset/mem_limit/shm_size (for rootless/LSF environments)
  --no-mem-limit  Drop mem_limit only; keep cpuset/shm_size/cap_add/security_opt
  --profile       Enable torch profiler (forwarded to generate-config.sh):
                  sets VLLM_TORCH_PROFILER_DIR per instance + mounts /profiler.
                  Forces --regenerate.
  --profiler-dir DIR  Host dir for profiler traces (forwarded). Forces regen.
  --no-wait       Start containers but don't wait for health checks
  --timeout N     Health check timeout in seconds (default: $HEALTH_TIMEOUT)
  -h, --help      Show this help

If generated/ directory doesn't exist, generate-config.sh runs automatically with defaults.
Pass configuration flags after --regenerate to customize (they forward to generate-config.sh).

Backend defaults to zentorch. Set BACKEND=native (or pass --native) to use the
native (non-zentorch) backend.

Examples:
  $0                                    # start with existing or default config
  $0 --regenerate -n 3 -c 24           # regenerate for 3 instances, 24 cores each
  $0 --regenerate --model /path/model  # regenerate with a local model
  $0 --native --regenerate             # build native image and regenerate config to use it
EOF
    exit 0
}

REGENERATE=false
NO_WAIT=false
# Backend selection: "zentorch" (default) or "native". Override via the
# BACKEND env var or the --native/--zentorch flags below.
BACKEND="${BACKEND:-zentorch}"
USE_TORCHAO=false
USE_LMCACHE=false
USE_MOONCAKE=false
GENERATE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regenerate) REGENERATE=true; shift ;;
        --native) BACKEND=native; shift ;;
        --zentorch) BACKEND=zentorch; shift ;;
        --torchao|--with-torchao) USE_TORCHAO=true; shift ;;
        --lmcache) USE_LMCACHE=true; shift ;;
        --lmcache-mooncake) USE_LMCACHE=true; USE_MOONCAKE=true; shift ;;
        --no-limits) GENERATE_ARGS+=("--no-limits"); shift ;;
        --no-mem-limit) GENERATE_ARGS+=("--no-mem-limit"); shift ;;
        --profile) GENERATE_ARGS+=("--profile"); REGENERATE=true; shift ;;
        --profiler-dir) GENERATE_ARGS+=("--profiler-dir" "$2"); REGENERATE=true; shift 2 ;;
        --no-wait) NO_WAIT=true; shift ;;
        --timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) GENERATE_ARGS+=("$1"); shift ;;
    esac
done

case "$BACKEND" in
    zentorch|native) ;;
    *) echo "ERROR: invalid BACKEND='$BACKEND' (expected 'zentorch' or 'native')." >&2; exit 1 ;;
esac

if [[ "$BACKEND" == "native" ]]; then
    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        echo "ERROR: VLLM_IMAGE must be set to the base image the native image is built from." >&2
        exit 1
    fi
    BASE_IMAGE="$VLLM_IMAGE"
    NATIVE_IMAGE="${BASE_IMAGE}_native"
    TEMP_CONTAINER="vllm-native-build-$$"

    if podman image exists "$NATIVE_IMAGE" 2>/dev/null; then
        echo "--- Native image already exists: $NATIVE_IMAGE ---"
    else
        echo "--- Building native image (removing zentorch) ---"
        echo "  Base:   $BASE_IMAGE"
        echo "  Target: $NATIVE_IMAGE"

        podman run --entrypoint bash --name "$TEMP_CONTAINER" "$BASE_IMAGE" \
            -c "pip uninstall -y zentorch zentorch-weekly 2>/dev/null; echo 'zentorch packages removed'"

        podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$NATIVE_IMAGE"
        podman rm "$TEMP_CONTAINER"

        echo "  Native image built successfully."
    fi

    export VLLM_IMAGE="$NATIVE_IMAGE"
    GENERATE_ARGS+=("--image" "$NATIVE_IMAGE" "--native")
    REGENERATE=true
    echo ""
fi

if $USE_TORCHAO; then
    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        echo "ERROR: VLLM_IMAGE must be set to the base image the torchao image is built from." >&2
        exit 1
    fi
    BASE_IMAGE="$VLLM_IMAGE"
    TORCHAO_SPEC="${TORCHAO_VERSION:+torchao==${TORCHAO_VERSION}}"
    TORCHAO_SPEC="${TORCHAO_SPEC:-torchao>=0.10.0}"
    TORCHAO_TAG="${TORCHAO_VERSION:-latest}"
    TORCHAO_IMAGE="${BASE_IMAGE}_torchao-${TORCHAO_TAG}"
    TEMP_CONTAINER="vllm-torchao-build-$$"

    if podman image exists "$TORCHAO_IMAGE" 2>/dev/null; then
        echo "--- torchao image already exists: $TORCHAO_IMAGE ---"
    else
        echo "--- Building torchao image (installing ${TORCHAO_SPEC}) ---"
        echo "  Base:   $BASE_IMAGE"
        echo "  Target: $TORCHAO_IMAGE"

        podman run --entrypoint bash --name "$TEMP_CONTAINER" "$BASE_IMAGE" \
            -c "pip install --no-cache-dir '${TORCHAO_SPEC}' && python -c 'import torchao; print(\"torchao\", torchao.__version__)'"

        podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$TORCHAO_IMAGE"
        podman rm "$TEMP_CONTAINER"

        echo "  torchao image built successfully."
    fi

    export VLLM_IMAGE="$TORCHAO_IMAGE"
    GENERATE_ARGS+=("--image" "$TORCHAO_IMAGE")
    REGENERATE=true
    echo ""
fi

if $USE_LMCACHE; then
    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        echo "ERROR: VLLM_IMAGE must be set to the base image the LMCache image is built from." >&2
        exit 1
    fi
    BASE_IMAGE="$VLLM_IMAGE"
    LMCACHE_SRC="${LMCACHE_SRC:-$HOME/LMCache}"
    LMCACHE_IMAGE="${BASE_IMAGE}_lmcache"
    TEMP_CONTAINER="vllm-lmcache-build-$$"

    if podman image exists "$LMCACHE_IMAGE" 2>/dev/null; then
        echo "--- LMCache image already exists: $LMCACHE_IMAGE ---"
        echo "    (delete it to rebuild: podman rmi $LMCACHE_IMAGE)"
    else
        if [[ ! -d "$LMCACHE_SRC" ]]; then
            echo "ERROR: LMCACHE_SRC '$LMCACHE_SRC' is not a directory." >&2
            exit 1
        fi
        echo "--- Building LMCache image (installing $LMCACHE_SRC, NO_GPU_EXT) ---"
        echo "  Base:   $BASE_IMAGE"
        echo "  Source: $LMCACHE_SRC"
        echo "  Target: $LMCACHE_IMAGE"

        # Mount the checkout read-only, then copy it (minus .git/build) into a
        # writable dir: setuptools builds in-tree under --no-build-isolation, so
        # installing straight off the RO mount fails. --no-deps keeps the
        # image's pinned torch/vllm; the requirements/*.txt files don't pin
        # torch so they're safe to resolve normally. --no-same-owner: a single-
        # UID rootless userns cannot chown to a checkout's unmapped group (e.g.
        # an NFS project group), and tar running as root would fail on it.
        podman run --entrypoint bash --name "$TEMP_CONTAINER" \
            -v "$LMCACHE_SRC":/lmcache-src:ro,Z "$BASE_IMAGE" -c '
                set -e
                mkdir -p /tmp/lmcache-build
                tar -C /lmcache-src --exclude=./.git --exclude=./build \
                    --exclude="./*.egg-info" -cf - . \
                    | tar -C /tmp/lmcache-build --no-same-owner -xf -
                cd /tmp/lmcache-build
                pip install -r requirements/build.txt
                pip install -r requirements/common.txt
                pip install -r requirements/cli.txt
                NO_GPU_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 \
                    pip install . --no-deps --no-build-isolation
                # Patch vLLM so VLLM_DIST_IDENT can be overridden per instance.
                # By default cpu_worker.py hard-sets it to the distributed-init
                # port, which collides across identical containers and corrupts
                # the CPU-SHM segments when instances share one /dev/shm. Keeping
                # a preset value lets the compose supply a unique VLLM_DIST_IDENT
                # so a shared server can use SHM cross-instance without collision.
                # Only the left of the assignment is matched, so both the one-line
                # (vLLM <= 0.27) and parenthesised (>= 0.28) forms stay valid.
                # find_spec locates the package without importing it: importing
                # vllm prints plugin INFO lines on stdout, corrupting the path.
                CW=$(python -c "import importlib.util,os;print(os.path.join(importlib.util.find_spec(\"vllm\").submodule_search_locations[0],\"v1\",\"worker\",\"cpu_worker.py\"))")
                sed -i -E "s/os\.environ\[\"VLLM_DIST_IDENT\"\] = /os.environ[\"VLLM_DIST_IDENT\"] = os.environ.get(\"VLLM_DIST_IDENT\") or /" "$CW"
                grep -q "os.environ.get(\"VLLM_DIST_IDENT\") or" "$CW" && python -m py_compile "$CW" && echo "Patched cpu_worker.py: keep a preset VLLM_DIST_IDENT" || { echo "ERROR: VLLM_DIST_IDENT patch failed"; exit 1; }
                python -c "import lmcache, vllm; print(\"lmcache\", lmcache.__version__, \"vllm\", vllm.__version__)"
            '

        podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$LMCACHE_IMAGE"
        podman rm "$TEMP_CONTAINER"

        echo "  LMCache image built successfully."
    fi

    export VLLM_IMAGE="$LMCACHE_IMAGE"

    if $USE_MOONCAKE; then
        # Derive a Mooncake-enabled image using the PURE-CPU wheel
        # (mooncake-transfer-engine-non-cuda): no libcuda/libcudart linkage, so
        # the transfer engine never runs GPU code paths and cannot misclassify a
        # CPU KV buffer as GPU memory (which segfaulted the CUDA-built wheel).
        # Needs only host runtime libs. Ships mooncake_master + metadata server.
        MOONCAKE_IMAGE="${LMCACHE_IMAGE}_mooncake"
        TEMP_CONTAINER="vllm-mooncake-build-$$"
        if podman image exists "$MOONCAKE_IMAGE" 2>/dev/null; then
            echo "--- Mooncake image already exists: $MOONCAKE_IMAGE ---"
        else
            if [[ ! -d "$LMCACHE_SRC" ]]; then
                echo "ERROR: LMCACHE_SRC '$LMCACHE_SRC' is not a directory." >&2
                exit 1
            fi
            echo "--- Building Mooncake image (mooncake-transfer-engine-non-cuda, TCP/CPU) ---"
            echo "  Base:   $LMCACHE_IMAGE"
            echo "  Target: $MOONCAKE_IMAGE"
            MOONCAKE_SRC="${MOONCAKE_SRC:-$HOME/Mooncake}"
            podman run --entrypoint bash --user 0:0 --name "$TEMP_CONTAINER" \
                -v "$LMCACHE_SRC":/lmcache-src:ro,Z \
                -v "$MOONCAKE_SRC":/mooncake:ro,Z \
                -v "$SCRIPT_DIR/build_mooncake_ext.py":/tmp/build_ext.py:ro,Z \
                "$LMCACHE_IMAGE" -c '
                set -e
                # A single-UID rootless userns cannot switch to the _apt sandbox user.
                printf "APT::Sandbox::User \"root\";\n" > /etc/apt/apt.conf.d/99-rootless
                apt-get update -qq
                # runtime + C++ build deps for the native lmcache_mooncake L2 ext
                # NOTE: no build-essential -- the base image already ships a C++
                # toolchain, and the build-essential g++ postinst breaks its
                # update-alternatives (c++ is already a gcc slave). Dev headers only.
                # --no-install-recommends: rdma-core Recommends iproute2, whose pinned
                # .deb has been purged from the jammy-updates mirror (404). Only the
                # libraries listed here are needed.
                apt-get install -y -qq --no-install-recommends libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2 \
                    libgoogle-glog-dev libgflags-dev libyaml-cpp-dev libjsoncpp-dev \
                    libibverbs-dev librdmacm-dev libnuma-dev libboost-dev libmsgpack-dev \
                    libcurl4-openssl-dev libunwind-dev libxxhash-dev liburing-dev libzstd-dev
                # Install the LOCAL Mooncake build (same source tree the native
                # lmcache_mooncake ext links against) so the SDK, master and
                # metadata server are version-matched with the L2 client -- a pip
                # wheel of a different Mooncake build fails segment mount with
                # RPC_FAIL. This build is non-CUDA (no libcuda linkage).
                cp -r /mooncake/mooncake-wheel /tmp/mc-wheel && pip install /tmp/mc-wheel
                python -c "import mooncake,os; so=os.path.dirname(mooncake.__file__)+\"/store.so\"; import subprocess; out=subprocess.check_output([\"objdump\",\"-p\",so]).decode(); assert \"libcuda\" not in out, \"mooncake build links libcuda\"; from mooncake.store import MooncakeDistributedStore; print(\"mooncake (local build) import OK, no libcuda\")"
                # Build + install the native lmcache_mooncake L2 extension against
                # the Mooncake C++ SDK (mounted headers + prebuilt static libs).
                # Static libs are baked in; libasio is the only extra runtime .so.
                cp /mooncake/build/mooncake-common/libasio.so /usr/local/lib/ && ldconfig
                rm -rf /tmp/lmm_out && python /tmp/build_ext.py
                LMDIR=$(python -c "import importlib.util;print(importlib.util.find_spec(\"lmcache\").submodule_search_locations[0])")
                cp /tmp/lmm_out/lmcache/lmcache_mooncake*.so "$LMDIR/"
                python -c "import lmcache.lmcache_mooncake as m; assert \"LMCacheMooncakeClient\" in dir(m); print(\"lmcache_mooncake ext OK\")"
            '
            podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$MOONCAKE_IMAGE"
            podman rm "$TEMP_CONTAINER"
            echo "  Mooncake image built successfully."
        fi
        export VLLM_IMAGE="$MOONCAKE_IMAGE"
        GENERATE_ARGS+=("--image" "$MOONCAKE_IMAGE" "--lmcache-mooncake")
    else
        # Default to the shared-server SHM topology: one lmcache server + shared
        # /dev/shm across all instances (cross-instance reuse, zero-copy), made
        # collision-safe by the VLLM_DIST_IDENT patch above + per-instance idents.
        # Set LMCACHE_PICKLE=1 to force the shared-server pickle transport instead.
        if [[ "${LMCACHE_PICKLE:-0}" == "1" ]]; then
            GENERATE_ARGS+=("--image" "$LMCACHE_IMAGE" "--lmcache-server")
        else
            GENERATE_ARGS+=("--image" "$LMCACHE_IMAGE" "--lmcache-shm")
        fi
    fi
    REGENERATE=true
    echo ""
fi

if $REGENERATE || [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "--- Generating configuration ---"
    bash "$SCRIPT_DIR/generate-config.sh" "${GENERATE_ARGS[@]}"
    echo ""
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ERROR: $COMPOSE_FILE not found. Run generate-config.sh first."
    exit 1
fi

# --- HuggingFace pre-warm ---------------------------------------------------
# Pull the model once on the host before any vLLM container starts, so the
# shared HF cache (bind-mounted into every instance) is fully populated and
# all N workers load from disk in parallel without racing on hf_hub lockfiles.
# Hard-fails on download error: skipping the pre-warm and letting N containers
# race on the same download is exactly the bug this step exists to prevent.
ENV_FILE="$GENERATED_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${MODEL_NAME:-}" || -z "${HF_CACHE_DIR:-}" ]]; then
    echo "ERROR: MODEL_NAME or HF_CACHE_DIR not set after sourcing $ENV_FILE."
    exit 1
fi

if [[ "$MODEL_NAME" == /* ]]; then
    echo "--- Pre-warm skipped: model is a local path ($MODEL_NAME) ---"
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    echo "--- Pre-warm skipped: HF offline mode; using the populated local cache for $MODEL_NAME ---"
else
    echo "--- Pre-warming HF cache for $MODEL_NAME -> $HF_CACHE_DIR ---"
    # huggingface_hub 1.0+ replaced `huggingface-cli` with `hf`. Prefer `hf`
    # when present; fall back to `huggingface-cli` for older installs.
    if command -v hf >/dev/null 2>&1; then
        HF_CLI=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI=(huggingface-cli download)
    else
        echo "ERROR: neither 'hf' nor 'huggingface-cli' found on host. Install with:" >&2
        echo "  pip install huggingface_hub" >&2
        exit 1
    fi
    if ! HF_HOME="$HF_CACHE_DIR/huggingface" \
         HF_TOKEN="${HF_TOKEN:-}" \
         "${HF_CLI[@]}" "$MODEL_NAME"; then
        echo "ERROR: ${HF_CLI[*]} failed for '$MODEL_NAME'." >&2
        echo "  Check HF_TOKEN, model id, and network access." >&2
        exit 1
    fi
    echo "  Pre-warm complete."
fi

# gpt-oss models use the openai_harmony tokenizer, which tries to download the
# tiktoken o200k_base vocab from openaipublic.blob.core.windows.net at startup.
# That fails behind a firewall / when the instances run HF_HUB_OFFLINE, raising
# openai_harmony.HarmonyError. Pre-fetch the vocab to the shared HF cache (which
# is bind-mounted into every container) and let the compose env point the
# tiktoken caches at it. See:
#   https://huggingface.co/openai/gpt-oss-120b/discussions/39
if [[ "${MODEL_NAME,,}" == *gpt-oss* ]]; then
    # curl on this host can only write to $HOME, so pre-fetch the vocab there,
    # then copy it into the bind-mounted HF cache. The instances read the vocab
    # from the cache ($HF_CACHE_DIR -> /opt/app-root/src/.cache; compose env
    # points TIKTOKEN_CACHE_DIR/TIKTOKEN_RS_CACHE_DIR at
    # /opt/app-root/src/.cache/tiktoken_cache). Without the copy the first chat
    # request raises openai_harmony.HarmonyError and crashes the engine (nginx
    # then 502s on /health).
    TIKTOKEN_DIR="$HOME/tiktoken_cache"
    CACHE_TIKTOKEN_DIR="$HF_CACHE_DIR/tiktoken_cache"
    # SHA1 of the vocab URL string -- the exact filename tiktoken's Rust backend
    # looks up in TIKTOKEN_RS_CACHE_DIR (no extension).
    O200K_HASH="fb374d419588a4632f3f557e76b4b70aebbca790"
    O200K_URL="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    mkdir -p "$TIKTOKEN_DIR" "$CACHE_TIKTOKEN_DIR"
    if [[ ! -s "$TIKTOKEN_DIR/$O200K_HASH" ]]; then
        echo "--- Pre-fetching tiktoken vocab for gpt-oss -> $TIKTOKEN_DIR ---"
        if ! curl -fsSL "$O200K_URL" -o "$TIKTOKEN_DIR/$O200K_HASH"; then
            echo "ERROR: failed to download tiktoken vocab from $O200K_URL" >&2
            echo "  Download it manually and save as $TIKTOKEN_DIR/$O200K_HASH" >&2
            exit 1
        fi
        echo "  tiktoken vocab fetched."
    else
        echo "--- tiktoken vocab already present at $TIKTOKEN_DIR/$O200K_HASH ---"
    fi
    # Copy into the container-visible cache so every instance loads it offline.
    cp -f "$TIKTOKEN_DIR/$O200K_HASH" "$CACHE_TIKTOKEN_DIR/$O200K_HASH"
fi
# ---------------------------------------------------------------------------

num_instances=$(grep -c 'container_name: vllm-instance-' "$COMPOSE_FILE")
echo "--- Starting $num_instances vLLM instances + NGINX ---"

# Host-network mode has no compose bridge, so skip all bridge pre-create / CNI
# patching. Detect it straight from the generated compose file.
if grep -qE '^[[:space:]]*network_mode:[[:space:]]*host' "$COMPOSE_FILE"; then
    echo "  Network mode: host (skipping bridge pre-create and CNI patch)."
else

# Pre-create the compose network to avoid a known podman-compose race condition
# (multiple services in the same compose file all call "podman network exists"
# concurrently, all see "no", all try to create, only the first succeeds and
# the rest fail with exit 125 "already exists"). We create it here first so the
# network already exists by the time podman-compose runs.
COMPOSE_PROJECT="$(basename "$GENERATED_DIR")"
COMPOSE_NETWORK="${COMPOSE_PROJECT}_vllm-network"
# Pull the subnet straight out of the generated compose so the pre-created
# network matches the static IPs assigned to each service. Without a matching
# subnet, podman-compose's ipv4_address assignments would fall outside the
# network and fail. Static IPs are required because rootless aardvark-dns is
# unreachable here, so NGINX must reach instances by IP, not hostname.
COMPOSE_SUBNET="$(grep -E '^[[:space:]]+- subnet:' "$COMPOSE_FILE" | head -1 | awk '{print $NF}')"
SUBNET_ARG=()
if [[ -n "$COMPOSE_SUBNET" ]]; then
    SUBNET_ARG=(--subnet "$COMPOSE_SUBNET")
fi
# Older podman builds lack `network create --ignore`, so guard with an explicit
# existence check to stay idempotent across versions.
if ! podman network exists "$COMPOSE_NETWORK" 2>/dev/null; then
    podman network create \
        --label "io.podman.compose.project=${COMPOSE_PROJECT}" \
        --label "com.docker.compose.project=${COMPOSE_PROJECT}" \
        --driver bridge \
        "${SUBNET_ARG[@]}" \
        "$COMPOSE_NETWORK" >/dev/null
fi

# podman 3.4.x (CNI backend) writes new networks with cniVersion 1.0.0, but the
# CNI plugins installed on some hosts only support up to 0.4.0. The firewall
# plugin then rejects the config ("plugin firewall does not support config
# version 1.0.0") and containers fail to get networking. podman's own default
# network (87-podman.conflist) uses 0.4.0, so downgrade the generated conflist
# to match. Patch any pre-existing conflist too, since the existence check above
# won't recreate a network that was already made with 1.0.0.
for cni_dir in "$HOME/.config/cni/net.d" /etc/cni/net.d; do
    conflist="$cni_dir/${COMPOSE_NETWORK}.conflist"
    if [[ -f "$conflist" ]] && grep -q '"cniVersion":[[:space:]]*"1\.0\.0"' "$conflist"; then
        sed -i -E 's/("cniVersion":[[:space:]]*")1\.0\.0(")/\10.4.0\2/' "$conflist"
        echo "  Patched CNI conflist $conflist: cniVersion 1.0.0 -> 0.4.0"
    fi
done

fi  # end bridge-only network setup

# Pre-pull each unique image once before starting the stack. All N vLLM
# services reference the same image, and podman-compose pulls it concurrently
# per-service into the same containers-storage. Concurrent pulls of the same
# image race on layer locks; one pull dies and takes "podman-compose up -d"
# down with it (the log ends mid "Copying blob" with no container status), and
# a half-extracted image then fails every exec with "missing dynamic library".
# Serializing the pull here -- same rationale as the HF pre-warm above -- makes
# every service find the image already in storage.
mapfile -t COMPOSE_IMAGES < <(grep -E '^[[:space:]]+image:' "$COMPOSE_FILE" | awk '{print $2}' | sort -u)
for img in "${COMPOSE_IMAGES[@]}"; do
    if podman image exists "$img" 2>/dev/null; then
        echo "--- Image already present: $img ---"
    else
        echo "--- Pre-pulling image: $img ---"
        if ! podman pull "$img"; then
            echo "ERROR: failed to pull image '$img'." >&2
            exit 1
        fi
    fi
done

cd "$GENERATED_DIR"
# STAGGER_STARTUP=1: start support services (lmcache server, Mooncake) first,
# then vLLM instances one at a time, then nginx. Spreads each instance's heavy
# engine-core init (model load, torch.compile) so slow starters do not miss
# vLLM's startup handshake under I/O contention. Classified by compose service
# key, which STACK_TAG never changes.
if [[ "${STAGGER_STARTUP:-0}" == "1" ]]; then
    echo "--- Staggered startup: support -> instances -> nginx ---"
    _stg_support=(); _stg_instances=(); _stg_nginx=()
    while read -r _svc; do
        case "$_svc" in
            vllm-[0-9]*) _stg_instances+=("$_svc") ;;
            nginx)       _stg_nginx+=("$_svc") ;;
            *)           _stg_support+=("$_svc") ;;
        esac
    done < <(awk '
        /^services:/ { ins = 1; next }
        ins && /^[^[:space:]]/ { ins = 0 }
        ins && /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ { sub(/:.*/, ""); print $1 }
    ' "$COMPOSE_FILE")
    if (( ${#_stg_support[@]} )); then
        echo "  support: ${_stg_support[*]}"
        podman-compose up -d "${_stg_support[@]}"
        sleep "${STAGGER_SUPPORT_DELAY:-20}"
    fi
    for _svc in "${_stg_instances[@]}"; do
        echo "  instance: $_svc"
        podman-compose up -d "$_svc"
        sleep "${STAGGER_DELAY:-45}"
    done
    echo "  nginx: ${_stg_nginx[*]}"
    podman-compose up -d "${_stg_nginx[@]}"
else
    podman-compose up -d
fi
cd "$SCRIPT_DIR"

# podman-compose (3.4.x) and podman (3.4.x) silently IGNORE the compose `cpuset:`
# key -- containers end up unconstrained (cpuset.cpus.effective = all cores),
# which lets nginx/vLLM float across every core and pollutes benchmark numbers.
# Enforce the intended pinning directly on each container's cgroup v2 node. This
# is a no-op under --no-limits (generate-config emits no cpuset: lines then).
_apply_cpuset_pins() {
    local applied=0
    while IFS='|' read -r cname cpus; do
        [ -z "$cname" ] && continue
        [ -z "$cpus" ] && continue
        local pid rel cg
        pid=$(podman inspect "$cname" --format '{{.State.Pid}}' 2>/dev/null)
        if [ -z "$pid" ] || [ "$pid" = "0" ]; then
            echo "  WARN: $cname not running yet; cannot pin to $cpus"; continue
        fi
        rel=$(sed -n 's/^0::\(.*\)/\1/p' "/proc/$pid/cgroup" 2>/dev/null)
        cg="/sys/fs/cgroup${rel}"
        if [ -w "$cg/cpuset.cpus" ]; then
            if echo "$cpus" > "$cg/cpuset.cpus" 2>/dev/null; then
                echo "  pinned $cname -> $cpus (effective $(cat "$cg/cpuset.cpus.effective" 2>/dev/null))"
                applied=$((applied+1))
            else
                echo "  WARN: failed writing cpuset $cpus for $cname"
            fi
        else
            echo "  WARN: $cg/cpuset.cpus not writable for $cname"
        fi
    done < <(awk '
        /container_name:/ { name=$2 }
        /cpuset:/ { val=$2; gsub(/"/,"",val); if(name!=""){ print name"|"val; name="" } }
    ' "$COMPOSE_FILE")
    if (( applied > 0 )); then
        echo "--- cpuset pinning enforced on $applied container(s) ---"
    fi
}
if grep -q 'cpuset:' "$COMPOSE_FILE" 2>/dev/null; then
    echo "--- Enforcing cpuset pinning via cgroup v2 (podman-compose ignores compose cpuset:) ---"
    _apply_cpuset_pins
fi

echo ""
echo "--- Container status ---"
podman ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

if $NO_WAIT; then
    echo ""
    echo "Started without waiting for health checks (--no-wait)."
    echo "Run $SCRIPT_DIR/test-setup.sh to verify the setup once instances are ready."
    exit 0
fi

echo ""
echo "--- Waiting for all instances to become healthy (timeout: ${HEALTH_TIMEOUT}s) ---"
echo "    vLLM model loading can take several minutes..."

# Required containers gate the wait loop. nginx-lb is shown for visibility
# only -- its self-healthcheck (wget through the upstream) can be flaky
# while vLLM workers are still warming up, even when the proxy is working.
required_containers=()
for i in $(seq 1 "$num_instances"); do
    required_containers+=("vllm-instance-$i${STACK_TAG:+-$STACK_TAG}")
done
optional_containers=("vllm-nginx-lb${STACK_TAG:+-$STACK_TAG}")

# Podman moved the health status field: older releases (e.g. 3.4.x) expose it at
# .State.Healthcheck.Status, newer ones at .State.Health.Status. Try both so the
# wait loop reads a real value instead of an empty string (which never matches
# "healthy" and hangs the loop until timeout).
container_health() {
    local c="$1" h
    # A bad --format field prints nothing to stdout but still exits 0 (podman
    # 3.4.x), so branch on empty output rather than exit status.
    h=$(podman inspect --format '{{.State.Health.Status}}' "$c" 2>/dev/null)
    [[ -z "$h" ]] && h=$(podman inspect --format '{{.State.Healthcheck.Status}}' "$c" 2>/dev/null)
    [[ -z "$h" ]] && h="missing"
    # Rootless podman without a systemd user session (e.g. inside an LSF job)
    # cannot schedule the healthcheck timer, so .State.Health stays "starting"
    # forever even after vLLM is serving; podman-compose 1.0.6 also mistranslates
    # the compose `test: ["CMD", ...]` probe, so `podman healthcheck run` is
    # unreliable here too. When the status is not yet conclusive, probe the
    # endpoint directly with a one-off exec (needs no systemd and no stored
    # healthcheck): the vLLM API answers on :8000/health, nginx on /health.
    if [[ "$h" != "healthy" && "$h" != "unhealthy" ]]; then
        if [[ "$(podman inspect --format '{{.State.Running}}' "$c" 2>/dev/null)" == "true" ]]; then
            case "$c" in
                *nginx*) podman exec "$c" wget -q -O /dev/null http://localhost/health 2>/dev/null && h="healthy" ;;
                *)       podman exec "$c" curl -fsS -o /dev/null http://localhost:8000/health 2>/dev/null && h="healthy" ;;
            esac
        fi
    fi
    echo "$h"
}

start_time=$(date +%s)

while true; do
    elapsed=$(( $(date +%s) - start_time ))
    if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
        echo ""
        echo "ERROR: Timed out after ${HEALTH_TIMEOUT}s waiting for vLLM instances to become healthy."
        echo "Check logs with: podman logs <container-name>"
        for c in "${required_containers[@]}" "${optional_containers[@]}"; do
            status=$(container_health "$c")
            echo "  $c: $status"
        done
        exit 1
    fi

    all_healthy=true
    status_line=""
    for c in "${required_containers[@]}"; do
        # vLLM instances have no restart policy, so an exited one never comes
        # back; stop now instead of waiting out HEALTH_TIMEOUT.
        if [[ "$(podman inspect --format '{{.State.Status}}' "$c" 2>/dev/null)" =~ ^(exited|stopped)$ ]]; then
            echo ""
            echo "ERROR: $c exited with code $(podman inspect --format '{{.State.ExitCode}}' "$c") before becoming healthy. Last log lines:"
            podman logs --tail 40 "$c" 2>&1
            exit 1
        fi
        health=$(container_health "$c")
        short_name="${c#vllm-}"
        if [[ "$health" == "healthy" ]]; then
            status_line="$status_line [$short_name:OK]"
        else
            status_line="$status_line [$short_name:$health]"
            all_healthy=false
        fi
    done
    for c in "${optional_containers[@]}"; do
        health=$(container_health "$c")
        short_name="${c#vllm-}"
        # Suffix with '*' so it's visually obvious this one is informational.
        status_line="$status_line [$short_name:${health}*]"
    done

    printf "\r  [%3ds]%s" "$elapsed" "$status_line"

    if $all_healthy; then
        echo ""
        echo ""
        echo "All vLLM instances are healthy. (nginx-lb status is informational only.)"
        break
    fi

    sleep "$HEALTH_INTERVAL"
done

# The loop above probes each vLLM container from inside itself, which says
# nothing about the proxy that benchmarks actually talk to. nginx marks an
# upstream down after repeated connect failures, so it can still be serving
# 502s for a few seconds after the last instance starts listening. Gate on the
# public endpoint before handing off.
echo ""
echo "--- Waiting for load balancer to serve the backend pool ---"
lb_url="http://localhost:${NGINX_PORT:-8080}/v1/models"
lb_deadline=$(( $(date +%s) + ${LB_READY_TIMEOUT:-120} ))
until curl -fsS -o /dev/null "$lb_url" 2>/dev/null; do
    if [[ $(date +%s) -ge $lb_deadline ]]; then
        echo ""
        echo "ERROR: vLLM instances are healthy but $lb_url never returned 200."
        echo "Check the proxy with: podman logs vllm-nginx-lb${STACK_TAG:+-$STACK_TAG}"
        exit 1
    fi
    printf "\r  waiting for %s ..." "$lb_url"
    sleep 2
done
echo ""
echo "Load balancer is serving the backend pool."

echo ""
echo "--- Final status ---"
podman ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "NGINX endpoint: http://localhost:${NGINX_PORT:-8080}"
echo "Run $SCRIPT_DIR/test-setup.sh to validate the full setup."
