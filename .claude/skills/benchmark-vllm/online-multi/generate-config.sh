#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# generate-config.sh -- emit docker-compose.yml + nginx.conf + .env for the
# ONLINE multi-instance quadrant: N cpuset-pinned vLLM `serve` containers
# behind an NGINX round-robin load balancer on one host port.
#
# Adapted (trimmed) from ZenDNN_tools/vllm_multiinstance/generate-config.sh
# and PR #81 (amd/skills vllm-multiinstance). Differences:
#   - container names are prefixed "bench-vllm-" so mem_poll.sh can find them
#   - each instance runs /bench/incontainer_serve.sh (LD_PRELOAD tuning)
#   - HF cache mounts at /root/.cache/huggingface (amdih ubuntu image)
#
# Static per-instance IPs on a dedicated /24 sidestep rootless podman DNS.
#
set -euo pipefail

NUM_INSTANCES="${NUM_INSTANCES:-4}"
CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
# 3-band layout default: nginx 0-15, guidellm client 16-31, vLLM from core 32.
NGINX_CORES="${NGINX_CORES:-0-15}"
VLLM_START_CORE="${VLLM_START_CORE:-32}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
MEM_LIMIT="${MEM_LIMIT:-100g}"
NGINX_MEM_LIMIT="${NGINX_MEM_LIMIT:-5g}"
SHM_SIZE="${SHM_SIZE:-16g}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/amdih/zendnn_zentorch:vllm_v0.24.0_zentorch_v2.11.0.3_ubuntu22.04_2026_ww28}"
VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-90}"
NGINX_PORT="${NGINX_PORT:-8080}"
VLLM_SUBNET="${VLLM_SUBNET:-10.201.0.0/24}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# Optional: explicit per-instance cpusets (semicolon-separated, one per
# instance), e.g. "0-31;32-63". When set, these are used verbatim for both
# `cpuset:` and VLLM_CPU_OMP_THREADS_BIND, NUM_INSTANCES is derived from the
# count, and the naive --start-core contiguous math is bypassed. run_sweep.sh
# passes physical-core-aware slices from check_hardware.sh here (fixes SMT
# sibling pinning on interleaved topologies).
INSTANCE_CPUSETS="${INSTANCE_CPUSETS:-}"
OUTPUT_DIR="generated"
DRY_RUN=false
NO_LIMITS="${NO_LIMITS:-false}"
NO_MEM_LIMIT="${NO_MEM_LIMIT:-false}"

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]
  -n, --num-instances N       (default $NUM_INSTANCES)
  -c, --cores-per-instance N  (default $CORES_PER_INSTANCE)
      --nginx-cores RANGE     (default $NGINX_CORES)
      --start-core N          (default $VLLM_START_CORE)
      --instance-cpusets STR  explicit per-instance cpusets, ';'-separated
                              (e.g. "0-31;32-63"); overrides --start-core math
                              and sets NUM_INSTANCES from the count
  -m, --model NAME            (default $MODEL_NAME)
      --image IMAGE           (default newest amdih tag)
      --kv-cache-space N      (default $VLLM_KV_CACHE_SPACE)
      --nginx-port PORT       (default $NGINX_PORT)
      --max-model-len N       (default $MAX_MODEL_LEN)
      --hf-cache-dir DIR      (default $HF_CACHE_DIR)
      --no-limits             skip ALL cgroup limits: cpuset/mem_limit/shm/caps
                              (rootless/LSF)
      --no-mem-limit          drop only the per-instance mem_limit while keeping
                              cpuset/shm/caps -- use when the memory cgroup
                              OOM-kills a container despite ample free host RAM
  -o, --output-dir DIR        (default generated/)
      --dry-run               print core allocation, write nothing
  -h, --help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-instances) NUM_INSTANCES="$2"; shift 2 ;;
    -c|--cores-per-instance) CORES_PER_INSTANCE="$2"; shift 2 ;;
    --nginx-cores) NGINX_CORES="$2"; shift 2 ;;
    --start-core) VLLM_START_CORE="$2"; shift 2 ;;
    -m|--model) MODEL_NAME="$2"; shift 2 ;;
    --image) VLLM_IMAGE="$2"; shift 2 ;;
    --kv-cache-space) VLLM_KV_CACHE_SPACE="$2"; shift 2 ;;
    --nginx-port) NGINX_PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --instance-cpusets) INSTANCE_CPUSETS="$2"; shift 2 ;;
    --no-limits) NO_LIMITS=true; shift ;;
    --no-mem-limit) NO_MEM_LIMIT=true; shift ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/../scripts" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"

_subnet_base="${VLLM_SUBNET%/*}"; IP_PREFIX="${_subnet_base%.*}"; IP_BASE="${IP_BASE:-10}"
mkdir -p "$HF_CACHE_DIR"; HF_CACHE_DIR="$(cd "$HF_CACHE_DIR" && pwd)"

# Explicit per-instance cpusets override the contiguous --start-core math.
INST_CPUSET_ARR=()
if [[ -n "$INSTANCE_CPUSETS" ]]; then
  IFS=';' read -ra INST_CPUSET_ARR <<< "$INSTANCE_CPUSETS"
  NUM_INSTANCES="${#INST_CPUSET_ARR[@]}"
fi

# i-th instance cpuset: explicit if provided, else contiguous fallback.
instance_cpuset() {
  local i="$1"
  if ((${#INST_CPUSET_ARR[@]})); then
    echo "${INST_CPUSET_ARR[$((i-1))]}"
  else
    local start=$(( VLLM_START_CORE + (i-1)*CORES_PER_INSTANCE ))
    local end=$(( start + CORES_PER_INSTANCE - 1 ))
    echo "${start}-${end}"
  fi
}

last_core=$(( VLLM_START_CORE + NUM_INSTANCES * CORES_PER_INSTANCE - 1 ))

echo "=== Online multi-instance config ==="
if ((${#INST_CPUSET_ARR[@]})); then
  echo "  instances:      $NUM_INSTANCES (explicit cpusets: $INSTANCE_CPUSETS)"
else
  echo "  instances:      $NUM_INSTANCES x ${CORES_PER_INSTANCE} cores  (vllm cores ${VLLM_START_CORE}-${last_core})"
fi
echo "  nginx cores:    $NGINX_CORES   port: $NGINX_PORT"
if $NO_LIMITS; then
  echo "  mem limit:      none (--no-limits: all cgroup limits dropped)"
  echo "  !! WARNING: --no-limits also drops cpuset pinning -- instances are NOT" >&2
  echo "  !!          core-pinned, so throughput/latency numbers are unreliable" >&2
  echo "  !!          and instances may contend for the same cores. Use only when" >&2
  echo "  !!          cgroups are unavailable (rootless/LSF); prefer --no-mem-limit." >&2
elif $NO_MEM_LIMIT; then
  echo "  mem limit:      none (--no-mem-limit: kept cpuset/shm/caps)"
else
  echo "  mem limit:      $MEM_LIMIT / instance"
fi
echo "  model:          $MODEL_NAME"
echo "  image:          $VLLM_IMAGE"
echo "  hf cache:       $HF_CACHE_DIR"
echo "  bench scripts:  $BENCH_SCRIPTS_DIR (mounted at /bench)"
echo "===================================="

if $DRY_RUN; then
  for i in $(seq 1 "$NUM_INSTANCES"); do
    echo "  bench-vllm-instance-$i: cores $(instance_cpuset "$i") -> ${IP_PREFIX}.$((IP_BASE+i))"
  done
  echo "  nginx: cores $NGINX_CORES -> ${IP_PREFIX}.${IP_BASE}"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/.env" <<EOF
MODEL_NAME=$MODEL_NAME
HF_TOKEN=$HF_TOKEN
MEM_LIMIT=$MEM_LIMIT
HF_CACHE_DIR=$HF_CACHE_DIR
NGINX_PORT=$NGINX_PORT
EOF

# --- nginx.conf ---
cat > "$OUTPUT_DIR/nginx.conf" <<'NGINX_HEADER'
user root;
pid /tmp/nginx.pid;
events { worker_connections 4096; }
http {
    upstream vllm_backend {
NGINX_HEADER
for i in $(seq 1 "$NUM_INSTANCES"); do
  echo "        server ${IP_PREFIX}.$(( IP_BASE + i )):8000 max_fails=10 fail_timeout=10s;" >> "$OUTPUT_DIR/nginx.conf"
done
cat >> "$OUTPUT_DIR/nginx.conf" <<'NGINX_BODY'
    }
    log_format up '$remote_addr [$time_local] "$request" $status upstream=$upstream_addr rt=$upstream_response_time';
    access_log /var/log/nginx/access.log up;
    error_log /var/log/nginx/error.log warn;
    server {
        listen 80; server_name _;
        proxy_connect_timeout 2s; proxy_send_timeout 300s;
        proxy_read_timeout 300s; send_timeout 300s;
        proxy_buffer_size 128k; proxy_buffers 8 256k; proxy_busy_buffers_size 512k;
        location /health { proxy_pass http://vllm_backend/health; proxy_set_header Connection ""; }
        location / {
            proxy_pass http://vllm_backend; proxy_http_version 1.1;
            proxy_set_header Host $host; proxy_set_header Connection "";
            proxy_buffering off; proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            client_max_body_size 50M;
        }
        location /nginx_status { stub_status on; access_log off; }
    }
}
NGINX_BODY

# --- docker-compose.yml ---
COMPOSE="$OUTPUT_DIR/docker-compose.yml"
cat > "$COMPOSE" <<EOF
version: '3.8'
services:
EOF

DEPENDS_LIST=""
for i in $(seq 1 "$NUM_INSTANCES"); do
  inst_cs="$(instance_cpuset "$i")"
  DEPENDS_LIST="${DEPENDS_LIST}      - vllm-$i
"
  cat >> "$COMPOSE" <<EOF
  vllm-$i:
    image: $VLLM_IMAGE
    container_name: bench-vllm-instance-$i
EOF
  if ! $NO_LIMITS; then
    cat >> "$COMPOSE" <<EOF
    cpuset: "$inst_cs"
EOF
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
      - HF_HOME=/root/.cache/huggingface
      - HF_TOKEN=\${HF_TOKEN}
      - VLLM_CPU_KVCACHE_SPACE=$VLLM_KV_CACHE_SPACE
      - TORCHINDUCTOR_FREEZING=1
      - VLLM_CPU_OMP_THREADS_BIND=$inst_cs
    volumes:
      - $HF_CACHE_DIR:/root/.cache/huggingface:z
      - $BENCH_SCRIPTS_DIR:/bench:ro,z
    entrypoint:
      - bash
      - /bench/incontainer_serve.sh
    command:
      - --model
      - \${MODEL_NAME:-$MODEL_NAME}
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --max-model-len
      - "$MAX_MODEL_LEN"
      - --trust-remote-code
EOF
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    for arg in $EXTRA_VLLM_ARGS; do printf '      - %s\n' "$arg" >> "$COMPOSE"; done
  fi
  cat >> "$COMPOSE" <<EOF
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 30s
      retries: 20
      start_period: 900s
    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.$(( IP_BASE + i ))

EOF
done

cat >> "$COMPOSE" <<EOF
  nginx:
    image: docker.io/library/nginx:alpine
    container_name: bench-vllm-nginx-lb
EOF
if ! $NO_LIMITS; then
  cat >> "$COMPOSE" <<EOF
    cpuset: "$NGINX_CORES"
    mem_limit: $NGINX_MEM_LIMIT
EOF
fi
cat >> "$COMPOSE" <<EOF
    ports:
      - "\${NGINX_PORT:-$NGINX_PORT}:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro,Z
    depends_on:
${DEPENDS_LIST}    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.${IP_BASE}
    restart: unless-stopped

networks:
  vllm-network:
    driver: bridge
    ipam:
      config:
        - subnet: $VLLM_SUBNET
EOF

echo "Written: $OUTPUT_DIR/{docker-compose.yml,nginx.conf,.env}"
