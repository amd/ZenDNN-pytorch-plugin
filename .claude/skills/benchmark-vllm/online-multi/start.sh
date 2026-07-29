#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# start.sh -- bring up the generated online-multi stack (N vLLM + NGINX) and
# wait for the load balancer to report healthy.
#
#   ./start.sh [--generated-dir DIR] [--nginx-port PORT] [--timeout SECONDS]
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/../scripts/common.sh"

GENERATED_DIR="${SELF_DIR}/generated"
NGINX_PORT="${NGINX_PORT:-8080}"
TIMEOUT="${HEALTH_TIMEOUT:-1200}"
INSTANCES=0   # when >0, wait for all N instance containers to be healthy first

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --nginx-port) NGINX_PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --instances) INSTANCES="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--generated-dir DIR] [--nginx-port PORT] [--timeout S] [--instances N]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

detect_runtime || exit 1
[[ -z "$CCOMPOSE" ]] && { echo "ERROR: no compose (podman-compose / 'podman compose') available." >&2; exit 1; }
[[ -f "${GENERATED_DIR}/docker-compose.yml" ]] || { echo "ERROR: ${GENERATED_DIR}/docker-compose.yml not found; run generate-config.sh first." >&2; exit 1; }

echo "Starting stack with: $CCOMPOSE (dir: $GENERATED_DIR)"
( cd "$GENERATED_DIR" && $CCOMPOSE up -d )

deadline=$(( SECONDS + TIMEOUT ))

# Gate on ALL N backends being healthy. The LB /health passes as soon as ONE
# backend is up (nginx proxy_next_upstream), which would let the client start
# against a half-warmed cluster; count healthy instance containers instead.
if (( INSTANCES > 0 )); then
  echo "Waiting for all ${INSTANCES} instance containers to be healthy (timeout ${TIMEOUT}s) ..."
  until (( $("$CRUNTIME" ps --filter "name=bench-vllm-instance-" --filter "health=healthy" --format '{{.Names}}' 2>/dev/null | grep -c . ) >= INSTANCES )); do
    if (( SECONDS >= deadline )); then
      echo "ERROR: only $("$CRUNTIME" ps --filter "name=bench-vllm-instance-" --filter "health=healthy" --format '{{.Names}}' 2>/dev/null | grep -c .)/${INSTANCES} instances healthy in ${TIMEOUT}s." >&2
      ( cd "$GENERATED_DIR" && $CCOMPOSE ps ) || true
      exit 5
    fi
    sleep 10
  done
  echo "All ${INSTANCES} instances healthy."
fi

TARGET="http://localhost:${NGINX_PORT}"
echo "Waiting for ${TARGET}/health (timeout ${TIMEOUT}s) ..."
until curl -fsS "${TARGET}/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: LB not healthy in ${TIMEOUT}s." >&2
    ( cd "$GENERATED_DIR" && $CCOMPOSE ps ) || true
    exit 5
  fi
  sleep 5
done
echo "Stack healthy at ${TARGET}"
