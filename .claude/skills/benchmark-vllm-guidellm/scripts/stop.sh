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

# Compose network name (mirrors start.sh): "<project>_vllm-network" where the
# project is the basename of the generated dir. Used for the force-cleanup that
# releases netavark IPAM leases (see clear_stale_leases below).
COMPOSE_PROJECT="$(basename "$GENERATED_DIR")"
COMPOSE_NETWORK="${COMPOSE_PROJECT}_vllm-network"

# Ensure podman uses an OCI runtime that actually exists on this host before any
# podman/podman-compose call (avoids 'default OCI runtime "runc" not found').
# shellcheck source=podman-runtime-setup.sh
# source "$SCRIPT_DIR/podman-runtime-setup.sh"
# ensure_oci_runtime

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Stop the multi-instance vLLM + NGINX stack.

Options:
  --clean         Remove volumes (model caches) after stopping
  --purge         Remove volumes AND delete generated config files
  --log-dir DIR   Directory to dump container logs into before teardown.
                  Defaults to \$LOG_DIR if set, else
                  sweep-logs/<timestamp>/container-logs/.
                  Pass --no-logs to skip the dump entirely.
  --no-logs       Skip dumping container logs before teardown.
  -h, --help      Show this help
EOF
    exit 0
}

CLEAN=false
PURGE=false
CLI_LOG_DIR=""
DUMP_LOGS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=true; shift ;;
        --purge) PURGE=true; CLEAN=true; shift ;;
        --log-dir) CLI_LOG_DIR="$2"; shift 2 ;;
        --no-logs) DUMP_LOGS=false; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ----------------------------------------------------------------------------
# Dump podman logs for every stack container BEFORE we stop/remove them.
# Matches vllm-* and nginx* containers (the names used by
# generate-config.sh). Includes stopped containers so we still capture exit logs.
# ----------------------------------------------------------------------------
dump_container_logs() {
    if ! $DUMP_LOGS; then
        return 0
    fi

    local containers
    containers=$(podman ps -a \
        --filter "name=vllm-" \
        --filter "name=nginx" \
        --format "{{.Names}}" | sort -u)

    if [[ -z "$containers" ]]; then
        echo "  No vllm-* or nginx* containers found -- skipping log dump."
        return 0
    fi

    local out_dir
    if [[ -n "$CLI_LOG_DIR" ]]; then
        out_dir="$CLI_LOG_DIR"
    elif [[ -n "${LOG_DIR:-}" ]]; then
        out_dir="$LOG_DIR/container-logs"
    else
        out_dir="$SCRIPT_DIR/sweep-logs/$(date +%Y%m%d-%H%M%S)/container-logs"
    fi
    mkdir -p "$out_dir"

    echo "--- Dumping container logs to $out_dir ---"
    local c
    for c in $containers; do
        local log_file="$out_dir/${c}.log"
        if podman logs "$c" > "$log_file" 2>&1; then
            local size
            size=$(wc -c < "$log_file" | tr -d ' ')
            echo "  saved $log_file ($size bytes)"
        else
            echo "  WARNING: failed to dump logs for $c (see $log_file)"
        fi
    done
}

# ----------------------------------------------------------------------------
# Force-remove every stack container by name (vllm-* / nginx*),
# including ones that podman-compose down might miss (e.g. containers left
# behind when a prior run was SIGKILLed mid-startup). Runs synchronously so the
# subsequent network removal isn't blocked by still-attached endpoints.
# ----------------------------------------------------------------------------
force_remove_stack_containers() {
    local ids
    ids=$(podman ps -a \
        --filter "name=vllm-" \
        --filter "name=nginx" \
        --format "{{.ID}}" | sort -u)
    [[ -z "$ids" ]] && return 0
    echo "  Force-removing leftover stack containers..."
    # Single call with --depend: nginx-lb depends on the vllm-instance-* it
    # proxies, so removing them individually in name order fails with
    # "has dependent containers which must be removed before it". --depend
    # removes dependents first, and one call lets podman resolve the order.
    # shellcheck disable=SC2086
    podman rm -f --depend $ids >/dev/null 2>&1 || true
}

# ----------------------------------------------------------------------------
# Remove the shared SHM dir generate-config.sh creates for LMCache stacks. It
# holds the LMCache server's L1 pool and vLLM's segments, and tmpfs keeps their
# pages in RAM until the files are deleted. The next start recreates it.
# ----------------------------------------------------------------------------
remove_lmcache_shm_dir() {
    local dir="/dev/shm/lmc-shared${STACK_TAG:+-$STACK_TAG}"
    [[ -d "$dir" ]] || return 0
    echo "  Removing LMCache shared SHM dir $dir..."
    rm -rf "$dir"
}

# ----------------------------------------------------------------------------
# Remove the compose network so netavark drops its per-network IPAM bucket.
# This is what actually frees fixed IPs (e.g. NGINX's 10.201.0.10). Must run
# AFTER all containers are detached, or netavark keeps the lease bucket alive.
# ----------------------------------------------------------------------------
force_remove_network() {
    podman network exists "$COMPOSE_NETWORK" 2>/dev/null || return 0
    echo "  Removing network $COMPOSE_NETWORK..."
    podman network rm -f "$COMPOSE_NETWORK" >/dev/null 2>&1 || true
}

# ----------------------------------------------------------------------------
# Detect and clear leaked netavark IPAM leases. When a container holding a
# fixed IP is SIGKILLed or interrupted mid-teardown, netavark fails to release
# its lease in runroot/networks/ipam.db. The dead container ID then keeps
# owning the IP, and the next start fails with:
#   "IPAM error: requested ip address <ip> is already allocated to container <id>"
# Removing the network does NOT free the lease (verified: the orphan survives
# `podman network rm`). The only reliable fix is to reset the lease db, which
# netavark rebuilds from scratch on the next network op.
#
# Deleting ipam.db drops leases for EVERY network, so we only do it when no
# containers remain on the host (safe: nothing is relying on a live lease).
# Otherwise we warn with the exact manual reset command.
# ----------------------------------------------------------------------------
clear_stale_leases() {
    local run_root ipam_db
    run_root="$(podman info --format '{{.Store.RunRoot}}' 2>/dev/null)"
    [[ -z "$run_root" ]] && return 0
    ipam_db="$run_root/networks/ipam.db"
    [[ -f "$ipam_db" ]] || return 0

    # Container IDs (full length) that still exist.
    local live_ids
    live_ids="$(podman ps -aq --no-trunc 2>/dev/null)"

    # Flag any 64-hex lease owner recorded in the db that no longer exists.
    local stale_found=false id
    while IFS= read -r id; do
        [[ -z "$id" ]] && continue
        if ! grep -qxF "$id" <<<"$live_ids"; then
            stale_found=true
            break
        fi
    done < <(strings "$ipam_db" 2>/dev/null | grep -oE '[0-9a-f]{64}' | sort -u)

    $stale_found || return 0

    local remaining
    remaining="$(podman ps -aq 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$remaining" == "0" ]]; then
        echo "  Clearing orphaned netavark IPAM lease(s) (no containers present)..."
        rm -f "$ipam_db"
        rm -f "$run_root"/networks/aardvark-dns/* 2>/dev/null || true
        echo "  netavark IPAM db reset; leases rebuild clean on next start."
    else
        echo "  WARNING: netavark IPAM db has orphaned lease(s) but $remaining container(s)"
        echo "           are still present, so it was not reset automatically ($ipam_db)."
        echo "           If the next start fails with an 'IPAM error: ... already"
        echo "           allocated' message, reset the lease db while podman is idle:"
        echo "             podman rm -f -a"
        echo "             rm -f '$ipam_db'"
        echo "             rm -f '$run_root'/networks/aardvark-dns/*"
    fi
}

dump_container_logs

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "No compose file found at $COMPOSE_FILE."
    echo "Attempting to stop containers by name..."
    force_remove_stack_containers
    remove_lmcache_shm_dir
    force_remove_network
    clear_stale_leases
    echo "Done."
    exit 0
fi

echo "--- Stopping multi-instance vLLM stack ---"

cd "$GENERATED_DIR"

if $CLEAN; then
    echo "  Stopping containers and removing volumes..."
    podman-compose down -v || true
else
    echo "  Stopping containers (volumes preserved)..."
    podman-compose down || true
fi

cd "$SCRIPT_DIR"

# podman-compose down can miss containers/networks left by a SIGKILLed run and
# does not guarantee netavark releases fixed-IP leases. Force-remove any
# leftover stack containers, drop the compose network (frees the IPAM bucket),
# and warn about any lease that survived.
force_remove_stack_containers
remove_lmcache_shm_dir
force_remove_network
clear_stale_leases

echo ""
echo "--- Remaining vLLM containers ---"
remaining=$(podman ps -a --filter "name=vllm-" --format "{{.Names}}" | wc -l)
if [[ "$remaining" -eq 0 ]]; then
    echo "  None (clean shutdown)."
else
    podman ps -a --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}"
fi

if $PURGE; then
    echo ""
    echo "--- Removing generated config ---"
    rm -rf "$GENERATED_DIR"
    echo "  Deleted $GENERATED_DIR"
fi

echo ""
echo "Stack stopped."
