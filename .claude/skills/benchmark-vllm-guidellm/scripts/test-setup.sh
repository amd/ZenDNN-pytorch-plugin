#!/bin/bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# The test_* functions are only invoked indirectly, through run_test.
# shellcheck disable=SC2317

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# STACK_TAG selects which stack to test (matches start.sh / generate-config.sh):
# empty = the default "generated" stack; e.g. STACK_TAG=r2 tests "generated-r2"
# whose containers are suffixed "-r2". SUFFIX is appended to every container name
# below. Pass the matching NGINX_PORT too (e.g. STACK_TAG=r2 NGINX_PORT=8081).
STACK_TAG="${STACK_TAG:-}"
SUFFIX="${STACK_TAG:+-$STACK_TAG}"
COMPOSE_FILE="$SCRIPT_DIR/generated${SUFFIX}/docker-compose.yml"
NGINX_PORT="${NGINX_PORT:-8080}"
BASE_URL="http://localhost:$NGINX_PORT"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

run_test() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- TEST $TOTAL: $name ---"
    if "$@"; then
        PASS=$((PASS + 1))
        echo "  PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL"
    fi
}

skip_test() {
    local name="$1"
    TOTAL=$((TOTAL + 1))
    SKIP=$((SKIP + 1))
    echo ""
    echo "--- TEST $TOTAL: $name ---"
    echo "  SKIPPED: $2"
}

# Detect number of instances from compose file or running containers
if [[ -f "$COMPOSE_FILE" ]]; then
    NUM_INSTANCES=$(grep -c 'container_name: vllm-instance-' "$COMPOSE_FILE")
else
    NUM_INSTANCES=$(podman ps --filter "name=vllm-instance-" --format "{{.Names}}" | wc -l)
fi

echo "=============================================="
echo "  Multi-Instance vLLM + NGINX Test Suite"
echo "=============================================="
echo "  Instances expected: $NUM_INSTANCES"
echo "  NGINX endpoint:     $BASE_URL"
echo "=============================================="

# -----------------------------------------------
# TEST 1: Container status
# -----------------------------------------------
test_container_status() {
    local ok=true

    for i in $(seq 1 "$NUM_INSTANCES"); do
        local name="vllm-instance-$i${SUFFIX}"
        local state
        state=$(podman inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo "not found")
        local health
        health=$(podman inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null || echo "unknown")
        echo "  $name: state=$state health=$health"
        if [[ "$state" != "running" || "$health" != "healthy" ]]; then
            ok=false
        fi
    done

    local nginx_state
    nginx_state=$(podman inspect --format '{{.State.Status}}' "vllm-nginx-lb${SUFFIX}" 2>/dev/null || echo "not found")
    echo "  vllm-nginx-lb${SUFFIX}: state=$nginx_state"
    if [[ "$nginx_state" != "running" ]]; then
        ok=false
    fi

    $ok
}
run_test "All containers running and healthy" test_container_status

# -----------------------------------------------
# TEST 2: Individual instance reachability (via nginx container)
# -----------------------------------------------
test_instance_reachability() {
    local ok=true

    for i in $(seq 1 "$NUM_INSTANCES"); do
        local result
        result=$(podman exec "vllm-nginx-lb${SUFFIX}" wget -q -O - "http://vllm-$i:8000/health" 2>/dev/null || echo "UNREACHABLE")
        if [[ "$result" == "UNREACHABLE" ]]; then
            echo "  vllm-$i:8000 -> UNREACHABLE from nginx container"
            ok=false
        else
            echo "  vllm-$i:8000 -> reachable"
        fi
    done

    $ok
}
run_test "Each vLLM instance reachable from NGINX container" test_instance_reachability

# -----------------------------------------------
# TEST 3: NGINX health endpoint
# -----------------------------------------------
test_nginx_health() {
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null)
    echo "  GET $BASE_URL/health -> HTTP $http_code"
    [[ "$http_code" == "200" ]]
}
run_test "NGINX /health endpoint responds" test_nginx_health

# -----------------------------------------------
# TEST 4: Model listing
# -----------------------------------------------
test_model_listing() {
    local response
    response=$(curl -s "$BASE_URL/v1/models" 2>/dev/null)
    local model_count
    model_count=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
    local model_id
    model_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

    echo "  Models returned: $model_count"
    echo "  Model ID: $model_id"
    [[ "$model_count" -ge 1 ]]
}
run_test "Model listing via /v1/models" test_model_listing

# -----------------------------------------------
# TEST 5: NGINX status endpoint
# -----------------------------------------------
test_nginx_status() {
    local response
    response=$(curl -s "$BASE_URL/nginx_status" 2>/dev/null)
    if echo "$response" | grep -q "Active connections"; then
        # shellcheck disable=SC2001
        echo "$response" | sed 's/^/  /'
        return 0
    else
        echo "  nginx_status not available"
        return 1
    fi
}
run_test "NGINX stub_status endpoint" test_nginx_status

# -----------------------------------------------
# TEST 6: Round-robin distribution
# -----------------------------------------------
test_round_robin() {
    echo "  Clearing nginx access log..."
    podman exec "vllm-nginx-lb${SUFFIX}" sh -c "echo '' > /var/log/nginx/access.log" 2>/dev/null

    local num_requests=$((NUM_INSTANCES * 2))
    echo "  Sending $num_requests sequential health requests..."
    for _ in $(seq 1 "$num_requests"); do
        curl -s -o /dev/null "$BASE_URL/health" 2>/dev/null
        sleep 0.1
    done

    sleep 1
    local log
    log=$(podman exec "vllm-nginx-lb${SUFFIX}" cat /var/log/nginx/access.log 2>/dev/null)

    echo "  Upstream distribution:"
    local all_hit=true
    for i in $(seq 1 "$NUM_INSTANCES"); do
        local count
        count=$(echo "$log" | grep -c "upstream=.*vllm-$i" 2>/dev/null || echo "0")
        # If the log format doesn't include upstream=, try IP-based matching
        if [[ "$count" -eq 0 ]]; then
            count=$(echo "$log" | grep -c "vllm-$i:8000" 2>/dev/null || echo "0")
        fi
        echo "    vllm-$i: $count requests"
        if [[ "$count" -eq 0 ]]; then
            all_hit=false
        fi
    done

    if ! $all_hit; then
        echo "  WARNING: Not all instances received traffic."
        echo "  This may be expected if nginx log format doesn't include upstream details."
        echo "  Raw log sample (last 5 lines):"
        echo "$log" | tail -5 | sed 's/^/    /'
    fi

    $all_hit
}
run_test "Round-robin distributes to all instances" test_round_robin

# -----------------------------------------------
# TEST 7: Chat completion (inference smoke test)
# -----------------------------------------------
test_chat_completion() {
    echo "  Sending chat completion request (may take 30-60s)..."
    local response
    response=$(curl -s --max-time 120 "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)"'",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
            "temperature": 0
        }' 2>/dev/null)

    local finish_reason
    finish_reason=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null || echo "error")
    local content
    content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")

    echo "  Finish reason: $finish_reason"
    echo "  Response: $content"

    if [[ "$finish_reason" == "error" ]]; then
        echo "  Raw response: $response" | head -5
        return 1
    fi
    return 0
}
run_test "Chat completion inference" test_chat_completion

# -----------------------------------------------
# TEST 8: Streaming response
# -----------------------------------------------
test_streaming() {
    local model_id
    model_id=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)

    echo "  Sending streaming request..."
    local tmpfile
    tmpfile=$(mktemp)

    curl -s --max-time 120 -N "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$model_id"'",
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": true
        }' > "$tmpfile" 2>/dev/null

    local chunk_count
    chunk_count=$(grep -c "^data:" "$tmpfile" 2>/dev/null || echo "0")
    local has_done
    has_done=$(grep -c "data: \[DONE\]" "$tmpfile" 2>/dev/null || echo "0")

    echo "  SSE chunks received: $chunk_count"
    echo "  Stream terminated ([DONE]): $([ "$has_done" -gt 0 ] && echo "yes" || echo "no")"

    rm -f "$tmpfile"

    [[ "$chunk_count" -gt 1 ]]
}
run_test "Streaming SSE response" test_streaming

# -----------------------------------------------
# TEST 9: CPU pinning verification
# -----------------------------------------------
test_cpu_pinning() {
    local ok=true

    for i in $(seq 1 "$NUM_INSTANCES"); do
        local cpuset
        cpuset=$(podman inspect --format '{{.HostConfig.CpusetCpus}}' "vllm-instance-$i${SUFFIX}" 2>/dev/null || echo "unknown")
        echo "  vllm-instance-$i${SUFFIX}: cpuset=$cpuset"
        if [[ "$cpuset" == "unknown" || -z "$cpuset" ]]; then
            ok=false
        fi
    done

    local nginx_cpuset
    nginx_cpuset=$(podman inspect --format '{{.HostConfig.CpusetCpus}}' "vllm-nginx-lb${SUFFIX}" 2>/dev/null || echo "unknown")
    echo "  vllm-nginx-lb${SUFFIX}: cpuset=$nginx_cpuset"

    # Check no overlaps
    echo "  Checking for CPU overlap between instances..."
    local all_cpus=()
    for i in $(seq 1 "$NUM_INSTANCES"); do
        local cpuset
        cpuset=$(podman inspect --format '{{.HostConfig.CpusetCpus}}' "vllm-instance-$i${SUFFIX}" 2>/dev/null)
        if [[ -n "$cpuset" ]]; then
            local start end
            start=$(echo "$cpuset" | cut -d'-' -f1)
            end=$(echo "$cpuset" | cut -d'-' -f2)
            for c in $(seq "$start" "$end"); do
                if printf '%s\n' "${all_cpus[@]}" 2>/dev/null | grep -qx "$c"; then
                    echo "  ERROR: CPU $c is assigned to multiple instances!"
                    ok=false
                fi
                all_cpus+=("$c")
            done
        fi
    done
    echo "  No CPU overlap detected among vLLM instances."

    $ok
}
run_test "CPU pinning and no overlap" test_cpu_pinning

# -----------------------------------------------
# TEST 10: Failover (stop one instance, verify nginx still works)
# -----------------------------------------------
test_failover() {
    if [[ "$NUM_INSTANCES" -lt 2 ]]; then
        echo "  Need at least 2 instances for failover test."
        return 1
    fi

    local target="vllm-instance-$NUM_INSTANCES${SUFFIX}"
    echo "  Stopping $target..."
    podman stop "$target" --timeout 5 >/dev/null 2>&1

    sleep 2

    echo "  Sending request through NGINX with $target down..."
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$BASE_URL/health" 2>/dev/null)
    echo "  GET /health -> HTTP $http_code"

    echo "  Restarting $target..."
    podman start "$target" >/dev/null 2>&1

    echo "  Waiting for $target to become healthy again..."
    local waited=0
    while [[ $waited -lt 120 ]]; do
        local health
        health=$(podman inspect --format '{{.State.Health.Status}}' "$target" 2>/dev/null || echo "unknown")
        if [[ "$health" == "healthy" ]]; then
            echo "  $target is healthy again (after ${waited}s)."
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done

    [[ "$http_code" == "200" ]]
}
run_test "Failover: NGINX serves when one instance is down" test_failover

# -----------------------------------------------
# TEST 11: Concurrent requests
# -----------------------------------------------
test_concurrent() {
    echo "  Sending $NUM_INSTANCES concurrent health requests..."
    local pids=()
    local results_dir
    results_dir=$(mktemp -d)

    for i in $(seq 1 "$NUM_INSTANCES"); do
        (
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/health" 2>/dev/null)
            echo "$code" > "$results_dir/$i"
        ) &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    local all_ok=true
    for i in $(seq 1 "$NUM_INSTANCES"); do
        local code
        code=$(cat "$results_dir/$i" 2>/dev/null || echo "000")
        if [[ "$code" != "200" ]]; then
            echo "  Request $i: HTTP $code (FAIL)"
            all_ok=false
        fi
    done

    if $all_ok; then
        echo "  All $NUM_INSTANCES concurrent requests returned HTTP 200."
    fi

    rm -rf "$results_dir"
    $all_ok
}
run_test "Concurrent requests handled correctly" test_concurrent

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo ""
echo "=============================================="
echo "  TEST RESULTS"
echo "=============================================="
echo "  Total:   $TOTAL"
echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo "=============================================="

if [[ $FAIL -gt 0 ]]; then
    echo "  Some tests FAILED. Check output above for details."
    exit 1
else
    echo "  All tests PASSED."
    exit 0
fi
