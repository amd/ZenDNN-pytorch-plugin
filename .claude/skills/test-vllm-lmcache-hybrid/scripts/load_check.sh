#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# usage: load_check.sh <label> <model>
#
# One run_sweep.sh pass (LMCache SHM, N instances, GuideLLM) for <model>, then
# a verdict from the GuideLLM results and the dumped container logs. Passes
# when every benchmark has 0 errored requests, no instance logged an
# AssertionError / EngineDeadError / HTTP 500 before teardown, and the LMCache
# server retrieved at least once. Exits 0 on pass, 1 otherwise. The harness
# recipe comes from the caller's env.
#
# Env: HARNESS_DIR (default: the benchmark-vllm-guidellm skill's scripts/),
# N (default 5), WORKLOAD (default chat_shared), GUIDELLM_RATES (default
# [32,64]), GUIDELLM_MAX_SECONDS (default 300), TAG (default <label> up to the
# first '-'). <label>-zentorch-<tag> must stay <= 30 chars (Ansible test_name
# limit).
set -uo pipefail
label=$1 model=$2
# The crash and retrieve counts below come from rg; without it they read as 0.
command -v rg >/dev/null || { echo "ERROR: ripgrep (rg) is required" >&2; exit 2; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=${HARNESS_DIR:-$(cd "$here/../../benchmark-vllm-guidellm/scripts" && pwd)}
n=${N:-5} workload=${WORKLOAD:-chat_shared} tag=${TAG:-${label%%-*}}
cd "$H" || exit 1
export GUIDELLM_RATES="${GUIDELLM_RATES:-[32,64]}" GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-300}"
# The harness default KV space (90 GiB at a 200 GiB limit) plus Qwen3.6 bf16's
# 72 GB of weights gets the worker OOM-killed during KV allocation.
export MEM_LIMIT="${MEM_LIMIT:-200g}" VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-16}" TEST_NAME_PREFIX="$label"

# Clean up first: stop.sh dumps a leftover stack's logs, which must not count as this run's.
./stop.sh --clean > /dev/null 2>&1
marker=$(mktemp)
./run_sweep.sh --lmcache -N "$n" -w "$workload" -m "$model | $tag" > "sweep-logs/load-$label.log" 2>&1
echo "LOAD[$label]: run_sweep rc=$? (log: $H/sweep-logs/load-$label.log)"

fail=0
bench=$(find vllm-cpu-perf-eval/results/llm -path "*/$workload-*-$label-*/external-endpoint/benchmarks.json" \
    -newer "$marker" 2>/dev/null | sort | awk 'END {print}')
if [[ -z $bench ]]; then
    echo "  guidellm: no benchmarks.json"
    fail=1
else
    python3 - "$bench" <<'EOF' || fail=1
import json, sys
d = json.load(open(sys.argv[1]))
bad = False
for bm in d["benchmarks"]:
    t = bm.get("request_totals") or bm.get("metrics", {}).get("request_totals") or {}
    print(f"  guidellm: successful={t.get('successful')} errored={t.get('errored')} incomplete={t.get('incomplete')}")
    bad |= (t.get("errored") or 0) > 0 or not t.get("successful")
sys.exit(1 if bad else 0)
EOF
fi

# stop.sh dumps container logs once per teardown; take this run's dump.
logs=""
for d in $(find sweep-logs -mindepth 2 -maxdepth 2 -type d -name container-logs -newer "$marker" | sort); do
    [[ -f $d/vllm-instance-$n.log ]] && logs=$d && break
done
rm -f "$marker"
if [[ -z $logs ]]; then
    echo "  container logs: not found"
    exit 1
fi
echo "  container logs: $H/$logs"
for f in "$logs"/vllm-instance-*.log; do
    # Teardown force-kills EngineCore, which logs EngineDeadError; count only what precedes it.
    live=$(awk '/\[shutdown\] API server: shutdown triggered/ {exit} {print}' "$f")
    crashes=$(rg -c 'AssertionError|EngineDeadError|EngineCore encountered a fatal error|" 500 ' <<< "$live" || echo 0)
    hit=$(rg -o 'External prefix cache hit rate: [0-9.]+%' "$f" | awk 'END {print $NF}')
    echo "  $(basename "$f" .log): crashes=$crashes lmcache_hit=${hit:-none}"
    [[ $crashes == 0 ]] || fail=1
done
retrieved=$(rg -c 'Retrieved [0-9]+ tokens' "$logs/vllm-lmcache-server.log" || echo 0)
echo "  server retrieves: $retrieved"
[[ $retrieved -gt 0 ]] || fail=1
echo "LOAD[$label]: $([[ $fail == 0 ]] && echo PASS || echo FAIL)"
exit $fail
