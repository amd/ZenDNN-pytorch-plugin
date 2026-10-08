#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# usage: correctness_check.sh <label> <model> [extra start.sh args...]
#
# Starts two vLLM instances with LMCache (shared server, SHM) through the
# harness and sends one greedy recall prompt three times:
#   A: instance 1, cold
#   B: instance 2, restored from LMCache (cross-instance hit)
#   D: instance 1 again, restored from vLLM's own prefix cache
# Passes when every answer names the right code word, B == D, and (unless
# STRICT=0) A == B. Exits 0 on pass, 1 otherwise. The harness recipe (image,
# prefix caching, chunk size, extra vLLM args) comes from the caller's env.
#
# Env: HARNESS_DIR (default: the benchmark-vllm-guidellm skill's scripts/),
# OUT_DIR (default: $HARNESS_DIR/hybrid-checks), STRICT (default 1), STACK_TAG.
set -uo pipefail
label=$1 model=$2
shift 2
# The error and retrieve counts below come from rg; without it they read as 0.
command -v rg >/dev/null || { echo "ERROR: ripgrep (rg) is required" >&2; exit 2; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=${HARNESS_DIR:-$(cd "$here/../../benchmark-vllm-guidellm/scripts" && pwd)}
out=${OUT_DIR:-$H/hybrid-checks}/$label
sfx=${STACK_TAG:+-$STACK_TAG}
mkdir -p "$out"
cd "$H" || exit 1
export NUM_INSTANCES=2 VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export MEM_LIMIT="${MEM_LIMIT:-200g}" VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-16}"

./stop.sh --clean > "$out/stop0.log" 2>&1
if ! ./start.sh --regenerate --timeout 1200 -m "$model" --dtype bfloat16 --lmcache "$@" > "$out/start.log" 2>&1; then
    echo "CHECK[$label]: FAIL (start.sh failed; see $out/start.log)"
    ./stop.sh --clean > "$out/stop1.log" 2>&1
    exit 1
fi

# ~2.6k-token prompt: long enough for whole LMCache chunks at N=640 and 1152.
python3 - "$model" "$out" <<'EOF'
import json, random, sys
model, out = sys.argv[1], sys.argv[2]
random.seed(7)
syl = ["ka", "zor", "vin", "tel", "mu", "rax", "pli", "dom", "sue", "fen", "gor", "lia"]
words = ["".join(random.choice(syl) for _ in range(3)) for _ in range(150)]
facts = "\n".join(f"Fact {i}: the code word for item {i} is {w}." for i, w in enumerate(words, 1))
prompt = (f"Below is a list of facts.\n{facts}\nQuestion: what is the code word for item 77?\n"
          "Answer: the code word for item 77 is")
json.dump({"model": model, "prompt": prompt, "max_tokens": 24, "temperature": 0},
          open(f"{out}/req.json", "w"))
open(f"{out}/expected.txt", "w").write(words[76])
EOF
ask() {
    local t0=$SECONDS
    podman exec -i "vllm-instance-$1$sfx" curl -s -X POST localhost:8000/v1/completions \
        -H 'Content-Type: application/json' --data-binary @- < "$out/req.json" > "$out/$2.json"
    echo "  $2 took $((SECONDS - t0))s"
}
ask 1 A && ask 2 B && ask 1 D

python3 - "$out" "$label" "${STRICT:-1}" <<'EOF'
import json, sys
out, label, strict = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
expected = open(f"{out}/expected.txt").read()
text = {}
for k in "ABD":
    raw = open(f"{out}/{k}.json").read()
    r = json.loads(raw) if raw.strip() else {}
    text[k] = r["choices"][0]["text"] if "choices" in r else f"<no completion: {raw[:200]!r}>"
    print(f"  {k}: {text[k]!r}")
correct = all(expected in text[k] for k in "ABD")
ok = correct and text["B"] == text["D"] and (text["A"] == text["B"] or not strict)
print(f"CHECK[{label}]: {'PASS' if ok else 'FAIL'} (answer '{expected}' in all: {correct}, "
      f"cold==lmcache: {text['A'] == text['B']}, lmcache==vllm_cache: {text['B'] == text['D']})")
open(f"{out}/verdict", "w").write("PASS" if ok else "FAIL")
EOF
echo "  server: $(podman logs "vllm-lmcache-server$sfx" 2>&1 | rg -o 'Stored [0-9]+ tokens|Retrieved [0-9]+ tokens' | sort | uniq -c | tr '\n' ';')"
errors=0
for i in 1 2; do
    n=$(podman logs "vllm-instance-$i$sfx" 2>&1 | rg -c '(ValueError|RuntimeError|AssertionError|Traceback)' || echo 0)
    echo "  instance-$i errors: $n"
    errors=$((errors + n))
done
retrieved=$(podman logs "vllm-lmcache-server$sfx" 2>&1 | rg -c 'Retrieved [0-9]+ tokens' || echo 0)
./stop.sh --clean > "$out/stop1.log" 2>&1
[[ $(cat "$out/verdict") == PASS && $errors == 0 && $retrieved -gt 0 ]]
