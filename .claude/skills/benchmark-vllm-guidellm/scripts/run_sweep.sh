#!/bin/bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# ============================================================================
# Multi-model, multi-variant (native + zentorch) benchmark sweep
#
# For each (variant, model):
#   1. Stop any running stack
#   2. Start the stack with the chosen image (zentorch or native) and model
#   3. Run guidellm benchmark via ansible at rates [32, 64, 96]
#   4. Stop the stack
#
# Results are tagged: test_name=<PREFIX>_<variant>_<sanitized-model>
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# 0. ARG PARSING
# ----------------------------------------------------------------------------
DRY_RUN=false
CLI_MODELS=()
CLI_MODELS_DIR=""
CLI_NUM_INSTANCES=""
CLI_QUANT=false
CLI_TORCHAO=false
CLI_NO_MEM_LIMIT=false
CLI_NO_LIMITS=false
CLI_LMCACHE=false
CLI_VARIANT=""
CLI_PROFILE=false
CLI_WORKLOADS=()
CLI_DTYPE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true; shift ;;
        --lmcache)
            CLI_LMCACHE=true; shift ;;
        --no-limits)
            CLI_NO_LIMITS=true; shift ;;
        --native)
            CLI_VARIANT="native"; shift ;;
        --zentorch)
            CLI_VARIANT="zentorch"; shift ;;
        -m|--model)
            CLI_MODELS+=("$2"); shift 2 ;;
        --models-dir)
            CLI_MODELS_DIR="$2"; shift 2 ;;
        -N|--num-instances)
            CLI_NUM_INSTANCES="$2"; shift 2 ;;
        --quant)
            CLI_QUANT=true; shift ;;
        --torchao)
            CLI_TORCHAO=true; shift ;;
        --no-mem-limit)
            CLI_NO_MEM_LIMIT=true; shift ;;
        --profile)
            CLI_PROFILE=true; shift ;;
        -w|--workload)
            CLI_WORKLOADS+=("$2"); shift 2 ;;
        -d|--dtype)
            CLI_DTYPE="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Sweeps every (variant, model) pair and runs the guidellm benchmark.

Options:
  -m, --model ENTRY      Model to sweep, as "path-or-repo | short-tag" (the
                         tag is optional). Repeat to add several. Overrides
                         the built-in MODELS list when given at least once.
  --models-dir DIR       Host dir holding local model folders. Bind-mounted
                         into each container at the same path; model ENTRY
                         left-sides are resolved relative to it.
  -N, --num-instances N  Number of vLLM instances (also seeds the default
                         test_name prefix EPYC<N>).
  --native               Sweep only the native (non-zentorch) variant.
                         Overrides the VARIANTS array.
  --zentorch             Sweep only the zentorch variant (default).
                         Overrides the VARIANTS array.
  --quant                Quantized-model run (default: llm-compressor format).
                         Uses the base image as-is, NO torchao install. Models
                         may be HF repo ids (pulled from the hub, needs
                         HF_TOKEN) or local dirs via --models-dir (or MODELS_DIR
                         env).
  --torchao              Only meaningful with --quant. Builds the derived
                         torchao-enabled image (passes --torchao to start.sh).
                         Use for models with quant_method=torchao. Without this,
                         --quant does NOT change the image.
  --no-mem-limit         Drop the per-instance mem_limit while keeping cpuset
                         pinning (forwarded to start.sh). Use when the memory
                         cgroup OOM-kills containers despite ample free host RAM.
  --profile              Enable the vLLM torch profiler. Forwards --profile to
                         start.sh (sets VLLM_TORCH_PROFILER_DIR + /profiler
                         mount per instance) and wraps the ansible benchmark
                         with POST /start_profile,/stop_profile to all N
                         instances via podman exec. Traces land in
                         PROFILER_DIR (default profiler-traces/) under a
                         per-run subdir. NOTE: wrapping the full multi-rate
                         run can produce multi-GB traces per instance.
  -w, --workload NAME    Workload profile to benchmark (chat, rag, code,
                         summarization, ...). Repeat to run several against the
                         same running stack. Overrides BASE_WORKLOAD env.
                         Defaults to "chat". Also sets the default
                         --max-model-len (see VLLM_MAX_MODEL_LEN below).
                         Special: "-w audio" benchmarks ASR (Whisper) via
                         audio-benchmark.yml against /v1/audio/transcriptions.
                         Not combinable with other workloads. Default model
                         openai/whisper-small; scenario via AUDIO_SCENARIO.
  -d, --dtype TYPE       dtype forwarded to "vllm serve --dtype". Defaults to
                         bfloat16; use "half" for fp16 (or float32/auto). Also
                         settable via the DTYPE env var; set DTYPE="" to omit.
  --no-limits            Skip cpuset/mem_limit/shm_size entirely (forwarded to
                         start.sh). Required on rootless hosts where the cpuset
                         cgroup controller is not delegated (containers fail to
                         start with 'cpuset.cpus: no such file or directory').
  --lmcache              Build the LMCache image (local checkout, NO_GPU_EXT)
                         and run a shared lmcache server wired to every
                         instance via the MP connector (forwarded to start.sh).
                         Zero-copy SHM transport; LMCACHE_PICKLE=1 selects the
                         ZMQ/pickle transport. LMCACHE_SRC overrides the
                         checkout path.
  --dry-run, -n          Validate paths/config and print every command that
                         would run, without starting containers, calling
                         ansible, or touching sudo. Safe with HF_TOKEN unset.
  -h, --help             Show this help.

Benchmarking window:
  GUIDELLM_MAX_SECONDS caps how long each rate runs. It takes either a single
  number, shared by every rate, or a "<rate>:<seconds>" map that gives each rate
  its own window:

    GUIDELLM_MAX_SECONDS=300                             one 300s window
    GUIDELLM_MAX_SECONDS="32:600,64:800,128:1100"        one window per rate

  Use the map when a single number cannot serve the whole sweep: a request takes
  longer at higher concurrency, so a window with several completions per stream
  at rate 32 leaves one (or none) at rate 256, and that rate's latency
  percentiles then come from a single sample. Rates missing from the map get
  GUIDELLM_MAX_SECONDS_DEFAULT (300).

  The map runs one guidellm invocation per rate, because max_duration is a
  run-wide constraint. They share one results directory and are merged back into
  benchmarks.json / benchmarks.csv / guidellm.log when the workload finishes, so
  the result layout matches a single-window run. The per-rate files
  (benchmarks-<rate>.json, ...) are kept alongside. Needs guidellm v0.7+.

Context window:
  vLLM starts once per model and serves every -w workload from that one stack,
  so --max-model-len must cover the largest of them. It defaults to the biggest
  context the selected workloads need, read from test-workloads.yml (the
  workload's own --max-model-len when it declares one, else isl+osl+128):

    -w chat                                 -> 2048
    -w code                                 -> 4096
    -w chat -w summarization -w code        -> 4096
    -w chat -w summarization -w code -w rag -> 8320

  Relevant environment variables:
    VLLM_MAX_MODEL_LEN     override the derived value outright
    VLLM_KV_CACHE_SPACE    KV cache GiB per instance (default 90); the context
                           window must fit in it or vLLM aborts at startup
    MAX_MODEL_LEN_HEADROOM tokens added to isl+osl (default 128)
    KV_FIT_CHECK           set false to skip the pre-flight fit check
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 1 ;;
    esac
done

# --quant marks a quantized run (llm-compressor by default, no image change).
# --torchao (only with --quant) opts into the derived torchao image build.
QUANT="$CLI_QUANT"
TORCHAO="$CLI_TORCHAO"
if [[ "$TORCHAO" == "true" && "$QUANT" != "true" ]]; then
    echo "ERROR: --torchao requires --quant." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. MODELS TO SWEEP  --  fill in your 20 models here
# ----------------------------------------------------------------------------
# Format per entry:  "hf-path-or-relative-dir | short-tag"
#   - Left side  = passed to vLLM as --model (HF repo id, or path under MODELS_DIR)
#   - Right side = used in test_name (must be [A-Za-z0-9-] and short enough that
#                  "${PREFIX}-${variant}-${tag}" stays <= 100 chars).
# The ansible playbook enforces: test_name matches ^[A-Za-z0-9-]{1,100}$.
MODELS=(
    "Llama-3.1-8B-Instruct                | llama31-8b"
    "gpt-oss-20b-BF16                     | gptoss-20b"
)

# CLI -m/--model entries override the built-in list above.
if (( ${#CLI_MODELS[@]} > 0 )); then
    MODELS=("${CLI_MODELS[@]}")
fi

# ----------------------------------------------------------------------------
# 2. VARIANTS  --  comment out one if you only want native or only zentorch
# ----------------------------------------------------------------------------
VARIANTS=(
    "zentorch"
#    "native"
)

# CLI --native/--zentorch overrides the VARIANTS array above.
if [[ -n "$CLI_VARIANT" ]]; then
    VARIANTS=("$CLI_VARIANT")
fi

# ----------------------------------------------------------------------------
# 3. BENCHMARK CONFIG
# ----------------------------------------------------------------------------
GUIDELLM_RATES="${GUIDELLM_RATES:-[32,64]}"
# GUIDELLM_RATES="${GUIDELLM_RATES:-[64,96,128,256,364,512,1024]}"

# Benchmarking window. Two accepted forms:
#   GUIDELLM_MAX_SECONDS=300                one window shared by every rate
#   GUIDELLM_MAX_SECONDS="32:600,64:800"    one window per rate
# A request takes longer at higher concurrency, so a window that gives several
# completions per stream at 32 gives one (or none) at 256, and the latency
# percentiles for that rate then rest on a single sample. max_duration is a
# run-wide GuideLLM constraint, so the map form runs one GuideLLM invocation per
# rate instead of a single one covering the whole list. The per-rate runs share
# one results directory and their outputs are merged back into benchmarks.json
# afterwards, so the layout on disk is the same either way.
# Rates the map does not mention fall back to GUIDELLM_MAX_SECONDS_DEFAULT.
GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-300}"
# Max requests before a run stops. The perf-eval default (1000) is far too small
# for high concurrency -- it terminates runs before steady state, so max_seconds
# never binds. Set high so the run is time-bounded, not request-bounded.
GUIDELLM_MAX_REQUESTS="${GUIDELLM_MAX_REQUESTS:-100000}"
GUIDELLM_MAX_SECONDS_DEFAULT="${GUIDELLM_MAX_SECONDS_DEFAULT:-300}"

# GUIDELLM_RATES as a bash array, for the per-rate loop and the pre-flight table.
GUIDELLM_RATE_LIST=()
IFS=',' read -r -a GUIDELLM_RATE_LIST <<< "${GUIDELLM_RATES//[\[\] ]/}"

declare -A GUIDELLM_RATE_SECONDS=()
PER_RATE_WINDOW=false
if [[ "$GUIDELLM_MAX_SECONDS" == *:* ]]; then
    PER_RATE_WINDOW=true
    IFS=',' read -r -a _window_entries <<< "${GUIDELLM_MAX_SECONDS// /}"
    for _entry in "${_window_entries[@]}"; do
        [[ -z "$_entry" ]] && continue
        if [[ "$_entry" != *:* ]] \
           || [[ ! "${_entry%%:*}" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
           || [[ ! "${_entry##*:}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: GUIDELLM_MAX_SECONDS entry '$_entry' is not <rate>:<seconds>." >&2
            echo "       Example: GUIDELLM_MAX_SECONDS=\"32:600,64:800,128:1100,256:1650\"" >&2
            exit 1
        fi
        GUIDELLM_RATE_SECONDS["${_entry%%:*}"]="${_entry##*:}"
    done
    unset _entry _window_entries
fi

# Window for one rate: its map entry, else the fallback default.
guidellm_window_for() {
    echo "${GUIDELLM_RATE_SECONDS[$1]:-$GUIDELLM_MAX_SECONDS_DEFAULT}"
}

# Filename-safe form of a rate, used as the per-rate output suffix.
sanitize_rate() {
    echo "${1//[^0-9A-Za-z.]/_}"
}

# The window as one line: "300s (all rates)" or "32:600s 64:800s 128:1100s".
window_summary() {
    if ! $PER_RATE_WINDOW; then
        echo "${GUIDELLM_MAX_SECONDS}s (all rates)"
        return
    fi
    local out="" rate
    for rate in "${GUIDELLM_RATE_LIST[@]}"; do
        out+="${rate}:$(guidellm_window_for "$rate")s "
    done
    echo "${out% }"
}

# Max concurrent in-flight requests. Must be >= the largest GUIDELLM_RATES
# value or the concurrent profile clamps high rates to this cap. Forwarded to
# ansible as guidellm_max_concurrency (the playbook defaults to 128).
GUIDELLM_MAX_CONCURRENCY="${GUIDELLM_MAX_CONCURRENCY:-1024}"
# Safety timeout (seconds) for the ansible poll that waits for the GuideLLM
# container to finish. The role's auto-formula (num_rates*(max_seconds+70)+300)
# under-budgets the per-rate drain time for high-concurrency runs on slow CPU
# backends, so we override it with a generous fixed value. Forwarded to ansible
# as guidellm_wait_timeout_seconds (extra-vars take precedence over set_fact).
# With a per-rate window each ansible run covers one rate, so an unset timeout
# is derived from that rate's window instead of using the sweep-wide value.
GUIDELLM_WAIT_TIMEOUT_EXPLICIT=false
if [[ -n "${GUIDELLM_WAIT_TIMEOUT:-}" ]]; then
    GUIDELLM_WAIT_TIMEOUT_EXPLICIT=true
fi
GUIDELLM_WAIT_TIMEOUT="${GUIDELLM_WAIT_TIMEOUT:-3600}"

# Safety timeout for a single-rate run of $1 seconds. Keeps the 3600s floor the
# fixed default had, and the role's own 14400s ceiling.
guidellm_wait_timeout_for() {
    local derived=$(( $1 + 900 ))
    if $GUIDELLM_WAIT_TIMEOUT_EXPLICIT; then
        echo "$GUIDELLM_WAIT_TIMEOUT"
        return
    fi
    (( derived < 3600 )) && derived=3600
    (( derived > 14400 )) && derived=14400
    echo "$derived"
}
# Workloads to benchmark per (variant, model). Each reuses the same running
# stack -- only the ansible step re-runs per workload. CLI -w/--workload wins,
# then the space-separated BASE_WORKLOAD env var, then default "chat".
read -r -a BASE_WORKLOADS <<< "${BASE_WORKLOAD:-chat}"
if (( ${#CLI_WORKLOADS[@]} > 0 )); then
    BASE_WORKLOADS=("${CLI_WORKLOADS[@]}")
fi

# ----------------------------------------------------------------------------
# 3b. AUDIO MODE
# ----------------------------------------------------------------------------
# The "audio" workload benchmarks ASR (Whisper transcription) instead of a
# generative chat model. Like embedding, it swaps the load-generator half of the
# run: upstream's audio-benchmark.yml driving GuideLLM's audio backend against
# /v1/audio/transcriptions, instead of llm-benchmark-concurrent-load.yml driving
# GuideLLM against /v1/chat/completions. The stack half (start.sh -> N instances
# behind nginx) is reused as-is -- nginx's catch-all "location /" already proxies
# /v1/audio/transcriptions, and audio-benchmark.yml is run in EXTERNAL mode so it
# targets the nginx endpoint rather than starting its own single vLLM server.
#
# Whisper is an encoder-decoder model with its own serve requirements, so unlike
# embedding this DOES set a few stack-side defaults (max-model-len, trust-remote-
# code, the V1 CPU workaround). All remain env-overridable.
#
# Both "-w audio" and "BASE_WORKLOAD=audio" reach this.
IS_AUDIO=false
for _wl in "${BASE_WORKLOADS[@]}"; do
    [[ "$_wl" == "audio" ]] && IS_AUDIO=true
done
unset _wl

# A single stack serves one model with one runner, so audio cannot share an
# invocation with a generative (or embedding) workload.
if $IS_AUDIO && (( ${#BASE_WORKLOADS[@]} > 1 )); then
    echo "ERROR: the 'audio' workload cannot be combined with other workloads." >&2
    echo "       Got: ${BASE_WORKLOADS[*]}" >&2
    echo "       Run audio separately: $0 -w audio" >&2
    exit 1
fi

# Whisper models to sweep when no -m/--model was given. Same "path | tag" format
# as MODELS above; tags stay <= 85 chars so "${PREFIX}-${variant}-${tag}" fits
# the ansible test_name limit. whisper-small is the guide's recommended CPU
# baseline; add whisper-tiny / whisper-medium via -m to cover a size range.
AUDIO_MODELS=(
    "openai/whisper-small | whisper-small"
)

# Scenario driven by audio-benchmark.yml. "transcription-throughput" is the
# primary one (sequential baseline + concurrent-2/4/8 stages). Others:
# transcription-latency, transcription-quality, audio-duration-scaling,
# constant-rate-stress, format-comparison, quick-test.
AUDIO_SCENARIO="${AUDIO_SCENARIO:-transcription-throughput}"

# Files processed per stage. Leave unset to inherit the scenario YAML's own
# num_samples (100 for throughput, 5 for quick-test); export to override. Only
# when set do we pass -e audio_num_files so it doesn't shadow the scenario value.
AUDIO_NUM_FILES="${AUDIO_NUM_FILES:-}"

# P95 latency target (seconds) for the enterprise report the playbook prints.
AUDIO_P95_TARGET="${AUDIO_P95_TARGET:-2.0}"

# vLLM v0.25.1's V1 engine has CPU encoder-decoder bugs; the guide's workaround
# is to disable it. Folded into SET_ENV_VAR_BEFORE_VLLM_SERVE so it runs INSIDE
# the container (before "vllm serve"). Set AUDIO_VLLM_ENV="" to drop it once vLLM
# fixes the V1 CPU path.
AUDIO_VLLM_ENV="${AUDIO_VLLM_ENV:-export VLLM_USE_V2_MODEL_RUNNER=0 &&}"

if $IS_AUDIO; then
    if (( ${#CLI_MODELS[@]} == 0 )); then
        MODELS=("${AUDIO_MODELS[@]}")
    fi

    # Whisper's context window is 448. audio is not in test-workloads.yml, so the
    # derivation below would otherwise fall back to 4096 and vLLM would reject the
    # model (max seq len 448 < requested 4096). Set before the derivation so an
    # explicit VLLM_MAX_MODEL_LEN still wins.
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-448}"

    # audio-benchmark.yml appends its own "audio-models/" segment to both
    # bench_config.results_dir (the working path) and local_results_base (the
    # collection path). local_results_base is the bare results/ dir, so leaving
    # bench_config.results_dir at its llm default splits the run across
    # results/llm/audio-models/ and results/audio-models/ -- and the summary the
    # sweep prints, plus extract_audio_metrics.py, both read the latter. "." keeps
    # the base dir; realpath in the inventory collapses it away.
    export RESULTS_SUBDIR="${RESULTS_SUBDIR:-.}"

    # Whisper needs --trust-remote-code. Prepend to any user-supplied extra args.
    if [[ "${EXTRA_VLLM_ARGS:-}" != *"--trust-remote-code"* ]]; then
        EXTRA_VLLM_ARGS="--trust-remote-code ${EXTRA_VLLM_ARGS:-}"
    fi
    export EXTRA_VLLM_ARGS

    # Encoder-decoder KV cache is tiny; the 90 GiB generative default is wasteful.
    VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-4}"

    # Prepend the V1 workaround to the in-container env prefix.
    if [[ -n "$AUDIO_VLLM_ENV" && "${SET_ENV_VAR_BEFORE_VLLM_SERVE:-}" != *"VLLM_USE_V2_MODEL_RUNNER"* ]]; then
        SET_ENV_VAR_BEFORE_VLLM_SERVE="$AUDIO_VLLM_ENV ${SET_ENV_VAR_BEFORE_VLLM_SERVE:-}"
    fi
fi

# ----------------------------------------------------------------------------
# 3c. EMBEDDING MODE
# ----------------------------------------------------------------------------
# The "embedding" workload (declared in test-workloads.yml) selects a pooling
# model instead of a generative one. It swaps the whole load-generator half of
# the run: upstream's embedding-benchmark.yml driving "vllm bench serve" against
# /v1/embeddings, instead of llm-benchmark-concurrent-load.yml driving GuideLLM
# against /v1/chat/completions. The stack half (start.sh -> generate-config.sh ->
# N instances behind nginx) is identical -- vLLM auto-detects the pooling runner
# from the model config, and nginx's catch-all "location /" already proxies
# /v1/embeddings with the same tuning the chat locations get.
#
# Both "-w embedding" and "BASE_WORKLOAD=embedding" reach this, because the CLI
# override above rewrites the same array.
IS_EMBEDDING=false
for _wl in "${BASE_WORKLOADS[@]}"; do
    [[ "$_wl" == "embedding" ]] && IS_EMBEDDING=true
done
unset _wl

# A single stack serves one model with one runner, so a pooling workload cannot
# share an invocation with a generative one.
if $IS_EMBEDDING && (( ${#BASE_WORKLOADS[@]} > 1 )); then
    echo "ERROR: the 'embedding' workload cannot be combined with other workloads." >&2
    echo "       Got: ${BASE_WORKLOADS[*]}" >&2
    echo "       Run embedding separately: $0 -w embedding" >&2
    exit 1
fi

# Embedding models to sweep when no -m/--model was given. Same "path | tag"
# format as MODELS above; tags stay <= 15 chars so "${PREFIX}-${variant}-${tag}"
# fits the ansible test_name limit.
# NOTE: nomic-embed-text-v1.5 ships custom modelling code and needs
#   export EXTRA_VLLM_ARGS="--trust-remote-code"
EMBEDDING_MODELS=(
    "RedHatAI/granite-embedding-english-r2 | granite-en-r2"
    "RedHatAI/nomic-embed-text-v1.5        | nomic-v15"
)

# These are always defined -- the script runs under "set -u" and the generative
# path still reads them when it builds its banner and pre-flight output.
#
# Scenarios run by the playbook: baseline (request-rate inf, then 25/50/75% of
# the measured max) and/or latency (concurrency 16/32/64/128/196).
EMBEDDING_SCENARIO="${EMBEDDING_SCENARIO:-all}"

# "vllm bench serve" has no time-based stop (unlike GuideLLM's max_seconds): a
# run lasts num_prompts/RPS seconds. Leave this unset to inherit the repo default
# (benchmark-tools.yml -> vllm_bench.num_prompts, currently 250); export
# EMBEDDING_NUM_PROMPTS to override. Only when set do we pass -e num_prompts so the
# extra-var doesn't shadow the repo's configured value.
EMBEDDING_NUM_PROMPTS="${EMBEDDING_NUM_PROMPTS:-}"

# The baseline "inf rate" test opens all num_prompts connections at once. With no
# ceiling the socket/nginx backlog refuses almost every request instantly (a
# 10000-prompt run died in ~7s with 9998 connection failures). Cap concurrency so
# inf means "as fast as the server drains" rather than "all at once". Matches the
# top of the latency scenario's concurrency sweep.
EMBEDDING_BASELINE_MAX_CONCURRENCY="${EMBEDDING_BASELINE_MAX_CONCURRENCY:-196}"

# Load-generator pinning. Default to the free core window between NGINX_CORES
# (1-15) and VLLM_START_CORE (32); an unpinned client would otherwise float onto
# the vLLM cores and understate throughput. NUMA is left unset (all nodes) on
# purpose -- binding the client to node 0 is what OOM-killed GuideLLM's workers,
# see the GUIDELLM_NUMA_NODE note below.
EMBEDDING_BENCH_CPUS="${EMBEDDING_BENCH_CPUS:-16-31}"
EMBEDDING_BENCH_NUMA="${EMBEDDING_BENCH_NUMA:-}"

# Path to the vllm CLI inside the load-generator image. Upstream hardcodes the
# vllm-openai-cpu layout (/opt/venv/bin/vllm); the zentorch image has it on PATH.
EMBEDDING_BENCH_BINARY="${EMBEDDING_BENCH_BINARY:-vllm}"

if $IS_EMBEDDING; then
    if (( ${#CLI_MODELS[@]} == 0 )); then
        MODELS=("${EMBEDDING_MODELS[@]}")
    fi

    # Pooling models hold no KV cache; the upstream model matrix asks for 1GiB.
    # Set before the export below so its 90GB generative default never applies.
    VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-4}"

    # Results land in results/embedding/ rather than results/llm/. Read by the
    # patched bench_config.results_dir in the ansible inventory.
    export RESULTS_SUBDIR="${RESULTS_SUBDIR:-embedding}"
fi

# --no-mem-limit: CLI flag wins over the NO_MEM_LIMIT env var. Forwarded to
# start.sh.
if [[ "$CLI_NO_MEM_LIMIT" == "true" ]]; then
    NO_MEM_LIMIT=true
fi
export NO_MEM_LIMIT="${NO_MEM_LIMIT:-false}"

# --profile: CLI flag wins over the PROFILE env var. When on, start.sh gets
# --profile (server-side trace dir + mount) and run_one wraps the ansible
# benchmark with /start_profile,/stop_profile to every instance.
if [[ "$CLI_PROFILE" == "true" ]]; then
    PROFILE=true
fi
export PROFILE="${PROFILE:-false}"

# dtype forwarded to start.sh -> generate-config.sh -> "vllm serve --dtype".
# CLI -d/--dtype wins over the DTYPE env var; defaults to bfloat16 ("half" for
# fp16). Set DTYPE="" in the env to omit the flag and let vLLM decide.
DTYPE="${CLI_DTYPE:-${DTYPE-bfloat16}}"

# Shell prefix spliced before "vllm serve" in each instance (for testing env
# vars without editing generate-config.sh). Exported so start.sh's child
# generate-config.sh inherits it and bakes it into the compose command, so it
# takes effect INSIDE the podman container. Provide the full prefix including
# the trailing "&&", e.g.:
#   export SET_ENV_VAR_BEFORE_VLLM_SERVE="export ZENDNNL_MATMUL_WEIGHT_CACHE=2 &&"
export SET_ENV_VAR_BEFORE_VLLM_SERVE="${SET_ENV_VAR_BEFORE_VLLM_SERVE:-}"
# Resolve instance count early: CLI -N wins, then env, then default 1.
# It seeds both the stack layout and the default test_name prefix below.
NUM_INSTANCES="${CLI_NUM_INSTANCES:-${NUM_INSTANCES:-1}}"

# test_name must be 1-100 chars, [A-Za-z0-9-] only.
# Format used below: "${PREFIX}-${variant}-${tag}"  (variant = "zentorch"|"native")
# Budget: "EPYC7" (5) + "-zentorch-" (10) + tag = 15 + len(tag), so tag <= 85.
TEST_NAME_PREFIX="${TEST_NAME_PREFIX:-EPYC${NUM_INSTANCES}}"

# If MODELS_DIR is set, each model name in the MODELS array is expected to be
# a relative path under MODELS_DIR (e.g. "meta-llama/Llama-3.1-8B-Instruct"
# resolves to "$MODELS_DIR/meta-llama/Llama-3.1-8B-Instruct"). The directory
# is bind-mounted at the same path inside the container, so the absolute path
# is passed straight to "vllm serve".
# Leave MODELS_DIR unset to pull models from the HuggingFace hub as before.
# CLI --models-dir wins over the MODELS_DIR env var.
MODELS_DIR="${CLI_MODELS_DIR:-${MODELS_DIR:-}}"

# ----------------------------------------------------------------------------
# 4. STACK / CPU LAYOUT  (passed to start.sh --regenerate)
# ----------------------------------------------------------------------------
export NUM_INSTANCES
export CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
export VLLM_START_CORE="${VLLM_START_CORE:-32}"
export NGINX_CORES="${NGINX_CORES:-1-15}"
export NGINX_PORT="${NGINX_PORT:-8080}"
export MEM_LIMIT="${MEM_LIMIT:-200g}"
export VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-90}"

# Zentorch image (base). The native variant is built from this image by
# uninstalling zentorch -- start.sh --native handles this automatically.
# No default is provided on purpose: the user must set VLLM_IMAGE explicitly.
# Mirror the HF_TOKEN handling below -- prompt interactively rather than
# hard-exiting so a tmux pane that launched the script stays alive.
export VLLM_IMAGE="${VLLM_IMAGE:-}"
if ! $DRY_RUN && [[ -z "${VLLM_IMAGE:-}" ]]; then
    echo ""
    echo "VLLM_IMAGE is not set. Set it to the vLLM/zentorch container image to use,"
    echo "e.g. export VLLM_IMAGE=<registry>/<repo>/<image>:<tag>"
    if [[ -t 0 ]]; then
        read -r -p "Enter the VLLM_IMAGE to use (or press Enter to abort): " VLLM_IMAGE
    fi
    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        echo "No image provided. Aborting (tmux session preserved)." >&2
        exit 1
    fi
    export VLLM_IMAGE
fi

# The embedding load generator runs "vllm bench serve" inside a container. Point
# it at the same image the server runs (upstream's own stated intent for this
# variable) so client and server can't skew on the embeddings API, and so no
# second image has to be pulled.
VLLM_BENCH_CONTAINER_IMAGE="${VLLM_BENCH_CONTAINER_IMAGE:-$VLLM_IMAGE}"
# Exporting an empty value would still be picked up by the inventory's
# lookup('env', ...), so only export it once it actually resolves.
if [[ -n "$VLLM_BENCH_CONTAINER_IMAGE" ]]; then
    export VLLM_BENCH_CONTAINER_IMAGE
fi

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"

# ----------------------------------------------------------------------------
# 5. ANSIBLE / ENDPOINT CONFIG
# ----------------------------------------------------------------------------
# GuideLLM load-generator CPU / NUMA pinning (optional passthrough to ansible).
# The load generator is otherwise strictly memory-bound to NUMA node 0
# (cpuset_mems default '0'); with many vLLM instances node 0 can run out of
# memory and the client's worker processes get OOM-killed at startup. Set
# GUIDELLM_NUMA_NODE to a range spanning all nodes (e.g. "0-7") or to a node
# with free RAM, and optionally GUIDELLM_CPUS to matching cores.
GUIDELLM_CPUS="${GUIDELLM_CPUS:-}"
GUIDELLM_NUMA_NODE="${GUIDELLM_NUMA_NODE:-}"

export VLLM_ENDPOINT_MODE="${VLLM_ENDPOINT_MODE:-external}"
export VLLM_ENDPOINT_URL="${VLLM_ENDPOINT_URL:-http://localhost:${NGINX_PORT}}"
export LOADGEN_HOSTNAME="${LOADGEN_HOSTNAME:-localhost}"
export DUT_HOSTNAME="${DUT_HOSTNAME:-localhost}"
export ANSIBLE_SSH_USER="${ANSIBLE_SSH_USER:-$(whoami)}"
export ANSIBLE_SSH_KEY="${ANSIBLE_SSH_KEY:-$HOME/.ssh/id_rsa}"

# HF_TOKEN is only needed for gated models (e.g. Llama); public models and a
# pre-populated offline cache work without one. Prompt on a TTY, otherwise
# continue anonymously. Never fall back to a placeholder token: HuggingFace
# rejects an invalid token even for public repos. Skipped in --dry-run.
if ! $DRY_RUN && [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "HF_TOKEN is not set. Gated models (e.g. Llama) need it; public models do not."
    if [[ -t 0 ]]; then
        read -r -s -p "Paste your HuggingFace token (or press Enter to continue without one): " HF_TOKEN
        echo ""
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Continuing without HF_TOKEN."
    fi
fi
export HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANSIBLE_DIR="${ANSIBLE_DIR:-$SCRIPT_DIR/vllm-cpu-perf-eval/automation/test-execution/ansible}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/sweep-logs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"

# Profiler trace root. Forwarded to start.sh/generate-config.sh as
# --profiler-dir so the host mount and run_one's trace-collection paths agree.
# Per-run subdirs (keyed by test_name) are created under this in run_one.
PROFILER_DIR="${PROFILER_DIR:-$SCRIPT_DIR/profiler-traces}"
if [[ "$PROFILE" == "true" ]]; then
    mkdir -p "$PROFILER_DIR"
    PROFILER_DIR="$(cd "$PROFILER_DIR" && pwd)"
fi
export PROFILER_DIR

if [[ -n "$MODELS_DIR" ]]; then
    if [[ ! -d "$MODELS_DIR" ]]; then
        echo "ERROR: MODELS_DIR='$MODELS_DIR' does not exist or is not a directory." >&2
        exit 1
    fi
    MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"
fi

# ----------------------------------------------------------------------------
# 6. CONTEXT WINDOW  (--max-model-len)
# ----------------------------------------------------------------------------
# In external-endpoint mode vLLM is started ONCE PER MODEL (start.sh ->
# generate-config.sh) and every workload in BASE_WORKLOADS is then driven
# against that one stack, so the per-workload vllm_args in test-workloads.yml
# never reach vLLM. The single context window therefore has to satisfy the
# largest selected workload, or vLLM rejects its requests with HTTP 400.
#
# Per workload we read isl/osl (isl_max/osl_max when variability: true) out of
# test-workloads.yml and take that workload's own "--max-model-len=N" when it
# declares one, else isl+osl+MAX_MODEL_LEN_HEADROOM. This mirrors the file's own
# convention: rag declares 8320 for 7680+512=8192, i.e. +128 for tokenization
# overhead. The default is the max across the selected workloads; an explicit
# VLLM_MAX_MODEL_LEN in the environment always wins.
WORKLOADS_FILE="${WORKLOADS_FILE:-$ANSIBLE_DIR/inventory/group_vars/all/test-workloads.yml}"
MAX_MODEL_LEN_HEADROOM="${MAX_MODEL_LEN_HEADROOM:-128}"
# Used only when no selected workload could be resolved -- the value this
# harness hardcoded before --max-model-len became configurable.
MAX_MODEL_LEN_FALLBACK="${MAX_MODEL_LEN_FALLBACK:-4096}"

# Print the context window one workload needs; return 1 if it isn't in
# test-workloads.yml. Parsed with awk rather than yq or python on purpose:
# neither yq nor a PyYAML-capable interpreter is guaranteed on the benchmark
# nodes (ansible here is pinned to /usr/bin/python3), while the block we need is
# a fixed 2-space-indented mapping under "test_configs:".
workload_max_model_len() {
    local wl="$1"
    [[ -f "$WORKLOADS_FILE" ]] || return 1
    awk -v want="$wl" -v headroom="$MAX_MODEL_LEN_HEADROOM" '
        function val(s) { gsub(/[",]/, "", s); return s }
        /^[^[:space:]#]/ { in_cfg = ($0 ~ /^test_configs:/); cur = ""; next }
        /^  [A-Za-z0-9_]+:[[:space:]]*$/ {
            if (in_cfg) { cur = $1; sub(/:$/, "", cur) }
            next
        }
        cur != want              { next }
        /^    isl:/              { isl = val($2) }
        /^    osl:/              { osl = val($2) }
        /^    isl_max:/          { isl_max = val($2) }
        /^    osl_max:/          { osl_max = val($2) }
        /^    variability:/      { variable = val($2) }
        /--max-model-len=[0-9]+/ {
            match($0, /--max-model-len=[0-9]+/)
            declared = substr($0, RSTART + 16, RLENGTH - 16) + 0
        }
        END {
            if (isl == "" || osl == "") exit 1
            i = (variable == "true" && isl_max != "") ? isl_max : isl
            o = (variable == "true" && osl_max != "") ? osl_max : osl
            need = i + o + headroom
            if (declared > need) need = declared
            print need
        }
    ' "$WORKLOADS_FILE"
}

MAX_MODEL_LEN_SOURCE="explicit VLLM_MAX_MODEL_LEN"
if [[ -z "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    _derived=0
    _unresolved=()
    for _wl in "${BASE_WORKLOADS[@]}"; do
        if _need=$(workload_max_model_len "$_wl") && [[ -n "$_need" ]]; then
            (( _need > _derived )) && _derived=$_need
        else
            _unresolved+=("$_wl")
        fi
    done
    if (( ${#_unresolved[@]} > 0 )); then
        echo "WARNING: workload(s) not found in $WORKLOADS_FILE: ${_unresolved[*]}" >&2
        echo "         their context needs are NOT reflected in the derived --max-model-len." >&2
    fi
    if (( _derived > 0 )); then
        VLLM_MAX_MODEL_LEN=$_derived
        MAX_MODEL_LEN_SOURCE="derived from workloads: ${BASE_WORKLOADS[*]}"
    else
        VLLM_MAX_MODEL_LEN=$MAX_MODEL_LEN_FALLBACK
        MAX_MODEL_LEN_SOURCE="fallback (no workload resolved from $WORKLOADS_FILE)"
    fi
fi
export VLLM_MAX_MODEL_LEN

# Resolve a model entry to the actual path/repo passed to vLLM.
# - With MODELS_DIR set: returns "$MODELS_DIR/$model" (absolute local path)
# - Without MODELS_DIR : returns "$model" unchanged (HF repo id)
resolve_model_path() {
    local m="$1"
    if [[ -n "$MODELS_DIR" ]]; then
        echo "$MODELS_DIR/$m"
    else
        echo "$m"
    fi
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
sanitize_model() {
    # meta-llama/Llama-3.1-8B-Instruct  ->  meta-llama__Llama-3.1-8B-Instruct
    # Used only for filesystem-safe log file names, NOT for test_name.
    echo "${1//\//__}"
}

# Trim leading/trailing whitespace.
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

# Split a MODELS entry "hf/path | tag" -> sets global MODEL_PATH and MODEL_TAG.
parse_model_entry() {
    local entry="$1"
    local left right
    left="${entry%%|*}"
    right="${entry##*|}"
    if [[ "$left" == "$entry" ]]; then
        # No '|' separator -- fall back to using the path as the tag too.
        MODEL_PATH="$(trim "$entry")"
        MODEL_TAG="$(trim "$entry")"
    else
        MODEL_PATH="$(trim "$left")"
        MODEL_TAG="$(trim "$right")"
    fi
}

# Validate a test_name against the ansible playbook's rules.
# Returns 0 if OK, 1 otherwise and prints a reason on stderr.
#
# 100, not 30: the playbooks this script runs are llm-benchmark-concurrent-load
# (which imports llm-benchmark-auto) and audio-benchmark, and the only test_name
# assert on that path is llm-benchmark-auto's "length <= 100". The <= 30 assert
# lives in llm-core-sweep-auto, which nothing here invokes, so a label driven
# through that playbook (cpueval --tag) still has to stay under 30.
validate_test_name() {
    local name="$1"
    local len=${#name}
    if (( len < 1 || len > 100 )); then
        echo "  test_name '$name' is $len chars (must be 1-100)" >&2
        return 1
    fi
    if [[ ! "$name" =~ ^[A-Za-z0-9-]+$ ]]; then
        echo "  test_name '$name' contains non-[A-Za-z0-9-] characters" >&2
        return 1
    fi
    return 0
}

# Shared HF cache that start.sh pre-warms and generate-config.sh bind-mounts
# into every instance. Read here to locate a model's config.json, and exported
# so the GuideLLM role mounts it read-only to load the tokenizer offline; the
# huggingface/ subdir must exist for that bind mount.
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/hf-shared}"
mkdir -p "$HF_CACHE_DIR/huggingface"
# Set to false to skip the KV-cache fit check below (e.g. for an architecture
# the estimate below models badly).
KV_FIT_CHECK="${KV_FIT_CHECK:-true}"

# Path to a model's config.json (local dir or the shared HF cache), or 1 if the
# model hasn't been downloaded yet.
model_config_json() {
    local model="$1" repo cfg
    if [[ "$model" == /* ]]; then
        [[ -f "$model/config.json" ]] || return 1
        echo "$model/config.json"
        return 0
    fi
    repo="models--${model//\//--}"
    # shellcheck disable=SC2012
    cfg=$(ls -1 "$HF_CACHE_DIR/huggingface/hub/$repo/snapshots"/*/config.json 2>/dev/null | head -1)
    [[ -n "$cfg" ]] || return 1
    echo "$cfg"
}

# First integer value of a top-level-or-nested JSON key.
json_int() {
    grep -oE "\"$2\"[[:space:]]*:[[:space:]]*[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'
}

# Tokens that fit in VLLM_KV_CACHE_SPACE GiB for the model described by
# config.json, or 1 if the shape can't be read. vLLM sizes the CPU KV cache as
#   bytes/token = 2 (K and V) * layers * kv_heads * head_dim * 2 (bf16)
# and refuses to start when max_model_len exceeds the resulting token count.
# Hybrid models (Qwen3.5 makes only every 4th layer full-attention) fit more
# than this, so the estimate is deliberately on the pessimistic side.
kv_cache_token_capacity() {
    local cfg="$1" layers kvh hd hs ah
    layers=$(json_int "$cfg" num_hidden_layers)
    kvh=$(json_int "$cfg" num_key_value_heads)
    hd=$(json_int "$cfg" head_dim)
    if [[ -z "$hd" ]]; then
        hs=$(json_int "$cfg" hidden_size)
        ah=$(json_int "$cfg" num_attention_heads)
        if [[ -n "$hs" && -n "$ah" ]] && (( ah > 0 )); then
            hd=$(( hs / ah ))
        fi
    fi
    [[ -n "$layers" && -n "$kvh" && -n "$hd" ]] || return 1
    awk -v gib="$VLLM_KV_CACHE_SPACE" -v l="$layers" -v k="$kvh" -v d="$hd" \
        'BEGIN { bpt = 2*l*k*d*2; if (bpt <= 0) exit 1; printf "%d\n", gib*1024*1024*1024/bpt }'
}

banner() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

# POST a profiler endpoint (start_profile|stop_profile) to every instance from
# inside its own netns via podman exec (per-instance ports are not published).
# $1 = endpoint name, $2 = run_log to append container output to.
profile_post_all() {
    local endpoint="$1" log="$2" i
    for i in $(seq 1 "$NUM_INSTANCES"); do
        echo "  [profile] POST /$endpoint -> vllm-instance-$i"
        podman exec "vllm-instance-$i" \
            curl -sS -X POST "http://localhost:8000/$endpoint" \
            >> "$log" 2>&1 \
            || echo "  WARNING: /$endpoint failed on vllm-instance-$i (see $log)"
    done
}

# Block until every instance's trace dir stops growing, so we never tear the
# stack down mid-serialization. /stop_profile returns before the trace is
# fully flushed for large captures. $1 = host trace root for this run.
profile_wait_flush() {
    local trace_root="$1"
    local stable_needed=3 interval=10 max_wait=1800
    local prev="" cur elapsed=0 stable=0
    echo "  [profile] waiting for traces to finish flushing in $trace_root ..."
    while (( elapsed < max_wait )); do
        # Total bytes across all instance trace dirs; "" if none yet.
        cur=$(find "$trace_root" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
        if [[ -n "$cur" && "$cur" -gt 0 && "$cur" == "$prev" ]]; then
            stable=$((stable + 1))
            if (( stable >= stable_needed )); then
                echo "  [profile] traces stable at ${cur} bytes."
                return 0
            fi
        else
            stable=0
        fi
        prev="$cur"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo "  WARNING: trace flush did not stabilize within ${max_wait}s; proceeding anyway."
}

# Rebuild the single-run file layout from the per-rate outputs of one run id.
# The dashboard, extract_benchmark_csv.py and the playbook's timing extraction
# all look for the literal names benchmarks.json / benchmarks.csv / guidellm.log,
# which no per-rate invocation writes on its own.
# $1 = shared test_run_id, $2 = log to append tool output to.
merge_rate_results() {
    local run_id="$1" log="$2"
    local results_root="$ANSIBLE_DIR/../../../results/llm"
    local -a run_dirs=()
    local f d

    shopt -s nullglob
    for f in "$results_root"/*/*"-${run_id}"/*/benchmarks-*.json; do
        d=$(dirname "$f")
        [[ " ${run_dirs[*]:-} " == *" $d "* ]] || run_dirs+=("$d")
    done
    shopt -u nullglob

    if (( ${#run_dirs[@]} == 0 )); then
        echo "  WARNING: no per-rate results for run id $run_id under $results_root."
        return 1
    fi

    for d in "${run_dirs[@]}"; do
        echo "  [$(date '+%H:%M:%S')] --- merging per-rate results in $d ---"
        if ! python3 "$SCRIPT_DIR/merge_guidellm_rates.py" "$d" >> "$log" 2>&1; then
            echo "  WARNING: merging per-rate results failed for $d (see $log)."
            continue
        fi
        # The playbook ran its timing extraction against a benchmarks.json that
        # did not exist yet, so redo it now that the merged file is in place.
        python3 "$ANSIBLE_DIR/../scripts/ansible/extract_benchmark_timings.py" \
            "$d/benchmarks.json" "$d/test-metadata.json" >> "$log" 2>&1 || true
    done
}

run_one() {
    local variant="$1"
    local entry="$2"

    parse_model_entry "$entry"
    local model="$MODEL_PATH"
    local tag="$MODEL_TAG"

    # test_name uses ONLY [A-Za-z0-9-] and is <= 100 chars (ansible validates).
    local test_name="${TEST_NAME_PREFIX}-${variant}-${tag}"

    # Log file name can use the sanitized HF path -- not subject to test_name rules.
    local log_tag
    log_tag=$(sanitize_model "$model")
    local run_log="$LOG_DIR/${variant}_${log_tag}.log"

    # Resolve the actual path/repo to load (local absolute path or HF repo id).
    local model_path
    model_path=$(resolve_model_path "$model")

    banner "[$variant] $model"
    echo "  test_name  = $test_name (${#test_name} chars)"
    echo "  model_path = $model_path"
    if $IS_AUDIO; then
        echo "  scenario   = $AUDIO_SCENARIO"
        echo "  num_files  = ${AUDIO_NUM_FILES:-<scenario default>}"
        echo "  p95_target = ${AUDIO_P95_TARGET}s"
        echo "  endpoint   = http://localhost:${NGINX_PORT} (external, via nginx)"
    elif $IS_EMBEDDING; then
        echo "  scenario   = $EMBEDDING_SCENARIO"
        echo "  prompts    = ${EMBEDDING_NUM_PROMPTS:-<repo default>}"
        echo "  bench_cpus = ${EMBEDDING_BENCH_CPUS:-<unpinned>}"
        echo "  bench_numa = ${EMBEDDING_BENCH_NUMA:-<all nodes>}"
    else
        echo "  rates      = $GUIDELLM_RATES"
        echo "  window     = $(window_summary)"
        echo "  max_conc   = $GUIDELLM_MAX_CONCURRENCY"
        if $PER_RATE_WINDOW && ! $GUIDELLM_WAIT_TIMEOUT_EXPLICIT; then
            echo "  wait_tmout = per rate (window + 900s, min 3600s)"
        else
            echo "  wait_tmout = ${GUIDELLM_WAIT_TIMEOUT}s"
        fi
    fi
    echo "  workloads  = ${BASE_WORKLOADS[*]}"
    echo "  max_ctx    = $VLLM_MAX_MODEL_LEN ($MAX_MODEL_LEN_SOURCE)"
    echo "  dtype      = ${DTYPE:-<vllm default>}"
    echo "  log        = $run_log"

    # Build start.sh args (same for dry-run and real run).
    # --torchao is only added when both --quant and --torchao were passed.
    local start_args=(--regenerate --timeout "$HEALTH_TIMEOUT" -m "$model_path")
    if [[ "$TORCHAO" == "true" ]]; then
        start_args=(--torchao "${start_args[@]}")
    fi
    if [[ -n "$MODELS_DIR" ]]; then
        start_args+=(--models-dir "$MODELS_DIR")
    fi
    if [[ "$variant" == "native" ]]; then
        start_args=(--native "${start_args[@]}")
    fi
    if [[ "${NO_MEM_LIMIT:-false}" == "true" ]]; then
        start_args+=(--no-mem-limit)
    fi
    if [[ -n "${DTYPE:-}" ]]; then
        start_args+=(--dtype "$DTYPE")
    fi
    # Per-run trace dir keyed by test_name so multi-model/variant runs don't
    # overwrite each other. Forwarded to start.sh -> generate-config.sh so the
    # host mount and the collection path below agree.
    local run_profiler_dir=""
    if [[ "$PROFILE" == "true" ]]; then
        run_profiler_dir="$PROFILER_DIR/$test_name"
        start_args+=(--profile --profiler-dir "$run_profiler_dir")
    fi
    if [[ "$CLI_LMCACHE" == "true" ]]; then
        start_args+=(--lmcache)
    fi
    if [[ "$CLI_NO_LIMITS" == "true" ]]; then
        start_args+=(--no-limits)
    fi

    # Without cgroup cpuset delegation (--no-limits) the GuideLLM container
    # cannot be cpuset-pinned, so pin the whole Ansible/GuideLLM process tree.
    local ansible_prefix=()
    if [[ "$CLI_NO_LIMITS" == "true" && -n "$GUIDELLM_CPUS" ]]; then
        ansible_prefix=(taskset -c "$GUIDELLM_CPUS")
    fi

    if $DRY_RUN; then
        echo ""
        echo "  [dry-run] would call: $SCRIPT_DIR/stop.sh --clean"
        echo "  [dry-run] would call: $SCRIPT_DIR/start.sh ${start_args[*]}"
        if [[ "$PROFILE" == "true" ]]; then
            echo "  [dry-run] profiler ON: traces -> $run_profiler_dir"
            echo "  [dry-run] would POST /start_profile to vllm-instance-1..$NUM_INSTANCES before ansible"
            echo "  [dry-run] would POST /stop_profile  to vllm-instance-1..$NUM_INSTANCES after ansible"
            echo "  [dry-run] would wait for trace flush before stop.sh --clean"
        fi
        echo "  [dry-run] would cd:   $ANSIBLE_DIR"
        for workload in "${BASE_WORKLOADS[@]}"; do
        echo "  [dry-run] would run ansible-playbook (workload=$workload):"
        if $IS_AUDIO; then
        cat <<EOF
    ansible-playbook -i inventory/hosts.yml \\
        audio-benchmark.yml \\
        --connection=local \\
        -e "ansible_become=false" \\
        -e "ansible_python_interpreter=/usr/bin/python3" \\
        -e "test_model=$model_path" \\
        -e "test_scenario=$AUDIO_SCENARIO" \\
        -e "test_name=$test_name" \\
        -e "requested_cores=$CORES_PER_INSTANCE" \\
        -e "vllm_port=$NGINX_PORT" \\
        -e '{"vllm_endpoint":{"mode":"external","external":{"url":"http://localhost:$NGINX_PORT"}}}' \\
        ${AUDIO_NUM_FILES:+-e \"audio_num_files=$AUDIO_NUM_FILES\" }-e "audio_p95_target=$AUDIO_P95_TARGET" \\
        -e "vllm_audio_deps_bundled=true"
EOF
        continue
        fi
        if $IS_EMBEDDING; then
        cat <<EOF
    SCENARIO=$EMBEDDING_SCENARIO \\
    ansible-playbook -i inventory/hosts.yml \\
        embedding-benchmark.yml \\
        --connection=local \\
        -e "ansible_become=false" \\
        -e "ansible_python_interpreter=/usr/bin/python3" \\
        -e "test_model=$model_path" \\
        -e "test_name=$test_name" \\
        -e "requested_cores=$CORES_PER_INSTANCE" \\
        ${EMBEDDING_NUM_PROMPTS:+-e \"num_prompts=$EMBEDDING_NUM_PROMPTS\" }-e "embedding_baseline_max_concurrency=$EMBEDDING_BASELINE_MAX_CONCURRENCY" \\
        -e "vllm_bench_binary=$EMBEDDING_BENCH_BINARY" \\
        ${EMBEDDING_BENCH_CPUS:+-e \"vllm_bench_cpus=$EMBEDDING_BENCH_CPUS\" }${EMBEDDING_BENCH_NUMA:+-e \"vllm_bench_numa_node=$EMBEDDING_BENCH_NUMA\" }\\
        -e '{"health_check":{"timeout":600,"interval":5}}'
EOF
        continue
        fi
        local dry_batches=("$GUIDELLM_RATES") dry_run_id="" dry_batch
        if $PER_RATE_WINDOW; then
            dry_batches=("${GUIDELLM_RATE_LIST[@]}")
            dry_run_id="<timestamp>-${test_name}"
        fi
        for dry_batch in "${dry_batches[@]}"; do
        local dry_rates="$GUIDELLM_RATES" dry_seconds="$GUIDELLM_MAX_SECONDS"
        local dry_timeout="$GUIDELLM_WAIT_TIMEOUT" dry_extra=""
        if $PER_RATE_WINDOW; then
            dry_rates="[$dry_batch]"
            dry_seconds=$(guidellm_window_for "$dry_batch")
            dry_timeout=$(guidellm_wait_timeout_for "$dry_seconds")
            dry_extra="-e \"test_run_id=$dry_run_id\" -e \"guidellm_output_suffix=-$(sanitize_rate "$dry_batch")\" "
        fi
        cat <<EOF
    ${ansible_prefix[*]:+${ansible_prefix[*]} }ansible-playbook -i inventory/hosts.yml \\
        llm-benchmark-concurrent-load.yml \\
        --connection=local \\
        -e "ansible_python_interpreter=/usr/bin/python3" \\
        -e "test_model=$model_path" \\
        -e "base_workload=$workload" \\
        -e "skip_phase_2=true" \\
        -e "skip_phase_3=true" \\
        -e "guidellm_rate=$dry_rates" \\
        -e "guidellm_max_seconds=$dry_seconds" \\
        -e "guidellm_max_requests=$GUIDELLM_MAX_REQUESTS" \\
        -e "guidellm_max_concurrency=$GUIDELLM_MAX_CONCURRENCY" \\
        -e "guidellm_wait_timeout_seconds=$dry_timeout" \\
        -e "guidellm_no_cgroup_limits=$CLI_NO_LIMITS" \\
        -e "test_name=$test_name" \\
        ${dry_extra}${GUIDELLM_CPUS:+-e \"guidellm_cpus=$GUIDELLM_CPUS\" }${GUIDELLM_NUMA_NODE:+-e \"guidellm_numa_node=$GUIDELLM_NUMA_NODE\" }\\
        -e '{"health_check":{"timeout":600,"interval":5}}'
EOF
        done
        if $PER_RATE_WINDOW; then
            echo "  [dry-run] would merge per-rate outputs: merge_guidellm_rates.py <results>/*-${dry_run_id}/*"
        fi
        done
        echo "  [dry-run] would call: $SCRIPT_DIR/stop.sh --clean"
        echo "  [dry-run] would run:  sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches"
        echo "  [dry-run] Done: $test_name"
        return 0
    fi

    # --- 1. Stop any previous stack -----------------------------------------
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Stopping any previous stack ---"
    "$SCRIPT_DIR/stop.sh" --clean >> "$run_log" 2>&1 || true

    # --- 2. Start fresh stack with the right image + model ------------------
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Starting stack ($variant, $model) ---"
    if ! "$SCRIPT_DIR/start.sh" "${start_args[@]}" >> "$run_log" 2>&1; then
        echo "  ERROR: start.sh failed for $variant / $model_path -- skipping."
        echo "  See $run_log for details."
        echo "  Containers left running for inspection. To inspect:"
        echo "    podman ps -a --filter 'name=vllm-'"
        echo "    podman logs vllm-instance-1"
        echo "  When done, manually clean with: $SCRIPT_DIR/stop.sh --clean"
        return 1
    fi

    # --- 2b. Start torch profiler on all instances (wraps the whole run) ----
    if [[ "$PROFILE" == "true" ]]; then
        echo "--- Starting torch profiler (all instances) ---"
        profile_post_all start_profile "$run_log"
    fi

    # --- 3. Run ansible benchmark sweep -------------------------------------
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Running ansible benchmark ---"
    pushd "$ANSIBLE_DIR" > /dev/null

    # Optional GuideLLM load-generator pinning overrides.
    local guidellm_extra_args=()
    if [[ -n "$GUIDELLM_CPUS" ]]; then
        guidellm_extra_args+=(-e "guidellm_cpus=$GUIDELLM_CPUS")
    fi
    if [[ -n "$GUIDELLM_NUMA_NODE" ]]; then
        guidellm_extra_args+=(-e "guidellm_numa_node=$GUIDELLM_NUMA_NODE")
    fi

    # Optional audio-benchmark override: only pass audio_num_files when set so an
    # unset value inherits the scenario YAML's own num_samples.
    local audio_extra_args=()
    if [[ -n "$AUDIO_NUM_FILES" ]]; then
        audio_extra_args+=(-e "audio_num_files=$AUDIO_NUM_FILES")
    fi

    # Optional vllm-bench overrides (embedding path). num_prompts is only passed
    # when EMBEDDING_NUM_PROMPTS is set; otherwise the repo default applies (an
    # empty -e num_prompts= would shadow it with "").
    local embedding_extra_args=()
    if [[ -n "$EMBEDDING_NUM_PROMPTS" ]]; then
        embedding_extra_args+=(-e "num_prompts=$EMBEDDING_NUM_PROMPTS")
    fi
    # Mount the same pre-warmed HF cache into the vllm-bench container so the
    # tokenizer loads offline (the bench node has no internet).
    embedding_extra_args+=(-e "bench_hf_cache_dir=$HF_CACHE_DIR")
    # On hosts without a subuid mapping (e.g. Jenkins) the bench image's baked-in
    # non-root user can't write to the bind-mounted results/cache dirs, so run
    # the container as 0:0 (== host user under rootless). Mirrors start.sh's
    # FORCE_ROOT_USER=auto gating for the server container.
    if grep -q "^${USER:-$(id -un)}:" /etc/subuid 2>/dev/null; then
        embedding_extra_args+=(-e "bench_run_as_root=false")
    else
        embedding_extra_args+=(-e "bench_run_as_root=true")
    fi
    if [[ -n "$EMBEDDING_BENCH_CPUS" ]]; then
        embedding_extra_args+=(-e "vllm_bench_cpus=$EMBEDDING_BENCH_CPUS")
    fi
    if [[ -n "$EMBEDDING_BENCH_NUMA" ]]; then
        embedding_extra_args+=(-e "vllm_bench_numa_node=$EMBEDDING_BENCH_NUMA")
    fi

    if (( ${#ansible_prefix[@]} )); then
        echo "  --no-limits: pinning Ansible + GuideLLM with: ${ansible_prefix[*]}"
    fi
    local benchmark_failed=false

    for workload in "${BASE_WORKLOADS[@]}"; do
        local wl_log="$LOG_DIR/${variant}_${log_tag}_${workload}.log"
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] --- workload=$workload (log: $wl_log) ---"

        if $IS_AUDIO; then
            # External mode against nginx: vllm_port points bench_config and the
            # connectivity checks at the nginx port; vllm_endpoint.external.url
            # covers the model-detection path. vllm_audio_deps_bundled=true skips
            # the audio image build (the zentorch image already ships vllm[audio]).
            #
            # ansible_become=false: the inventory derives ansible_become from
            # ansible_user (default ec2-user), so these plays default to
            # "become: true" -- upstream targets an EC2 DUT over SSH. This sweep
            # is the opposite shape: --connection=local against the rootless stack
            # start.sh just brought up as the invoking user, writing into this
            # checkout. Nothing on the external-endpoint path needs root, and
            # escalating breaks wherever $HOME is NFS-backed -- root_squash maps
            # root to nobody, so "Setup results directory on load generator"
            # (file + mode + recurse) fails chmod'ing our own files with
            # "[Errno 1] Operation not permitted".
            if ! ansible-playbook -i inventory/hosts.yml \
                    audio-benchmark.yml \
                    --connection=local \
                    -e "ansible_become=false" \
                    -e "ansible_python_interpreter=/usr/bin/python3" \
                    -e "test_model=$model_path" \
                    -e "test_scenario=$AUDIO_SCENARIO" \
                    -e "test_name=$test_name" \
                    -e "requested_cores=$CORES_PER_INSTANCE" \
                    -e "vllm_port=$NGINX_PORT" \
                    -e "{\"vllm_endpoint\":{\"mode\":\"external\",\"external\":{\"url\":\"http://localhost:$NGINX_PORT\"}}}" \
                    "${audio_extra_args[@]}" \
                    -e "audio_p95_target=$AUDIO_P95_TARGET" \
                    -e "vllm_audio_deps_bundled=true" \
                    >> "$wl_log" 2>&1; then
                echo "  WARNING: ansible audio benchmark failed for $variant / $model."
                echo "  See $wl_log for details."
                benchmark_failed=true
            fi
            continue
        fi

        if $IS_EMBEDDING; then
            # SCENARIO goes via the environment because the playbook reads it
            # that way. num_prompts is folded into embedding_extra_args only when
            # overridden, so an unset value inherits the repo default.
            # ansible_become=false for the reason spelled out in the audio
            # invocation above; on this path the first casualty is "Create test
            # run base directory", which fails under NFS root_squash with
            # "[Errno 13] Permission denied" on the results dir.
            if ! SCENARIO="$EMBEDDING_SCENARIO" \
                 ansible-playbook -i inventory/hosts.yml \
                    embedding-benchmark.yml \
                    --connection=local \
                    -e "ansible_become=false" \
                    -e "ansible_python_interpreter=/usr/bin/python3" \
                    -e "test_model=$model_path" \
                    -e "test_name=$test_name" \
                    -e "requested_cores=$CORES_PER_INSTANCE" \
                    -e "embedding_baseline_max_concurrency=$EMBEDDING_BASELINE_MAX_CONCURRENCY" \
                    -e "vllm_bench_binary=$EMBEDDING_BENCH_BINARY" \
                    "${embedding_extra_args[@]}" \
                    -e '{"health_check":{"timeout":600,"interval":5}}' \
                    >> "$wl_log" 2>&1; then
                echo "  WARNING: ansible embedding benchmark failed for $variant / $model."
                echo "  See $wl_log for details."
                benchmark_failed=true
            fi
            continue
        fi

        # With a per-rate window, each rate is its own GuideLLM invocation and
        # they must land in one results directory, so the run id is fixed here
        # rather than generated per playbook run. The format matches the one
        # llm-benchmark-auto.yml generates ("<timestamp>-<test_name>") so the
        # result paths the extract scripts parse stay unchanged.
        local batches=("$GUIDELLM_RATES") run_id="" batch
        if $PER_RATE_WINDOW; then
            batches=("${GUIDELLM_RATE_LIST[@]}")
            run_id="$(date +%Y%m%d-%H%M%S)-${test_name}"
        fi

        for batch in "${batches[@]}"; do
            local batch_rates="$GUIDELLM_RATES"
            local batch_seconds="$GUIDELLM_MAX_SECONDS"
            local batch_timeout="$GUIDELLM_WAIT_TIMEOUT"
            local batch_args=() batch_label=""
            if $PER_RATE_WINDOW; then
                batch_rates="[$batch]"
                batch_seconds=$(guidellm_window_for "$batch")
                batch_timeout=$(guidellm_wait_timeout_for "$batch_seconds")
                batch_args=(
                    -e "test_run_id=$run_id"
                    -e "guidellm_output_suffix=-$(sanitize_rate "$batch")"
                )
                batch_label=" / rate $batch"
                echo "    [$(date '+%H:%M:%S')] rate=$batch window=${batch_seconds}s timeout=${batch_timeout}s"
            fi

            if ! "${ansible_prefix[@]}" ansible-playbook -i inventory/hosts.yml \
                    llm-benchmark-concurrent-load.yml \
                    --connection=local \
                    -e "ansible_python_interpreter=/usr/bin/python3" \
                    -e "test_model=$model_path" \
                    -e "base_workload=$workload" \
                    -e "skip_phase_2=true" \
                    -e "skip_phase_3=true" \
                    -e "guidellm_rate=$batch_rates" \
                    -e "guidellm_max_seconds=$batch_seconds" \
                    -e "guidellm_max_requests=$GUIDELLM_MAX_REQUESTS" \
                    -e "guidellm_max_concurrency=$GUIDELLM_MAX_CONCURRENCY" \
                    -e "guidellm_wait_timeout_seconds=$batch_timeout" \
                    -e "guidellm_no_cgroup_limits=$CLI_NO_LIMITS" \
                    -e "test_name=$test_name" \
                    "${batch_args[@]}" \
                    "${guidellm_extra_args[@]}" \
                    -e '{"health_check":{"timeout":600,"interval":5}}' \
                    >> "$wl_log" 2>&1; then
                echo "  WARNING: ansible benchmark failed for $variant / $model / $workload$batch_label."
                echo "  See $wl_log for details."
                benchmark_failed=true
            fi
        done

        if $PER_RATE_WINDOW; then
            merge_rate_results "$run_id" "$wl_log" || true
        fi
    done

    popd > /dev/null

    # --- 3b. Stop torch profiler + wait for traces to flush -----------------
    # Must finish BEFORE teardown: stop.sh --clean kills the containers and
    # would truncate a still-serializing trace.
    if [[ "$PROFILE" == "true" ]]; then
        echo "--- Stopping torch profiler (all instances) ---"
        profile_post_all stop_profile "$run_log"
        profile_wait_flush "$run_profiler_dir"
        echo "  [profile] traces for $test_name in: $run_profiler_dir"
    fi

    # --- 4. Tear down -------------------------------------------------------
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Stopping stack ---"
    "$SCRIPT_DIR/stop.sh" --clean >> "$run_log" 2>&1 || true

    # --- 5. Drop page cache / dentries / inodes -----------------------------
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- Dropping page cache ---"
    sync
    # shellcheck disable=SC2024
    if echo 3 | sudo -n tee /proc/sys/vm/drop_caches >> "$run_log" 2>&1; then
        echo "  Page cache dropped."
    else
        echo "  WARNING: failed to drop page cache (need passwordless sudo for 'tee /proc/sys/vm/drop_caches')."
    fi

    if $benchmark_failed; then
        echo "  ERROR: benchmark failed: $test_name"
        return 1
    fi
    echo "  Done: $test_name"
}

# ----------------------------------------------------------------------------
# Pre-flight validation
# ----------------------------------------------------------------------------
banner "Pre-flight checks"

preflight_ok=true
check_path() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        echo "  [OK]   $label : $path"
    else
        echo "  [MISS] $label : $path"
        preflight_ok=false
    fi
}

check_path "SCRIPT_DIR    " "$SCRIPT_DIR"
check_path "start.sh      " "$SCRIPT_DIR/start.sh"
check_path "stop.sh       " "$SCRIPT_DIR/stop.sh"
check_path "LOG_DIR       " "$LOG_DIR"
check_path "ANSIBLE_DIR   " "$ANSIBLE_DIR"
if $IS_AUDIO; then
    check_path "playbook      " "$ANSIBLE_DIR/audio-benchmark.yml"
elif $IS_EMBEDDING; then
    check_path "playbook      " "$ANSIBLE_DIR/embedding-benchmark.yml"
else
    check_path "playbook      " "$ANSIBLE_DIR/llm-benchmark-concurrent-load.yml"
fi
check_path "inventory     " "$ANSIBLE_DIR/inventory/hosts.yml"

if (( ${#MODELS[@]} == 0 )); then
    echo "  [MISS] MODELS array is empty"
    preflight_ok=false
else
    echo "  [OK]   MODELS array: ${#MODELS[@]} entries"
fi
if (( ${#VARIANTS[@]} == 0 )); then
    echo "  [MISS] VARIANTS array is empty"
    preflight_ok=false
else
    echo "  [OK]   VARIANTS array: ${VARIANTS[*]}"
fi

if command -v ansible-playbook >/dev/null 2>&1; then
    echo "  [OK]   ansible-playbook found at: $(command -v ansible-playbook)"
else
    echo "  [MISS] ansible-playbook not found in PATH"
    preflight_ok=false
fi
if command -v podman >/dev/null 2>&1; then
    echo "  [OK]   podman found at: $(command -v podman)"
else
    echo "  [WARN] podman not found in PATH (start.sh will fail)"
fi

if $DRY_RUN; then
    echo "  [INFO] HF_TOKEN check skipped (dry-run)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  [OK]   HF_TOKEN is set (${#HF_TOKEN} chars)"
else
    echo "  [WARN] HF_TOKEN is not set"
fi

if [[ -n "$MODELS_DIR" ]]; then
    echo "  [OK]   MODELS_DIR    : $MODELS_DIR (will be bind-mounted into containers)"
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        if [[ -d "$MODELS_DIR/$MODEL_PATH" ]]; then
            echo "  [OK]   model dir    : $MODELS_DIR/$MODEL_PATH"
        else
            echo "  [MISS] model dir    : $MODELS_DIR/$MODEL_PATH"
            preflight_ok=false
        fi
    done
else
    echo "  [INFO] MODELS_DIR not set -- models will be downloaded from HuggingFace hub"
fi

if [[ "$QUANT" == "true" ]]; then
    if [[ "$TORCHAO" == "true" ]]; then
        echo "  [OK]   --quant ON     : torchao image (start.sh --torchao)"
    else
        echo "  [OK]   --quant ON     : llm-compressor (base image, no torchao)"
    fi
else
    echo "  [INFO] --quant OFF    : base image, no torchao"
fi

# A per-rate window needs the harness to give each rate its own output name,
# which it takes from guidellm_output_suffix. A checkout without that support
# writes plain benchmarks.json for every rate into the one run directory the
# rates share, so each rate overwrites the previous one and only the last
# survives -- indistinguishable, afterwards, from a sweep that only ever ran its
# highest concurrency. Rather than lose the rest of the ladder, drop back to one
# window covering all of it, which every checkout handles. The window has to
# collapse to a single number too: the map form is what selects the per-rate
# path, and ansible would otherwise receive the map as max_seconds.
if $PER_RATE_WINDOW \
   && ! grep -q 'guidellm_output_suffix' \
        "$ANSIBLE_DIR/roles/benchmark_guidellm/tasks/main.yml" 2>/dev/null; then
    _fallback_window=0
    for _rate in "${GUIDELLM_RATE_LIST[@]}"; do
        _rate_window=$(guidellm_window_for "$_rate")
        (( _rate_window > _fallback_window )) && _fallback_window=$_rate_window
    done
    echo "  WARN: this vllm-cpu-perf-eval checkout does not read" >&2
    echo "        guidellm_output_suffix, so a per-rate run would keep only the" >&2
    echo "        last rate's results. Apply vllm-cpu-perf-eval.patch to use" >&2
    echo "        per-rate windows; falling back to one ${_fallback_window}s" >&2
    echo "        window for every rate so the whole ladder is measured." >&2
    PER_RATE_WINDOW=false
    GUIDELLM_MAX_SECONDS="$_fallback_window"
    unset _rate _rate_window _fallback_window
fi

# The benchmarking window each rate will get, and what that costs in wall time.
if ! $IS_AUDIO && ! $IS_EMBEDDING; then
    echo ""
    echo "  --- guidellm benchmarking window ---"
    if $PER_RATE_WINDOW; then
        echo "  [OK]   mode          : per rate (one guidellm run per rate, merged afterwards)"
    else
        echo "  [OK]   mode          : single window for all rates (one guidellm run)"
    fi
    window_total=0
    for rate in "${GUIDELLM_RATE_LIST[@]}"; do
        if $PER_RATE_WINDOW; then
            rate_seconds=$(guidellm_window_for "$rate")
            if [[ -n "${GUIDELLM_RATE_SECONDS[$rate]:-}" ]]; then
                printf "  [OK]   rate %-9s: %ss\n" "$rate" "$rate_seconds"
            else
                printf "  [INFO] rate %-9s: %ss (not in the map, using GUIDELLM_MAX_SECONDS_DEFAULT)\n" \
                    "$rate" "$rate_seconds"
            fi
        else
            rate_seconds="$GUIDELLM_MAX_SECONDS"
            printf "  [OK]   rate %-9s: %ss\n" "$rate" "$rate_seconds"
        fi
        window_total=$(( window_total + rate_seconds ))
    done
    printf "  [INFO] benchmark time: ~%d min per (variant, model, workload), excluding startup\n" \
        $(( window_total / 60 ))
    unset window_total rate rate_seconds
fi

# Context window vs KV cache. vLLM aborts at startup with "The model's max seq
# len (N) is larger than the maximum number of tokens that can be stored in KV
# cache (M)" -- catch that here instead of 20 minutes into the sweep.
echo ""
echo "  --- context window / KV-cache fit ---"
echo "  [OK]   --max-model-len : $VLLM_MAX_MODEL_LEN ($MAX_MODEL_LEN_SOURCE)"
echo "  [INFO] KV cache space  : ${VLLM_KV_CACHE_SPACE} GiB per instance"
if [[ -f "$WORKLOADS_FILE" ]]; then
    echo "  [OK]   workloads file : $WORKLOADS_FILE"
else
    echo "  [WARN] workloads file : $WORKLOADS_FILE (missing)"
fi
if $IS_AUDIO; then
    echo "  [INFO] fit check skipped (audio: whisper encoder-decoder KV shape not modeled)"
elif [[ "$KV_FIT_CHECK" != "true" ]]; then
    echo "  [INFO] fit check disabled (KV_FIT_CHECK=$KV_FIT_CHECK)"
else
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        kv_model_path=$(resolve_model_path "$MODEL_PATH")
        if ! kv_cfg=$(model_config_json "$kv_model_path"); then
            echo "  [INFO] $MODEL_TAG: not in $HF_CACHE_DIR yet -- fit unverified until first download"
        elif ! kv_capacity=$(kv_cache_token_capacity "$kv_cfg"); then
            echo "  [INFO] $MODEL_TAG: no layer/head counts in $kv_cfg -- fit unverified"
        elif (( VLLM_MAX_MODEL_LEN > kv_capacity )); then
            kv_needed=$(awk -v g="$VLLM_KV_CACHE_SPACE" -v m="$VLLM_MAX_MODEL_LEN" \
                            -v c="$kv_capacity" 'BEGIN { printf "%d", int(g*m/c) + 1 }')
            echo "  [MISS] $MODEL_TAG: max-model-len $VLLM_MAX_MODEL_LEN > ~$kv_capacity tokens that fit in ${VLLM_KV_CACHE_SPACE} GiB"
            echo "         vLLM would abort at startup. Either export VLLM_KV_CACHE_SPACE=$kv_needed"
            echo "         (or higher), or lower VLLM_MAX_MODEL_LEN / drop the longest workload."
            echo "         Export KV_FIT_CHECK=false to override this check."
            preflight_ok=false
        else
            echo "  [OK]   $MODEL_TAG: $VLLM_MAX_MODEL_LEN <= ~$kv_capacity tokens of KV cache"
        fi
    done
fi

if $IS_AUDIO; then
    echo ""
    echo "  --- audio benchmark ---"
    echo "  [OK]   scenario      : $AUDIO_SCENARIO"
    echo "  [OK]   num_files     : ${AUDIO_NUM_FILES:-<scenario default>}"
    echo "  [OK]   p95 target    : ${AUDIO_P95_TARGET}s"
    echo "  [OK]   endpoint      : http://localhost:${NGINX_PORT} (external, via nginx)"
    echo "  [OK]   max-model-len : $VLLM_MAX_MODEL_LEN (whisper)"
    echo "  [OK]   extra args    : ${EXTRA_VLLM_ARGS:-<none>}"
    echo "  [OK]   in-container  : ${SET_ENV_VAR_BEFORE_VLLM_SERVE:-<none>}"
    echo "  [OK]   results subdir: results/audio-models"
    scenario_file="$ANSIBLE_DIR/../../../tests/audio-models/${AUDIO_SCENARIO}.yaml"
    if [[ -f "$scenario_file" ]]; then
        echo "  [OK]   scenario file : $scenario_file"
    else
        echo "  [MISS] scenario file : $scenario_file"
        preflight_ok=false
    fi
fi

if $IS_EMBEDDING; then
    echo ""
    echo "  --- embedding benchmark ---"
    echo "  [OK]   scenario      : $EMBEDDING_SCENARIO"
    echo "  [OK]   num_prompts   : ${EMBEDDING_NUM_PROMPTS:-<repo default>}"
    echo "  [OK]   bench image   : ${VLLM_BENCH_CONTAINER_IMAGE:-<unset>}"
    echo "  [OK]   bench binary  : $EMBEDDING_BENCH_BINARY"
    echo "  [OK]   bench cpuset  : ${EMBEDDING_BENCH_CPUS:-<unpinned>} / mems ${EMBEDDING_BENCH_NUMA:-<all>}"
    echo "  [OK]   results subdir: results/$RESULTS_SUBDIR"
    # The latency scenario tops out at concurrency 196. Below ~20 waves the run
    # is dominated by ramp-up and the percentiles are noise.
    if [[ "$EMBEDDING_SCENARIO" != "baseline" && -n "$EMBEDDING_NUM_PROMPTS" ]] && (( EMBEDDING_NUM_PROMPTS < 20 * 196 )); then
        echo "  [WARN] num_prompts $EMBEDDING_NUM_PROMPTS is under 20x the max concurrency (196)."
        echo "         The latency points will be ramp-dominated. Consider EMBEDDING_NUM_PROMPTS>=3920."
    fi
    if [[ "$EMBEDDING_BENCH_CPUS" == *"$VLLM_START_CORE"* ]]; then
        echo "  [WARN] EMBEDDING_BENCH_CPUS ($EMBEDDING_BENCH_CPUS) may overlap the vLLM cores"
        echo "         (VLLM_START_CORE=$VLLM_START_CORE). The client would contend with the server."
    fi
fi

# Validate every (variant, model) -> test_name fits ansible's rules.
echo ""
echo "  --- test_name validation ---"
for variant in "${VARIANTS[@]}"; do
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        candidate="${TEST_NAME_PREFIX}-${variant}-${MODEL_TAG}"
        if validate_test_name "$candidate" 2>/dev/null; then
            printf "  [OK]   %-30s  (%d chars)\n" "$candidate" "${#candidate}"
        else
            printf "  [BAD]  %-30s  (%d chars, [A-Za-z0-9-] only, 1-100 chars)\n" \
                "$candidate" "${#candidate}"
            preflight_ok=false
        fi
    done
done

if ! $preflight_ok; then
    echo ""
    echo "Pre-flight failed. Fix the [MISS] items above before running." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
mode_label="online (guidellm)"
if $IS_AUDIO; then
    mode_label="audio (guidellm)"
elif $IS_EMBEDDING; then
    mode_label="embedding (vllm bench serve)"
fi
if $DRY_RUN; then
    banner "DRY RUN -- ${#MODELS[@]} models x ${#VARIANTS[@]} variants -- $mode_label (no commands will execute)"
else
    banner "Benchmark sweep -- ${#MODELS[@]} models x ${#VARIANTS[@]} variants -- $mode_label"
fi
echo "  Variants : ${VARIANTS[*]}"
echo "  Models   :"
for entry in "${MODELS[@]}"; do
    parse_model_entry "$entry"
    printf "    - %-40s -> %s\n" "$MODEL_PATH" "$MODEL_TAG"
done
if $IS_AUDIO; then
    echo "  Scenario : $AUDIO_SCENARIO"
    echo "  NumFiles : ${AUDIO_NUM_FILES:-<scenario default>}"
elif $IS_EMBEDDING; then
    echo "  Scenario : $EMBEDDING_SCENARIO"
    echo "  Prompts  : ${EMBEDDING_NUM_PROMPTS:-<repo default>}"
else
    echo "  Rates    : $GUIDELLM_RATES"
    echo "  Window   : $(window_summary)"
    if $PER_RATE_WINDOW && ! $GUIDELLM_WAIT_TIMEOUT_EXPLICIT; then
        echo "  WaitTmout: per rate (window + 900s, min 3600s)"
    else
        echo "  WaitTmout: ${GUIDELLM_WAIT_TIMEOUT}s"
    fi
fi
echo "  Workloads: ${BASE_WORKLOADS[*]}"
echo "  MaxCtxLen: $VLLM_MAX_MODEL_LEN ($MAX_MODEL_LEN_SOURCE)"
echo "  Log dir  : $LOG_DIR"
if [[ "$PROFILE" == "true" ]]; then
    echo "  Profiler : ON -> $PROFILER_DIR/<test_name> (per-instance subdirs)"
fi

total=$(( ${#MODELS[@]} * ${#VARIANTS[@]} ))
i=0
failed=()

for variant in "${VARIANTS[@]}"; do
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        i=$((i + 1))
        echo ""
        echo ">>> [$i/$total] variant=$variant model=$MODEL_PATH"
        if ! run_one "$variant" "$entry"; then
            failed+=("$variant / $MODEL_PATH")
        fi
    done
done

banner "Sweep complete"
echo "  Total runs : $total"
echo "  Failed     : ${#failed[@]}"
for f in "${failed[@]}"; do echo "    - $f"; done
echo "  Logs       : $LOG_DIR"
echo ""
echo "Results from each run are in:"
if $IS_AUDIO; then
    echo "  $ANSIBLE_DIR/../../../results/audio-models/<sanitized-model>/<scenario>-<test_name>-<timestamp>/<stage>/"
    echo ""
    echo "Extract a CSV with:"
    echo "  python3 $SCRIPT_DIR/extract_audio_metrics.py $ANSIBLE_DIR/../../../results/audio-models"
elif $IS_EMBEDDING; then
    # test_name is a PREFIX on the embedding run directory, a SUFFIX on the LLM
    # one -- see test_run_id in embedding-benchmark.yml.
    echo "  $ANSIBLE_DIR/../../../results/embedding/<sanitized-model>/<test_name>-<timestamp>/{baseline,latency}/"
    echo ""
    echo "Extract a CSV with:"
    echo "  python3 $SCRIPT_DIR/extract_embedding_metrics.py $ANSIBLE_DIR/../../../results/embedding"
else
    echo "  $ANSIBLE_DIR/../../../results/llm/<sanitized-model>/<workload>-<timestamp>-<test_name>/external-endpoint/"
fi

if (( ${#failed[@]} > 0 )); then
    exit 1
fi
