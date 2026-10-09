<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Multi-Instance vLLM + NGINX Benchmarking Setup

Automated scripts for deploying multiple CPU-pinned vLLM instances behind an NGINX load balancer, designed for RedHat multi-instance benchmarking on AMD EPYC (Turin) CPUs.

All commands below run from this skill's `scripts/` directory:

```bash
cd .claude/skills/benchmark-vllm-guidellm/scripts
```

The agent workflow (stages, the one-time patch setup and its expected conflicts) is in [SKILL.md](SKILL.md).

For cross-instance KV-cache sharing (a shared LMCache server, optionally backed by a Mooncake L2 pool, vs vLLM's per-instance prefix caching), see [LMCACHE.md](LMCACHE.md).

## Architecture

```
              Benchmark Client (GuideLLM)
                       |
                       v
              +----------------+
              |  NGINX :8080   |  CPUs 1-15 (configurable)
              |  round-robin   |
              +----------------+
               / |    |    | \
              v  v    v    v  v
          vllm-1  vllm-2  vllm-3  vllm-4  vllm-5
          :8000   :8000   :8000   :8000   :8000
         CPUs    CPUs    CPUs    CPUs    CPUs
         32-63   64-95   96-127  128-159 160-191
```

All containers share a Docker bridge network (`vllm-network`). NGINX resolves each vLLM instance by service name and distributes requests via round-robin. Only NGINX exposes a port to the host.

## Prerequisites

- `podman` and `podman-compose` installed on the DUT
- Sufficient CPU cores for the desired number of instances
- HuggingFace token (for gated models like Llama)

## Quick Start

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# 2. Generate config, start the stack, wait for healthy
export NUM_INSTANCES=3
export CORES_PER_INSTANCE=32
export VLLM_START_CORE=32
export NGINX_CORES=1-15
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export VLLM_IMAGE=<registry>/<repo>:<tag>   # required, no default
export MEM_LIMIT=200G
export VLLM_KV_CACHE_SPACE=90
export VLLM_MAX_MODEL_LEN=4096
export NGINX_PORT=8080
export OUTPUT_DIR=generated
export DRY_RUN=false
./start.sh

# 3. Validate everything works
./test-setup.sh

# 4. Run your benchmarks against http://localhost:8080
git clone https://github.com/redhat-et/vllm-cpu-perf-eval.git
# The patch is generated against this upstream commit; later upstream
# revisions do not apply cleanly.
git -C vllm-cpu-perf-eval checkout 2bacf71ce6329eecc58e2a5ec2d7b8c98ea67174

# Apply local fixes (rootless GuideLLM user, local model bind-mount,
# shared-prefix workloads, offline tokenizer cache, etc.). SKILL.md covers the
# embedding/audio patches that stack on top with --3way.
git -C vllm-cpu-perf-eval apply ../../patches/vllm-cpu-perf-eval.patch

cd vllm-cpu-perf-eval/automation/test-execution/ansible

export VLLM_ENDPOINT_MODE=external
export VLLM_ENDPOINT_URL=http://localhost:8080
export LOADGEN_HOSTNAME=localhost
export DUT_HOSTNAME=localhost
export ANSIBLE_SSH_USER=amd
export ANSIBLE_SSH_KEY=/home/<user>/.ssh/id_rsa
export HF_TOKEN=<your-huggingface-token>

ansible-playbook -i inventory/hosts.yml       llm-benchmark-concurrent-load.yml       --connection=local       -e "ansible_python_interpreter=/usr/bin/python3"       -e "test_model=meta-llama/Llama-3.1-8B-Instruct"       -e "base_workload=chat"       -e "skip_phase_2=true"       -e "skip_phase_3=true"       -e "guidellm_rate=[32]"       -e "guidellm_max_seconds=300"       -e "test_name=EPYC-NO-SMT-5INST" -e '{"health_check":{"timeout":600,"interval":5}}' -e "guidellm_profile=concurrent"

# 5. Tear down when done
./stop.sh

# 6. Results will be saved in the `results/` directory.
# Example:
vllm-cpu-perf-eval/results/llm/meta-llama__Llama-3.1-8B-Instruct/chat-20260512-054112-EPYC-NO-SMT-5INST/external-endpoint/
```


## Loading Models: HuggingFace vs Local

vLLM accepts either a HuggingFace repo id or an absolute filesystem path as the
model. The `-m/--model` flag (or `MODEL_NAME`) selects which.

### From HuggingFace Hub

Pass the repo id. The model downloads into each instance's HF cache on first
start (set `HF_TOKEN` for gated models like Llama).

```bash
export HF_TOKEN=hf_your_token_here
./start.sh --regenerate -m meta-llama/Llama-3.1-8B-Instruct
```

### From a local directory

Pre-download or copy the model onto the host, then expose its parent dir to the
containers with `--models-dir` and point `-m` at the **absolute path inside that
dir**. `--models-dir DIR` bind-mounts `DIR` (host) into every container at the
same absolute path, read-only, so `vllm serve` loads straight off disk — no
network, no HF cache.

```bash
# Model lives at /data/models/Llama-3.1-8B-Instruct on the host
./start.sh --regenerate \
  --models-dir /data/models \
  -m /data/models/Llama-3.1-8B-Instruct
```

`-m` must be an absolute path under `--models-dir` (the same path is valid on
host and in the container). `HF_TOKEN` is not needed for a local model.

### Sweeping multiple models (`run_sweep.sh`)

`run_sweep.sh` runs the guidellm benchmark across a list of models. Entries are
`"path-or-repo | short-tag"`; with `--models-dir` the left side is resolved
relative to it.

```bash
# Two local models, 7 instances (test_name prefix defaults to EPYC7)
./run_sweep.sh \
  --models-dir /data/models \
  -N 7 \
  -m "Llama-3.1-8B-Instruct | llama31-8b" \
  -m "gpt-oss-20b-BF16      | gptoss-20b"

# One HuggingFace model (no --models-dir)
./run_sweep.sh -m "meta-llama/Llama-3.1-8B-Instruct | llama31-8b"

# Validate config without launching anything
./run_sweep.sh --dry-run --models-dir /data/models -m "Llama-3.1-8B-Instruct | l8b"
```

#### Benchmarking window per rate

`GUIDELLM_MAX_SECONDS` caps how long each rate runs. A single number applies to
every rate, which under-serves the high ones: a request takes longer at higher
concurrency, so a window that gives several completions per stream at rate 32
gives one (or none) at rate 256, leaving that rate's latency percentiles resting
on a single sample. Pass a `<rate>:<seconds>` map to size each rate separately:

```bash
GUIDELLM_RATES="[32,64,128,256]" \
GUIDELLM_MAX_SECONDS="32:600,64:800,128:1100,256:1650" \
  ./run_sweep.sh -N 7 -m "Llama-3.1-8B-Instruct | llama31-8b"
```

Rates missing from the map use `GUIDELLM_MAX_SECONDS_DEFAULT` (300). Because
`max_duration` is a run-wide GuideLLM constraint, the map runs one GuideLLM
invocation per rate; they share a results directory, write
`benchmarks-<rate>.json` / `guidellm-<rate>.log`, and are merged back into the
usual `benchmarks.json` / `benchmarks.csv` / `guidellm.log` by
`merge_guidellm_rates.py` when the workload finishes. The merged `guidellm.log`
has the same one-table-per-title shape as a single-window run, so the perf
dashboard and every extract script read it unchanged. Needs GuideLLM v0.7+.
See [SKILL.md](SKILL.md#stage-5-load-parameters) for how to size the windows and
what the merge guarantees for result collection.

#### Running normal vs quantized models

Two CLI flags on `run_sweep.sh` control quantization handling. They only affect
the **container image**; the model-load path (HF repo vs local dir) is the same
in every case.

| Run type | Flags | Image used | Model source |
|----------|-------|------------|--------------|
| **Normal** (bf16/fp16) | *(none)* | base zentorch image as-is | HF repo id or `--models-dir` |
| **Quant — llm-compressor** | `--quant` | base image as-is (no rebuild) | HF repo id or `--models-dir` |
| **Quant — torchao** | `--quant --torchao` | derived image with `torchao` installed | HF repo id or `--models-dir` |

Rules:

- `--quant` marks a quantized run. By **default it means llm-compressor format**
  and uses the base image unchanged (no torchao install, no image rebuild).
- `--torchao` is **only valid together with `--quant`**. It builds a derived
  image with `torchao>=0.10.0` pre-installed — required for models with
  `quant_method=torchao`. Passing `--torchao` without `--quant` is an error.
- A local model dir is **never mandatory**. Any run type can pull the model from
  the HuggingFace hub (needs `HF_TOKEN`) or load it from `--models-dir`.

```bash
# 1. Normal (non-quant) model from HuggingFace
HF_TOKEN=hf_xxx ./run_sweep.sh \
  -N 5 -m "meta-llama/Llama-3.1-8B-Instruct | llama31-8b"

# 2. Normal model from a local dir
./run_sweep.sh --models-dir ~/models \
  -N 5 -m "Llama-3.1-8B-Instruct | llama31-8b"

# 3. Quant, llm-compressor — base image, model from HuggingFace
HF_TOKEN=hf_xxx ./run_sweep.sh --quant \
  -N 5 -m "RedHatAI/Llama-3.1-8B-Instruct-quantized.w8a8 | l8b-w8a8"

# 4. Quant, llm-compressor — base image, model from a local dir
./run_sweep.sh --quant --models-dir ~/models \
  -N 5 -m "Llama-3.1-8B-Instruct-w8a8 | l8b-w8a8"

# 5. Quant, torchao — builds the torchao image, model from HuggingFace
HF_TOKEN=hf_xxx ./run_sweep.sh --quant --torchao \
  -N 5 -m "pytorch/Llama-3.1-8B-Instruct-int4 | l8b-int4"
```

Notes:

- The `MODELS=(...)` array near the top of `run_sweep.sh` is the default model
  list. CLI `-m`/`--model` entries override it.
- For torchao runs you can pin the version via `TORCHAO_VERSION=0.17.0` in the
  environment (consumed by `start.sh`). Without `--torchao`, this env var is a
  no-op.
- For gated HuggingFace models (e.g. Llama) a real `HF_TOKEN` is required. For
  local-only runs, any placeholder (e.g. `local-models-no-token-needed`) works.

### Audio models (`-w audio`)

`run_sweep.sh -w audio` sweeps **ASR** (Whisper transcription) models instead of
generative ones. As with `-w embedding`, the stack half of the run is reused — N
vLLM instances behind nginx, same `start.sh`/`generate-config.sh` — because
nginx's catch-all `location /` already proxies `/v1/audio/transcriptions`. What
swaps out is the load-generator half: upstream's `audio-benchmark.yml` (run in
**external mode** against the nginx endpoint) driving GuideLLM's audio backend,
in place of `llm-benchmark-concurrent-load.yml` against `/v1/chat/completions`.

Whisper is an encoder-decoder model, so unlike embedding `-w audio` also sets a
few **stack-side** defaults before calling `start.sh` (all env-overridable):
`--max-model-len 448`, `--trust-remote-code`, `VLLM_KV_CACHE_SPACE=4`, and the
V1-engine CPU workaround `VLLM_USE_V2_MODEL_RUNNER=0`. The zentorch image already
ships `vllm[audio]`, so no audio image is built (`vllm_audio_deps_bundled=true`).

```bash
# Default audio model (openai/whisper-small), throughput scenario, 8 instances
./run_sweep.sh -w audio -N 8

# One model, quick validation scenario, whisper-tiny
AUDIO_SCENARIO=quick-test \
  ./run_sweep.sh -w audio -N 1 -m "openai/whisper-tiny | whisper-tiny"

# Latency scenario
AUDIO_SCENARIO=transcription-latency \
  ./run_sweep.sh -w audio -N 4 -m "openai/whisper-small | whisper-small"

# Validate without launching anything
./run_sweep.sh --dry-run -w audio -N 1
```

`BASE_WORKLOAD=audio` is equivalent to `-w audio`. The workload cannot be
combined with a generative or embedding one in the same invocation — a single
stack serves one model with one runner — so `-w audio -w chat` is a hard error.

When no `-m` is given, the default model is `openai/whisper-small`.

| Env var | Default | Meaning |
|---|---|---|
| `AUDIO_SCENARIO` | `transcription-throughput` | Scenario from `tests/audio-models/`: `transcription-throughput` (sequential + concurrent-2/4/8), `transcription-latency`, `transcription-quality` (WER/CER), `audio-duration-scaling`, `constant-rate-stress`, `format-comparison`, `quick-test` |
| `AUDIO_NUM_FILES` | *(scenario default)* | Files per stage; unset inherits the scenario YAML's `num_samples` |
| `AUDIO_P95_TARGET` | `2.0` | P95 latency target (s) used by the playbook's enterprise report |
| `AUDIO_VLLM_ENV` | `export VLLM_USE_V2_MODEL_RUNNER=0 &&` | In-container env prefix folded into `SET_ENV_VAR_BEFORE_VLLM_SERVE`; set `""` to drop the V1 CPU workaround |
| `VLLM_MAX_MODEL_LEN` | `448` | Whisper context window |
| `VLLM_KV_CACHE_SPACE` | `4` | GiB; encoder-decoder holds little KV, so the 90 GiB generative default is dropped |

`transcription-quality` additionally needs the controller to have
`pip install jiwer datasets soundfile requests` for the WER/CER evaluation step.

Results land under `results/audio-models/`, one directory per run — `test_name`
(the `EPYC<N>-<variant>-<tag>` token) is threaded into the run-dir name so
instance-count and backend sweeps stay distinguishable:

```
results/audio-models/openai__whisper-small/
  transcription-throughput-EPYC8-zentorch-whisper-small-20260804-051500/
    test-metadata.json
    sequential/benchmarks.json  benchmarks.csv  stage-metadata.json
    concurrent-2/ ... concurrent-8/
```

Flatten a whole tree into one CSV — the `instances` and `backend` columns come
from the run directory name, so rows from several sweeps can be concatenated and
plotted against instance count:

```bash
python3 extract_audio_metrics.py results/audio-models -o audio-sweep.csv
### Embedding models (`-w embedding`)

`run_sweep.sh -w embedding` sweeps **pooling** (embedding) models instead of
generative ones. The stack half of the run is unchanged — N vLLM instances behind
nginx, same `start.sh`/`generate-config.sh` — because vLLM picks the pooling
runner up from the model config on its own, and nginx's catch-all `location /`
already proxies `/v1/embeddings`. What swaps out is the load-generator half:
upstream's `embedding-benchmark.yml` driving `vllm bench serve` against
`/v1/embeddings`, in place of `llm-benchmark-concurrent-load.yml` driving
GuideLLM against `/v1/chat/completions`.

```bash
# Default embedding model list, 8 instances
./run_sweep.sh -w embedding -N 8

# One model, baseline scenario only, short run
EMBEDDING_SCENARIO=baseline EMBEDDING_NUM_PROMPTS=500 \
  ./run_sweep.sh -w embedding -N 1 \
    -m "RedHatAI/granite-embedding-english-r2 | granite-en-r2"

# nomic-embed-text-v1.5 ships custom modelling code
EXTRA_VLLM_ARGS="--trust-remote-code" \
  ./run_sweep.sh -w embedding -N 8 \
    -m "RedHatAI/nomic-embed-text-v1.5 | nomic-v15"

# Validate without launching anything
./run_sweep.sh --dry-run -w embedding -N 1
```

`BASE_WORKLOAD=embedding` is equivalent to `-w embedding`. The workload cannot be
combined with a generative one in the same invocation — a single stack serves one
model with one runner — so `-w embedding -w chat` is a hard error.

When no `-m` is given, the default list is `granite-en-r2` and `nomic-v15`.

| Env var | Default | Meaning |
|---|---|---|
| `EMBEDDING_SCENARIO` | `all` | `baseline` (rate `inf`, then 25/50/75 % of the measured max), `latency` (concurrency 16/32/64/128/196), or `all` for both |
| `EMBEDDING_NUM_PROMPTS` | *(repo default)* | Requests per measurement point; unset inherits `benchmark-tools.yml`'s `num_prompts` (250) |
| `EMBEDDING_BENCH_CPUS` | `16-31` | `--cpuset-cpus` for the load-generator container |
| `EMBEDDING_BENCH_NUMA` | *(unset — all nodes)* | `--cpuset-mems` for the load generator |
| `EMBEDDING_BENCH_BINARY` | `vllm` | Path to the `vllm` CLI inside the load-generator image (upstream's default is `/opt/venv/bin/vllm`) |
| `VLLM_BENCH_CONTAINER_IMAGE` | `$VLLM_IMAGE` | Load-generator image; defaults to the same image the server runs |
| `VLLM_KV_CACHE_SPACE` | `4` | GiB; pooling models hold no KV cache, so the 90 GiB generative default is dropped |
| `RESULTS_SUBDIR` | `embedding` | Leaf of the results tree |

`EMBEDDING_NUM_PROMPTS` matters more here than it looks. `vllm bench serve` has
no time-based stop (there is no GuideLLM `max_seconds` equivalent): a point runs
for `num_prompts / RPS` seconds. Upstream's default of 250 finishes in about a
second against a multi-instance stack — pure ramp, no steady state — and the
concurrency-196 latency point would be 1.3 waves. For the latency points, set it
to at least 20× the largest concurrency level (3920); the pre-flight warns when a
value you set is below that.

Results land under `results/embedding/`, one directory per run:

```
results/embedding/RedHatAI__granite-embedding-english-r2/
  EPYC8-zentorch-granite-en-r2-20260804-051500/
    test-metadata.json
    baseline/sweep-inf.json  sweep-25pct.json  sweep-50pct.json  sweep-75pct.json
    latency/concurrent-16.json ... concurrent-196.json
```

Flatten a whole tree into one CSV — the `instances` and `backend` columns come
from the `EPYC<N>-<variant>-<tag>` run directory name, so rows from several
sweeps can be concatenated and plotted against instance count:

```bash
python3 extract_embedding_metrics.py results/embedding -o embedding-sweep.csv
```

## Scripts

### `generate-config.sh` — Generate Configuration

Creates `docker-compose.yml`, `nginx.conf`, and `.env` inside the `generated/` directory.

```bash
# Default: 5 instances, 32 cores each, starting at core 32
./generate-config.sh

# Custom layout for a smaller CPU
./generate-config.sh -n 3 -c 24 --start-core 16 --nginx-cores 0-15

# Preview core allocation without writing files
./generate-config.sh -n 8 -c 16 --dry-run

# Use a local model and custom image
./generate-config.sh \
  -m /models/Llama-3.1-8B-Instruct \
  --image amdih/zendnn_zentorch:vllm_v0.18.0_zentorch_v5.2.1_ubuntu22.04_r5.2.1
```

**Parameters:**

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-n, --num-instances` | `NUM_INSTANCES` | 5 | Number of vLLM instances |
| `-c, --cores-per-instance` | `CORES_PER_INSTANCE` | 32 | CPU cores per instance |
| `--nginx-cores` | `NGINX_CORES` | 1-15 | CPU range for NGINX |
| `--start-core` | `VLLM_START_CORE` | 32 | First core for vLLM instances |
| — | `VLLM_CPU_OMP_THREADS_BIND` | unset | OpenMP binding mode: per-instance range when unset; `auto` selects from taskset-restricted cores; `nobind` leaves affinity/thread count to inherited OpenMP settings |
| — | `VLLM_RESERVED_CPUS_PER_INSTANCE` | 1 with `auto`, else 0 | Cores per instance left out of the OpenMP worker bind for vLLM's serving processes; ignored with `nobind` |
| `-m, --model` | `MODEL_NAME` | meta-llama/Llama-3.1-8B-Instruct | HF repo id, or absolute path to a local model dir |
| `--models-dir` | `MODELS_DIR` | — | Host dir bind-mounted into each container at the same absolute path (for local models) |
| `--image` | `VLLM_IMAGE` | — (required) | Container image |
| `--mem-limit` | `MEM_LIMIT` | 100g | Memory limit per instance |
| `--kv-cache-space` | `VLLM_KV_CACHE_SPACE` | 63 | KV cache space (GB) |
| `--max-model-len` | `VLLM_MAX_MODEL_LEN` | 4096 | Context window for `vllm serve`. Must cover the largest `isl+osl` benchmarked against the stack and still fit in `--kv-cache-space`. `run_sweep.sh` derives it from the selected `-w` workloads. |
| `--nginx-port` | `NGINX_PORT` | 8080 | Host port for NGINX |
| `-o, --output-dir` | — | generated/ | Output directory |
| `--native` | — | — | Skip zentorch-specific env vars |
| `--no-limits` | — | — | Skip cpuset/mem_limit/shm_size (for LSF) |
| `--no-mem-limit` | — | — | Drop `mem_limit` only; keep cpuset/shm_size/cap_add/security_opt |
| `--dry-run` | — | — | Print layout without writing |

All flags can also be set as environment variables.

### NUMA memory pinning

Memory pinning is derived automatically from each vLLM instance's CPU range;
there is no additional flag. `generate-config.sh` reads
`/sys/devices/system/cpu/cpu*/node*`, writes the detected node(s) to the
generated service's `x-cpuset-mems`, and shows `mems <nodes>` in `--dry-run`
output. After the containers start, `start.sh` enforces both `cpuset.cpus` and
`cpuset.mems` directly on each cgroup because older podman-compose versions
ignore the Compose CPU setting and do not understand the memory-node extension.
On cgroup v2, `cpuset.mems` constrains allocations made after the write; it does
not migrate pages that the process allocated earlier.

This applies to vLLM instances when limits are enabled. `--no-limits` disables
both CPU and memory pinning. If the NUMA node for an instance's CPU range cannot
be detected, configuration continues without memory pinning for that instance
and prints a warning. At startup, verify lines such as:

```text
pinned vllm-instance-1 -> 32-63 (effective 32-63)
pinned vllm-instance-1 memory -> NUMA node 1 (effective 1)
```

### Reserve cores for online serving

The container cpuset and the OpenMP worker binding are separate. For CPU online
serving, vLLM recommends leaving 1-2 cores available to the API server, engine
core, tokenization and networking:

```bash
# Four 32-core instances on cores 128-255. Each container keeps all 32 cores,
# but its OpenMP worker gets 30 (128-157, 160-189, 192-221, 224-253).
NUM_INSTANCES=4 \
VLLM_START_CORE=128 \
VLLM_RESERVED_CPUS_PER_INSTANCE=2 \
./run_sweep.sh -w chat -m "RedHatAI/gemma-4-E4B-it | gemma-e4b"
```

When `VLLM_CPU_OMP_THREADS_BIND` is unset, the generated bind for an instance
on `<start>-<end>` is `<start>-<end-N>`, where `N` is
`VLLM_RESERVED_CPUS_PER_INSTANCE`. The container cpuset remains
`<start>-<end>`. With the default `N=0`, generated configs are unchanged.

### vLLM auto-bind on multi-NUMA hosts

Auto-bind needs restricted process affinity and the correct memory node. This
harness additionally requires container limits as a safety policy:

```bash
NUM_INSTANCES=4 \
VLLM_START_CORE=128 \
VLLM_CPU_OMP_THREADS_BIND=auto \
./run_sweep.sh -w chat -m "RedHatAI/gemma-4-E4B-it | gemma-e4b"
```

Auto mode reserves one core per instance by default. Set
`VLLM_RESERVED_CPUS_PER_INSTANCE` explicitly to override it.

For each instance, `generate-config.sh`:

1. Reads the NUMA node of its host core range from
   `/sys/devices/system/cpu/cpu*/node*`.
2. Exports `VLLM_CPU_OMP_THREADS_BIND=auto`.
3. Exports `VLLM_CPU_NUM_OF_RESERVED_CPU=N`.
4. Exports `CPU_VISIBLE_MEMORY_NODES=<detected node>`.
5. Starts `vllm serve` through `taskset -c <instance range>`, so process affinity is
   already active when vLLM resolves auto binding. This is required on older
   podman-compose versions that ignore the Compose `cpuset:` key until
   `start.sh` applies a cgroup pin after container startup.

`CPU_VISIBLE_MEMORY_NODES` is required when several independent rank-0 vLLM
instances run on different NUMA nodes. Without it, auto-bind selects the first
visible node for every instance; an instance whose allowed cores are on another
node can receive an empty OpenMP core list.

As a safety policy, auto-bind is rejected when:

- `--no-limits` disables the per-container resource limits;
- an instance's core range spans multiple NUMA nodes;
- the host NUMA node cannot be read.

This topology check is independent of model type and NPS mode. On NPS=4, each
instance must still fit within one NUMA node. Use the explicit reserved-core
mode instead when a block spans nodes.

### `start.sh` — Start the Stack

Launches the multi-instance stack and waits for all containers to pass health checks.

```bash
# Start with existing config (generates defaults if none exists)
./start.sh

# Regenerate config and start (pass any generate-config.sh flags after --regenerate)
./start.sh --regenerate -n 3 -c 24

# Start without waiting for health checks
./start.sh --no-wait

# Custom health check timeout (default: 600s / 10 min)
./start.sh --timeout 900
```

Model loading typically takes 3-8 minutes per instance depending on model size.

### `test-setup.sh` — Validate the Setup

Runs 11 tests to verify the entire stack is working correctly:

| # | Test | What it checks |
|---|------|---------------|
| 1 | Container status | All containers running and healthy |
| 2 | Instance reachability | Each vLLM instance reachable from NGINX container |
| 3 | NGINX health | `/health` endpoint returns HTTP 200 |
| 4 | Model listing | `/v1/models` returns at least one model |
| 5 | NGINX status | `stub_status` endpoint shows connection stats |
| 6 | Round-robin | Requests distributed across all instances (via access log) |
| 7 | Chat completion | End-to-end inference request through NGINX |
| 8 | Streaming | SSE streaming response with multiple chunks |
| 9 | CPU pinning | Each container has correct cpuset, no overlaps |
| 10 | Failover | NGINX serves traffic when one instance is stopped |
| 11 | Concurrent | Parallel requests all return HTTP 200 |

```bash
./test-setup.sh
```

Exit code is 0 if all tests pass, 1 if any fail.

### `stop.sh` — Stop the Stack

```bash
# Stop containers, keep model cache volumes
./stop.sh

# Stop and remove volumes (frees disk space, models will re-download)
./stop.sh --clean

# Stop, remove volumes, and delete generated config files
./stop.sh --purge
```

## Integration with vllm-cpu-perf-eval

Once the stack is running, use the `vllm-cpu-perf-eval` ansible automation to run benchmarks against it:

```bash
# On the control host, point to the NGINX endpoint
export VLLM_ENDPOINT_MODE=external
export VLLM_ENDPOINT_URL=http://<DUT-hostname>:8080
export LOADGEN_HOSTNAME=<DUT-hostname>
export ANSIBLE_SSH_USER=ec2-user
export ANSIBLE_SSH_KEY=~/<your-key>.pem
export HF_TOKEN=<your-huggingface-token>

# Run benchmarks
cd vllm-cpu-perf-eval/automation/test-execution/ansible

for workload in chat chat_lite summarization; do
    ansible-playbook -i inventory/hosts.yml \
      llm-benchmark-concurrent-load.yml \
      -e "test_model=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" \
      -e "base_workload=$workload" \
      -e "skip_phase_2=true" \
      -e "skip_phase_3=true" \
      -e "guidellm_rate=[32,64,96,160]" \
      -e "guidellm_max_seconds=300" \
      -e "test_name=EPYC-NO-SMT"
done
```

In `external` mode, ansible skips container management entirely — it sends benchmark traffic to the NGINX endpoint, which distributes it across the 5 vLLM instances.

## Example CPU Layouts

### 192-core Turin (default)
```bash
./generate-config.sh -n 5 -c 32 --start-core 32 --nginx-cores 1-15
# nginx:  cores 1-15
# vllm-1: cores 32-63
# vllm-2: cores 64-95
# vllm-3: cores 96-127
# vllm-4: cores 128-159
# vllm-5: cores 160-191
```

### 128-core system
```bash
./generate-config.sh -n 3 -c 32 --start-core 32 --nginx-cores 0-15
# nginx:  cores 0-15
# vllm-1: cores 32-63
# vllm-2: cores 64-95
# vllm-3: cores 96-127
```

### 64-core system
```bash
./generate-config.sh -n 2 -c 24 --start-core 16 --nginx-cores 0-7
# nginx:  cores 0-7
# vllm-1: cores 16-39
# vllm-2: cores 40-63
```

## Troubleshooting

**Containers stay "starting" for a long time:**
Model download and loading is slow. Check logs:
```bash
podman logs -f vllm-instance-1
```

**NGINX returns 502 Bad Gateway:**
vLLM instances haven't finished loading yet. Wait for health checks to pass:
```bash
curl http://localhost:8080/health   # returns 200 when at least one instance is ready
```

**Round-robin test shows uneven distribution:**
This is normal for a small number of requests. NGINX round-robin is per-connection, and retries on failure can skew counts. Send more requests for a clearer distribution.

**GuideLLM container exits with code 126 — `cannot setuid to unmapped uid 1001 in user namespace`:**
The GuideLLM container image (`ghcr.io/vllm-project/guidellm:v0.6.0`) has `USER 1001:0` baked in. In rootless podman, UID 1001 is not mapped in the user namespace, causing the container to crash immediately. The fix is to add `user: "0:0"` to the `podman_container` task in `vllm-cpu-perf-eval/automation/test-execution/ansible/roles/benchmark_guidellm/tasks/main.yml` (the "Start GuideLLM benchmark container" task), which runs the container as root and bypasses the UID mapping issue.

**Permission denied on podman:**
Ensure your user can run rootless podman, or run with `sudo`.

**`cpuset.cpus: no such file or directory` (cgroup v2 + rootless podman):**
Rootless podman needs the `cpuset` controller delegated to the user. Fix (requires root):
```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
echo -e '[Service]\nDelegate=cpu cpuset io memory pids' | sudo tee /etc/systemd/system/user@.service.d/delegate.conf
sudo systemctl daemon-reload
sudo systemctl restart user@$(id -u).service
```
Then re-run `./start.sh --regenerate`.

The same controller is required for NUMA memory pinning. A
`failed writing cpuset.mems` warning means CPU pinning may still be active, but
memory allocation is not restricted to the instance's detected NUMA node.

**`rootless needs no limits + no cgrouppath when no permission is granted for cgroups` (LSF environments):**
On LSF-managed machines, the cgroup hierarchy is owned by LSF and users cannot create sub-cgroups. Use the `--no-limits` flag to skip all cgroup-dependent resource constraints (cpuset, mem_limit, shm_size):
```bash
./start.sh --regenerate --no-limits -n 3 -c 32
```
CPU pinning is disabled with `--no-limits`, but containers will start. This is the only option when running inside LSF jobs. `run_sweep.sh --no-limits` forwards the flag to `start.sh`; with `GUIDELLM_CPUS` set it pins Ansible and GuideLLM with `taskset` instead.

**Container OOM-killed during warmup/compilation (host has plenty of free RAM):**
This is the memory cgroup (`mem_limit`) capping the container, not the host running out of RAM. Either raise `--mem-limit` / `MEM_LIMIT`, or drop the memory cap entirely while keeping CPU pinning intact with `--no-mem-limit`:
```bash
./start.sh --regenerate --no-mem-limit -n 3 -c 32
```
Unlike `--no-limits`, this keeps `cpuset`, `shm_size`, `cap_add` (SYS_NICE), and `security_opt` (seccomp), so CPU pinning and performance settings are preserved — only the per-instance `mem_limit` is removed. The NGINX load balancer's own `mem_limit` is unaffected.

**`exec: "--model": executable file not found in $PATH`:**
The container image has no `ENTRYPOINT` set. The `generate-config.sh` script adds `entrypoint: ["vllm", "serve"]` to the compose file automatically. If using an older generated config, regenerate:
```bash
./start.sh --regenerate
```

**If the podman logs is not working run the following command:**
```bash
cat >> ~/.config/containers/containers.conf <<'EOF'
[containers]
log_driver = "k8s-file"
EOF
```

**If the GuideLLM container is not working run the following command:**
```bash
cd .claude/skills/benchmark-vllm-guidellm/scripts
ANS=vllm-cpu-perf-eval/automation/test-execution/ansible
#3a — resilient health-check until (your version was fine)
sed -i 's/until: health_check_result\.status == 200/until: health_check_result.status | default(0) == 200/' \
"$ANS/roles/common/tasks/vllm-health-check.yml" "$ANS/health-check.yml"
#3b — disable sudo escalation for load_generator (FIXED: hardcode false, no #-clash)
grep -q '^        ansible_become: false' "$ANS/inventory/hosts.yml" \
|| sed -i 's#^\(        \)bench_config:#\1ansible_become: false\n\1bench_config:#' "$ANS/inventory/hosts.yml"
#4 — run GuideLLM container as root (your version was fine)
grep -q '^    user: "0:0"' "$ANS/roles/benchmark_guidellm/tasks/main.yml" \
|| sed -i 's#^\(    \)image: "{{ guidellm_cfg.container_image }}"#\1image: "{{ guidellm_cfg.container_image }}"\n\1user: "0:0"#' \
"$ANS/roles/benchmark_guidellm/tasks/main.yml"
```
**cgroupv2 systemd warnings (`The cgroupv2 manager is set to systemd but there is no systemd user session available`):**
These are non-fatal warnings — podman falls back to cgroupfs automatically. To silence them permanently:
```bash
podman system migrate --cgroup-manager=cgroupfs
```

**A container that ran before now fails with `exec ...: no such file or directory` (podman storage under `/tmp`):**
For example `exec /opt/app-root/bin/guidellm: no such file or directory` from the GuideLLM container: files were deleted from the image on disk. AMD grid hosts run an hourly `/tmp` cleaner (`ls /etc/cron.hourly/cleantmp-ng` to check) that deletes files whose access time is more than 15 days old. Pulled image layers keep the timestamps they were built with, so an image pulled into a store under `/tmp` is eligible at once, and files the host never reads, such as the venv `python*` symlinks behind GuideLLM's entrypoint, are gone after the next hourly run. The cleaner never descends into `/tmp/amdbench_*` (`no_check_path` in `/etc/cron.hourly/cleantmp-ng.ini`), so keep the store there:
```bash
podman system reset --force   # discard the damaged store while storage.conf still points at it
mkdir -p /tmp/amdbench_$USER/containers
# then in ~/.config/containers/storage.conf, [storage] section:
#   graphroot = "/tmp/amdbench_<user>/containers"
```
The next run pulls the images again and rebuilds the LMCache images.

## File Structure

```
benchmark-vllm-guidellm/
├── SKILL.md                            # agent workflow
├── README.md                           # this file
├── LMCACHE.md                          # KV-cache sharing guide (APC / LMCache SHM / Mooncake L2)
├── patches/
│   ├── vllm-cpu-perf-eval.patch        # core fixes for the external-endpoint flow
│   ├── embedding.patch                 # -w embedding fixes (stack on top, --3way)
│   └── audio-model.patch               # -w audio fixes (stack on top, --3way)
└── scripts/
    ├── generate-config.sh              # generates compose + nginx config
    ├── start.sh                        # launches the stack (online/serve)
    ├── run_sweep.sh                    # benchmark sweep driver (guidellm via ansible)
    ├── test-setup.sh                   # validates the setup (11 tests)
    ├── stop.sh                         # tears down the stack
    ├── extract_guidellm_metrics.py     # guidellm.log -> CSV
    ├── extract_benchmark_csv.py        # parse guidellm benchmarks.json
    ├── extract_embedding_metrics.py    # results/embedding tree -> CSV
    ├── extract_audio_metrics.py        # results/audio-models tree -> CSV
    ├── merge_guidellm_rates.py         # merge per-rate guidellm outputs into benchmarks.json
    ├── lmcache_cpu_bench.sh            # single-instance LMCache connector smoke benchmark
    ├── lmcache_cluster_bench.sh        # N-instance no-cache / APC / LMCache A/B via lmcache bench
    ├── lmcache_chart_bench.sh          # native / zentorch / zentorch+LMCache via run_sweep.sh
    ├── build_mooncake_ext.py           # builds the native lmcache_mooncake L2 extension
    ├── vllm-cpu-perf-eval/             # cloned automation (gitignored)
    └── generated/                      # auto-generated (gitignored)
        ├── docker-compose.yml
        ├── nginx.conf
        └── .env
```

Some additional notes:
```bash
To check the container status:
podman ps

To get vllm server log for each instance:
export RUN_DIR=$PWD
for i in 1 2 3 4 5; do     podman logs vllm-instance-$i > "$RUN_DIR/vllm-instance-$i.log" 2>&1;   done

To get the number of completions for each instance:
for f in "$RUN_DIR"/vllm-instance-*.log; do     echo "$f: $(grep -cE 'POST /v1/(chat/)?completions' "$f")";   done
```