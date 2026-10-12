---
name: benchmark-vllm-guidellm
description: >-
  Runs the multi-instance vLLM benchmark sweep on AMD EPYC CPUs: N
  cpuset-pinned `vllm serve` containers (zentorch or native backend) behind an
  NGINX round-robin load balancer, driven by the redhat-et/vllm-cpu-perf-eval
  ansible automation in external-endpoint mode. One invocation sweeps a list of
  models, bringing up a fresh stack per model and running every selected
  workload against it: GuideLLM for generative workloads (chat, rag, code,
  summarization, reasoning, variable-length, context-scaling), `vllm bench
  serve` for embedding models, and GuideLLM's audio backend for Whisper ASR.
  Loads models from Hugging Face or a local directory, handles llm-compressor
  and torchao quantized checkpoints, can capture vLLM torch-profiler traces,
  and compares cross-instance KV-cache sharing (vLLM prefix caching vs a
  shared LMCache server over zero-copy SHM) on
  shared-prefix workloads. Validates a running stack with an 11-test self-check
  and flattens results into CSV. Use when asked to run the multi-instance /
  GuideLLM sweep, to benchmark several models, quantized models, embedding
  models or audio models on CPU across instance counts, or to benchmark LMCache
  / prefix caching across instances. For the interactive single-model run with
  topology-sized core slices and a throughput/latency profile, use
  benchmark-vllm instead.
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Benchmark vLLM with GuideLLM (multi-instance sweep)

```
load generator (GuideLLM / vllm bench serve container)
    -> NGINX :8080 round-robin        (cores 1-15)
         -> vllm-instance-1           (cores 32-63)
         -> vllm-instance-2           (cores 64-95)
         -> ...                       (one contiguous core block each)
```

`scripts/run_sweep.sh` is the entry point. For each (variant, model) pair it
stops any previous stack, starts a fresh one (`start.sh` -> `generate-config.sh`
-> `podman-compose up` -> health wait), runs every selected workload through the
perf-eval ansible playbooks against `http://localhost:8080`, tears the stack
down and drops the page cache.

[README.md](README.md) next to this file is the user-facing reference: every
script flag, example CPU layouts and the full troubleshooting list. This file is
the agent workflow.

## Sections
- [Setup (one time)](#setup-one-time)
- [Golden rule: confirm, then dry-run](#golden-rule-confirm-then-dry-run)
- [Fail fast](#fail-fast)
- [Stage 1: workload mode and workloads](#stage-1-workload-mode-and-workloads)
- [Stage 2: models](#stage-2-models)
- [Stage 3: image, variant, quantization](#stage-3-image-variant-quantization)
- [Stage 4: stack layout and memory](#stage-4-stack-layout-and-memory)
- [Stage 5: load parameters](#stage-5-load-parameters)
- [Stage 6: confirm, dry-run, run](#stage-6-confirm-dry-run-run)
- [Embedding models (`-w embedding`)](#embedding-models--w-embedding)
- [Audio models (`-w audio`)](#audio-models--w-audio)
- [Profiling (`--profile`)](#profiling---profile)
- [KV-cache sharing (LMCache)](#kv-cache-sharing-lmcache)
- [Manual stack control](#manual-stack-control)
- [Results](#results)
- [Verify](#verify)
- [Gotchas](#gotchas)
- [Files](#files)

## Setup (one time)

Host requirements (the DUT and the load generator are the same host; every
playbook runs with `--connection=local`):

- `podman` and `podman-compose` (rootless works).
- `ansible-playbook` driving `/usr/bin/python3`, with the `containers.podman`
  and `ansible.posix` collections (`ansible-galaxy collection list`).
- `hf` or `huggingface-cli` on the host (`pip install huggingface_hub`):
  `start.sh` downloads every Hugging Face model into the shared cache
  (`HF_CACHE_DIR`, default `~/.cache/hf-shared`) before any container starts,
  and the instances then run with `HF_HUB_OFFLINE=1`.
- `HF_TOKEN` for gated models (e.g. Llama). Without one, `run_sweep.sh`
  prompts on a TTY and otherwise continues anonymously, which works for public
  models and a pre-populated cache. Never set a placeholder token: Hugging Face
  rejects an invalid token even for public repos.
- Optional: passwordless `sudo` for `tee /proc/sys/vm/drop_caches` between runs.
  Without it the sweep prints a warning and continues.

No zentorch Python environment is needed: everything that imports zentorch runs
inside the container image.

Clone the perf-eval automation where `run_sweep.sh` expects it, **pin it to
`2bacf71`** (the main patch is generated against that commit; later upstream
revisions do not apply), and apply the three patches in order. Stage after each
one, because `git apply --3way` refuses to run against an index that differs
from the working tree:

```bash
cd .claude/skills/benchmark-vllm-guidellm
PE=scripts/vllm-cpu-perf-eval
git clone https://github.com/redhat-et/vllm-cpu-perf-eval.git "$PE"
git -C "$PE" checkout 2bacf71ce6329eecc58e2a5ec2d7b8c98ea67174

git -C "$PE" apply "$PWD/patches/vllm-cpu-perf-eval.patch"          # applies cleanly
git -C "$PE" add -A
git -C "$PE" apply --3way "$PWD/patches/embedding.patch"            # resolve, then:
git -C "$PE" add -A
git -C "$PE" apply --3way "$PWD/patches/audio-model.patch"          # resolve, then:
git -C "$PE" add -A
```

| Patch | Needed for | What it changes |
|-------|------------|-----------------|
| `vllm-cpu-perf-eval.patch` | every run | GuideLLM container as `user: "0:0"` (rootless), `ansible_become: false`, `results_dir` leaf from `RESULTS_SUBDIR`, tolerant health-check `until`, local-model bind mount for GuideLLM, a safety-timeout stop instead of a hard fail, embedding `vllm_bench_binary` + measured P95, `test_name` in audio run dirs, the shared-prefix workloads (passed to GuideLLM 0.7 as `prefix_buckets`), the shared HF cache mounted read-only into GuideLLM (offline tokenizer), no GuideLLM cpuset under `--no-limits`, and `guidellm_output_suffix` for per-rate output files |
| `embedding.patch` | `-w embedding` | `realpath` on `results_dir`; mounts the shared HF cache into the `vllm bench serve` container (offline tokenizer); `bench_run_as_root` instead of a hard-coded root user |
| `audio-model.patch` | `-w audio` | GuideLLM dataset cache under `$HOME` instead of `/tmp`; `HF_HUB_DISABLE_XET=1` and a download timeout |

`embedding.patch` and `audio-model.patch` overlap the main patch, so `--3way`
stops on conflicts. On `2bacf71` they are exactly these:

- `embedding.patch` -> `ansible/inventory/hosts.yml`: keep both comment blocks,
  and merge the two `results_dir` lines so the `RESULTS_SUBDIR` leaf is kept and
  `realpath` is applied to the whole path:
  `... else (playbook_dir ~ '/../../../results/' ~ (lookup('env', 'RESULTS_SUBDIR') | default('llm', true))) | realpath }}"`.
- `embedding.patch` -> `roles/benchmark_embedding/tasks/{baseline,latency}.yml`
  (two hunks each): take `embedding.patch`'s
  `{% if bench_run_as_root | bool %}--user 0:0{% endif %}` over the hard-coded
  `--user 0:0`, and keep the main patch's
  `{{ vllm_bench_binary | default('/opt/venv/bin/vllm') }} bench serve`.
- `audio-model.patch` -> `ansible/inventory/hosts.yml`: keep the new
  `cache_dir:` block and the main patch's `results/<subdir>` comment line.

Then check the tree:

```bash
git -C "$PE" grep -nE '^(<<<<<<<|>>>>>>>)' || echo "no conflict markers"
(cd "$PE/automation/test-execution/ansible" && for pb in llm-benchmark-concurrent-load.yml \
    llm-benchmark-auto.yml embedding-benchmark.yml audio-benchmark.yml health-check.yml; do
  ansible-playbook -i inventory/hosts.yml --syntax-check "$pb"; done)
```

If the conflicts differ from the list above, the clone is not on `2bacf71`:
check it out again rather than resolving new conflicts by hand.

## Golden rule: confirm, then dry-run

Walk the user through Stages 1-5 **one at a time**, offering the default at
each. Run nothing until the Stage 6 confirmation. `run_sweep.sh --dry-run`
runs every pre-flight check and prints every `start.sh` and `ansible-playbook`
command without starting a container, calling ansible or touching `sudo`. It is
safe with `HF_TOKEN` and `VLLM_IMAGE` unset, so always show it before the real
run.

## Fail fast

`run_sweep.sh` does **not** stop at the first failure. After a failed
`start.sh` it leaves the containers up for inspection and moves on to the next
(variant, model); after a failed playbook it prints a warning, finishes that
model's remaining workloads, and marks the model failed. It exits non-zero at
the end if any (variant, model) failed. `start.sh` gives up as soon as an
instance exits during start-up and prints its last log lines. Treat any of
these as a failure, **stop**, and report:

- a non-zero exit from `run_sweep.sh`, an `ERROR:` line, or
  `WARNING: ansible ... benchmark failed`;
- a non-zero `Failed` count in the `Sweep complete` banner;
- an instance that never becomes healthy, a container OOM, or a
  `vllm serve` / GuideLLM traceback;
- an empty or zero result.

Surface the exact error and the relevant log: `scripts/sweep-logs/<ts>/` holds
`<variant>_<model>.log` (stack) and `<variant>_<model>_<workload>.log` (ansible),
plus `container-logs/` dumped by `stop.sh`. Also check `podman ps -a` and
`podman logs vllm-instance-1`. Tear down with `scripts/stop.sh --clean` and
give the likely cause and fix. Never present partial numbers as a complete
sweep.

## Stage 1: workload mode and workloads

Each invocation runs in exactly one mode. A stack serves one model with one
runner, so mixing modes (`-w embedding -w chat`, `-w audio -w chat`) is a hard
error.

| Mode | `-w` | Load generator | Endpoint | Default models |
|------|------|----------------|----------|----------------|
| generative | `chat` (default) or any `test_configs` key; repeatable | GuideLLM via `llm-benchmark-concurrent-load.yml` | `/v1/chat/completions` | `MODELS` array in `run_sweep.sh` (two local-dir entries; pass `-m`) |
| embedding | `embedding` | `vllm bench serve` via `embedding-benchmark.yml` | `/v1/embeddings` | `granite-en-r2`, `nomic-v15` |
| audio | `audio` | GuideLLM audio backend via `audio-benchmark.yml` | `/v1/audio/transcriptions` | `openai/whisper-small` |

Generative workloads come from the clone's
`automation/test-execution/ansible/inventory/group_vars/all/test-workloads.yml`:
`chat` (512/512), `chat_lite` (128/128), `rag` (7680/512), `code` (1024/1024),
`summarization` (2048/256), `reasoning` (256/2048), `chat_var`, `code_var`,
`summarization_var` (normal-distributed lengths), and `context_scaling_1k/4k/8k`.
`BASE_WORKLOAD="chat rag"` is equivalent to `-w chat -w rag`.

The main patch adds **shared-prefix workloads** for KV-cache sharing runs:
`chat_shared`, `codegen_shared`, `summarization_shared`, `rag_shared`,
`smoke_shared` (each request is one of a pool of shared prefixes plus a unique
suffix), and the prefix-reuse sweep `hit_000`, `hit_025`, `hit_050`, `hit_075`,
`hit_097` (same 2112-token input, 0-97% of it shared). Plain `chat` prompts are
random, so caching shows no benefit on them. See
[KV-cache sharing (LMCache)](#kv-cache-sharing-lmcache).

All workloads for a model run against **one warm stack**. Adding a workload
costs one load run, not another multi-minute start. The context window is sized
for the largest selected workload, though (see Stage 4).

## Stage 2: models

- `-m "path-or-repo | tag"`, repeatable. Any `-m` replaces the built-in list.
  Without `|`, the path doubles as the tag.
- **Hugging Face**: pass the repo id; `start.sh` downloads it on the host first.
- **Local**: `--models-dir DIR` plus a path relative to it, e.g.
  `--models-dir ~/models -m "Llama-3.1-8B-Instruct | llama31-8b"`. `DIR` is
  bind-mounted read-only at the same absolute path in every container; no
  `HF_TOKEN` and no download. Pre-flight checks each `DIR/<path>` exists.
- `test_name` is `${TEST_NAME_PREFIX:-EPYC<N>}-<variant>-<tag>` and must match
  `^[A-Za-z0-9-]{1,100}$`. Pre-flight validates every pair, so keep tags short
  and `[A-Za-z0-9-]` only.
- Custom modelling code (e.g. `nomic-embed-text-v1.5`): `EXTRA_VLLM_ARGS="--trust-remote-code"`.
  `EXTRA_VLLM_ARGS` is appended verbatim to every `vllm serve`.
- `-d/--dtype`: `bfloat16` by default, `half` for fp16; `DTYPE=""` omits the flag.
- gpt-oss models need the tiktoken `o200k_base` vocab offline. `start.sh`
  pre-fetches it and the compose file points the instances at it automatically.

## Stage 3: image, variant, quantization

**Always ask for the image. There is no default.** Export `VLLM_IMAGE` before
running: without it `run_sweep.sh` prompts on a TTY and aborts otherwise, and
`generate-config.sh` (and `start.sh --native`/`--torchao`) exit with an error. Prefer an image already on
the host (`podman images`); public zentorch tags are at
https://hub.docker.com/r/amdih/zendnn_zentorch/tags .

Variant (one per invocation):

- `--zentorch` (default): the image as-is.
- `--native`: the same image with zentorch pip-uninstalled, committed as
  `<image>_native` on first use and cached afterwards. Deriving it from the same
  image keeps vLLM, torch and compiler flags identical, so a delta is zentorch's.
  For an A/B, run two invocations back to back on the same host, identical apart
  from `--native`. The variant is part of `test_name`, so their results don't
  collide. Run zentorch first and stop if it fails.

Quantization (affects only the image, never how the model is loaded):

| Run | Flags | Image |
|-----|-------|-------|
| bf16 / fp16 | none | base image |
| llm-compressor (w8a8, w4a16, ...) | `--quant` | base image, no rebuild |
| torchao (`quant_method=torchao`) | `--quant --torchao` | derived `<image>_torchao-<ver>` with `torchao>=0.10.0` (pin with `TORCHAO_VERSION=0.17.0`) |

`--torchao` without `--quant` is an error.

To test in-container env vars without editing scripts, set
`SET_ENV_VAR_BEFORE_VLLM_SERVE` to a shell prefix that ends in `&&`, e.g.
`"export ZENDNNL_MATMUL_WEIGHT_CACHE=2 &&"`.

## Stage 4: stack layout and memory

| Variable / flag | Default | Meaning |
|-----------------|---------|---------|
| `-N` / `NUM_INSTANCES` | `1` | vLLM instances; also seeds the `EPYC<N>` prefix |
| `CORES_PER_INSTANCE` | `32` | cores per instance |
| `VLLM_START_CORE` | `32` | first core of instance 1 |
| `VLLM_CPU_OMP_THREADS_BIND` | unset | OpenMP binding mode: unset = per-instance core range, `auto` = vLLM selects from taskset-restricted cores on that instance's NUMA node, `nobind` = vLLM leaves affinity/thread count to inherited OpenMP settings |
| `VLLM_RESERVED_CPUS_PER_INSTANCE` | `1` with `auto`, else `0` | cores in every instance range not assigned to OpenMP workers; use 1-2 for CPU online serving; ignored with `nobind` |
| `NGINX_CORES` | `1-15` | NGINX cpuset |
| `NGINX_PORT` | `8080` | host port of the endpoint |
| `MEM_LIMIT` | `200g` | per-instance memory cgroup cap |
| `--no-mem-limit` | off | drop only `mem_limit`; keep cpuset, shm, caps |
| `VLLM_KV_CACHE_SPACE` | `90` (4 for embedding/audio) | KV cache GiB **per instance** |
| `GUIDELLM_CPUS` / `GUIDELLM_NUMA_NODE` | unset | optional GuideLLM cpuset / memory nodes |
| `--no-limits` | off | drop cpuset, mem_limit and shm entirely (no cgroup delegation); with `GUIDELLM_CPUS` set, Ansible + GuideLLM are pinned with `taskset` |
| `STAGGER_STARTUP=1` | off | start support services, then instances one at a time, then NGINX (`STAGGER_DELAY`, default 45 s) |

Instance `i` gets the contiguous logical CPUs
`VLLM_START_CORE + (i-1)*CPI` to `VLLM_START_CORE + i*CPI - 1`. podman-compose
3.4 ignores the compose `cpuset:` key, so `start.sh` writes each container's
cpuset into its cgroup v2 node after `podman-compose up` and prints
`pinned <container> -> <cpus>`.

For each vLLM instance, `generate-config.sh` also derives the NUMA node(s) of
that CPU range from `/sys/devices/system/cpu/cpu*/node*` and emits
`x-cpuset-mems`. `start.sh` writes it to the container's `cpuset.mems` cgroup
file and prints `pinned <container> memory -> NUMA node <nodes>`. On cgroup v2,
this constrains allocations made after the write but does not migrate pages
allocated earlier. The pinning is automatic; there is no separate setting.
`--no-limits` disables both CPU and memory pinning. An unknown NUMA node leaves
memory unpinned for that instance, while a
`WARN: ... cpuset.cpus not writable` or `failed writing cpuset.mems` line means
the corresponding cgroup pin was not enforced.

Neither script checks that the layout fits, so do it before confirming:

- `VLLM_START_CORE + N*CPI - 1` must be below `nproc`.
- Check the blocks against `lscpu -e=CPU,NODE,SOCKET,CORE`; a block that
  straddles NUMA nodes measures cross-node traffic.
- Keep `16-31` free for the load generator (`EMBEDDING_BENCH_CPUS`
  default) and out of `NGINX_CORES`.
- `N * VLLM_KV_CACHE_SPACE` plus the weights must fit in `free -g`.

**Reserve cores for online serving.** vLLM's CPU documentation recommends
leaving 1-2 cores for the API server, engine core, tokenization and networking.
Keep the container cpuset unchanged and remove the cores only from the OpenMP
worker binding:

```bash
VLLM_RESERVED_CPUS_PER_INSTANCE=2 ./run_sweep.sh -N 4 ...
```

For a 32-core instance on `128-159`, this produces
`VLLM_CPU_OMP_THREADS_BIND=128-157`; cores `158-159` remain available to the
other vLLM processes. The setting is model-independent. Enable it first on CPU
online / GuideLLM jobs; validate embedding and audio separately.

**Auto-bind by NUMA node.** To let vLLM choose the workers and reserve one core:

```bash
VLLM_CPU_OMP_THREADS_BIND=auto \
./run_sweep.sh -N 4 ...
```

Auto mode reserves one core per instance by default. Set
`VLLM_RESERVED_CPUS_PER_INSTANCE` explicitly to override it.

`generate-config.sh` reads every instance's NUMA node from
`/sys/devices/system/cpu/cpu*/node*` and exports both
`VLLM_CPU_NUM_OF_RESERVED_CPU=N` and
`CPU_VISIBLE_MEMORY_NODES=<that instance's node>`. The latter is required for
several single-rank vLLM instances on different NUMA nodes: without it, each
rank-0 worker selects the first visible node and an instance on another node can
receive an empty core list. The generated entrypoint also starts `vllm serve`
through `taskset -c <instance range>`. This makes the affinity effective before
vLLM resolves `auto`, even on podman-compose 3.4 where `start.sh` must apply the
container cgroup cpuset after startup. As a safety policy, auto-bind is disabled
with `--no-limits`, and it fails early if an instance spans NUMA nodes. This
also makes the setting safe for NPS=4 when each instance range fits in one
NUMA node.

Announce the resolved blocks (`generate-config.sh -n N -c CPI --dry-run` prints
them) and get a yes.

**Context window.** vLLM starts once per model, so one `--max-model-len` must
cover every selected workload. `run_sweep.sh` derives it from
`test-workloads.yml`: a workload's own `--max-model-len` if it declares one,
else `isl+osl+128` (`MAX_MODEL_LEN_HEADROOM`), taking the max:

| `-w` selection | Derived `--max-model-len` |
|----------------|---------------------------|
| `chat` | 2048 |
| `code` | 4096 |
| `chat summarization code` | 4096 |
| `chat summarization code rag` | 8320 |

`VLLM_MAX_MODEL_LEN` overrides it; never leave it to vLLM (131072 for
Llama-3.1-8B would not fit the KV cache). Pre-flight estimates the fit as
`2 x layers x kv_heads x head_dim x 2` bytes per token from the cached
`config.json` and fails with the minimum `VLLM_KV_CACHE_SPACE` needed.
`KV_FIT_CHECK=false` skips it. The check needs the model's `config.json` in the
cache, so it is reported unverified until the model has been downloaded.

## Stage 5: load parameters

Generative mode (embedding and audio have their own knobs, below):

| Variable | Default | Meaning |
|----------|---------|---------|
| `GUIDELLM_RATES` | `[32,64]` | concurrency levels, **absolute at the load balancer** (`[64]` with `-N 4` is ~16 per instance) |
| `GUIDELLM_MAX_SECONDS` | `300` | benchmarking window: one number for every rate, or a `<rate>:<seconds>` map (below); 60-120 only for smoke tests |
| `GUIDELLM_MAX_SECONDS_DEFAULT` | `300` | window for rates the map does not mention |
| `GUIDELLM_MAX_CONCURRENCY` | `1024` | must be >= the largest rate or high rates are clamped |
| `GUIDELLM_MAX_REQUESTS` | `100000` | request cap per rate; kept high so runs end on `GUIDELLM_MAX_SECONDS`, not the perf-eval default of 1000 |
| `GUIDELLM_WAIT_TIMEOUT` | `3600` | ansible poll timeout for the GuideLLM container; with a window map it is derived per rate as window + 900 s (3600-14400 s) unless set explicitly |
| `HEALTH_TIMEOUT` | `1200` | seconds `start.sh` waits for all instances |

High rates (32, 64 and up) measure saturation throughput; low rates (1, 2, 4,
8) measure per-request latency (TTFT, ITL, TPOT). Ask which the user wants and
confirm the list.

**Benchmarking window per rate.** A request takes longer the higher the
concurrency, so one window rarely suits every rate. With the default 300 s, a
Qwen3.6-35B run completed 35 requests at rate 32 but 1 at rate 256 (one request
takes ~480 s there); GuideLLM cancelled the rest when the window closed, so that
rate's latency and requests/s rest on a single sample. For a wide rate ladder,
propose a map instead:

```bash
GUIDELLM_RATES="[32,64,128,256]" GUIDELLM_MAX_SECONDS="32:600,64:800,128:1100,256:1650" \
  scripts/run_sweep.sh -N 3 -m "meta-llama/Llama-3.1-8B-Instruct | llama31-8b"
```

Size each window so every stream finishes at least 3 requests:
`3 x request_latency` at that concurrency, plus the warm-up (10 % of the window)
and cool-down (30 s). `--dry-run` prints the resolved window per rate and the
total benchmark time. A malformed map entry fails pre-flight.

`max_duration` is run-wide in GuideLLM, so a map runs **one GuideLLM invocation
per rate**. They share one results directory (a run id fixed by `run_sweep.sh`)
and write `benchmarks-<rate>.json`, `benchmarks-<rate>.csv` and
`guidellm-<rate>.log`; when the workload finishes, `merge_guidellm_rates.py`
rebuilds the usual `benchmarks.json`, `benchmarks.csv` and `guidellm.log`,
splicing every rate's rows into the first log's tables so it has the same
one-table-per-title shape as a single-window run. Read results exactly as
before; never glob `benchmarks-*.json` too, or every rate counts twice. A rate
that dies before printing its tables is reported on stderr and is absent from
the merged log; its own `guidellm-<rate>.log` holds the failure.

The per-rate filenames need the `guidellm_output_suffix` support from
`vllm-cpu-perf-eval.patch` and GuideLLM v0.7+ (the default image is v0.7.2; on
older images the run fails early). On an unpatched clone `run_sweep.sh` warns
and falls back to one window covering the whole ladder (the longest in the
map), so every rate is still measured.

## Stage 6: confirm, dry-run, run

Show the assembled command, get a yes, dry-run it, then run:

```bash
cd .claude/skills/benchmark-vllm-guidellm
export VLLM_IMAGE=<image> HF_TOKEN=<token>

# bf16 model from Hugging Face, two workloads on one warm stack
GUIDELLM_RATES="[32,64]" scripts/run_sweep.sh --dry-run -N 4 -w chat -w rag \
    -m "meta-llama/Llama-3.1-8B-Instruct | llama31-8b"

# several local models, llm-compressor quant
scripts/run_sweep.sh --dry-run --quant --models-dir ~/models -N 4 \
    -m "Llama-3.1-8B-Instruct-w8a8 | l8b-w8a8" -m "Qwen3-8B-w8a8 | q3-8b-w8a8"
```

Per (variant, model), a real run takes the model start (typically 3-8 min, plus
the download on first use) plus `workloads x rates x GUIDELLM_MAX_SECONDS` (with
a window map, the sum of the windows per workload; `--dry-run` prints it). Run
it detached so a dropped session does not kill it, and wait for the
`Sweep complete` banner:

```bash
mkdir -p scripts/sweep-logs
nohup scripts/run_sweep.sh -N 4 -w chat -m "meta-llama/Llama-3.1-8B-Instruct | llama31-8b" \
    > scripts/sweep-logs/run_sweep.out 2>&1 &
```

| Goal | Flags |
|------|-------|
| bf16 / fp16 model | none |
| llm-compressor quant | `--quant` |
| torchao quant | `--quant --torchao` |
| local model dir | `--models-dir DIR` (with any of the above) |
| instances | `-N N` |
| workloads | `-w NAME` (repeatable) |
| embedding / audio | `-w embedding` / `-w audio` (alone) |
| native variant | `--native` |
| per-instance OOM with free host RAM | `--no-mem-limit` |
| no cgroup cpuset delegation (rootless, LSF) | `--no-limits` (+ `GUIDELLM_CPUS` to pin GuideLLM) |
| vLLM per-instance prefix caching (APC) | `ENABLE_PREFIX_CACHING=true` |
| shared LMCache server | `--lmcache` (`LMCACHE_PICKLE=1` for the ZMQ/pickle transport) |
| torch profiler | `--profile` |
| validate only | `--dry-run` |

`scripts/run_sweep.sh --help` lists every flag.

## Embedding models (`-w embedding`)

The stack is unchanged: vLLM picks the pooling runner from the model config, and
NGINX's catch-all `location /` proxies `/v1/embeddings`. The load generator is
`vllm bench serve` in a container, by default the same image as the server.

```bash
# one instance on one CCD (8 cores), baseline scenario only
CORES_PER_INSTANCE=8 EMBEDDING_SCENARIO=baseline EMBEDDING_NUM_PROMPTS=10000 \
  scripts/run_sweep.sh -w embedding -N 1 -m "RedHatAI/granite-embedding-english-r2 | granite-en-r2"
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_SCENARIO` | `all` | `baseline` (rate `inf`, then 25/50/75 % of the measured max), `latency` (concurrency 16/32/64/128/196), or `all` |
| `EMBEDDING_NUM_PROMPTS` | repo default (250) | requests per point |
| `EMBEDDING_BASELINE_MAX_CONCURRENCY` | `196` | caps the `inf`-rate point so it does not open every connection at once |
| `EMBEDDING_BENCH_CPUS` | `16-31` | load-generator cpuset |
| `EMBEDDING_BENCH_NUMA` | unset (all nodes) | load-generator memory nodes |
| `EMBEDDING_BENCH_BINARY` | `vllm` | `vllm` CLI path inside the load-generator image |
| `VLLM_BENCH_CONTAINER_IMAGE` | `$VLLM_IMAGE` | load-generator image |

`vllm bench serve` has no time-based stop: a point lasts `num_prompts / RPS`
seconds, so the 250 default finishes in about a second against a multi-instance
stack. Set `EMBEDDING_NUM_PROMPTS` to at least 20x the largest concurrency
(3920 for the latency scenario); pre-flight warns below that.

Nothing asserts that zentorch supports BERT-style encoders. With a stack up,
confirm the runner loaded before a long sweep; if it fails, retry with
`--native` to separate zentorch from vLLM-CPU:

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"RedHatAI/granite-embedding-english-r2","input":"hello"}' | jq '.data[0].embedding | length'
```

## Audio models (`-w audio`)

Whisper ASR through `audio-benchmark.yml` in external mode against the NGINX
endpoint. Whisper is encoder-decoder, so `-w audio` also sets stack defaults
(all env-overridable): `--max-model-len 448`, `--trust-remote-code`,
`VLLM_KV_CACHE_SPACE=4`, and the CPU workaround `VLLM_USE_V2_MODEL_RUNNER=0`.
The zentorch image ships `vllm[audio]`, so no audio image is built.

```bash
AUDIO_SCENARIO=quick-test scripts/run_sweep.sh -w audio -N 1 -m "openai/whisper-tiny | whisper-tiny"
for n in 1 2 4 8; do scripts/run_sweep.sh -w audio -N "$n" -m "openai/whisper-small | whisper-small"; done
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `AUDIO_SCENARIO` | `transcription-throughput` | `tests/audio-models/<name>.yaml`: also `transcription-latency`, `transcription-quality` (WER/CER), `audio-duration-scaling`, `constant-rate-stress`, `format-comparison`, `quick-test` |
| `AUDIO_NUM_FILES` | scenario default | files per stage |
| `AUDIO_P95_TARGET` | `2.0` | P95 latency target (s) for the playbook's report |
| `AUDIO_VLLM_ENV` | `export VLLM_USE_V2_MODEL_RUNNER=0 &&` | in-container prefix; `""` drops the workaround |

`transcription-quality` also needs `pip install jiwer datasets soundfile requests`
on the host. To check the endpoint is live, run
`curl -s -o /dev/null -w '%{http_code}\n' -X OPTIONS localhost:8080/v1/audio/transcriptions`;
`200` or `405` means it is.

## Profiling (`--profile`)

`--profile` enables the vLLM torch profiler in every instance, POSTs
`/start_profile` before the first workload and `/stop_profile` after the last,
then waits for the traces to stop growing before teardown. Traces land in
`scripts/profiler-traces/<test_name>/instance-<i>/` (`PROFILER_DIR` overrides
the root), with the key-averages table grouped by input shape. Wrapping a full
multi-rate run can produce multi-GB traces per instance, so profile a short run
(one workload, one rate, low `GUIDELLM_MAX_SECONDS`).

## KV-cache sharing (LMCache)

Compares vLLM's per-instance prefix caching with one LMCache server shared by
every instance, on the shared-prefix workloads. [LMCACHE.md](LMCACHE.md) is the
full guide: prerequisites, every mode, the prefix-reuse sweep, the knobs, and
how to read the results. Read it before proposing an LMCache run.

| Mode | Flags | What runs |
|------|-------|-----------|
| no cache | none (the default) | instances only, vLLM prefix caching off |
| APC | `ENABLE_PREFIX_CACHING=true` | each instance caches its own prefixes |
| LMCache SHM | `--lmcache` | + one shared `lmcache server`, zero-copy KV through a shared `/dev/shm` |
| LMCache pickle | `LMCACHE_PICKLE=1 --lmcache` | same server, KV over ZMQ |

Ask before running:

- **LMCache checkout** at `LMCACHE_SRC` (default `$HOME/LMCache`): branch
  `zen-lmcache/cpu-shm` of `amd/LMCache_zen`, tested at `0bfa083c` (older
  builds crash EngineCore when GuideLLM cancels requests between rates).
  `--lmcache` builds `<VLLM_IMAGE>_lmcache` from it once and reuses it, so
  `podman rmi` that image after changing the checkout.
- **Memory**: the server's L1 pool (`LMCACHE_L1_SIZE_GB`, default 100 GiB) comes
  on top of every instance's `VLLM_KV_CACHE_SPACE`, and SHM mode holds it in
  the host's `/dev/shm`.
- **`LMCACHE_CHUNK_SIZE`** must be a multiple of every KV-cache group's block
  size (gemma-4 on CPU needs 256).
- **Hybrid models** (Qwen3.5 / Qwen3.6): `--lmcache` asks vLLM in the image for
  the unified block size N (`scripts/hybrid_block_size.py`), then turns on vLLM
  prefix caching, sets `LMCACHE_CHUNK_SIZE=N` and adds
  `--max-num-batched-tokens N`. Without prefix caching vLLM keeps no reusable
  Mamba state and LMCache refuses the model at startup ("MambaSpec with
  mamba_cache_mode='none'"). Values you set are kept; an explicit
  `ENABLE_PREFIX_CACHING=false`, or a chunk size that is not a multiple of N,
  is an error. See `test-vllm-lmcache-hybrid`.

Give each mode its own `TEST_NAME_PREFIX` (not a substring of another) so the
result directories tell them apart, use the same `VLLM_IMAGE` for every mode,
and run one workload per invocation so every point starts from a cold cache:

```bash
cd .claude/skills/benchmark-vllm-guidellm
M="meta-llama/Llama-3.1-8B-Instruct | llama31"
ENABLE_PREFIX_CACHING=true TEST_NAME_PREFIX=APC scripts/run_sweep.sh -N 5 -w chat_shared -m "$M"
TEST_NAME_PREFIX=SHM scripts/run_sweep.sh --lmcache -N 5 -w chat_shared -m "$M"
```

LMCache modes need bridge networking (`VLLM_NETWORK_MODE=host` is rejected).
The server takes the static address just below NGINX (`10.201.0.9`).
`stop.sh` deletes the shared SHM directory (`/dev/shm/lmc-shared[-<STACK_TAG>]`),
which frees the RAM it holds.

## Manual stack control

For debugging, or to drive ansible by hand (this is what `run_sweep.sh` does per
workload):

```bash
cd .claude/skills/benchmark-vllm-guidellm/scripts
export VLLM_IMAGE=<image> HF_TOKEN=<token>
./start.sh --regenerate -n 3 -c 32 --start-core 32 --nginx-cores 1-15 \
    -m meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096
./test-setup.sh                       # 11 checks; exit 0 only if all pass
curl -s localhost:8080/v1/models

cd vllm-cpu-perf-eval/automation/test-execution/ansible
export VLLM_ENDPOINT_MODE=external VLLM_ENDPOINT_URL=http://localhost:8080 \
       LOADGEN_HOSTNAME=localhost DUT_HOSTNAME=localhost ANSIBLE_SSH_USER="$(whoami)"
ansible-playbook -i inventory/hosts.yml llm-benchmark-concurrent-load.yml --connection=local \
    -e "ansible_python_interpreter=/usr/bin/python3" \
    -e "test_model=meta-llama/Llama-3.1-8B-Instruct" -e "base_workload=chat" \
    -e "skip_phase_2=true" -e "skip_phase_3=true" \
    -e "guidellm_rate=[32]" -e "guidellm_max_seconds=300" \
    -e "test_name=EPYC3-zentorch-llama31-8b" -e '{"health_check":{"timeout":600,"interval":5}}'
cd - && ./stop.sh                     # --clean also drops volumes; --purge also deletes generated/
```

To drive load from a separate control host instead, point `VLLM_ENDPOINT_URL`
and `LOADGEN_HOSTNAME` at the DUT, set `ANSIBLE_SSH_USER`/`ANSIBLE_SSH_KEY`, and
drop `--connection=local`.

Per-instance request counts confirm the round-robin actually spread load:

```bash
for i in 1 2 3; do echo "vllm-instance-$i: $(podman logs vllm-instance-$i 2>&1 | grep -cE 'POST /v1/(chat/)?completions')"; done
```

## Results

Inside the clone (`R=scripts/vllm-cpu-perf-eval/results`):

```
$R/llm/<org>__<model>/<workload>-<ts>-<test_name>/external-endpoint/   guidellm.log, benchmarks.json
                                                    (+ guidellm-<rate>.log, benchmarks-<rate>.json with a window map)
$R/embedding/<org>__<model>/<test_name>-<ts>/{baseline,latency}/*.json
$R/audio-models/<org>__<model>/<scenario>-<test_name>-<ts>/<stage>/benchmarks.json
```

Alongside: `scripts/sweep-logs/<ts>/` (per-run stack and ansible logs, plus
`container-logs/`).

```bash
python3 scripts/extract_guidellm_metrics.py "$R/llm/<model>/<run>/external-endpoint/guidellm.log"
python3 scripts/extract_benchmark_csv.py "$R/llm/<model>"      # writes <run>/benchmark_summary.csv per run
python3 scripts/extract_embedding_metrics.py "$R/embedding" -o embedding-sweep.csv
python3 scripts/extract_audio_metrics.py "$R/audio-models" -o audio-sweep.csv
```

For generative runs, report from `guidellm.log` (Server Throughput Statistics
for throughput, Request Latency Statistics for latency) via
`extract_guidellm_metrics.py`; treat `extract_benchmark_csv.py` as a
cross-check. The embedding and audio extractors recover `instances` and
`backend` from the `EPYC<N>-<variant>-<tag>` run-dir name, so CSVs from
several sweeps can be concatenated and plotted against instance count.

## Verify

- `scripts/run_sweep.sh --dry-run ...` passes pre-flight (`[OK]` everywhere, no
  `[MISS]`/`[BAD]`) and prints the expected `start.sh` and `ansible-playbook`
  commands.
- `scripts/generate-config.sh -n N -c CPI --dry-run` prints the core blocks and
  their detected NUMA memory nodes.
- With a stack up, `scripts/test-setup.sh` (see the test 2 and 6 gotcha).
- After a run: no `Traceback` or `No space left` under the run's
  `guidellm.log`, and every rate has results.

## Gotchas

- **Tests 2 and 6 of `test-setup.sh` address instances by service name**
  (`vllm-<i>`), but `generate-config.sh` points NGINX at static IPs because
  rootless aardvark-dns is often unreachable. They can report FAIL on a working
  stack; rely on the other tests and the per-instance request counts.
- **cgroup v2 + rootless podman**: `cpuset.cpus: no such file or directory`
  means the `cpuset` controller is not delegated to the user. The one-time root
  fix is in [README.md](README.md#troubleshooting).
- **LSF / no cgroup permission**: pass `--no-limits` (skip cpuset, mem_limit,
  shm) to `run_sweep.sh`. Container CPU pinning is off in that mode; set
  `GUIDELLM_CPUS` so at least Ansible and GuideLLM are pinned with `taskset`.
- **Container OOM with free host RAM** is the per-instance `mem_limit`. Use
  `--no-mem-limit` or lower `VLLM_KV_CACHE_SPACE` (it is per instance).
- **NGINX `upstream timed out` on rootless LSF nodes**: the bridge drops
  inter-container traffic. Use `VLLM_NETWORK_MODE=host`.
- **`IPAM error: requested ip address ... is already allocated`**: a killed run
  leaked a netavark lease. `stop.sh` resets the lease db when no containers are
  left, and otherwise prints the manual reset commands.
- **GuideLLM load generator OOM-killed at start** with many instances: it is
  memory-bound to NUMA node 0 by default. Set `GUIDELLM_NUMA_NODE` to span
  nodes (e.g. `0-7`).
- **`podman logs` empty**: set `log_driver = "k8s-file"` under `[containers]`
  in `~/.config/containers/containers.conf`.
- **Gated model fails to download**: `HF_TOKEN` is missing or lacks access;
  `start.sh` hard-fails the pre-warm rather than letting N containers race.
  With `HF_HUB_OFFLINE=1` exported, the pre-warm is skipped and the model (and
  its tokenizer, which GuideLLM loads offline from `HF_CACHE_DIR`) must already
  be in the cache.
- **`exec ...: no such file or directory` from a container that ran before**
  (e.g. GuideLLM): the podman store is under `/tmp` and a `/tmp` cleaner deleted
  image files. Move the store; see [README.md](README.md#troubleshooting).
- **An instance exits during start-up** (OOM, bad flag): `start.sh` now stops
  immediately and prints its last 40 log lines instead of waiting out
  `HEALTH_TIMEOUT`. Under I/O contention some instances can miss vLLM's
  start-up handshake; `STAGGER_STARTUP=1` spreads them out.
- **Do not compare across stack shapes.** `N`, CPI, the core bands, the image
  and `--max-model-len` all move the numbers; change one thing at a time.
- **The `_native`, `_torchao-*` and `_lmcache` images are
  cached** by tag. If the base image or the LMCache checkout changes, delete the
  derived image or the run uses the stale one. For `_lmcache`, `start.sh` prints
  the image's `lmcache.commit` label and warns when the checkout differs.

## Files

```
.claude/skills/benchmark-vllm-guidellm/
  SKILL.md                         # this file (agent workflow)
  README.md                        # user reference: flags, CPU layouts, troubleshooting
  LMCACHE.md                       # KV-cache sharing guide (APC / LMCache SHM)
  scripts/
    run_sweep.sh                   # entry point: pre-flight, models x variants x workloads
    start.sh                       # derived-image builds, HF pre-warm, compose up, cpuset pinning, health wait
    stop.sh                        # log dump, compose down, network + IPAM lease + LMCache SHM cleanup
    generate-config.sh             # docker-compose.yml + nginx.conf + .env (+ LMCache server)
    hybrid_block_size.py           # run in the image by start.sh --lmcache: hybrid model? vLLM block size N
    test-setup.sh                  # 11-test validation of a running stack
    extract_guidellm_metrics.py    # guidellm.log -> CSV
    extract_benchmark_csv.py       # benchmarks.json -> CSV
    extract_embedding_metrics.py   # results/embedding tree -> CSV
    extract_audio_metrics.py       # results/audio-models tree -> CSV
    merge_guidellm_rates.py        # per-rate GuideLLM outputs -> benchmarks.json/.csv + guidellm.log
    lmcache_cpu_bench.sh           # single-instance LMCache connector smoke test (host venv)
    lmcache_cluster_bench.sh       # N instances: no cache / APC / LMCache via `lmcache bench engine`
    lmcache_chart_bench.sh         # native / zentorch / zentorch+LMCache via run_sweep.sh
    vllm-cpu-perf-eval/            # the clone (gitignored, pinned to 2bacf71)
    generated*/ sweep-logs/ profiler-traces/ chart-bench-*/   # run artifacts (gitignored)
  patches/
    vllm-cpu-perf-eval.patch       # core fixes for the external-endpoint flow + shared-prefix workloads
    embedding.patch                # -w embedding: offline tokenizer, root gating, realpath
    audio-model.patch              # -w audio: dataset cache dir, HF download settings
```
