<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# KV-cache sharing with LMCache — Guide

How to benchmark cross-instance KV-cache sharing on the multi-instance stack:
the cache modes and the command for each, the shared-prefix workloads, and the
prefix-reuse sweep. For the stack itself see [SKILL.md](SKILL.md) and
[README.md](README.md). Commands below run from this skill's `scripts/`
directory (`cd .claude/skills/benchmark-vllm-guidellm/scripts`).

## Cache modes

Every mode runs the same N vLLM instances behind NGINX; only the scope at which
a prompt prefix can be reused changes.

| Mode | What runs | vLLM prefix caching | Flags |
|------|-----------|---------------------|-------|
| No cache | instances only | off | *(none, the default)* |
| APC | instances only; each caches its own prefixes | on | `ENABLE_PREFIX_CACHING=true` |
| LMCache SHM | + one shared `lmcache server`; zero-copy KV through a shared `/dev/shm` | off | `--lmcache` |
| LMCache pickle | + the same server; KV over ZMQ, no shared memory | off | `LMCACHE_PICKLE=1 --lmcache` |
| Mooncake L2 | + the SHM server backed by a Mooncake Store pool (master + metadata services) over TCP | off | `--lmcache-mooncake` |

```
GuideLLM ──> NGINX :8080 ──> vllm-instance-1..N ──MP connector──> vllm-lmcache-server ──L2 adapter──> mooncake-master
                                                                  (L1 pool, shared /dev/shm)          mooncake-metadata
```

- Every instance talks only to the server (`LMCacheMPConnector`); in L2 mode the
  server holds the single Mooncake client and stores KV straight into the pool
  (`--l2-store-policy skip_l1`).
- vLLM prefix caching stays off under LMCache so each hit is attributable to one
  cache layer.
- `--lmcache` builds `<VLLM_IMAGE>_lmcache` (LMCache installed, plus a one-line
  vLLM patch that lets each instance take a unique `VLLM_DIST_IDENT`);
  `--lmcache-mooncake` builds `<VLLM_IMAGE>_lmcache_mooncake` on top. Both are
  built once and reused; `podman rmi` them to rebuild. The vLLM inside is the
  same as in `VLLM_IMAGE`, so use one `VLLM_IMAGE` for every mode you compare.
- The server and Mooncake services sit on the compose bridge at static
  addresses `IP_BASE-1..IP_BASE-3` (default `10.201.0.9`, `.8`, `.7`). LMCache
  modes need bridge networking; `VLLM_NETWORK_MODE=host` is rejected.

## Prerequisites

- **vLLM image** (`VLLM_IMAGE`): APC, LMCache SHM and Mooncake L2 were run end
  to end on `vllm_v0.27.1_zentorch_v2.13.0.0:ubuntu22.04_2026_ww36` with
  LMCache `ad10d542`, and LMCache SHM with `d9aa0fdc`
  on `vllm_v0.29.0_zentorch_v2.13.0.2:ubuntu26.04_2026_ww39`. The image build's
  `VLLM_DIST_IDENT` patch also applies to the vLLM 0.28 `cpu_worker.py`.

- **LMCache** with the CPU connector: branch `feat/cpu-connector` of
  `AMD-Zenai/LMCache` at `d9aa0fdc`, checked out at `$LMCACHE_SRC` (default
  `$HOME/LMCache`). Earlier commits crash EngineCore
  (`assert req_id in self.requests`) when a client cancels a request whose KV
  is still loading, which GuideLLM does as each rate in `GUIDELLM_RATES` ends;
  `podman rmi` a `<VLLM_IMAGE>_lmcache` built from one so the next run rebuilds
  it. The repository is private, so clone it with an account in the AMD-Zenai
  organization; `gh auth setup-git` lets `git` use your `gh` login for the
  HTTPS URL.

  ```bash
  git clone -b feat/cpu-connector https://github.com/AMD-Zenai/LMCache.git ~/LMCache
  git -C ~/LMCache checkout d9aa0fdc
  ```

- **Mooncake** (only for `--lmcache-mooncake`): a non-CUDA build of
  `kvcache-ai/Mooncake` (validated at `98ff4e47`) at `$MOONCAKE_SRC` (default
  `$HOME/Mooncake`). Build it inside the vLLM image you benchmark with
  (`VLLM_IMAGE`, see below) so its libraries match the image the extension is
  compiled in:

  ```bash
  git clone https://github.com/kvcache-ai/Mooncake.git ~/Mooncake
  git -C ~/Mooncake checkout 98ff4e47
  git -C ~/Mooncake submodule update --init --recursive
  podman run --rm --entrypoint /bin/bash -v ~/Mooncake:/workspace/Mooncake \
    -w /workspace/Mooncake "$VLLM_IMAGE" -lc '
      set -euxo pipefail
      printf "APT::Sandbox::User \"root\";\n" > /etc/apt/apt.conf.d/99-rootless
      apt-get update
      apt-get install -y git wget unzip ninja-build pkg-config patchelf \
        libibverbs-dev librdmacm-dev libgoogle-glog-dev libgflags-dev libgtest-dev \
        libjsoncpp-dev libunwind-dev libnuma-dev libboost-dev libssl-dev libgrpc-dev \
        libgrpc++-dev libprotobuf-dev libyaml-cpp-dev protobuf-compiler-grpc \
        libcurl4-openssl-dev libhiredis-dev liburing-dev libjemalloc-dev \
        libmsgpack-dev libzmq3-dev libzstd-dev libasio-dev libxxhash-dev
      cmake -S extern/yalantinglibs -B extern/yalantinglibs/build \
        -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
      cmake --build extern/yalantinglibs/build -j"$(nproc)"
      cmake --install extern/yalantinglibs/build
      cmake -S . -B build -DUSE_CUDA=OFF -DWITH_STORE=ON -DWITH_TE=ON \
        -DUSE_ETCD=OFF -DBUILD_UNIT_TESTS=OFF -DWITH_STORE_RUST=OFF
      cmake --build build -j"$(nproc)"
      NON_CUDA_BUILD=1 bash scripts/build_wheel.sh || echo "wheel repair failed; mooncake-wheel/ is still populated"'
  ```

  `start.sh` needs `build/mooncake-store/src/libmooncake_store.a`,
  `build/mooncake-common/libasio.so` and a populated `mooncake-wheel/` from it.
  The wheel-repair step of `build_wheel.sh` can fail harmlessly: the populated
  `mooncake-wheel/` directory is what gets installed.

- **Memory.** The server's L1 pool is `LMCACHE_L1_SIZE_GB` (default 100 GiB,
  container limit L1 + 8 GiB) on top of every instance's `VLLM_KV_CACHE_SPACE`.
  In SHM and L2 modes the pool is shared through the host's `/dev/shm`
  (bind-mounted into the server and every instance), so leave room for it
  there.

## Commands for each mode

`run_sweep.sh` starts the stack, runs GuideLLM, and tears down, once per model.
Give each mode its own `TEST_NAME_PREFIX` so the result directories
(`<workload>-<timestamp>-<prefix>-<variant>-<tag>`) tell the modes apart;
prefixes must not be substrings of one another.

```bash
export VLLM_IMAGE=<registry>/<vllm-zentorch-image>:<tag>   # required; use one image for every mode
M="meta-llama/Llama-3.1-8B-Instruct | llama31"

# No cache (the harness default: vLLM prefix caching off, no connector)
TEST_NAME_PREFIX=NOC ./run_sweep.sh -N 5 -w chat_shared -m "$M"

# APC: vLLM's own per-instance prefix caching
ENABLE_PREFIX_CACHING=true TEST_NAME_PREFIX=APC ./run_sweep.sh -N 5 -w chat_shared -m "$M"

# LMCache SHM: one shared server, zero-copy /dev/shm transport
TEST_NAME_PREFIX=SHM ./run_sweep.sh --lmcache -N 5 -w chat_shared -m "$M"

# LMCache pickle: same server, KV over ZMQ
LMCACHE_PICKLE=1 TEST_NAME_PREFIX=PKL ./run_sweep.sh --lmcache -N 5 -w chat_shared -m "$M"

# Mooncake L2: shared server backed by a distributed Mooncake Store pool
STAGGER_STARTUP=1 TEST_NAME_PREFIX=L2 ./run_sweep.sh --lmcache-mooncake -N 5 -w chat_shared -m "$M"
```

`STAGGER_STARTUP=1` starts the support services first, then the instances one
at a time, then NGINX. Use it when some instances time out during engine-core
start-up (model load, `torch.compile`) under I/O contention; L2 stacks start the
slowest.

To bring a stack up without running GuideLLM (for manual requests or
inspection), use `start.sh` with the same mode flags:

```bash
./start.sh --regenerate -n 5 -m meta-llama/Llama-3.1-8B-Instruct                                   # no cache
ENABLE_PREFIX_CACHING=true ./start.sh --regenerate -n 5 -m meta-llama/Llama-3.1-8B-Instruct        # APC
./start.sh --regenerate --lmcache -n 5 -m meta-llama/Llama-3.1-8B-Instruct                         # LMCache SHM
LMCACHE_PICKLE=1 ./start.sh --regenerate --lmcache -n 5 -m meta-llama/Llama-3.1-8B-Instruct        # LMCache pickle
./start.sh --regenerate --lmcache-mooncake -n 5 -m meta-llama/Llama-3.1-8B-Instruct                # Mooncake L2
```

On rootless hosts without cgroup cpuset delegation, add `--no-limits`. It drops
cpuset/memory limits from every service, including the LMCache and Mooncake
ones. With `GUIDELLM_CPUS` set, `run_sweep.sh` then pins Ansible and GuideLLM
with `taskset` instead:

```bash
GUIDELLM_CPUS=0-15 TEST_NAME_PREFIX=SHM ./run_sweep.sh --lmcache --no-limits -N 5 -w chat_shared -m "$M"
```

### Checking that the cache is doing the work

- **Hit counters per instance.** APC hits are in `vllm:prefix_cache_hits_total`
  / `vllm:prefix_cache_queries_total`; hits served through LMCache are in
  `vllm:external_prefix_cache_hits_total` / `..._queries_total`. Each result's
  `vllm-metrics.json` records them as sampled through NGINX (one backend), not
  as a fleet total. To read one instance directly:

  ```bash
  podman exec vllm-instance-1 curl -s localhost:8000/metrics \
    | grep -E 'vllm:(external_)?prefix_cache_(hits|queries)_total'
  ```

- **Server traffic.** L1 chunk writes and cross-instance reads:

  ```bash
  podman exec vllm-lmcache-server curl -s localhost:9090/metrics \
    | grep -E 'lmcache_mp_l1_(read|write)_chunks_total'
  ```

- **Mooncake L2 actually serving.** With `skip_l1` a chunk leaves L1 once its
  Mooncake store completes, so retrievals should report `(0 L1, N L2)`. Count
  the split:

  ```bash
  podman logs vllm-lmcache-server 2>&1 \
    | grep -oE 'retained keys \([0-9]+ L1, [0-9]+ L2\)' | sort | uniq -c
  ```

  LMCache tries that removal once and skips a chunk another request is reading
  at that moment; the chunk then stays in L1 until evicted, and hits on it are
  served from there. Prefixes that several requests use at once are the ones
  caught: chat-style runs have served about 70% of their hit chunks from L1
  (codegen, summarization and RAG runs under 6%), and a single-prefix smoke run
  can be served entirely from L1. Check the split before reading an L2 run as
  Mooncake performance.

## Shared-prefix workloads

Defined in `test-workloads.yml` (added by `vllm-cpu-perf-eval.patch`). Each
request is one of `prefix_count` shared prefixes plus a unique suffix, so the
input is `prefix + suffix` tokens. `run_sweep.sh` derives `--max-model-len`
from the workload, as for any other workload.

| Workload | Shared prefix | Unique suffix | Output | Prefix pool | `--max-model-len` |
|----------|--------------:|--------------:|-------:|------------:|------------------:|
| `chat_shared` | 2048 | 64 | 16 | 64 | 4096 |
| `codegen_shared` | 4096 | 256 | 512 | 64 | 8192 |
| `summarization_shared` | 6144 | 128 | 256 | 64 | 8192 |
| `rag_shared` | 6144 | 1024 | 256 | 64 | 8192 |
| `smoke_shared` | 2048 | 64 | 16 | 1 | 4096 |

GuideLLM 0.7 (the default image) receives them as a JSON `--data` object with
one prefix bucket:
`{"kind":"synthetic_text","prompt_tokens":<suffix>,"output_tokens":<output>,"prefix_buckets":[{"prefix_tokens":<prefix>,"prefix_count":<pool>}]}`.
Its CLI rejects flat `prefix_tokens` / `prefix_count` keys. GuideLLM 0.6 takes
them flat in `GUIDELLM_DATA`:
`prompt_tokens=<suffix>,output_tokens=<output>,prefix_tokens=<prefix>,prefix_count=<pool>`.
Workloads without `prefix_tokens` render exactly as before. Under
`run_sweep.sh`, which exports `HF_CACHE_DIR`, the GuideLLM container mounts
`$HF_CACHE_DIR/huggingface` read-only and loads the tokenizer offline from it,
so the tokenizer must already be in that cache. For HF models `start.sh`
pre-warms it; with `HF_HUB_OFFLINE=1` it must be pre-populated.

### How to think about them

- **Warm-up cost is the whole story.** A per-instance cache has to see every
  prefix on every instance before it can hit. With the load balancer spreading
  requests, many land on an instance that has not seen their prefix, so the
  fleet pays up to `N x prefix_count` cold prefills. A shared cache pays
  `prefix_count` once for the whole fleet. The gap therefore grows with the
  instance count and the prefix pool, and shrinks when warm-up is short
  compared with the run.
- **Check that the pool fits.** A pool of `prefix_count x prefix_tokens` tokens
  must fit in each instance's KV cache (APC) and in the server L1 (LMCache).
  Otherwise you are measuring eviction, not reuse. For Llama-3.1-8B in bf16 a
  token costs 2 (K+V) x 32 layers x 8 KV heads x 128 x 2 bytes = 128 KiB, so:

  | Workload | Pool tokens | Pool size |
  |----------|------------:|----------:|
  | `chat_shared` | 131,072 | 16 GiB |
  | `codegen_shared` | 262,144 | 32 GiB |
  | `summarization_shared`, `rag_shared` | 393,216 | 48 GiB |

- **Compare fairly.**
  - Same `VLLM_IMAGE` for every mode.
  - The same offered concurrency per instance: scale `GUIDELLM_RATES` with `-N`.
    The reference chat runs used 64 concurrent streams per instance, so
    `GUIDELLM_RATES="[320]"` for `-N 5`. Keep `GUIDELLM_MAX_CONCURRENCY` at or
    above it.
  - One workload per invocation, so every (mode, workload) pair starts from a
    fresh stack and a cold cache. Several `-w` in one run share one stack.
- **Read more than one number.**
  - Aggregate tokens/s (input + output) is dominated by the shared prefix.
    Report generated tokens/s next to it.
  - Report the completion rate with every throughput. A saturated run leaves
    requests incomplete, so its throughput is a floor, not a ceiling.
  - Report TTFT at p50/p95/p99. Sharing mostly removes the tail: the request
    that lands on a cold instance and pays full prefill.

## Prefix-reuse sweep

Five workloads with the same 2112-token input and 16-token output, differing
only in how much of the input is a shared prefix (pool of 64):

| Workload | Shared prefix | Unique suffix | Reusable fraction |
|----------|--------------:|--------------:|------------------:|
| `hit_000` | 0 | 2112 | 0% |
| `hit_025` | 512 | 1600 | 24.2% |
| `hit_050` | 1024 | 1088 | 48.5% |
| `hit_075` | 1536 | 576 | 72.7% |
| `hit_097` | 2048 | 64 | 97.0% |

Run every point on a fresh stack, once per mode:

```bash
M="meta-llama/Llama-3.1-8B-Instruct | llama31"
for w in hit_000 hit_025 hit_050 hit_075 hit_097; do
  ENABLE_PREFIX_CACHING=true TEST_NAME_PREFIX=APC ./run_sweep.sh -N 5 -w "$w" -m "$M"
  TEST_NAME_PREFIX=SHM ./run_sweep.sh --lmcache -N 5 -w "$w" -m "$M"
  STAGGER_STARTUP=1 TEST_NAME_PREFIX=L2 ./run_sweep.sh --lmcache-mooncake -N 5 -w "$w" -m "$M"
done
```

Do not pass all five with `-w ... -w ...` in one invocation: the points would
share one running stack, and a later point could start with KV cached by an
earlier one.

### How to read it

- **`hit_000` is the control.** There is nothing to share, so every mode should
  match within run-to-run noise. A gap there is harness or connector overhead,
  not caching.
- **Observed vs configured reuse.** The observed hit rate (hits / queries from
  the counters above) shows how much of the configured reuse each cache
  converts into hits. A shared cache tracks the configured fraction; a
  per-instance cache falls short because every instance warms separately.
- **Where it pays off.** The throughput gap opens as reuse rises. Where it
  becomes worth the extra moving parts depends on the fleet size and the prefix
  pool.

Reference points (Llama-3.1-8B bf16, 15 instances x 32 cores on a 2-socket,
512-core server, 300 s per point; hit rate sampled from one backend):

| Reuse | APC tok/s | SHM tok/s | L2 tok/s | Hit rate APC / SHM / L2 |
|------:|----------:|----------:|---------:|------------------------:|
| 0% | 5,873 | 5,787 | 5,845 | 0 / 0 / 0% |
| 24.2% | 6,397 | 7,369 | 7,383 | 10.6 / 22.5 / 21.6% |
| 48.5% | 7,030 | 10,020 | 9,962 | 19.9 / 45.6 / 44.8% |
| 72.7% | 8,846 | 17,338 | 16,272 | 37.6 / 69.3 / 67.8% |
| 97.0% | 11,878 | 65,507 | 33,942 | 57.4 / 94.4 / 91.6% |

## Tuning knobs (env vars)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_PREFIX_CACHING` | `false` | `true` = vLLM per-instance prefix caching (APC mode) |
| `LMCACHE_SRC` | `$HOME/LMCache` | LMCache checkout installed into the derived image; read only when an image is built |
| `MOONCAKE_SRC` | `$HOME/Mooncake` | Built Mooncake tree (`--lmcache-mooncake`) |
| `LMCACHE_PICKLE` | `0` | `1` = ZMQ/pickle transport instead of SHM |
| `LMCACHE_L1_SIZE_GB` | `100` | Server L1 pool (GiB); container limit is L1 + 8 GiB |
| `LMCACHE_CHUNK_SIZE` | `128` | Tokens per KV chunk on the server; must be a multiple of every KV-cache group's `tokens_per_block` |
| `LMCACHE_SERVER_CORES` | `NGINX_CORES` | cpuset for the lmcache server |
| `MOONCAKE_CORES` | `LMCACHE_SERVER_CORES` | cpuset for the Mooncake master and metadata services |
| `MOONCAKE_PROTOCOL` | `tcp` | `rdma` on RDMA-capable hosts |
| `MOONCAKE_L1_GB` | `4` | Sizes the Mooncake global segment and local buffer (each `MOONCAKE_L1_GB + 1` GiB) |
| `STAGGER_STARTUP` | `0` | `1` = support services, then instances one by one, then NGINX |
| `STAGGER_SUPPORT_DELAY`, `STAGGER_DELAY` | `20`, `45` | Seconds after the support services / after each instance |
| `GUIDELLM_CPUS` | *(unset)* | With `--no-limits`: `taskset` CPUs for Ansible + GuideLLM |
| `HF_CACHE_DIR` | `$HOME/.cache/hf-shared` | Shared HF cache; also mounted read-only into GuideLLM |

## Standalone drivers

`lmcache_cluster_bench.sh` and `lmcache_chart_bench.sh` require `VLLM_IMAGE`.

- `lmcache_cpu_bench.sh`: one vLLM instance on a host venv (`$VENV`, no
  containers), vLLM alone vs vLLM + an `lmcache server`, driven by
  `lmcache bench engine` (long-doc-qa). A quick connector smoke test.
- `lmcache_cluster_bench.sh`: N instances behind NGINX; runs no cache, APC and
  LMCache with `lmcache bench engine` through NGINX and prints the server's L1
  read/write chunk counters.
- `lmcache_chart_bench.sh`: native, zentorch, and zentorch + LMCache through
  `run_sweep.sh` (GuideLLM, default `chat_shared`), scored with
  `extract_guidellm_metrics.py`.

## Notes

- `generate-config.sh` recreates the shared SHM directory
  (`/dev/shm/lmc-shared`, or `/dev/shm/lmc-shared-<STACK_TAG>`) on every
  generation, which discards the segments of a live stack with the same tag.
  Use a distinct `STACK_TAG` for a second stack. `stop.sh` deletes the
  directory, which frees the RAM its tmpfs pages hold.
- The Ansible side of these workloads ships in `vllm-cpu-perf-eval.patch`, which
  applies to the pinned `vllm-cpu-perf-eval` commit in [README.md](README.md).
