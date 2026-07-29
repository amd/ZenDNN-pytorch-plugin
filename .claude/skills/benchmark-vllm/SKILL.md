---
name: benchmark-vllm
description: Benchmark vLLM (with the zentorch/ZenDNN CPU backend) across four quadrants -- online vs offline, single vs multi-instance -- by running the amdih/zendnn_zentorch container image with podman and switching the container entrypoint per benchmark type. Offline uses `vllm bench throughput`; online uses a running `vllm serve` driven by GuideLLM (multi-instance adds an NGINX round-robin load balancer). ALWAYS interactive: ask the user for the quadrant, check the machine with lscpu, ask instances/cores-per-instance and validate the layout (recommend a fitting config if it does not), ask the image (default newest amdih tag) and the workload params, then confirm the exact command before running. Use when asked to benchmark vLLM throughput/latency/serving, compare zentorch vs native, or size a multi-instance CPU serving run.
version: 0.1.0
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Benchmark vLLM (zentorch / ZenDNN CPU backend)

A thin, **interactive** orchestrator. It runs the prebuilt
`amdih/zendnn_zentorch` image with **podman** and switches the container
entrypoint per benchmark type. It never guesses a layout that overloads the
machine -- it checks `lscpu`, validates `N * cores_per_instance` against the
physical cores, and recommends a fitting config when the request does not fit.

Four quadrants:

| Quadrant | Client | How it runs | Runner |
|----------|--------|-------------|--------|
| offline single | `vllm bench throughput` | 1 container, entrypoint overridden | `scripts/run_offline_single.sh` |
| offline multi  | `vllm bench throughput` | 1 container, N taskset-pinned procs | `scripts/run_offline_multi.sh` |
| online single  | GuideLLM | `vllm serve` container + host client | `scripts/run_online_single.sh` |
| online multi   | GuideLLM | podman-compose: N serve + NGINX LB | `online-multi/run_sweep.sh` |

## Sections
- [Flow](#flow)
- [Golden rule: ask at every stage](#golden-rule-ask-at-every-stage)
- [Stage 1-2: quadrant](#stage-1-2-quadrant)
- [Stage 3: hardware check + config validation](#stage-3-hardware-check--config-validation)
- [Stage 4: image](#stage-4-image)
- [Stage 5: workload params](#stage-5-workload-params)
- [Stage 6: confirm + run](#stage-6-confirm--run)
- [Entrypoint-switch matrix](#entrypoint-switch-matrix)
- [Results](#results)
- [Verify](#verify)
- [Gotchas](#gotchas)
- [Files](#files)

## Flow

Interactive, one stage at a time. Nothing runs until the Stage 6 confirmation,
and the hardware gate re-prompts rather than overloading the machine.

See [benchmark-vllm-flow.mmd](benchmark-vllm-flow.mmd) for the raw Mermaid
benchmark workflow.

## Golden rule: ask at every stage
Drive the user through the stages **one at a time**, offering a sensible default
at each. Do NOT run anything until the final confirmation in Stage 6. Every
runner accepts `--dry-run`, which prints the exact podman command without
executing -- show that to the user before the real run.

## Fail fast: stop the run on any failure
If **any step fails, STOP immediately** -- do not continue to the next stage,
the next quadrant in a sweep, or the next rate/instance. A "failure" includes:
a non-zero exit from any script/command, an image pull error, a container that
does not become healthy within the timeout, a `vllm serve`/GuideLLM crash or
traceback, a `0 tok/s` / missing-`Throughput:` result, or a container OOM.
When something fails: surface the exact error and the relevant log
(`server.log`, `guidellm.log`, `result_inst*.txt`), tear down anything that was
started (stop containers / `online-multi/stop.sh`), and report to the user with
the likely cause and suggested fix (e.g. `--hf-offline`, reduce
`VLLM_CPU_KVCACHE_SPACE`, `--no-mem-limit`). Only re-run after the user confirms
the fix. Never silently push past an error or fabricate results.

## Stage 1-2: quadrant
Ask two things:
1. **online or offline?** Offline measures max throughput (`vllm bench
   throughput`, no server). Online measures serving under load (a live `vllm
   serve` driven by GuideLLM: throughput + TTFT/ITL/TPOT + latency percentiles).
2. **single or multi-instance?**

## Stage 3: hardware check + config validation
Run the detector and show the summary:
```bash
scripts/check_hardware.sh                 # topology + recommendation only
```
Then ask the user for **N instances** and **cores per instance (CPI)** (for
single quadrants, N=1 -- ask only CPI). Suggest the detected defaults. Validate:
```bash
# multi: validate a specific request; exits non-zero + prints a recommendation
# if it does not fit. Emits KEY=VALUE (incl. CPUSET) on stdout.
scripts/check_hardware.sh --mode multi -n <N> -c <CPI>
scripts/check_hardware.sh --mode single -c <CPI>
```
Rules enforced/announced:
- `N * CPI <= physical cores` (physical, not SMT threads). If it fails, **do not
  run** -- present the recommended `N`/`CPI` (one instance per NUMA node, or the
  largest N that fits) and re-prompt.
- Warn when `CPI` exceeds or does not divide the NUMA-node core count (instances
  would straddle NUMA nodes).
- For **online multi**, reserve cores for NGINX with `--nginx-cores` (default
  `0-15`) and for the host GuideLLM client with `--guidellm-cores` (default
  `16-31`); both bands are excluded from the vLLM pool and the fit check is
  validated against the remaining physical cores (vLLM then starts at core 32).
  Confirm these with the user (see "Online core reservation" below).

The runners call `check_hardware.sh` themselves when `--cpuset` is not supplied,
so passing `--n/--cpi` is enough -- but confirm the numbers with the user first.
`online-multi/run_sweep.sh` uses `check_hardware.sh --exclude <nginx,guidellm>
--slices` to derive each instance's cpuset from the physical-core list (first
thread per core), so instances never land on SMT sibling threads regardless of
topology. For the offline runner you may also pass `--cpuset` alone (one instance
spanning that set) or with `--n`/`--cpi` for explicit slicing.

### Online core reservation (3-band layout) -- CONFIRM WITH THE USER
Online quadrants run a host-side GuideLLM client (and, for multi, an NGINX load
balancer) that must not fight the vLLM server for cores. The defaults mirror
`ZenDNN_tools/vllm_multiinstance`, but **always confirm the layout with the
user** because the vLLM core start shifts with the reservations:

- **online multi:** `--nginx-cores 0-15`, `--guidellm-cores 16-31`, vLLM instances
  on the remaining physical cores (i.e. from core `32` up). Both bands are
  excluded from the vLLM pool, and the host GuideLLM client is `taskset`-pinned
  to `--guidellm-cores`. Confirm `nginx-cores`, `guidellm-cores`, `N`, and `CPI`
  -- changing any of them changes where the vLLM instances start.
- **online single:** no NGINX. The host GuideLLM client is pinned to the **last
  ~3 physical cores** (`--guidellm-tail 3`, or an explicit `--guidellm-cores`),
  which are reserved off the vLLM server's cpuset. Confirm the tail size / cores.

Announce the resolved bands (e.g. "nginx 0-15, guidellm 16-31, vLLM 32-...") and
get a yes before running.

## Stage 4: image
**Always ask for the image.** Default to the newest `amdih/zendnn_zentorch`
tag (see `DEFAULT_IMAGE` in `scripts/common.sh`, currently
`vllm_v0.24.0_zentorch_v2.11.0.3_ubuntu22.04_2026_ww28`). Browse tags at
https://hub.docker.com/r/amdih/zendnn_zentorch/tags . The runners
`podman pull` the image if it is not present locally. Pass a chosen image with
`--image <ref>`.

## Stage 5: workload params
Ask only the params relevant to the quadrant, each with a default:
- **offline:** `--input` (in tokens, 128), `--output` (128), `--prompts` (128),
  `--max-seqs` (128).
- **online:** `--input`/`--output` tokens, `--port` (single) / `--nginx-port`
  (multi), plus two params you must **always confirm with the user** (for both
  online single and online multi):
  - **`--rates`** -- the GuideLLM concurrency list. Do not assume; ask. Suggested
    defaults: `1,2,4,8` (single) and `32,64` (multi).
  - **`--max-seconds`** -- seconds per rate. **Default is `300`** (not 60/120);
    ask the user and offer 300 as the default.
- **online multi memory:** offer `--no-mem-limit` when relevant (see Gotchas):
  it drops only the per-instance `mem_limit` cgroup cap while keeping cpuset/
  shm/caps. Distinct from `--no-limits` (which drops everything).
- **model:** default `meta-llama/Llama-3.1-8B-Instruct`. Gated models need
  `HF_TOKEN` exported (forwarded into the container). A shared host HF cache is
  mounted (`--hf-cache-dir`, default `~/.cache/huggingface`).
- **air-gapped / HF-blocked hosts:** if the model is already cached but the host
  cannot reach `huggingface.co`, pass `--hf-offline` (sets `HF_HUB_OFFLINE=1`
  inside the container). Without it, vLLM still contacts the Hub to validate the
  repo and a fully-cached run can fail with a silent `0 tok/s`.
- Optional zentorch knobs are forwarded when exported: `ZENDNNL_MATMUL_ALGO`,
  `USE_ZENDNN_MATMUL_DIRECT`, `ZENTORCH_FP16_OPS`, `THP_MODE`.

## Stage 6: confirm + run
Show the assembled command (use `--dry-run`), get a yes, then run without it.

```bash
# offline single
scripts/run_offline_single.sh --model <M> --cpi <CPI> --image <IMG> [--input .. --output .. --prompts .. --max-seqs ..] [--dry-run]

# offline multi
scripts/run_offline_multi.sh  --model <M> --n <N> --cpi <CPI> --image <IMG> [workload...] [--dry-run]

# online single (host GuideLLM pinned to the last ~3 cores via --guidellm-tail,
# or pass --guidellm-cores <set>; those cores are reserved off the vLLM cpuset)
scripts/run_online_single.sh  --model <M> --cpi <CPI> --image <IMG> \
    --rates 1,2,4,8 --max-seconds 300 [--guidellm-cores <set> | --guidellm-tail 3] [--dry-run]

# online multi (nginx + guidellm cores are reserved from the vLLM pool; slices
# are derived from physical cores automatically; host GuideLLM is taskset-pinned)
online-multi/run_sweep.sh     --model <M> --n <N> --cpi <CPI> --image <IMG> \
    --nginx-cores 0-15 --guidellm-cores 16-31 --rates 32,64 --max-seconds 300 \
    [--no-mem-limit] [--hf-offline] [--dry-run]
```

## Entrypoint-switch matrix
The whole point of the skill: the **same image**, different entrypoint per type.

| Quadrant | podman entrypoint | Command inside |
|----------|-------------------|----------------|
| offline single/multi | `bash /bench/offline_launcher.sh` | N x `taskset ... vllm bench throughput` (N=1 for single) |
| online single | `bash /bench/incontainer_serve.sh` | `vllm serve ...` (then GuideLLM from host) |
| online multi | `bash /bench/incontainer_serve.sh` (per compose service) | `vllm serve ...` behind NGINX; GuideLLM from host |

`incontainer_env.sh` (sourced by the in-container scripts) discovers tcmalloc +
libiomp5 and sets `LD_PRELOAD`, `TORCHINDUCTOR_FREEZING`,
`VLLM_CPU_KVCACHE_SPACE`. Core pinning is `--cpuset-cpus` on the container plus
`taskset`/`VLLM_CPU_OMP_THREADS_BIND` per instance.

## Results
- **offline** -> `scripts/parse_offline.py <results_dir>` writes
  `throughput_summary.csv` (per-instance + summed throughput, grouped by run
  tag). Runners call it automatically.
- **online** -> the runners tee GuideLLM output to `guidellm.log` and call the
  **parse-guidellm** skill's parser at
  `~/.claude/skills/parse-guidellm/scripts/parse_guidellm_log.py` if present
  (throughput + TTFT/ITL/TPOT + latency). Follow that skill to build comparison
  tables.
- **peak memory** -> `scripts/mem_poll.sh` samples `podman stats` for all
  `bench-vllm-*` containers during every run and appends a `PEAK` line.
- Results land under `./bench_results/<quadrant>_<timestamp>/`.

## Verify
- `check_hardware.sh` alone prints a topology summary and a recommended layout.
- Every runner with `--dry-run` prints the exact command(s) and exits.
- Smoke test (offline single, small run):
  `scripts/run_offline_single.sh --model facebook/opt-125m --cpi 8 --prompts 8 --dry-run`
  then drop `--dry-run` on a host with the image pulled.

## Gotchas
- **Container OOM / memory cgroup**: if a container is OOM-killed while the host
  still has plenty of free RAM, it is the per-instance `mem_limit` cgroup cap
  (online multi), not the host. Two knobs, in order:
  1. **Reduce the KV cache** -- ask the user to re-test once with a smaller
     `VLLM_CPU_KVCACHE_SPACE` (default 90). This is the direct vLLM memory knob
     and often the real fix; export it before the run, e.g.
     `VLLM_CPU_KVCACHE_SPACE=45`.
  2. If lowering KV cache is not desired, raise `MEM_LIMIT` or pass
     `--no-mem-limit` (online multi) to drop only the `mem_limit` cap while
     keeping cpuset/shm/caps. `--no-mem-limit` is distinct from `--no-limits`
     (which drops all cgroup limits, for rootless/LSF). Always inform the user of
     the KV-cache option first.
- **`--no-limits` -- WARN the user**: it drops *all* cgroup limits, **including
  cpuset core-pinning**. Instances are then unpinned, so throughput/latency
  numbers are unreliable and instances may contend for the same cores. Only use
  it when cgroups genuinely aren't available (rootless/LSF). Whenever you (or the
  user) pass `--no-limits`, explicitly warn that pinning is off and prefer
  `--no-mem-limit` if the goal is just to avoid a memory-cgroup OOM. The runners
  also print this warning at runtime.
- **Podman required** for online multi (compose + NGINX). `common.sh` falls back
  to docker for the single-container quadrants but online-multi needs
  `podman-compose` or `podman compose`.
- **GuideLLM runs on the host**, not in the image. Install it
  (`pip install guidellm`) or set `BENCH_GUIDELLM_BIN` (do **not** use a
  `GUIDELLM_*` name -- GuideLLM treats those as its own config and warns). The
  runners call `guidellm benchmark run ...` explicitly (works on GuideLLM 0.6
  and 0.7+, which dropped the implicit default subcommand); other flags still
  vary by version, so adjust if needed. Both online runners `taskset`-pin the
  host client to its reserved cores (`--guidellm-cores` / `--guidellm-tail`) so
  it does not steal cores from the vLLM server.
- **Host-side HF offline**: `--hf-offline` only sets `HF_HUB_OFFLINE=1` *inside*
  the container. GuideLLM runs on the host and builds its synthetic dataset with
  the model's tokenizer -- on an HF-blocked host, also export `HF_HUB_OFFLINE=1`
  and `HF_HOME=<cache>` in the shell before calling the online runners, or
  GuideLLM fails resolving the tokenizer.
- **Rootless podman DNS**: online-multi assigns static IPs on a dedicated /24
  (`VLLM_SUBNET`, default `10.201.0.0/24`) so NGINX reaches instances by IP.
- **SMT/NUMA**: all quadrants pin to physical cores (first thread per core) --
  offline/online-single via `check_hardware.sh`'s `CPUSET`, online-multi via its
  `--slices` output. Keep `CPI` within a NUMA node for best throughput.
- **Container names must stay `bench-vllm-*`** so `mem_poll.sh` can find them.
- **HF cache** is bind-mounted and shared across instances; first run downloads
  the model. Use `HF_TOKEN` for gated models, `--hf-offline` when air-gapped.
- **Extra-arg quoting**: `--serve-args`/`EXTRA_VLLM_ARGS` are split on
  whitespace, so an individual argument value containing a space is not
  preserved. Fine for simple flags; avoid embedded spaces in a single value.

## Files
```
.claude/skills/benchmark-vllm/
  SKILL.md                         # this file
  benchmark-vllm-flow.mmd          # raw Mermaid source for the Flow diagram
  scripts/
    common.sh                      # runtime detect, image pull, perf env, helpers
    check_hardware.sh              # lscpu topology, N/CPI validation, recommendation
    incontainer_env.sh             # (in container) LD_PRELOAD + tuning
    incontainer_serve.sh           # (in container) entrypoint for online serve
    offline_launcher.sh            # (in container) N x vllm bench throughput
    run_offline_single.sh          # host: offline single (N=1 wrapper)
    run_offline_multi.sh           # host: offline multi
    run_online_single.sh           # host: online single (serve + GuideLLM)
    parse_offline.py               # offline throughput -> CSV
    mem_poll.sh                    # peak memory sampler
  online-multi/
    generate-config.sh             # compose + nginx.conf + .env
    start.sh                       # compose up + health wait
    stop.sh                        # compose down
    run_sweep.sh                   # generate -> start -> GuideLLM -> stop -> parse
```
