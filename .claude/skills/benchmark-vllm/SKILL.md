---
name: benchmark-vllm
description: Measures vLLM serving performance on the zentorch/ZenDNN CPU backend on AMD EPYC. Brings up a multi-instance stack -- N cpuset-pinned `vllm serve` containers, one core slice each, behind an NGINX round-robin load balancer -- and drives it with GuideLLM through the redhat-et/vllm-cpu-perf-eval ansible automation in external-endpoint mode. Sizes the instance layout from the machine's real topology (SMT siblings kept inside one instance), derives the server context window from the selected workloads, and sweeps a list of request rates against one warm stack, so throughput numbers (high concurrency, 32/64+) and latency numbers (low concurrency, 1/2/4/8) come from an identical stack and stay comparable. Can also run a native (non-zentorch) A/B leg from the same image with zentorch uninstalled, so a zentorch-vs-native delta is not confounded by a different vLLM commit, torch build or compiler flags. Runs the perf-eval workload set (chat, rag, code, summarization, reasoning, variable-length and context-scaling profiles) against any Hugging Face model the image can serve, and produces per-run GuideLLM results, a resident-memory trace and a per-instance provenance banner (pip list, exported env, exact serve command). Use when asked to benchmark vLLM throughput, latency or serving performance on CPU with zentorch, or to size a multi-instance CPU serving run.
version: 1.0.0
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Benchmark vLLM (zentorch / ZenDNN CPU backend)

An **interactive** orchestrator around one stack shape:

```
GuideLLM (container, own cores) -> NGINX round-robin (own cores)
                                     -> N x `vllm serve` containers, one cpuset each
```

There is no second mode. Peak-throughput ("offline") numbers are the same stack
measured at high concurrency (`--rates 32,64` and up); latency numbers are the
same stack at low concurrency (`--rates 1,2,4,8`). Nothing else changes, so the
two profiles are directly comparable and a single warm stack can serve both.

The run itself is executed by [redhat-et/vllm-cpu-perf-eval](https://github.com/redhat-et/vllm-cpu-perf-eval)
(ansible + GuideLLM) in `VLLM_ENDPOINT_MODE=external`: our compose stack is the
endpoint, ansible only drives load and collects metrics. The harness that wires
the two together is vendored under `harness/` from
[amd/skills PR #81](https://github.com/amd/skills/pull/81) and kept close to
upstream so re-syncs stay cheap.

## Sections
- [Setup (one time)](#setup-one-time)
- [Golden rule: ask at every stage](#golden-rule-ask-at-every-stage)
- [Fail fast](#fail-fast)
- [Stage 1: profile, rates, workloads, duration](#stage-1-profile-rates-workloads-duration)
- [Stage 2: hardware check + core bands](#stage-2-hardware-check--core-bands)
- [Stage 3: image + backend (optional native A/B)](#stage-3-image--backend-optional-native-ab)
- [Stage 4: model + memory](#stage-4-model--memory)
- [Stage 5: confirm + run](#stage-5-confirm--run)
- [What bench.sh adds on top of the harness](#what-benchsh-adds-on-top-of-the-harness)
- [Results](#results)
- [Verify](#verify)
- [Gotchas](#gotchas)
- [Files](#files)

## Setup (one time)

```bash
scripts/setup-harness.sh
```

Clones `vllm-cpu-perf-eval` into `harness/vllm-cpu-perf-eval/` (gitignored,
**deliberately unpinned**, same as upstream) and applies
`harness/vllm-cpu-perf-eval.patch` with `git apply --3way` (rootless guidellm
user fix, `/tmp` -> `BENCH_TMPDIR` redirect, local-model bind mount). Safe to
re-run. It also reports missing ansible collections
(`containers.podman`, `ansible.posix`, `community.general`) -- install any it
flags before the first run.

Pre-warm the HF cache once per model so the first instance does not download
under load:

```bash
HF_HOME=$HOME/.cache/hf-shared/huggingface hf download meta-llama/Llama-3.1-8B-Instruct
```

## Golden rule: ask at every stage
Drive the user through the stages **one at a time**, offering a sensible default
at each. Do NOT run anything until the Stage 5 confirmation. `bench.sh
--dry-run` prints the resolved configuration and every ansible command without
executing -- show that before the real run.

## Fail fast
If **any step fails, STOP immediately** -- do not continue to the next rate,
workload or A/B leg. A "failure" includes: a non-zero exit from any script, an
image pull error, an instance that does not become healthy within the timeout, a
`vllm serve`/GuideLLM crash or traceback, an empty/zero result, or a container
OOM. When something fails: surface the exact error and the relevant log
(`results/sweep_<label>.log`, the run's `guidellm.log`, `podman ps`), tear down
anything still up (`harness/stop.sh`), and report the likely cause and fix.
Never silently push past an error or fabricate results.

## Stage 1: profile, rates, workloads, duration

Ask for the **profile** first; it sets the rates and nothing else:

| Profile | `--rates` | Measures | Read from GuideLLM |
|---------|-----------|----------|--------------------|
| throughput ("offline") | `32,64` (and up: `96`, `128`) | saturation: tok/s across the whole stack | Server Throughput Statistics |
| latency | `1,2,4,8` | per-request behaviour: TTFT / ITL / TPOT | Request Latency Statistics |

Rates are **absolute at the load balancer**, not per instance: `--rates 64` with
`--n 4` is ~16 concurrent requests per instance. Ask the user to confirm the
list; do not assume.

Then ask the **workloads** (`-w`, repeatable, default `chat`). All of them run
against **one warm stack** -- one GuideLLM run each, one results dir each -- so
adding a workload costs a load run, not another ~15 min warm-up.

```bash
scripts/bench.sh --list-workloads     # isl / osl / declared / derived max-model-len
```

| Workload | isl / osl | Shape |
|----------|-----------|-------|
| `chat` | 512 / 512 | the primary working point |
| `chat_lite` | 128 / 128 | short turns |
| `rag` | 7680 / 512 | long retrieved context, short answer |
| `code` | 1024 / 1024 | code generation, near the practical CPU OSL ceiling |
| `summarization` | 2048 / 256 | long in, short out |
| `reasoning` | 256 / 2048 | short in, long out |
| `chat_var`, `code_var`, `summarization_var` | variable | realistic traffic (normal-distributed isl/osl) |
| `context_scaling_1k/4k/8k` | 1k/4k/8k / 256 | TTFT vs context length |
| `embedding` | 512 / 1 | embedding models only |

Mixing workloads is fine and cheap, but the context window is sized for the
**largest** of them (see below), so `-w chat -w rag` runs `chat` on an
8320-token window rather than 2048.

Finally the **duration**: `--max-seconds`, seconds per rate, **default 300**.
Offer 300; shorter runs (60-120) are for smoke tests only.

## Stage 2: hardware check + core bands

```bash
scripts/check_hardware.sh                                    # topology + recommendation
scripts/check_hardware.sh --mode multi -n <N> -c <CPI> \
    --exclude 1-15,16-31 --slices                            # validate + print slices
```

Ask for **N instances** and **cores per instance (CPI)**; suggest the detected
recommendation (typically one instance per NUMA node, or per CCD group).
Enforced:

- `N * CPI <= physical cores` remaining **after** the nginx and guidellm bands.
  If it does not fit, **do not run** -- show the recommended `N`/`CPI` and
  re-prompt.
- Warn when `CPI` does not divide the NUMA-node core count (instances would
  straddle NUMA nodes).

**Confirm the 3-band layout with the user** -- changing a band moves the vLLM
instances:

```
nginx  1-15   (--nginx-cores)     GuideLLM  16-31  (--guidellm-cpus)     vLLM  32+
```

`bench.sh` excludes both bands from the pool, then takes the per-instance
cpusets from `check_hardware.sh --slices`, which walks **physical** cores and
keeps SMT siblings inside one instance. Announce the resolved sets (e.g.
`cpusets: 32-47,160-175;48-63,176-191`) and get a yes.

## Stage 3: image + backend (optional native A/B)

**Always ask for the image.** Default: the newest `amdih/zendnn_zentorch` tag
(`DEFAULT_IMAGE` in `scripts/common.sh`, currently
`vllm_v0.24.0_zentorch_v2.11.0.3_ubuntu22.04_2026_ww28`); browse tags at
https://hub.docker.com/r/amdih/zendnn_zentorch/tags .

Prefer an image already present on the host (`podman images`) over pulling a new
tag, and note that comparing two separately-built images conflates zentorch with
the vLLM commit, torch build and compiler flags baked into each, so change one
thing at a time.

Then the **backend**, and **ask whether an A/B against native is wanted**:

- `--zentorch` (default) -- zentorch backend, the `ZENDNNL_*` and inductor knobs
  from `scripts/incontainer_env.sh` and `harness/generate-config.sh`.
- `--native` -- the *same image* with zentorch pip-uninstalled and committed as
  `<image>_native` (built by `harness/start.sh` on first use, cached after
  that), zentorch env knobs off. Deriving the native image rather than asking
  for a second one is the point: two separately-built images also differ in
  vLLM commit, torch build and compiler flags, and those differences show up in
  the numbers as if they were zentorch's doing.

Ask it plainly, e.g. *"zentorch only, or also a native (non-zentorch) run on the
same image so the two can be compared?"* If A/B is wanted, that is **two
`bench.sh` invocations, run back to back on the same host**, identical apart
from `--native` and `--run-tag`:

```bash
scripts/bench.sh ... --run-tag _zentorch
scripts/bench.sh ... --native --run-tag _native
```

Run zentorch first, and treat the pair as one job: confirm both commands at
Stage 5, and if the first leg fails, stop and report rather than running the
second (see [Fail fast](#fail-fast)). Comparing legs from different hosts, stack
shapes or images is not a zentorch-vs-native result. The first `--native` run
pays a one-off image build (a few minutes) on top of the usual warm-up.

## Stage 4: model + memory

- **model** (`-m`): HF repo id, or a path under `--models-dir` (bind-mounted).
  Default `meta-llama/Llama-3.1-8B-Instruct`. Gated models need `HF_TOKEN`
  exported. The `test_name` tag is derived from the basename; pass the upstream
  `"path | tag"` form to choose it yourself.
- **`--kv-cache-space N`** (GiB, per instance) -- `VLLM_CPU_KVCACHE_SPACE`. It
  is the direct memory knob and is per instance, so N instances multiply it.
  Raise for long-context workloads (`rag`, `context_scaling_8k`), lower it first
  when a container gets OOM-killed.
- **`--dtype` / `--block-size`** -- only when the user asks; defaults come from
  the image/vLLM. Both are passed straight to `vllm serve`.
- **`--max-model-len`** -- normally leave it alone (derived, see below). Set
  below what the workloads need, `bench.sh` warns and the run fills with HTTP
  400s.

## Stage 5: confirm + run

Show the assembled command, get a yes, dry-run it, then run for real:

```bash
# throughput profile, chat + rag on one warm stack
scripts/bench.sh --model meta-llama/Llama-3.1-8B-Instruct \
    --n 4 --cpi 32 --rates 32,64 --max-seconds 300 \
    -w chat -w rag --image <IMG> --kv-cache-space 40 [--dry-run]

# latency profile, same stack shape
scripts/bench.sh --model meta-llama/Llama-3.1-8B-Instruct \
    --n 4 --cpi 32 --rates 1,2,4,8 --max-seconds 300 -w chat --image <IMG>

# native leg of an A/B (identical apart from --native and --run-tag)
scripts/bench.sh ... --native --run-tag _native
```

`bench.sh --help` lists every flag. Use `--run-tag` so repeat sweeps do not
clobber each other's `results/` files.

Call chain:

```
scripts/bench.sh            flags -> env, cpusets, max-model-len, model tag
  scripts/run_combo.sh      mem poller + results/ + tmp/ + sweep log
    harness/run_sweep.sh    preflight, test_name, per-workload ansible loop
      harness/start.sh      generate-config.sh -> compose up -> health wait
      ansible (perf-eval)   GuideLLM against the NGINX endpoint
      harness/stop.sh       compose down
```

## What bench.sh adds on top of the harness

Three adjustments the upstream harness does not make for a zentorch-on-EPYC
run:

1. **`--max-model-len` is derived from the workloads**, not hardcoded to 4096.
   `scripts/workload_info.py` reads the vendored `test-workloads.yml` and
   computes `max(isl_max|isl + osl_max|osl + 128, declared --max-model-len)`
   across every `-w`, so a workload whose isl+osl exceeds the server's context
   window (`rag` needs 8320, `context_scaling_8k` 16384) is not sent to an
   undersized server.
2. **Instance cpusets come from `check_hardware.sh --slices`** (physical cores,
   SMT siblings kept together) instead of contiguous core arithmetic, which
   straddles sibling threads on SMT-enabled EPYC. If slicing fails, `bench.sh`
   warns and falls back to upstream's arithmetic.
3. **The `test_name` tag is derived and length-checked here**, so a long HF repo
   id cannot fail ansible's `^[A-Za-z0-9-]{1,30}$` validator at preflight.

Plus: a provenance banner per instance (`scripts/incontainer_serve.sh` prints
`pip list`, the exported env and the exact `vllm serve` command into the
instance log), and several workloads per warm stack.

## Results

Upstream layout, unchanged. Per GuideLLM run, inside the clone:

```
harness/vllm-cpu-perf-eval/results/llm/<model>/<workload>-<ts>-<test_name>/external-endpoint/
```

Alongside, from the wrapper (in the directory you ran `bench.sh` from):

```
results/sweep_<label>.log     full sweep transcript
results/mem_<label>.csv       podman stats samples + a final PEAK line
harness/sweep-logs/           per-run harness logs
```

Scoring rule: the authoritative numbers are in each run's `guidellm.log` under
**Server Throughput Statistics** (throughput profile) and **Request Latency
Statistics** (latency profile). Parse with:

```bash
scripts/parse_guidellm_log.py <path>/guidellm.log
```

`<label>` defaults to `n<N>-<backend>-<tag>` (e.g. `n4-zentorch-llama-3-1-8b`,
`n4-native-llama-3-1-8b`),
plus `--run-tag`.

## Verify

- `scripts/check_hardware.sh` alone prints topology + a recommended layout.
- `scripts/bench.sh --list-workloads` prints the workload table with the derived
  context window per workload.
- `scripts/bench.sh ... --dry-run` prints the resolved banner and every ansible
  invocation (with `test_name`, `base_workload`, `guidellm_cpus`) and exits
  without starting a container.

## Gotchas

- **`podman info` can hang on some hosts** (>25 s), and `harness/check-host.sh`
  calls it, so a run appears to stall before anything starts. Work around with
  `SKIP_HOST_CHECK=1 scripts/bench.sh ...` after confirming podman itself works
  (`podman ps`).
- **The vendored clone is unpinned** (same as upstream). If upstream drifts, the
  `--3way` apply can conflict; `setup-harness.sh` says so and leaves the tree for
  manual resolution in `harness/vllm-cpu-perf-eval/`. Re-generate
  `harness/vllm-cpu-perf-eval.patch` from the resolved diff if the fix should
  stick.
- **Missing ansible collections** are the most common first-run failure. Install
  whatever `setup-harness.sh` marks `MISS`.
- **`sudo` is not required.** The harness auto-detects it (`sudo -n true`) and
  passes `-e ansible_become=false` when unavailable; `--no-become` forces it.
- **Rates are absolute**, at the load balancer. `GUIDELLM_MAX_CONCURRENCY` is set
  to at least the largest rate, otherwise the concurrent profile silently clamps.
- **Container OOM while the host has free RAM** is the per-instance `mem_limit`
  cgroup cap. Lower `--kv-cache-space` first (per instance, so N instances
  multiply it); `run_combo.sh` already passes `--no-mem-limit` by default, which
  drops only that cap and keeps cpuset/shm/caps.
- **Rootless podman DNS**: instances get static IPs on a dedicated /24
  (`10.201.0.0/24`) because aardvark-dns is unreachable on the bridge; NGINX
  reaches them by IP.
- **HF offline**: `run_combo.sh` exports `HF_HUB_OFFLINE=1` and a placeholder
  `HF_TOKEN`. Pre-warm the cache (see Setup) or the run fails resolving the repo.
- **Do not compare across stack shapes.** N, CPI, the core bands and
  `--max-model-len` all move the numbers; change one thing at a time. For a
  zentorch-vs-native A/B this means the two legs differ only by `--native`, on
  the same host, back to back.
- **The `_native` image is cached** once built (`podman images | grep _native`).
  If the base image is rebuilt under the same tag, delete the stale
  `<image>_native` or the A/B compares against an old build.

## Files

```
.claude/skills/benchmark-vllm/
  SKILL.md                         # this file
  reference.md                     # upstream harness notes (from amd/skills PR #81)
  benchmark-vllm-flow.mmd          # raw Mermaid source for the flow diagram
  scripts/
    bench.sh                       # THE entry point: flags -> harness env
    setup-harness.sh               # one-time: clone + patch vllm-cpu-perf-eval
    check_hardware.sh              # lscpu topology, N/CPI validation, --slices
    workload_info.py               # workload table -> derived --max-model-len
    common.sh                      # DEFAULT_IMAGE + model-name helper
    run_combo.sh                   # mem poller + results/ + tmp/ + sweep log
    mem_poll.sh                    # podman stats sampler (PEAK line)
    incontainer_serve.sh           # (in container) entrypoint + provenance banner
    incontainer_env.sh             # (in container) LD_PRELOAD + tuning env
    parse_guidellm_log.py          # guidellm.log -> throughput / latency table
    extract_perf.py, detect.py     # upstream helpers
    Dockerfile.buildB              # upstream alternate image build
  harness/                         # vendored from amd/skills PR #81
    run_sweep.sh                   # preflight, test_name, per-workload ansible loop
    start.sh / stop.sh             # compose up (+ native image build) / down
    generate-config.sh             # compose + nginx.conf + .env
    check-host.sh                  # host preflight (see the podman info gotcha)
    vllm-cpu-perf-eval.patch       # patch applied to the clone
    vllm-cpu-perf-eval/            # the clone itself (gitignored, unpinned)
```
