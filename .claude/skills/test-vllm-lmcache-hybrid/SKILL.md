---
name: test-vllm-lmcache-hybrid
description: >-
  Serve and validate hybrid (Gated DeltaNet + attention) models such as
  Qwen3.5 / Qwen3.6 with LMCache SHM on vLLM CPU + zentorch through the
  benchmark-vllm-guidellm multi-instance harness. Use when running these models
  with LMCache, checking that LMCache restores are correct (cold vs restored vs
  vLLM prefix cache), load-testing cross-instance cache hits, or debugging
  "Mamba"/"hybrid KV cache groups" LMCache errors and garbage output on
  zentorch.
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Validate hybrid models with LMCache (vLLM CPU + zentorch)

## What this covers

Qwen3.5 / Qwen3.6 interleave linear-attention (Gated DeltaNet) layers with full
attention, so vLLM gives them two KV cache groups (attention + Mamba state).
Stock LMCache rejects that on CPU. Validated end to end on AMD EPYC Turin with
vLLM 0.29 and 0.30 for:

| Key | Checkpoint | N |
|---|---|---|
| q35bf | Qwen/Qwen3.5-9B | 640 |
| q35w8 | RedHatAI/Qwen3.5-9B-quantized.w8a8 | 640 |
| q35w4 | RedHatAI/Qwen3.5-9B-quantized.w4a16 | 640 |
| q36bf | RedHatAI/Qwen3.6-35B-A3B | 1152 |
| q36w8 | amd/Qwen3.6-35B-A3B-w8a8-llmcompressor | 1152 |
| q36w4 | amd/Qwen3.6-35B-A3B-w4a16-llmcompressor | 1152 |

N is vLLM's unified block size on CPU (logged as "Setting attention block size
to N tokens"). `start.sh --lmcache` gets it from vLLM in the image
(`scripts/hybrid_block_size.py`) and applies the recipe below on its own.

The checks drive the stack through the
[`benchmark-vllm-guidellm`](../benchmark-vllm-guidellm/SKILL.md) harness
(`start.sh --lmcache`, `run_sweep.sh --lmcache`, `-w chat_shared`). Set that
skill up first: its clone pinned to `2bacf71` with the patches applied. Its
[LMCACHE.md](../benchmark-vllm-guidellm/LMCACHE.md) explains the LMCache modes.

## Prerequisites

- **LMCache** with hybrid engine-driven transfer: branch
  `zen-lmcache/cpu-shm` of `amd/LMCache_zen` (the same checkout
  benchmark-vllm-guidellm uses).
  Upstream `dev` without it fails the first store with "engine-driven transfer
  does not support hybrid KV cache groups".
- **Image**: either load a prebuilt `_lmcache` image
  ```bash
  podman load -i <vllm_v0.30.0_zentorch_..._lmcache_<sha>.tar.gz>
  # start.sh --lmcache runs ${VLLM_IMAGE}_lmcache, so name it accordingly:
  podman tag localhost/vllm_v0.30.0_zentorch_v2.13.0.2:ubuntu26.04_2026_ww40_lmcache_b7b84e67 \
      localhost/vllm030:upstream_lmcache
  export VLLM_IMAGE=localhost/vllm030:upstream
  ```
  or let `start.sh` build `${VLLM_IMAGE}_lmcache` from
  `LMCACHE_SRC=<checkout of the branch>` (it skips the build when that image
  exists, so `podman rmi` it after changing the source).
- **zentorch**: use vLLM 0.31.0 with zentorch main (`26f305ed` or later). It
  gives correct text with default weight prepack for Qwen3.5-9B bf16 and w8a8
  (also with an fp8 KV cache) and for Qwen3-0.6B, so drop the
  `VLLM_ZENTORCH_WEIGHT_PREPACK=0` line from the recipe on it. The 0.30 ww40 image
  (zentorch-weekly 2.13.0.2.dev20260930) is correct for Qwen3.5-9B only with
  `VLLM_ZENTORCH_WEIGHT_PREPACK=0`, and Qwen3-0.6B is wrong on it even then,
  with or without LMCache. The 0.29 ww39 image
  (dev20260923) produces wrong text for every model (its
  `zentorch_linear_unary` is wrong at M=1; no env fixes it). Both images ship
  torch 2.13.0+cpu at the same commit, so for 0.29 copy
  `site-packages/zentorch` and its dist-info from the 0.30 image into a 0.29
  container and `podman commit` it. GuideLLM success counts do not check text,
  so always run the correctness check after changing images.
- Usual harness env (`HF_TOKEN`, `HF_CACHE_DIR`, podman-compose and Ansible on
  `PATH`, `CONTAINERS_STORAGE_CONF` for a private store). Run from a clone on
  local disk, not NFS: the harness writes `generated/`, `sweep-logs/`, results.
- `rg` (ripgrep) on `PATH`: the checks count errors and LMCache retrieves with
  it, and exit with an error when it is missing.

## Recipe

`start.sh --lmcache` sets the first three for a hybrid model unless they are
already set (an explicit `ENABLE_PREFIX_CACHING=false`, or a chunk size that is
not a multiple of N, stops it with an error):

```bash
ENABLE_PREFIX_CACHING=true          # vLLM picks Mamba cache mode 'align'
LMCACHE_CHUNK_SIZE=N
EXTRA_VLLM_ARGS="--max-num-batched-tokens N"
SET_ENV_VAR_BEFORE_VLLM_SERVE="export VLLM_ZENTORCH_WEIGHT_PREPACK=0 &&"   # 0.30 ww40 only
# w4a16 checkpoints: prepend "export VLLM_CPU_INT4_W4A8=1 && "
```

With `--max-num-batched-tokens 2N-1` (the upstream `qwen3_5` recipe) restores
are correct and throughput is ~5% higher, but the cold run computes the prompt
tail in a different-sized chunk and its text can drift from restored runs after
the first sentence. Budget N makes cold, LMCache-restored and vLLM-restored
output identical token for token.

## Validate

```bash
S=.claude/skills/test-vllm-lmcache-hybrid/scripts
export VLLM_IMAGE=localhost/vllm030:upstream     # harness uses ${VLLM_IMAGE}_lmcache
$S/validate_hybrid.sh correctness                # 2 instances, ~4-8 min per model
$S/validate_hybrid.sh load                       # 5 instances, GuideLLM chat_shared, ~20 min per model
ONLY=q35bf,q36w4 $S/validate_hybrid.sh correctness -030   # subset, label suffix
```

Single models: `correctness_check.sh <label> <model>` and
`load_check.sh <label> <model>` with the recipe env set. All three exit 0 only
on pass. `HARNESS_DIR` defaults to `.claude/skills/benchmark-vllm-guidellm/scripts`.

Pass criteria:

- **correctness**: the same greedy prompt cold on instance 1 (A), restored from
  LMCache on instance 2 (B), and restored from vLLM's prefix cache on instance 1
  (D) all name the right code word, A == B == D (`PREFILL_BUDGET=2n-1` relaxes
  to B == D), the server logs a `Retrieved` line, and no instance logs a
  traceback. Expect `Retrieved 2560 tokens` (Qwen3.5) / `2304` (Qwen3.6).
- **load**: 0 errored GuideLLM requests at every rate, no AssertionError /
  EngineDeadError / HTTP 500 before teardown, LMCache retrieves > 0. Reference
  (0.29 and 0.30, budget N): Qwen3.5 46-58% and Qwen3.6 19-28% per-instance
  LMCache hit rate. Qwen3.6 tops out lower because only one 1152-token chunk
  fits in each 2048-token shared prefix.

Run long jobs in tmux so they survive disconnects. The login shell may be csh,
which rejects `2>&1`, so pass the command to bash:

```bash
tmux new-session -d -s hybrid bash -c "$S/validate_hybrid.sh load 2>&1 | tee hybrid-load.log"
```

## Pitfalls

- **"MambaSpec with mamba_cache_mode='none'" at startup**: prefix caching is
  off, so vLLM forces Mamba cache mode 'none' (even with `--mamba-cache-mode
  align`). `start.sh --lmcache` turns it on for hybrid models; if you still see
  this, check the "Checking whether ... is a hybrid model" lines in its output.
- **Chunk size rejected**: `LMCACHE_CHUNK_SIZE` must be a multiple of N.
- **LMCache-driven mode** (`LMCACHE_MP_TRANSFER_MODE=lmcache_driven`) fails at
  registration ("requires a contiguous tensor"); do not relax that check, it
  would alias stale Mamba state. Use engine-driven (the default).
- **`ZENDNNL_MATMUL_WEIGHT_CACHE=0`** is not needed for serving and makes the
  Qwen3.6 w4a16 MoE path ~18x slower (requests time out at nginx's 300 s).
- **Disk**: with the VFS storage driver every container is a full image copy
  (~8.4 GB); a 5-instance stack needs ~60 GB free. A full disk shows up as
  `cannot write PCH file: No space left on device` during torch.compile warmup,
  and that instance's API server can hang instead of exiting.
- **`EngineDeadError` right after "[shutdown] API server: shutdown triggered"**
  is the harness teardown, not a failure; the load check ignores it.
- **Ansible rejects test names over 30 chars** (`<label>-zentorch-<tag>`); keep
  labels short.

## Files

```
.claude/skills/test-vllm-lmcache-hybrid/
  SKILL.md                     # this file
  scripts/
    validate_hybrid.sh         # runs a check for the six checkpoints, prints a summary
    correctness_check.sh       # 2 instances: cold vs LMCache-restored vs vLLM-restored text
    load_check.sh              # one run_sweep.sh --lmcache pass + verdict from results and logs
```
