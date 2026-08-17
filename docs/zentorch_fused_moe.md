(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch_fused_moe — Fused Mixture-of-Experts FFN Block (Out Variant)

## 1. Overview

`zentorch_fused_moe` is a single-call operator that executes the full Mixture-of-Experts (MoE) FFN block:

```
input [T, H]
   ├─ token-expert grouping (per-active-expert input buffers)
   ├─ W13 gate+up projection                 (per-expert GEMM, batched)
   ├─ gated activation (SiLU / GELU / SwigluOAI)
   ├─ W2 down projection                     (per-expert GEMM, batched)
   └─ router-weighted reduce into output [T, H]
```

The C++ op assembles the per-active-expert input buffers and the routing metadata, then delegates the actual GEMMs + post-ops to `zentorch_group_matmul_out_impl` (which wraps ZenDNN LowOHA's `group_matmul_direct`). The full chain runs inside one backend call.

The op is the C++ landing pad for vLLM's `CPUFusedMOE` forward (patched by `src/cpu/python/zentorch/vllm/__init__.py`); its schema mirrors vLLM's `cpu_fused_moe` signature so the patched dispatch can swap the op name without touching call sites.

**One op, three weight regimes.** The same schema and single backend call serve three weight formats, dispatched by the weight tensor dtype inside `GroupMatmul.cpp`:

| Regime | `w13` / `w2` dtype | Scales | Activation quant |
|--------|--------------------|--------|------------------|
| **bf16 / f32** | same as `input` | none | none |
| **DA8W8** | `torch.int8`, full-width `[E, N, K]` | per-channel or per-group | dynamic per-token s8 |
| **DA8W4** | packed s4: `torch.int32` `[E, N, K/8]` or `torch.int8` `[E, N, K/2]` | per-group `[E, G, N]` | dynamic per-token s8 |

Both DA8W4 containers hold the same s4 nibble stream, so they are interchangeable. `torch.int8` therefore serves both quantized regimes and is disambiguated by its last dim: full width is DA8W8, half width is packed s4.

The DA8W4 regime is symmetric-only (no zero-points) and needs no `expert_map` (CPU MoE is single-rank), so it maps 1:1 onto this schema: packed int4 weights go in `w13` / `w2` and per-group weight scales in `w13_scales` / `w2_scales`. See [§9 "DA8W4 weights"](#9-da8w4-weights).

> **Note:**
> - The op is an **out variant**: `output` is allocated by the caller and mutated in place. The schema marks it `Tensor(a!)` and the op returns `()`. The caller does **not** need to zero-initialise it — the weighted-reduce post-op writes every `[T, H]` element (the `k = 0` slot initialises, `k > 0` accumulate).
> - Shape / dtype / bias validation is performed once by the producing Python layer — the `CPUFusedMOE` patch (`vllm/__init__.py`) for bf16 / DA8W8, or the DA8W4 experts backend (`vllm/model_executor/layers/fused_moe/experts/zentorch_moe.py`) for DA8W4. The C++ op trusts its inputs (it only fails fast on the DA8W4-requires-bf16 and `E_a > 1` invariants).
> - Only experts that actually receive at least one routed token are materialised in Phase 1 and forwarded to the backend (the **active set** of size E_a ≤ E).

## 2. Motivation

vLLM's stock CPU MoE path picks between two implementations:

- `cpu_fused_moe` — hand-written AMX/VEC MicroGemm kernel with prepacked weights.
- `cpu_fused_moe_torch` — a per-expert `F.linear` loop.

The MicroGemm path requires offline prepacking of weights (and a separate quantization flow per dtype); the torch loop incurs per-expert kernel-launch overhead, allocates intermediates between W13, activation, W2, and reduce, and reads each expert's output buffer once for the reduce.

`zentorch_fused_moe` collapses the whole MoE FFN block into a single backend call:

- **Token-expert grouping in C++** with no atomics — positions for each routed `(t, k)` pair are pre-assigned during a cheap single-threaded sweep, then a `parallel_for` does the actual memcpy in Phase 1.
- **Active-set narrowing** — experts that receive zero routed tokens are skipped entirely; we forward only E_a slices of `w13` / `w2` / biases to the backend.
- **Buffer aliasing** — the per-expert input buffers are reused as W2 output buffers, since W13 has already consumed them by the time W2 writes. No second allocation per active expert.
- **Fused post-op chain** — W13 → gated activation → W2 → router-weighted reduce executes inside one `group_matmul_direct` call.
- **Standard `[E, ...]` weight layout** — no prepack step, weights are consumed in the same layout vLLM stores them in.

## 3. API

### Signature

```python
torch.ops.zentorch.zentorch_fused_moe(
    output,         # Tensor(a!), [T, H], same dtype as input, uninitialized
    input,          # Tensor, [T, H], bf16 / f32 / fp16 (bf16 only for DA8W8 and DA8W4), contiguous
    w13,            # Tensor, [E, 2*I, H] (bf16/f32/fp16/int8) or DA8W4 packed s4:
                    #   [E, 2*I, H/8] int32 or [E, 2*I, H/2] int8
    w2,             # Tensor, [E, H, I]   (bf16/f32/fp16/int8) or DA8W4 packed s4:
                    #   [E, H, I/8] int32 or [E, H, I/2] int8
    w13_bias,       # Optional[Tensor], [E, 2*I] or None
    w2_bias,        # Optional[Tensor], [E, H]   or None
    topk_weights,   # Tensor, [T, K], f32, contiguous
    topk_id,        # Tensor, [T, K], int32, contiguous, values in [0, E)
    skip_weighted,  # bool; if true, requires K == 1 (router weight pre-applied by caller)
    act,            # str: 'silu' | 'gelu' | 'gelu_tanh' | 'swigluoai'
    w13_scales=None, # Optional[Tensor], [E, N] / [E, G, N] (DA8W8) or [E, G, N] (DA8W4) or None
    w2_scales=None, # Optional[Tensor], [E, K_out] / [E, G, K_out] (DA8W8) or [E, G, H] (DA8W4) or None
    *, zentorch_op_name='zentorch::zentorch_fused_moe'
) -> None
```

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output` | Tensor (bf16/f32/f16) | `[T, H]`. **Out parameter**: allocated (uninitialised) by the caller. The op fully overwrites it with the per-token reduced expert outputs. |
| `input` | Tensor (bf16/f32/f16) | `[T, H]` token activations. Contiguous. bf16 only when weights are int8 (DA8W8) or packed s4 (the quantized kernels reject f32/fp16). |
| `w13` | Tensor (bf16/f32/f16/int8/int32) | `[E, 2*I, H]` gate+up projection weights (concatenated), or packed s4 for DA8W4: `[E, 2*I, H/8]` int32 or `[E, 2*I, H/2]` int8. Sliced per-active-expert via `select(0, e)`. When DA8W8 or DA8W4, `w13_scales` is required. |
| `w2` | Tensor (bf16/f32/f16/int8/int32) | `[E, H, I]` down-projection weights, or packed s4 for DA8W4: `[E, H, I/8]` int32 or `[E, H, I/2]` int8. Sliced per-active-expert via `select(0, e)`. When DA8W8 or DA8W4, `w2_scales` is required. |
| `topk_weights` | Tensor (f32) | `[T, K]` router weights used by the weighted-reduce post-op. |
| `topk_id` | Tensor (int32) | `[T, K]` expert ids, values in `[0, E)`. |
| `skip_weighted` | bool | If `true`, the caller has already multiplied `input` by the (K=1) router weight, so the reduce post-op is fed an all-ones weight vector. |
| `act` | str | Gated activation applied between W13 and W2. One of `'silu'`, `'gelu'`, `'gelu_tanh'`, `'swigluoai'`. Maps to `silu_and_mul`, `gelu_and_mul`, `swiglu_oai_mul` enums internally (`gelu_tanh` aliases `gelu_and_mul`). |
| `w13_bias` | Tensor? (bf16/f32) | `[E, 2*I]` or `None`. Default `None`. |
| `w2_bias` | Tensor? (bf16/f32) | `[E, H]` or `None`. Default `None`. |
| `w13_scales` | Tensor? (f32/bf16) | Per-expert quantization scales for quantized `w13`. DA8W8: `[E, N]` (per-channel) or `[E, G, N]` (per-group). DA8W4: per-group `[E, G, N]`. Default `None` (for bf16/f32/fp16). |
| `w2_scales` | Tensor? (f32/bf16) | Per-expert quantization scales for quantized `w2`. DA8W8: `[E, K_out]` (per-channel) or `[E, G, K_out]` (per-group). DA8W4: per-group `[E, G, H]`. Default `None` (for bf16/f32/fp16). |
| `zentorch_op_name` | str | Profiling / tracing name. Default `'zentorch::zentorch_fused_moe'`. |

> **DA8W4 detection.** The DA8W4 path is selected from `w13` alone, with no extra schema arg, and classification runs on the per-expert 2D slice (`w13.select(0, e)`), not on the stacked 3D tensor. For the pack-factor rules and the `w2` inheritance contract, see [zentorch_group_matmul.md](./zentorch_group_matmul.md) §4 "DA8W4 detection".

## 5. Input Contract (Constraints)

| Constraint | Condition |
|-----------|-----------|
| `output` | 2D `[T, H]`, same dtype as `input`. Incoming contents are fully overwritten, so zero-init is not required |
| `input` dtype | `torch.bfloat16`, `torch.float32`, or `torch.float16` (fp16 requires AVX-512 FP16 hardware support); bf16 **only** when weights are int8 (DA8W8) or packed s4 |
| `input` layout | 2D `[T, H]`, contiguous |
| `w13`, `w2` dtype | Same dtype as `input`, `torch.int8` at full K (dynamic A8W8), or packed s4 (`torch.int32` at K/8, `torch.int8` at K/2). `w13` and `w2` must share one dtype. |
| `w13`, `w2` shape | 3D, leading dim `E`; DA8W4 packs the K dim by the container's factor (`[E, N, K/8]` int32 or `[E, N, K/2]` int8) |
| `w13_scales` | Required when `w13` is DA8W8 or packed s4. DA8W8 `[E, N]`/`[E, G, N]`; DA8W4 `[E, G, N]` (per-group), f32/bf16. `None` for bf16/f32/fp16. |
| `w2_scales` | Required when `w2` is DA8W8 or packed s4. DA8W8 `[E, K_out]`/`[E, G, K_out]`; DA8W4 `[E, G, H]` (per-group), f32/bf16. `None` for bf16/f32/fp16. |
| `w13_bias`, `w2_bias` | `None` or 2D `[E, …]`, same dtype as `input`. |
| `topk_weights` | 2D `[T, K]`, `torch.float32`, contiguous |
| `topk_id` | 2D `[T, K]`, `torch.int32`, contiguous, values in `[0, E)` |
| `skip_weighted` | If `true`, requires `K == 1` |
| `act` | One of `'silu'`, `'gelu'`, `'gelu_tanh'`, `'swigluoai'` |
| Active experts | `E_a > 1` required (`group_matmul_direct` needs ≥ 2 active experts); `E_a == 1` raises |

Validation lives in the producing Python layer — `_moe_forward_zentorch` / patch install in `src/cpu/python/zentorch/vllm/__init__.py` for bf16/DA8W8, or the DA8W4 experts backend for DA8W4. The C++ op assumes the contract holds; it only fails fast on the packed-s4 layout of `w13`, on `w13.dtype == w2.dtype` and `input.is_bfloat16()` for packed-s4 weights, and on the `E_a > 1` grouped-GEMM precondition.

### Dynamic A8W8 quantization support

When `w13` or `w2` weights are `torch.int8`, the corresponding `w13_scales` / `w2_scales` tensors must be provided. The Python patch layer (`_patched_init`) automatically detects DA8W8 weights when `torchao` is installed: it checks `isinstance(w, Int8Tensor)` for each weight attribute, extracts `w.scale` with a `shape[-1] == 1` check (handles both 2D and 3D scale layouts), validates dtype (f32/bf16), and calls `replace_parameter(layer, weight_attr, w.qdata)` to replace the Int8Tensor with its raw int8 data. The forward path (`_moe_forward_zentorch`) fetches scales via `getattr(layer, "w13_scale", None)`.

## 6. Design

The op runs in two phases: **Phase 1 — Token-Expert Grouping** (custom C++ in `FusedMoE.cpp`), and **Phase 2 — Fused GEMM chain** (delegated to `zentorch_group_matmul_out_impl`).

### 6.1 Phase 1 — Token-Expert Grouping

For each routed pair `(t, k)`, expert `e = topk_id[t][k]` must receive a copy of `input[t]` in its per-expert input buffer. This is done with two sub-passes:

**Pass 1 — single-threaded bookkeeping (O(T·K·E_a)):**

1. Walk the T·K `(t, k)` pairs in flat order, `i = t*K + k`.
2. On first encounter of an expert `e`, append it to `active_expert_ids` (linear scan over the existing list to detect first-encounter; E_a is small in practice, the scan stays in L1). Push a new counter onto `tokens_per_active`.
3. For each `i`, record `topk_to_expert_row[i] = (a, pos)` where `a` is the active slot for `e` and `pos = tokens_per_active[a]++` is the deterministic row this pair will occupy in expert `a`'s eventual input buffer. **No atomics are needed in Pass 2** because positions are pre-assigned here.

`active_expert_ids` is the single source of truth for the active set — there is no parallel `expert_to_active[E]` reverse map.

**Allocation:**

After Pass 1, allocate `grouped_inputs[a] = at::empty({tokens_per_active[a], H})` for each active slot. The vector is size E_a and is handed directly to `group_matmul`.

**Pass 2 — parallel data movement (`at::parallel_for`):**

For each pair `i`, look up the pre-assigned `(a, pos)` and `memcpy` row `t = i / K` of `input` into row `pos` of `grouped_inputs[a]`. No locks, no atomics — Pass 1 guarantees every `(a, pos)` is unique.

### 6.2 Worked example

Setup: T = 3 tokens, H = 4, E = 5 experts, K = 2 top-k routing.

```
input  [T, H] = [[ t0_h0, t0_h1, t0_h2, t0_h3 ],   # token 0
                 [ t1_h0, t1_h1, t1_h2, t1_h3 ],   # token 1
                 [ t2_h0, t2_h1, t2_h2, t2_h3 ]]   # token 2

topk_id [T, K] = [[ 3, 0 ],     # token 0 -> experts {3, 0}
                  [ 3, 1 ],     # token 1 -> experts {3, 1}
                  [ 0, 1 ]]     # token 2 -> experts {0, 1}
```

After Pass 1 (single sweep over T·K = 6 pairs, first-encounter ordering):

```
active_expert_ids = [ 3, 0, 1 ]      # E_a = 3 (experts 2 and 4 never appear)
tokens_per_active = [ 2, 2, 2 ]      # final M_e per active slot

topk_to_expert_row [6 entries] = (active_idx, row_in_expert):
  i=0 (t=0,k=0) -> (a=0, row=0)   # expert 3
  i=1 (t=0,k=1) -> (a=1, row=0)   # expert 0
  i=2 (t=1,k=0) -> (a=0, row=1)   # expert 3
  i=3 (t=1,k=1) -> (a=2, row=0)   # expert 1
  i=4 (t=2,k=0) -> (a=1, row=1)   # expert 0
  i=5 (t=2,k=1) -> (a=2, row=1)   # expert 1
```

After Pass 2 (parallel memcpy using the pre-assigned positions):

```
grouped_inputs[0]  (active_idx 0 = expert 3, M=2) = [ input[0], input[1] ]
grouped_inputs[1]  (active_idx 1 = expert 0, M=2) = [ input[0], input[2] ]
grouped_inputs[2]  (active_idx 2 = expert 1, M=2) = [ input[1], input[2] ]
```

### 6.3 Phase 2 — Active-only weight slicing + fused execution

With the active set known, the op:

1. Builds size-E_a slice vectors for `w13`, `w2`, `w13_bias`, `w2_bias`, `w13_scales`, `w2_scales` via `select(0, e)` for each `e = active_expert_ids[a]`. Inactive experts contribute nothing to the backend call. Scale slicing only occurs when the corresponding weights are DA8W8 (checked via `torchao` availability and weight dtype).
2. Builds `row_ptrs[T·K]`: for each `i`, `row_ptrs[i] = &grouped_inputs[a].data[pos * row_bytes]`. The W2 down-projection writes per-expert outputs back into these same `grouped_inputs` buffers (W13 has already consumed them), so `row_ptrs[i]` is exactly where the `(t, k)`-th expert result will live by the time the weighted-reduce post-op runs.
3. If `skip_weighted` is set, substitutes an all-ones weight vector (router weights have been pre-applied to `input` by the caller).
4. Calls `zentorch_group_matmul_out_impl` once with `gemm_outputs={}` (backend allocates W13 outputs internally), `w2_outputs = grouped_inputs` (aliased), and the post-op metadata (`topk_weights`, `row_ptrs`, `moe_output = output`).

### 6.4 Buffer aliasing — why `w2_outputs == grouped_inputs` is safe

Within `group_matmul_direct`'s fused chain, the lifetime of each `grouped_inputs[a]` buffer is:

```
W13 reads grouped_inputs[a]   ──►   W13 outputs (internal buffer)
                                    ──► gated act ──► W2 inputs (internal buffer)
                                                       ──► W2 writes grouped_inputs[a]
                                                              ──► weighted reduce reads it
```

W13 has finished reading `grouped_inputs[a]` before W2 starts writing it, so reusing the buffer saves an `at::empty({M_e, H})` per active expert without aliasing hazards. The `row_ptrs` table targets the same buffers, so the post-op reads the W2 outputs directly without an extra copy.

### 6.5 Execution flow

```
zentorch_fused_moe()
  ├─ build_token_expert_mapping(input, topk_id):
  │     ├─ Pass 1 (single-threaded):
  │     │     ├─ Linear-scan registration into active_expert_ids
  │     │     └─ Assign deterministic (active_idx, pos) per (t, k) pair
  │     ├─ Allocate grouped_inputs[a] of shape [M_a, H] for each active slot
  │     └─ Pass 2 (at::parallel_for, grain=64): memcpy input rows into slots
  ├─ Build size-E_a slices: w13_slices, w2_slices, w13_bias_slices, w2_bias_slices,
  │                         w13_scale_slices (if int8), w2_scale_slices (if int8)
  ├─ Build row_ptrs[T*K]: pointers into grouped_inputs[a][pos]
  ├─ If skip_weighted: effective_topk_weights = ones_like(topk_weights)
  └─ zentorch_group_matmul_out_impl(
        gemm_outputs={},                         # backend allocates W13 dst internally
        inputs=grouped_inputs,                   # size E_a
        w13_weights=w13_slices,
        w2_weights=w2_weight_slices,             # fused W2 post-op
        moe_output=output,                       # weighted reduce target
        topk_weights=effective_topk_weights,
        row_ptrs=row_ptrs,
        activation=act,                          # gated act post-op
        w13_bias=w13_bias_slices,
        w2_bias=w2_bias_slices,
        w13_scales=w13_scale_slices,             # int8 w13 scales (or empty)
        w2_scales=w2_scale_slices,               # int8 w2 scales (or empty)
        zentorch_op_name=zentorch_op_name)
        # Backend runs W13 -> gated_act -> W2 -> weighted_reduce in one call
```

### 6.6 Optional two-pass split (`ZENTORCH_TWO_PASS`)

The single-call path above runs W13 → gated activation → W2 → weighted-reduce inside one
`group_matmul_direct` call. The op splits the chain into two backend calls when the `ZENTORCH_TWO_PASS` 
environment variable is set:

```
ZENTORCH_TWO_PASS=1
  ├─ Call 1: W13 + gated activation only
  │     gemm_outputs = per-expert [M_e, N] buffers (kernel writes the gated
  │                    result into the first I = N/2 columns)
  │     w2_weights   = {}      moe_output = None      row_ptrs = None
  │     → produces the activated intermediate, no W2, no reduce
  │
  └─ Call 2: W2 + MoE weighted-reduce only
        inputs       = first I columns of Call 1's output, made contiguous
        w13_weights  = w2 slices (W2 is handed in as the only matmul)
        gemm_outputs = grouped_inputs (so the pre-built row_ptrs still target
                       the correct W2 destination rows)
        activation   = "none"   moe_output = output   row_ptrs = row_ptrs
        → down projection + router-weighted reduce into output
```

When `ZENTORCH_TWO_PASS` is unset (the default), the
single-call fused path in 6.5 is used. This split path is exercised by
`test_int8_w13_and_w2_two_pass` when run with `ZENTORCH_TWO_PASS=1`.

## 7. Complexity

| Stage | Cost |
|-------|------|
| Pass 1 (registration + position assignment) | O(T·K·E_a), single-threaded, in-L1 |
| Allocation of per-active-expert input buffers | O(E_a) tensor allocs of total size `T·K·H · sizeof(dtype)` |
| Pass 2 (memcpy) | O(T·K) parallel `memcpy`s of `H · sizeof(dtype)` bytes |
| Slice / row_ptrs construction | O(E_a) + O(T·K) |
| Fused backend call | Dominant term — see `zentorch_group_matmul.md` |

E_a (number of active experts) is bounded by `min(E, T·K)` and in practice sits in the tens for typical inference workloads, so the linear scans inside Pass 1 stay cheap.

## 8. Test Plan

Tests for `zentorch_fused_moe` live in `test/unittests/op_tests/test_group_matmul.py` alongside the `zentorch_group_matmul.out` tests, within the `Test_GroupMatmul` class.

### 8.1 Hypothesis strategy

Tests are **Hypothesis-based**, decorated with
`@GroupMatmulTestCase.hypothesis_params_group_matmul_itr(...)` (the same composite strategy
`tensor_group_matmul_strategy` described in
[zentorch_group_matmul.md §7.1](./zentorch_group_matmul.md)). Each example draws randomized
dims plus a reproducible `tensor_seed`; dtype comes from `dtype_list=supported_dtypes`
(`"float32"`, plus `"bfloat16"` when BF16 is supported, plus `"float16"` when AVX-512 FP16
is supported). The DA8W8 tests override `k_list` to satisfy their shape constraints and exclude
`"float16"` and `"float32"` (the dynamic-int8 path quantizes activations from bf16 only).

### 8.2 Test matrix for `zentorch_fused_moe`

| Test | Weights | Post-ops | Notes |
|------|---------|----------|-------|
| `test_fused_moe_pipeline` (Output 2) | bf16/f32 | Full pipeline (silu activation + w2 + MoE reduce) | Single-call fused path |
| `test_int8_w13_and_w2_single_pass` (sub-test 3) | DA8W8 w13 + DA8W8 w2 | No activation + MoE reduce | `k_list = [4, 8]`, `K == K_out == N` |
| `test_int8_w13_and_w2_two_pass` | DA8W8 w13 + DA8W8 w2 | silu activation + w2 + MoE reduce | `k_list = [8, 16]`, `K == K_out`; exercises the `ZENTORCH_TWO_PASS` split path when run with `ZENTORCH_TWO_PASS=1`|

`test_fused_moe_pipeline` verifies two output paths per config: (1) low-level `zentorch_group_matmul.out` with inline MoE weighted-reduce, and (2) high-level `zentorch_fused_moe` (token grouping + full pipeline in a single op call). Both are compared against the same reference.

### 8.3 DA8W8 weight scale integration

When `w13_scales` or `w2_scales` is provided:

1. **Python layer** (`vllm/__init__.py`): At init time (`_patched_init`), loops over `("w13_weight", "w13_scale")` and `("w2_weight", "w2_scale")` pairs. For each, checks `isinstance(w, Int8Tensor)`, extracts `w.scale`, applies a `weight_scales.shape[-1] == 1` check (handles both 2D `[N, 1]` and 3D `[E, N, 1]` scale layouts), squeezes and validates dtype (f32/bf16), stores as `layer.<scale_attr>`, and calls `replace_parameter(layer, weight_attr, w.qdata)` inside the loop to replace the Int8Tensor with its raw int8 data. Forward path fetches via `getattr(layer, "w13_scale", None)`. Same for w2.
2. **FusedMoe.cpp**: Per-active-expert slicing via `w13_scales->select(0, e)` / `w2_scales->select(0, e)`. Passed to `zentorch_group_matmul_out_impl` as `w13_scales` / `w2_scales`.
3. **GroupMatmul.cpp**:
   - **Op1 (w13)**: `params[i].quant_params.wei_scale` populated from `w13_scales[i]`. `src_scale` buffer allocated by caller (kernel fills at runtime).
   - **Op2 (w2)**: `fused_moe.down_scale[i]` populated from `w2_scales[i]`. Op2 inherits `dynamic_quant`, `dtypes.compute`, `src_scale.dims` from `params[i]` — only the weight scale is per-pass. 1D scales `{K_out}` normalized to `{1, K_out}`.

### 8.4 Supported gated activation strings

The C++ `map_activation_to_gated_act` function and the vLLM Python layer (`_SUPPORTED_MOE_ACTIVATIONS`) both use the short-form strings. The mapping is:

| Input string | Enum | Description |
|--------------|------|-------------|
| `"silu"` | `grp_matmul_gated_act_t::silu_and_mul` | SiLU(gate) × up |
| `"gelu"` | `grp_matmul_gated_act_t::gelu_and_mul` | GELU(gate) × up (tanh-approx in fused kernels) |
| `"gelu_tanh"` | `grp_matmul_gated_act_t::gelu_and_mul` | Same as `"gelu"`; vLLM / Gemma-4 MoE name |
| `"swigluoai"` | `grp_matmul_gated_act_t::swiglu_oai_mul` | SwigluOAI variant |

The vLLM forward path normalizes `MoEActivation` enums to their `.value` string before passing to the C++ op.

### 8.5 DA8W4 tests

The DA8W4 regime is covered by `test/unittests/op_tests/test_fused_moe_da8w4.py` (`Test_FusedMoEDA8W4`), which drives `zentorch_fused_moe` with packed s4 weights + per-group scales over hypothesis-drawn shapes. The accuracy cases compare element-wise against a pure-PyTorch DA8W4 reference (`assertEqual` with `atol=rtol=5e-2`):

| Test | Checks |
|------|--------|
| `test_fused_moe_da8w4_accuracy` | Output matches the DA8W4 reference, with and without per-expert w13/w2 biases (`with_bias`) |
| `test_fused_moe_da8w4_requires_bf16` | A float32 activation against packed s4 weights must raise |

### 8.6 Known limitations

| Limitation | Detail |
|------------|--------|
| Mixed bf16-Op1 / DA8W8-Op2 | Unsupported — LowOHA enforces one quant scheme for both passes. Both must be DA8W8 or both bf16. |
| DA8W4 requires bf16 input | f32 activations are rejected for packed-s4 weights in either container (the DA8W4 kernel is s8-dynamic-quant only). |
| Mixed containers across Op1/Op2 | Unsupported — `w13` and `w2` must share one dtype, so an int32-packed `w13` cannot pair with an int8-packed `w2` even though the byte streams match. |
| DA8W4 is symmetric only | Asymmetric (`uint4`) / act-reordered int4 checkpoints are unsupported — use the native vLLM CPU WNA16 MoE path. |

## 9. DA8W4 weights

The DA8W4 regime runs the same MoE FFN block with **symmetric int4 weights** and a **dynamically per-token s8-quantized bf16 activation**: each GEMM computes `s8 x s4 -> bf16` (`compute = s8`) on ZenDNN's AOCL-DLP backend (the s4 weights are widened to s8 inside the kernel). It is the fused-MoE analogue of the linear DA8W4 op — reuse a W4A16 checkpoint (compressed-tensors / GPTQ, symmetric, no act-ordering) but quantize the activation dynamically at inference time.

It needs no DA8W4-specific schema: since it is symmetric-only (no zero-points) and CPU MoE is single-rank (no `expert_map`), every input maps directly onto the standard schema, and the op selects the DA8W4 kernel from `w13`'s packed layout in either container (see §4).

### 9.1 Argument mapping

| DA8W4 concept | `zentorch_fused_moe` slot |
|--------------|---------------------------|
| packed s4 gate+up weight `[E, 2*I, H/8]` int32 or `[E, 2*I, H/2]` int8 | `w13` |
| packed s4 down weight `[E, H, I/8]` int32 or `[E, H, I/2]` int8 | `w2` |
| per-group W13 scale `[E, G, N]` | `w13_scales` |
| per-group W2 scale `[E, G, H]` | `w2_scales` |
| (symmetric — no zero-points) | *(no such arg)* |
| (CPU single-rank — no expert map) | *(no such arg)* |
| `apply_router_weight_on_input` | `skip_weighted` |
| gated activation | `act` |
| optional expert bias | `w13_bias` / `w2_bias` (bf16; `None` when the checkpoint is bias-free) |

### 9.2 Weight packing

No DA8W4-specific packing op is added — expert weights are packed with the existing WOQ repack op and stacked:

```python
torch.ops.zentorch.zentorch_woq_repack_weight(
    unpacked_weight   # int8, [N, K], one signed s4 value per element in [-8, 7]
) -> Tensor           # int32, [N, K/8] (8 nibbles per int32)
```

The per-expert packed tensors are stacked into the `[E, N, K/8]` 3D tensors passed as `w13` / `w2`. The grouped GEMM reads each per-expert view directly as `w13.select(0, e)` → `[N, K/8]` int32 (no `.t()`); `GroupMatmul.cpp`'s DA8W4 branch takes the unpacked `K` / `ldb` from the activation's contraction dim and validates the packed columns against it.

### 9.3 Packed-s4 layout & DA8W4 metadata wiring

The per-expert weight/scale views are passed to the grouped impl **without a `.t()`**, because `GroupMatmul.cpp`'s DA8W4 branch derives the matmul metadata from the packed layout directly:

| Item | Tensor handed in | GroupMatmul derives |
|------|------------------|---------------------|
| W13 weight | `w13.select(0, e)` → `[N, K/8]` int32 or `[N, K/2]` int8 | `N = size(0)`, unpacked `K` = input's last dim (validated: `K / size(1)` must be 8 for int32, 2 for int8), `ldb = K`, `transB = true`, `dtypes.wei = s4` |
| W2 weight | `w2.select(0, e)` → `[H, I/8]` int32 or `[H, I/2]` int8 | `N_down = size(0)`, `ldb_down` = down-proj input dim `K_down` (**not** validated against `size(1)`; the regime comes from `w13`), `dtypes.wei = s4` (inherited by Op2) |
| W13 scale | `w13_scales.select(0, e)` → `[G, N]` | `wei_scale.dims = {G, N}` (per-group, unchanged) |
| W2 scale | `w2_scales.select(0, e)` → `[G2, H]` | `fused.down_scale.dims = {G2, H}` (per-group) |

Notes: the unpacked `K`/`ldb` is the activation's contraction dim in **nibble units**, independent of the container, and `run_dlp`'s `cvt_s4_to_s8` reads the buffer as a transposed s4 nibble stream. Both `w13` and `w2` must be contiguous in either container, since `ldb`/`ldb_down` come from that contraction dim rather than `stride(0)`. The per-group weight scale is passed through as-is (unlike the dynamic-A8W8 path, which normalizes `{N}` to `{1, N}`); the per-token source scale `{M, 1}` is filled by the kernel and broadcast across the `G` groups. Resulting per-expert config: `dtypes = {src: bf16, wei: s4, dst: bf16, compute: s8}`, `dynamic_quant = true`, `is_weights_const = true`.

### 9.4 Kernel / algo selection

DA8W4 (`s4` weight + `dynamic_quant = true`) is **not** in ZenDNN's M-tile (ALGO 2) / N-tile (ALGO 3) regime set (those cover BF16, weight-only S4/U4, and dynamic-A8W8). It runs through the grouped dispatcher's **legacy per-expert path** (ALGO 1) — one AOCL-DLP DA8W4 kernel call per expert GEMM. Deeper cross-op fusion for DA8W4 is a future optimization.

### 9.5 Constraints specific to DA8W4

| Constraint | Condition |
|-----------|-----------|
| `input` dtype | `torch.bfloat16` only — f32 rejected (the C++ op fails fast when `w13` is packed s4, in either container) |
| Container agreement | `w13.dtype == w2.dtype` — both must be the int32 container or both the int8 one |
| Packed `K` divisibility | int32 container: `K % 8 == 0`; int8 container: `K % 2 == 0`. Only `w13`'s `K` (the hidden size) is checked; `w2`'s intermediate size is not. |
| Zero points | Unsupported (symmetric only) — asymmetric `uint4` / act-reordered checkpoints must use the native vLLM CPU WNA16 MoE path |
| Expert map | Not applicable — CPU MoE is single-rank (`supports_expert_map()` is `False`); topk ids index the stacked weights directly |
| Group size (per GEMM) | `K % G == 0` and `(K / G) % 4 == 0` (AOCL sym_quant constraint) |
| Single-token + single-group W2 | The pathological combination `M_e == 1` **and** `inter == group_size` (W2 has a single quant group) trips an AOCL-DLP sym-quant edge case; real models (`inter >> group_size`) never hit it |

## 10. Reference

- Backend operator: [zentorch_group_matmul.md](./zentorch_group_matmul.md) — the parallel group-matmul + MoE post-op chain that this op delegates to (including the DA8W4 metadata contract).
- Source: `src/cpu/cpp/FusedMoE.cpp` (Phase 1 + dispatch) and `src/cpu/cpp/GroupMatmul.cpp` (backend wrapper, incl. the DA8W4 branch).
- vLLM integration: `src/cpu/python/zentorch/vllm/__init__.py` (`_moe_forward_zentorch`, `FusedMoEPatch`, `ZENTORCH_FUSED_MOE=1`) for bf16 / DA8W8; `vllm/model_executor/layers/fused_moe/experts/zentorch_moe.py` (`ZentorchExpertsInt4DA8W4`, `VLLM_CPU_INT4_W4A8`) for DA8W4.
- LowOHA Op2 quantization: [zentorch_group_matmul.md](./zentorch_group_matmul.md) §6.4 "Dynamic quantization — DA8W8 and DA8W4" — documents `fused.down_scale` and the Op2-inherits-Op1 quant contract.
