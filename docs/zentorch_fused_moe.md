(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch_fused_moe — Fused Mixture-of-Experts FFN Block (Out Variant)

## 1. Overview

`zentorch_fused_moe` is a single-call operator that executes the full Mixture-of-Experts (MoE) FFN block:

```
input [T, H]
   ├─ unique-token s8 quant (DA8W8 fused path; ZENTORCH_MOE_PREQUANT, default on)
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
| **DA8W8** | `torch.int8`, full-width `[E, N, K]` | per-channel or per-group | unique-token s8, then grouping (fused path; `ZENTORCH_MOE_PREQUANT=0` groups bf16 instead) |
| **DA8W4** | packed s4: `torch.int32` `[E, N, K/8]` or `torch.int8` `[E, N, K/2]` | per-group `[E, G, N]` | bf16 grouping, ZenDNN quantizes per expert row (`dynamic_quant=true`). Unique-token is DA8W8-only |

Both DA8W4 containers hold the same s4 nibble stream, so they are interchangeable. `torch.int8` therefore serves both quantized regimes and is disambiguated by its last dim: full width is DA8W8, half width is packed s4.

The DA8W4 regime is symmetric-only (no zero-points) and needs no `expert_map` (CPU MoE is single-rank), so it maps 1:1 onto this schema: packed int4 weights go in `w13` / `w2` and per-group weight scales in `w13_scales` / `w2_scales`. See [§9 "DA8W4 weights"](#9-da8w4-weights).

> **Note:**
> - The op is an **out variant**: `output` is allocated by the caller and mutated in place. The schema marks it `Tensor(a!)` and the op returns `()`. The caller does **not** need to zero-initialise it — the weighted-reduce post-op writes every `[T, H]` element (the `k = 0` slot initialises, `k > 0` accumulate).
> - Shape / dtype / bias validation is performed once by the producing Python layer — the `CPUFusedMOE` patch (`vllm/__init__.py`) for bf16 / DA8W8, or the DA8W4 experts backend (`vllm/model_executor/layers/fused_moe/experts/zentorch_moe.py`) for DA8W4. The C++ op still fail-fasts on `E_a > 1`, `topk_id` values in `[0, E)`, unique-token preconditions (contiguous bf16, AVX-512, `T>0`), and grouped-buffer contiguity. DA8W8/DA8W4 activation dtype is enforced in `GroupMatmul.cpp`.
> - GEMM work is the **active set** of size E_a ≤ E (experts that received at least one token). `w13` / `w2` lists passed to GroupMatmul are still sized **E** (active prefix + inactive prepack tail); bias, weight-scale, and `src_scales` lists are sized E_a.

## 2. Motivation

vLLM's stock CPU MoE path picks between two implementations:

- `cpu_fused_moe` — hand-written AMX/VEC MicroGemm kernel with prepacked weights.
- `cpu_fused_moe_torch` — a per-expert `F.linear` loop.

The MicroGemm path requires offline prepacking of weights (and a separate quantization flow per dtype); the torch loop incurs per-expert kernel-launch overhead, allocates intermediates between W13, activation, W2, and reduce, and reads each expert's output buffer once for the reduce.

`zentorch_fused_moe` collapses the whole MoE FFN block into a single backend call:

- **Token-expert grouping in C++** with no atomics — positions for each routed `(t, k)` pair are pre-assigned during a cheap single-threaded sweep (`expert_to_active[E]` lookup), then `torch::stable::parallel_for` copies unique rows in Pass 2.
- **Active-set GEMMs + prepack tail** — only E_a experts participate in this call's GEMMs. Inactive `w13` / `w2` slices are still appended so ZenDNN's weight-cache warmer sees every expert.
- **Scratchpad grouping buffers** — default `ZENTORCH_USE_SCRATCHPAD=1` packs per-expert `[M_e, H]` (and `[M_e, 1]` scales on the unique-token path) into a process-lifetime aligned block; set `0` for `new_empty` per expert.
- **Buffer aliasing** — the per-expert input buffers are reused as W2 output buffers, since W13 has already consumed them by the time W2 writes. No second allocation per active expert.
- **Fused post-op chain** — W13 → gated activation → W2 → router-weighted reduce executes inside one `group_matmul_direct` call.
- **Standard `[E, ...]` weight layout** — no offline prepack of the stacked tensors; per-expert views are cached (`ExpertSliceCache`) and consumed in the layout vLLM stores them in.

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

Validation of stacked 3D weights / biases / scales lives in the producing Python layer — `_moe_forward_zentorch` / patch install in `src/cpu/python/zentorch/vllm/__init__.py` for bf16/DA8W8, or the DA8W4 experts backend for DA8W4. The C++ fused-MoE op still checks `E_a > 1`, `topk_id[i] ∈ [0, E)`, unique-token preconditions, and grouped-buffer contiguity. Packed-s4 vs DA8W8 classification, `w13`/`w2` dtype agreement, and “DA8W4/DA8W8 require bf16 or int8 activations” are enforced in `zentorch_group_matmul_out_impl`.

### Dynamic A8W8 quantization support

When `w13` or `w2` weights are `torch.int8`, the corresponding `w13_scales` / `w2_scales` tensors must be provided. The Python patch layer (`_patched_init`) automatically detects DA8W8 weights when `torchao` is installed: it checks `isinstance(w, Int8Tensor)` for each weight attribute, extracts `w.scale` with a `shape[-1] == 1` check (handles both 2D and 3D scale layouts), validates dtype (f32/bf16), and calls `replace_parameter(layer, weight_attr, w.qdata)` to replace the Int8Tensor with its raw int8 data. The forward path (`_moe_forward_zentorch`) fetches scales via `getattr(layer, "w13_scale", None)`.

## 6. Design

The op runs in two phases: **Phase 1 — Token-Expert Grouping** (custom C++ in `FusedMoE.cpp`), and **Phase 2 — Fused GEMM chain** (delegated to `zentorch_group_matmul_out_impl`).

### 6.1 Phase 1 — Token-Expert Grouping

For each routed pair `(t, k)`, expert `e = topk_id[t][k]` must receive a copy of `input[t]` in its per-expert input buffer. This is done with two sub-passes:

**Pass 1 — single-threaded bookkeeping (O(T·K)):**

1. Walk the T·K `(t, k)` pairs in flat order, `i = t*K + k`. `topk_id[i]` must be in `[0, E)`.
2. On first encounter of an expert `e`, append it to `active_expert_ids` (first-seen order is unchanged) and set `expert_to_active[e] = a`. Already-seen experts resolve with that size-E table (`-1` = unseen), not a linear scan of the growing active list. Reserve each expert's source-token list to `T` on first encounter.
3. For each `i`, record `topk_to_expert_row[i] = (a, pos)` where `a` is the active slot for `e` and `pos = tokens_per_active[a]++` is the deterministic row this pair will occupy in expert `a`'s eventual input buffer. Append source token `t` to `source_tokens_per_active[a]`. **No atomics are needed in Pass 2** because positions are pre-assigned here.

`active_expert_ids` remains the ordered active set handed to later phases. `expert_to_active` exists only for the T·K walk and is discarded when mapping returns.

**Allocation:**

After Pass 1, allocate `grouped_inputs[a]` as `[tokens_per_active[a], H]` in the **activation dtype** (bf16 for unique-token). The vector is size E_a. Default `ZENTORCH_USE_SCRATCHPAD=1` places every `[M_e, H]` region (and `[M_e, 1]` scale region when unique-token scales are present) in one 64-byte-aligned process-lifetime block and exposes them as `from_blob` tensors. `ZENTORCH_USE_SCRATCHPAD=0` uses `new_empty` per expert.

**Pass 2 — parallel data movement (`torch::stable::parallel_for` over E_a):**

Each worker owns one destination buffer and walks `source_tokens_per_active[a]`. Float path: `memcpy` row `t` of `input` into row `pos` of `grouped_inputs[a]`. Unique-token (DA8W8/DA8W4 fused path): first quantize unique `[T, H]` bf16 tokens out-of-place to int8 + f32 `[T, 1]` scales via `dynamic_per_token_quant_bf16_s8_native`, then memcpy **H int8 bytes** per row into the leading `M_e*H` bytes of the bf16 grouping buffer (plus a scale broadcast into `grouped_src_scales`). Grouping never quantizes. Dest-base pointer setup is serial; the row copies use `parallel_for` with `grain_size=1`. No locks, no atomics — Pass 1 guarantees every `(a, pos)` is unique.

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

1. Builds **size-E** `w13` / `w2` slice lists (active experts in `active_expert_ids` order at `[0, E_a)`, then inactive experts in original `[0, E)` order) so ZenDNN's prepack warmer sees every expert. Bias and weight-scale lists stay **size E_a**. Slices come from `ExpertSliceCache` (process-lifetime per-tensor `select(0, e)` cache, keyed by `data_ptr()`; tests flush via `zentorch_flush_moe_weight_cache`). Scale slicing runs whenever `w13_scales` / `w2_scales` are defined (DA8W8 and DA8W4) — there is no torchao check in C++.
2. Builds `row_ptrs[T·K]`: for each `i`, `row_ptrs[i] = &grouped_inputs[a].data[pos * row_bytes]` with `row_bytes` from the **bf16/fp** grouping buffers. W2 writes per-expert outputs into those same buffers (float: src-reuse; unique-token: `gemm_outputs` / `dst_down` after W13 has consumed the packed int8 prefix), so `row_ptrs[i]` is where the `(t, k)`-th expert result lives when weighted-reduce runs.
3. If `skip_weighted` is set, substitutes an all-ones weight vector (router weights have been pre-applied to `input` by the caller).
4. Calls `zentorch_group_matmul_out_impl` once:
   - **float:** `gemm_outputs={}` (backend allocates W13 internally), `inputs=grouped_inputs` (W2 src-reuse), `src_scales=[]`.
   - **unique-token:** `inputs` = int8 `from_blob` views of `grouped_inputs`, filled `src_scales` from `grouped_src_scales`, `gemm_outputs=grouped_inputs` (bf16 W2 dests).

### 6.4 Buffer aliasing — why W2 dests can reuse `grouped_inputs`

**Float path** (`gemm_outputs={}`): ZenDNN src-reuse. Within `group_matmul_direct`'s fused chain:

```
W13 reads grouped_inputs[a]   ──►   W13 outputs (internal buffer)
                                    ──► gated act ──► W2 inputs (internal buffer)
                                                       ──► W2 writes grouped_inputs[a]
                                                              ──► weighted reduce reads it
```

W13 has finished reading `grouped_inputs[a]` before W2 starts writing it.

**Unique-token path:** `grouped_inputs[a]` is a bf16 `[M_e, H]` allocation. Packed int8 occupies only the leading `M_e*H` bytes (W13 reads an int8 view). Op1 dest is library-internal; W2 writes the full bf16 buffer via `gemm_outputs` / `dst_down`. Int8 payload is consumed before that write. `row_ptrs` target the bf16 rows.

### 6.5 Execution flow

```
zentorch_fused_moe()
  ├─ If DA8W8, ZENTORCH_TWO_PASS is off, and ZENTORCH_MOE_PREQUANT is on (default):
  │     unique-token dynamic_per_token_quant_bf16_s8_native → int8 [T,H] + f32 [T,1]
  │     (DA8W4 skips this and groups bf16)
  ├─ build_token_expert_mapping(input, topk_id, E, optional unique-token scales):
  │     ├─ Pass 1 (single-threaded O(T·K)):
  │     │     ├─ expert_to_active[E] lookup; first-seen order into active_expert_ids
  │     │     └─ Assign deterministic (active_idx, pos) per (t, k) pair
  │     ├─ Allocate grouped_inputs[a] [M_a, H] in activation dtype (scratchpad
  │     │     from_blob by default; new_empty if ZENTORCH_USE_SCRATCHPAD=0)
  │     └─ scatter_tokens_to_experts: parallel_for over E_a (bf16 rows, or packed
  │           int8 + scale broadcast)
  ├─ Unique-token: wrap grouped_inputs as int8 from_blob views; src_scale_slices
  │     from grouped_src_scales
  ├─ Build size-E w13/w2 slices (active prefix + inactive prepack tail) and
  │     size-E_a bias / weight-scale slices (ExpertSliceCache)
  ├─ Build row_ptrs[T*K]: pointers into grouped_inputs[a][pos] (bf16 W2 dests)
  ├─ If skip_weighted: effective_topk_weights = ones_like(topk_weights)
  └─ zentorch_group_matmul_out_impl(
        gemm_outputs={} or grouped_inputs,       # empty: float src-reuse; unique-token: dst_down
        inputs=grouped_inputs or int8 views,     # unique-token: kChar view of packed prefix
        w13_weights=w13_slices,                  # sized E
        w2_weights=w2_weight_slices,             # sized E; fused W2 post-op
        moe_output=output,
        topk_weights=effective_topk_weights,
        row_ptrs=row_ptrs,
        activation=act,
        w13_bias=w13_bias_slices,                # sized E_a
        w2_bias=w2_bias_slices,
        w13_scales=w13_scale_slices,
        w2_scales=w2_scale_slices,
        src_scales=[] or grouped unique-token scales,
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
single-call fused path in 6.5 is used. Unique-token pre-quant is **fused-path
only**: `use_prequant` is false whenever two-pass is set, so this split keeps
the original bf16 grouping and lets each GEMM dynamically quantize inside
ZenDNN. This split path is exercised by `test_int8_w13_and_w2_two_pass` when
run with `ZENTORCH_TWO_PASS=1`.

### 6.7 Unique-token kill switch (`ZENTORCH_MOE_PREQUANT`)

On the fused **DA8W8** path, unique-token s8 quant is on by default (`ZENTORCH_MOE_PREQUANT=1`).
**DA8W4 never unique-token quantizes**: ZenDNN `group_matmul_fused_moe` `op1_internal` allows mixed `src=s8` / `dst=bf16` only when `wei=s8`. Packed s4 (`wei=s4`) keeps bf16 grouping and `dynamic_quant=true`.
Set `ZENTORCH_MOE_PREQUANT=0` to keep **one** fused `group_matmul_direct` call but
group bf16 tokens on DA8W8 too (`dynamic_quant=true`).
`ZENTORCH_TWO_PASS=1` still forces unique-token off (two plugin GEMMs).

```
ZENTORCH_MOE_PREQUANT=1 (default) + TWO_PASS unset + DA8W8
  → unique-token quant, then int8 grouping, one fused call
ZENTORCH_MOE_PREQUANT=1 (default) + TWO_PASS unset + DA8W4
  → bf16 grouping, ZenDNN quantizes per expert row, one fused call
ZENTORCH_MOE_PREQUANT=0            + TWO_PASS unset
  → bf16 grouping for DA8W8 and DA8W4, one fused call
ZENTORCH_TWO_PASS=1                (PREQUANT ignored)
  → bf16 grouping, two plugin GEMMs
```

### 6.8 Environment variables

Read once at process start via `EnvReader` (`src/cpu/cpp/EnvReader.hpp`). Values other than `0`/`1` fall back to the default.

| Variable | Default | Effect |
|----------|---------|--------|
| `ZENTORCH_MOE_PREQUANT` | `1` | Unique-token s8 quant on fused **DA8W8** only. DA8W4 always groups bf16. `0` groups bf16 on DA8W8 too. |
| `ZENTORCH_TWO_PASS` | `0` | Split W13+act and W2+reduce into two `group_matmul_direct` calls. Forces unique-token off. |
| `ZENTORCH_USE_SCRATCHPAD` | `1` | Pack grouped `[M_e, H]` (and unique-token `[M_e, 1]` scales) into a reused aligned block. `0` = `new_empty` per expert. |
| `ZENTORCH_ENABLE_CHECKS` | `0` | GroupMatmul **weight-scale** validators only (`validate_weight_scales`, DA8W4 per-group scale shape). Shape/dtype/list-size checks and int8 `src_scales` checks still run. FusedMoE does not read this flag. |

## 7. Complexity

| Stage | Cost |
|-------|------|
| Pass 1 (registration + position assignment) | O(T·K) plus a size-E `expert_to_active` table, single-threaded |
| Allocation of per-active-expert input buffers | O(E_a) `from_blob` (scratchpad) or `new_empty`; total size `T·K·H · sizeof(dtype)` plus unique-token scales |
| Pass 2 (memcpy) | O(T·K) parallel `memcpy`s; unique-token copies `H` int8 bytes per row, float copies `H · sizeof(dtype)` |
| Slice / row_ptrs construction | O(E) weight-list fill + O(T·K) row_ptrs; slice `select` amortized after `ExpertSliceCache` warms |
| Fused backend call | Dominant term — see `zentorch_group_matmul.md` |

E_a is bounded by `min(E, T·K)`. Pass 1 does not scan the growing active list.

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
| `test_int8_w13_and_w2_single_pass` (sub-test 3) | DA8W8 w13 + DA8W8 w2 | silu + MoE reduce | Unique-token s8 + one fused `group_matmul_direct`; `k_list = [4, 8]` |
| `test_int8_w13_and_w2_two_pass` | DA8W8 w13 + DA8W8 w2 | silu activation + w2 + MoE reduce | `k_list = [8, 16]`, `K == K_out`; requires `ZENTORCH_TWO_PASS=1` (bf16 grouping, two plugin GEMMs; unique-token is off) |

`test_fused_moe_pipeline` verifies two output paths per config: (1) low-level `zentorch_group_matmul.out` with inline MoE weighted-reduce, and (2) high-level `zentorch_fused_moe` (token grouping + full pipeline in a single op call). Both are compared against the same reference.

### 8.3 DA8W8 weight scale integration

When `w13_scales` or `w2_scales` is provided:

1. **Python layer** (`vllm/__init__.py`): At init time (`_patched_init`), loops over `("w13_weight", "w13_scale")` and `("w2_weight", "w2_scale")` pairs. For each, checks `isinstance(w, Int8Tensor)`, extracts `w.scale`, applies a `weight_scales.shape[-1] == 1` check (handles both 2D `[N, 1]` and 3D `[E, N, 1]` scale layouts), squeezes and validates dtype (f32/bf16), stores as `layer.<scale_attr>`, and calls `replace_parameter(layer, weight_attr, w.qdata)` inside the loop to replace the Int8Tensor with its raw int8 data. Forward path fetches via `getattr(layer, "w13_scale", None)`. Same for w2.
2. **FusedMoe.cpp**: Per-active-expert slicing from `ExpertSliceCache` (`w13_scales.select(0, e)` / `w2_scales.select(0, e)` on first encounter of that tensor). Passed to `zentorch_group_matmul_out_impl` as `w13_scales` / `w2_scales`. Unique-token `src_scales` are the grouped `[M_e, 1]` f32 buffers, not a schema arg on `zentorch_fused_moe`.
3. **GroupMatmul.cpp**:
   - **Op1 (w13), unique-token:** int8 `inputs` + filled f32 `src_scales` from `dynamic_per_token_quant_bf16_s8_native` (converted to `wei_scale` dtype if they differ). `dynamic_quant = false`. Fused w2: caller bf16 `gemm_outputs` as `dst_down`.
   - **Op1 (w13), bf16/two-pass:** `src_scale` buffer allocated by the wrapper; `dynamic_quant = true`; ZenDNN fills the scales.
   - **Op2 (w2):** `fused_moe.down_scale[i]` from `w2_scales[i]`. Op2 inherits `dtypes.compute` / `dtypes.wei`. Unique-token Op1 (`src=s8`, `dynamic_quant=false`) makes ZenDNN re-enable Op2 `dynamic_quant` (W2 src is bf16 post-act). 1D scales `{K_out}` normalized to `{1, K_out}`.

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

Notes: the unpacked `K`/`ldb` is the activation's contraction dim in **nibble units**, independent of the container, and `run_dlp`'s `cvt_s4_to_s8` reads the buffer as a transposed s4 nibble stream. Both `w13` and `w2` must be contiguous in either container, since `ldb`/`ldb_down` come from that contraction dim rather than `stride(0)`. The per-group weight scale is passed through as-is (unlike the dynamic-A8W8 path, which normalizes `{N}` to `{1, N}`).

**DA8W4 fused path** always groups bf16 and uses Op1 `dynamic_quant=true` (kernel-filled `{M, 1}` scales, broadcast across `G`). Unique-token (`src=s8`, `dynamic_quant=false`) is DA8W8-only. **Two-pass:** both GEMMs are `src=bf16`, `dynamic_quant=true`.

### 9.4 Kernel / algo selection

DA8W4 (`s4` weight) is **not** in ZenDNN's M-tile (ALGO 2) / N-tile (ALGO 3) regime set (those cover BF16, weight-only S4/U4, and dynamic-A8W8). It runs through the grouped dispatcher's **legacy per-expert path** (ALGO 1) — one AOCL-DLP DA8W4 kernel call per expert GEMM — with Op1 `dynamic_quant=true`. Deeper cross-op fusion for DA8W4 is a future optimization.

### 9.5 Constraints specific to DA8W4

| Constraint | Condition |
|-----------|-----------|
| `input` dtype | `torch.bfloat16` only — f32/fp16 rejected in `GroupMatmul.cpp` when weights are packed s4 (unique-token also requires contiguous bf16 in `quantize_unique_tokens_s8`) |
| Container agreement | `w13.dtype == w2.dtype` — both must be the int32 container or both the int8 one |
| Packed `K` divisibility | int32 container: `K % 8 == 0`; int8 container: `K % 2 == 0`. Only `w13`'s `K` (the hidden size) is checked; `w2`'s intermediate size is not. |
| Zero points | Unsupported (symmetric only) — asymmetric `uint4` / act-reordered checkpoints must use the native vLLM CPU WNA16 MoE path |
| Expert map | Not applicable — CPU MoE is single-rank (`supports_expert_map()` is `False`); topk ids index the stacked weights directly |
| Group size (per GEMM) | `K % G == 0` and `(K / G) % 4 == 0` (AOCL sym_quant constraint) |
| Single-token + single-group W2 | The pathological combination `M_e == 1` **and** `inter == group_size` (W2 has a single quant group) trips an AOCL-DLP sym-quant edge case; real models (`inter >> group_size`) never hit it |

## 10. Reference

- Backend operator: [zentorch_group_matmul.md](./zentorch_group_matmul.md) — the parallel group-matmul + MoE post-op chain that this op delegates to (including the DA8W4 metadata contract).
- Source: `src/cpu/cpp/FusedMoE.cpp` (Phase 1 + dispatch) and `src/cpu/cpp/GroupMatmul.cpp` (backend wrapper, incl. the DA8W4 branch). Meta kernels: `_meta_registrations.py` (`zentorch_fused_moe` is an AOTI shim, not `make_fallback`; `zentorch_group_matmul.out` is `make_fallback`).
- Env: `src/cpu/cpp/EnvReader.hpp` (`ZENTORCH_MOE_PREQUANT`, `ZENTORCH_TWO_PASS`, `ZENTORCH_USE_SCRATCHPAD`, `ZENTORCH_ENABLE_CHECKS`).
- Tests: `test/unittests/op_tests/test_group_matmul.py`; flush hook `zentorch_flush_moe_weight_cache`.
- vLLM integration: `src/cpu/python/zentorch/vllm/__init__.py` (`_moe_forward_zentorch`, `FusedMoEPatch`, `ZENTORCH_FUSED_MOE=1`) for bf16 / DA8W8; `vllm/model_executor/layers/fused_moe/experts/zentorch_moe.py` (`ZentorchExpertsInt4DA8W4`, `VLLM_CPU_INT4_W4A8`) for DA8W4.
- LowOHA Op2 quantization: [zentorch_group_matmul.md](./zentorch_group_matmul.md) §6.4 "Dynamic quantization — DA8W8 and DA8W4" — documents `fused.down_scale` and the Op2-inherits-Op1 quant contract.
