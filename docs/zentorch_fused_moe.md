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

> **Note:**
> - The op is an **out variant**: `output` is allocated and zero-initialised by the caller and mutated in place. The schema marks it `Tensor(a!)` and the op returns `()`.
> - All shape / dtype / bias validation is performed once in the Python patch layer (`vllm/__init__.py`) — the C++ op trusts its inputs.
> - Only experts that actually receive at least one routed token are materialised in Phase 1 and forwarded to the backend (the **active set** of size E_a ≤ E).

## 2. Motivation

vLLM's stock CPU MoE path picks between two implementations:

- `cpu_fused_moe` — hand-written AMX/VEC MicroGemm kernel with prepacked weights.
- `cpu_fused_moe_torch` — a per-expert `F.linear` loop.

The MicroGemm path requires offline prepacking of weights (and a separate quantisation flow per dtype); the torch loop incurs per-expert kernel-launch overhead, allocates intermediates between W13, activation, W2, and reduce, and reads each expert's output buffer once for the reduce.

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
    output,         # Tensor(a!), [T, H], same dtype as input, ZERO-INITIALIZED
    input,          # Tensor, [T, H], bf16 / f32, contiguous
    w13,            # Tensor, [E, 2*I, H], same dtype as input (or int8)
    w2,             # Tensor, [E, H, I],   same dtype as input (or int8)
    w13_bias,       # Optional[Tensor], [E, 2*I] or None
    w2_bias,        # Optional[Tensor], [E, H]   or None
    topk_weights,   # Tensor, [T, K], f32, contiguous
    topk_id,        # Tensor, [T, K], int32, contiguous, values in [0, E)
    skip_weighted,  # bool; if true, requires K == 1 (router weight pre-applied by caller)
    act,            # str: 'silu' | 'gelu' | 'swigluoai'
    w13_scales=None, # Optional[Tensor], [E, N] or [E, G, N] or None (required for int8 w13)
    w2_scales=None, # Optional[Tensor], [E, K_out] or [E, G, K_out] or None (required for int8 w2)
    *, zentorch_op_name='zentorch::zentorch_fused_moe'
) -> None
```

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output` | Tensor (bf16/f32) | `[T, H]`. **Out parameter**: allocated and zero-initialised by the caller. The op accumulates per-token reduced expert outputs into it. |
| `input` | Tensor (bf16/f32) | `[T, H]` token activations. Contiguous. |
| `w13` | Tensor (bf16/f32/int8) | `[E, 2*I, H]` gate+up projection weights (concatenated). Sliced per-active-expert via `select(0, e)`. When int8, `w13_scales` is required. |
| `w2` | Tensor (bf16/f32/int8) | `[E, H, I]` down-projection weights. Sliced per-active-expert via `select(0, e)`. When int8, `w2_scales` is required. |
| `topk_weights` | Tensor (f32) | `[T, K]` router weights used by the weighted-reduce post-op. |
| `topk_id` | Tensor (int32) | `[T, K]` expert ids, values in `[0, E)`. |
| `skip_weighted` | bool | If `true`, the caller has already multiplied `input` by the (K=1) router weight, so the reduce post-op is fed an all-ones weight vector. |
| `act` | str | Gated activation applied between W13 and W2. One of `'silu'`, `'gelu'`, `'swigluoai'`. Maps to `silu_and_mul`, `gelu_and_mul`, `swiglu_oai_mul` enums internally. |
| `w13_bias` | Tensor? (bf16/f32) | `[E, 2*I]` or `None`. Default `None`. |
| `w2_bias` | Tensor? (bf16/f32) | `[E, H]` or `None`. Default `None`. |
| `w13_scales` | Tensor? (f32) | Per-expert quantization scales for int8 `w13`. Shape `[E, N]` (per-channel) or `[E, G, N]` (per-group). Default `None` (for bf16/f32). |
| `w2_scales` | Tensor? (f32) | Per-expert quantization scales for int8 `w2`. Shape `[E, K_out]` (per-channel) or `[E, G, K_out]` (per-group). Default `None` (for bf16/f32). |
| `zentorch_op_name` | str | Profiling / tracing name. Default `'zentorch::zentorch_fused_moe'`. |

## 5. Input Contract (Constraints)

| Constraint | Condition |
|-----------|-----------|
| `output` | 2D `[T, H]`, same dtype as `input`, **zero-initialised** by caller |
| `input` dtype | `torch.bfloat16` or `torch.float32` |
| `input` layout | 2D `[T, H]`, contiguous |
| `w13`, `w2` dtype | Same dtype as `input`, or `torch.int8` (dynamic int8 quantization) |
| `w13`, `w2` shape | 3D, leading dim `E` |
| `w13_scales` | Required when `w13` is int8. `[E, N]` (per-channel) or `[E, G, N]` (per-group), f32. `None` for bf16/f32. |
| `w2_scales` | Required when `w2` is int8. `[E, K_out]` (per-channel) or `[E, G, K_out]` (per-group), f32. `None` for bf16/f32. |
| `w13_bias`, `w2_bias` | `None` or 2D `[E, …]`, same dtype as `input` |
| `topk_weights` | 2D `[T, K]`, `torch.float32`, contiguous |
| `topk_id` | 2D `[T, K]`, `torch.int32`, contiguous, values in `[0, E)` |
| `skip_weighted` | If `true`, requires `K == 1` |
| `act` | One of `'silu'`, `'gelu'`, `'swigluoai'` |

Validation lives in the Python patch (`_moe_forward_zentorch` / patch install in `src/cpu/python/zentorch/vllm/__init__.py`); the C++ op assumes the contract holds.

### Dynamic int8 quantization support

When `w13` or `w2` weights are `torch.int8`, the corresponding `w13_scales` / `w2_scales` tensors must be provided. The Python patch layer (`_patched_init`) automatically detects int8 weights when `torchao` is installed: it checks `isinstance(w, Int8Tensor)` for each weight attribute, extracts `w.scale` with a `shape[-1] == 1` check (handles both 2D and 3D scale layouts), validates dtype (f32/bf16), and calls `replace_parameter(layer, weight_attr, w.qdata)` to replace the Int8Tensor with its raw int8 data. The forward path (`_moe_forward_zentorch`) fetches scales via `getattr(layer, "w13_scale", None)`.

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

1. Builds size-E_a slice vectors for `w13`, `w2`, `w13_bias`, `w2_bias`, `w13_scales`, `w2_scales` via `select(0, e)` for each `e = active_expert_ids[a]`. Inactive experts contribute nothing to the backend call. Scale slicing only occurs when the corresponding weights are int8 (checked via `torchao` availability and weight dtype).
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
(`"float32"`, plus `"bfloat16"` when BF16 is supported). The int8 tests override `k_list` to satisfy their shape
constraints.

### 8.2 Test matrix for `zentorch_fused_moe`

| Test | Weights | Post-ops | Notes |
|------|---------|----------|-------|
| `test_fused_moe_pipeline` (Output 2) | bf16/f32 | Full pipeline (silu activation + w2 + MoE reduce) | Single-call fused path |
| `test_int8_w13_and_w2_single_pass` (sub-test 3) | int8 w13 + int8 w2 | No activation + MoE reduce | `k_list = [4, 8]`, `K == K_out == N` |
| `test_int8_w13_and_w2_two_pass` | int8 w13 + int8 w2 | silu activation + w2 + MoE reduce | `k_list = [8, 16]`, `K == K_out`; exercises the `ZENTORCH_TWO_PASS` split path when run with `ZENTORCH_TWO_PASS=1`|

`test_fused_moe_pipeline` verifies two output paths per config: (1) low-level `zentorch_group_matmul.out` with inline MoE weighted-reduce, and (2) high-level `zentorch_fused_moe` (token grouping + full pipeline in a single op call). Both are compared against the same reference.

### 8.3 Int8 weight scale integration

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
| `"gelu"` | `grp_matmul_gated_act_t::gelu_and_mul` | GELU(gate) × up |
| `"swigluoai"` | `grp_matmul_gated_act_t::swiglu_oai_mul` | SwigluOAI variant |

The vLLM forward path normalizes `MoEActivation` enums to their `.value` string before passing to the C++ op.

### 8.5 Known limitations

| Limitation | Detail |
|------------|--------|
| Mixed bf16-Op1 / int8-Op2 | Unsupported — LowOHA enforces one quant scheme for both passes. Both must be int8 or both bf16. |

## 9. Reference

- Backend operator: [zentorch_group_matmul.md](./zentorch_group_matmul.md) — the parallel group-matmul + MoE post-op chain that this op delegates to.
- Source: `src/cpu/cpp/FusedMoE.cpp` (Phase 1 + dispatch) and `src/cpu/cpp/GroupMatmul.cpp` (backend wrapper).
- vLLM integration: `src/cpu/python/zentorch/vllm/__init__.py` (`_moe_forward_zentorch`, `FusedMoEPatch`). Enabled by `ZENTORCH_FUSED_MOE=1`.
- LowOHA Op2 quantization: [lowoha_group_matmul_operator.md](./zentorch_group_matmul.md) §"Op2 quantization (optional)" — documents `fused.down_scale` / `fused.down_zp` fields and the inheritance contract.
