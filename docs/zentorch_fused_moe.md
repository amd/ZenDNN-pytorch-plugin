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
    w13,            # Tensor, [E, 2*I, H], same dtype as input
    w2,             # Tensor, [E, H, I],   same dtype as input
    w13_bias,       # Optional[Tensor], [E, 2*I] or None
    w2_bias,        # Optional[Tensor], [E, H]   or None
    topk_weights,   # Tensor, [T, K], f32, contiguous
    topk_id,        # Tensor, [T, K], int32, contiguous, values in [0, E)
    skip_weighted,  # bool; if true, requires K == 1 (router weight pre-applied by caller)
    act,            # str: 'silu' | 'gelu' | 'swigluoai'
    *, zentorch_op_name='zentorch::zentorch_fused_moe'
) -> None
```

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `output` | Tensor (bf16/f32) | `[T, H]`. **Out parameter**: allocated and zero-initialised by the caller. The op accumulates per-token reduced expert outputs into it. |
| `input` | Tensor (bf16/f32) | `[T, H]` token activations. Contiguous. |
| `w13` | Tensor (bf16/f32) | `[E, 2*I, H]` gate+up projection weights (concatenated). Sliced per-active-expert via `select(0, e)`. |
| `w2` | Tensor (bf16/f32) | `[E, H, I]` down-projection weights. Sliced per-active-expert via `select(0, e)`. |
| `w13_bias` | Tensor? (bf16/f32) | `[E, 2*I]` or `None`. |
| `w2_bias` | Tensor? (bf16/f32) | `[E, H]` or `None`. |
| `topk_weights` | Tensor (f32) | `[T, K]` router weights used by the weighted-reduce post-op. |
| `topk_id` | Tensor (int32) | `[T, K]` expert ids, values in `[0, E)`. |
| `skip_weighted` | bool | If `true`, the caller has already multiplied `input` by the (K=1) router weight, so the reduce post-op is fed an all-ones weight vector. |
| `act` | str | Gated activation applied between W13 and W2. One of `'silu'`, `'gelu'`, `'swigluoai'`. |
| `zentorch_op_name` | str | Profiling / tracing name. Default `'zentorch::zentorch_fused_moe'`. |

## 5. Input Contract (Constraints)

| Constraint | Condition |
|-----------|-----------|
| `output` | 2D `[T, H]`, same dtype as `input`, **zero-initialised** by caller |
| `input` dtype | `torch.bfloat16` or `torch.float32` |
| `input` layout | 2D `[T, H]`, contiguous |
| `w13`, `w2` | 3D, leading dim `E`, same dtype as `input` |
| `w13_bias`, `w2_bias` | `None` or 2D `[E, …]`, same dtype as `input` |
| `topk_weights` | 2D `[T, K]`, `torch.float32`, contiguous |
| `topk_id` | 2D `[T, K]`, `torch.int32`, contiguous, values in `[0, E)` |
| `skip_weighted` | If `true`, requires `K == 1` |
| `act` | One of `'silu'`, `'gelu'`, `'swigluoai'` |

Validation lives in the Python patch (`_moe_forward_zentorch` / patch install in `src/cpu/python/zentorch/vllm/__init__.py`); the C++ op assumes the contract holds.

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

1. Builds size-E_a slice vectors for `w13`, `w2`, `w13_bias`, `w2_bias` via `select(0, e)` for each `e = active_expert_ids[a]`. Inactive experts contribute nothing to the backend call.
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
  ├─ Build size-E_a slices: w13_slices, w2_slices, w13_bias_slices, w2_bias_slices
  ├─ Build row_ptrs[T*K]: pointers into grouped_inputs[a][pos]
  ├─ If skip_weighted: effective_topk_weights = ones_like(topk_weights)
  └─ zentorch_group_matmul_out_impl(
        gemm_outputs={},                    # backend allocates W13 outputs
        inputs=grouped_inputs,              # size E_a
        weights=w13_slices,
        bias=w13_bias_slices,
        activation=act,                     # gated act post-op
        w2_weights=w2_weight_slices,        # fused W2 post-op
        w2_bias=w2_bias_slices,
        w2_outputs=grouped_inputs,          # aliased
        moe_output=output,                  # weighted reduce target
        topk_weights=effective_topk_weights,
        row_ptrs=row_ptrs,
        ...)
        # Backend runs W13 -> gated_act -> W2 -> weighted_reduce in one call
```

## 7. Complexity

| Stage | Cost |
|-------|------|
| Pass 1 (registration + position assignment) | O(T·K·E_a), single-threaded, in-L1 |
| Allocation of per-active-expert input buffers | O(E_a) tensor allocs of total size `T·K·H · sizeof(dtype)` |
| Pass 2 (memcpy) | O(T·K) parallel `memcpy`s of `H · sizeof(dtype)` bytes |
| Slice / row_ptrs construction | O(E_a) + O(T·K) |
| Fused backend call | Dominant term — see `zentorch_group_matmul.md` |

E_a (number of active experts) is bounded by `min(E, T·K)` and in practice sits in the tens for typical inference workloads, so the linear scans inside Pass 1 stay cheap.

## 8. Reference

- Backend operator: [zentorch_group_matmul.md](./zentorch_group_matmul.md) — the parallel group-matmul + MoE post-op chain that this op delegates to.
- Source: `src/cpu/cpp/FusedMoE.cpp` (Phase 1 + dispatch) and `src/cpu/cpp/GroupMatmul.cpp` (backend wrapper).
- vLLM integration: `src/cpu/python/zentorch/vllm/__init__.py` (`_moe_forward_zentorch`, `FusedMoEPatch`). Enabled by `ZENTORCH_FUSED_MOE=1`.
