# `recompute_w_u_fwd`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_recompute_w_u_fwd` |
| C++ symbol | `zentorch::zentorch_gdn_recompute_w_u_fwd` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/RecomputeWUFwd.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/wy_fast.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_recompute_w_u_fwd.py` |
| Profiler span | `zentorch::gdn::recompute_w_u_fwd` |
| Backends | `CPU`, `Meta` |

## What this op does

`recompute_w_u_fwd` is the fourth of the six FLA sub-kernels that
compose `chunk_gated_delta_rule`. It uses the WY-representation matrix
`A` (output of `solve_tril`) together with `k`, `v`, `β` and the
chunk-local cumulative gate `g` to compute:

```
u[t, :] = sum_j A[t, j] * (β[j] * v[j, :])                                ← shape (BT_eff, V)
w[t, :] = sum_j A[t, j] * (β[j] * exp(g[j]) * k[j, :])                    ← shape (BT_eff, K)
```

per chunk × `(b, h)`. The output `u` re-projects `v` through the WY
matrix gated by `β`, while `w` is the same thing for `k` but
additionally scaled by `exp(g)` (the per-token chunk-cumulative
log-decay). Together `(w, u)` are the inputs to the chunk-recurrent
state-update kernel `chunk_gated_delta_rule_fwd_h`.

## How `recompute_w_u_fwd` is wired into Qwen3.5 / Qwen3-Next GDN

In `chunk_gated_delta_rule`:

```python
w, u = recompute_w_u_fwd(
    k=k,                            # (1, T_total, Hg=num_k_heads, K)
    v=v,                            # (1, T_total, H=num_v_heads, V)
    beta=beta,                      # (1, T_total, H)
    A=A,                            # (1, T_total, H, BT) ← from solve_tril
    g_cumsum=g,                     # (1, T_total, H)     ← from chunk_local_cumsum
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
# w: (1, T_total, H, K)
# u: (1, T_total, H, V)
```

Both outputs use the V-head dim `H` (not the K-head dim `Hg`); `k` is
GQA-expanded internally (each V-head pulls `k` from its corresponding
K-head `h // (H // Hg)`).

## Schema

```
zentorch::gdn_recompute_w_u_fwd(
    Tensor k, Tensor v, Tensor beta, Tensor g_cumsum, Tensor A,
    Tensor cu_seqlens, Tensor chunk_indices,
    *, str zentorch_op_name='zentorch::gdn_recompute_w_u_fwd'
) -> (Tensor, Tensor)
```

Returns `(w, u)` where `w` is in `k.dtype` and `u` is in `v.dtype`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `k` | `(B, T, Hg, K)` with `B == 1` | fp32 or bf16 | inner dim unit-stride | none |
| `v` | `(B, T, H, V)` with `B == 1` | same as `k` | inner dim unit-stride | none |
| `beta` | `(B, T, H)` | floating (typically fp32) | matches `v` outer dims | none |
| `g_cumsum` | `(B, T, H)` | floating (typically fp32) | matches `beta` shape | none |
| `A` | `(B, T, H, BT)` | floating | inner dim unit-stride | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_indices` | `(NT, 2)` | int32 | contiguous | none |
| return `w` | `(B, T, H, K)` | `k.dtype` | fresh contiguous (`at::zeros`) | — |
| return `u` | `(B, T, H, V)` | `v.dtype` | fresh contiguous (`at::zeros`) | — |

## Math (per chunk × (b, h))

Let `c` be a chunk with token range `[chunk_start, chunk_end)`,
`BT_eff = chunk_end - chunk_start <= BT`, and `r = H // Hg` the GQA
ratio.

```python
A_block    = A[b, chunk_start:chunk_end, h, :BT_eff]            # (BT_eff, BT_eff)
v_block    = v[b, chunk_start:chunk_end, h]                     # (BT_eff, V)
k_block    = k[b, chunk_start:chunk_end, h // r]                # (BT_eff, K)
beta_block = beta[b, chunk_start:chunk_end, h]                  # (BT_eff,)
g_block    = g_cumsum[b, chunk_start:chunk_end, h]              # (BT_eff,)

# u branch
v_beta = v_block * beta_block.unsqueeze(-1)                      # (BT_eff, V)
u_block = A_block @ v_beta                                       # (BT_eff, V)

# w branch
k_beta_eg = k_block * beta_block.unsqueeze(-1) * exp(g_block).unsqueeze(-1)
w_block = A_block @ k_beta_eg                                    # (BT_eff, K)

w[b, chunk_start:chunk_end, h, :] = w_block       # cast to k.dtype
u[b, chunk_start:chunk_end, h, :] = u_block       # cast to v.dtype
```

All matmuls accumulate in fp32 (`tl.dot` accumulator); inputs upcast
internally; outputs cast back to `k.dtype`/`v.dtype` on store.

## Output buffer initialization

Invalid token positions (rows outside any sequence in varlen, or rows
beyond `T` in the last partial chunk) are left **uninitialised** by the
Triton kernel. The cpp op uses `at::zeros(...)` so the output is
deterministically `0` at those positions; the math at those positions
is logically zero anyway.

## Parallelization

- **Outer parallel axis:** `(chunk × head)` pairs. Independent.
- **Inside a chunk × head:** two small batched GEMMs of shapes
  `(BT_eff, BT_eff) × (BT_eff, V)` and `(BT_eff, BT_eff) × (BT_eff, K)`
  via `at::matmul`, plus the `β · exp(g) · k` pre-scale. With
  `BT = 64` and `K = V = 128`, both fit comfortably in L1.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_indices=None` (recompute internally) | none | not supported |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply, with `max(input_tol, fp32_tol)` slack since the matmul
accumulates in fp32. The inner reduction is `BT_eff = 64` long; well
within bf16 precision for the typical `A` magnitudes (which are bounded
since `A = (I + L)^{-1}` for nilpotent `L`).

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_recompute_w_u_fwd`.
Returns `(w, u)` with shapes `(B, T, H, K)` and `(B, T, H, V)` in
**`k.dtype`** and **`v.dtype`** respectively (matching the cpp op which
uses `at::zeros(..., k.options())` / `at::zeros(..., v.options())`).
Required by Inductor and FakeTensorMode for AOT graph capture; also
used by the `make_fallback` Inductor lowering that routes calls through
the dispatcher to the cpp impl.
