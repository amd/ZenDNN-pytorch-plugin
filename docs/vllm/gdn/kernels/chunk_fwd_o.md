# `chunk_fwd_o`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_chunk_fwd_o` |
| C++ symbol | `zentorch::zentorch_gdn_chunk_fwd_o` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/ChunkFwdO.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/chunk_o.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_chunk_fwd_o.py` |
| Profiler span | `zentorch::gdn::chunk_fwd_o` |
| Backends | `CPU`, `Meta` |

## What this op does

`chunk_fwd_o` is the sixth (and last) of the six FLA sub-kernels that
compose `chunk_gated_delta_rule`. It produces the per-token output `o`
of the chunked GDN attention. Per chunk × `(b, h)`:

```
o[t, :] = scale * (
    (q[t, :] · h_chunk^T)         * exp(g[t])                                  ← history
  + sum_{j ≤ t in chunk} (q[t, :] · k[j, :]) * exp(g[t] − g[j]) * v_new[j, :]    ← in-chunk
)
```

where `h_chunk` is the pre-chunk state snapshot
(`chunk_gated_delta_rule_fwd_h`'s output `h[i_t]`) and `v_new` is the
value-corrected `v` (`chunk_gated_delta_rule_fwd_h`'s `v_new`). The two
contributions are independent terms in the chunked-recurrent attention
sum:

- The **history term** captures the contribution from all tokens in
  *prior* chunks via the running state `h_chunk`.
- The **in-chunk term** captures the standard self-attention within the
  current chunk, with a causal mask (`j ≤ t`, including the diagonal —
  this is *different* from `chunk_scaled_dot_kkt_fwd`'s strict `j < t`
  mask) and a log-decay reweighting `exp(g[t] − g[j])`.

The final `* scale` is the standard `1/sqrt(K)` normalization.

## How `chunk_fwd_o` is wired into Qwen3.5 / Qwen3-Next GDN

In `chunk_gated_delta_rule`:

```python
o = chunk_fwd_o(
    q=q,           # (1, T_total, Hg=num_k_heads, K)
    k=k,           # (1, T_total, Hg, K)
    v=v_new,       # (1, T_total, H=num_v_heads, V) ← chunk_gated_delta_rule_fwd_h output
    h=h,           # (1, NT_total, H, V, K)         ← chunk_gated_delta_rule_fwd_h output
    g=g,           # (1, T_total, H)                ← chunk_local_cumsum output
    scale=scale,   # K^{-1/2}
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
)
# o: (1, T_total, H, V)
```

The output `o` is the chunked GDN attention output that
`chunk_gated_delta_rule` returns to the caller.

## GQA layout

Same convention as the other building blocks:

| Symbol | Meaning | Qwen3.5 / Qwen3-Next |
|---|---|---|
| `Hg` | number of K-heads (head dim of `q` and `k`) | 16 |
| `H` | number of V-heads (head dim of `v` and `o`) | 32 |
| `r = H / Hg` | GQA ratio | 2 |

Each output V-head `h_idx` reads `q[..., h_idx // r, :]` and
`k[..., h_idx // r, :]` (the K-head it belongs to) and uses its own
`v[..., h_idx, :]`, `h[..., h_idx, :, :]`, `g[..., h_idx]`.

## Schema

```
zentorch::gdn_chunk_fwd_o(
    Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, float scale,
    Tensor cu_seqlens, Tensor chunk_offsets, int chunk_size,
    *, str zentorch_op_name='zentorch::gdn_chunk_fwd_o'
) -> Tensor
```

Returns a fresh tensor of shape `(B, T, H, V)` in `v.dtype`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `q` | `(B, T, Hg, K)` with `B == 1` | fp32 or bf16 | inner dim unit-stride | none |
| `k` | `(B, T, Hg, K)` | same as `q` | inner dim unit-stride | none |
| `v` | `(B, T, H, V)` | same as `q` | inner dim unit-stride | none |
| `h` | `(B, NT_total, H, V, K)` | same as `q` | inner dim unit-stride | none |
| `g` | `(B, T, H)` | floating (typically fp32) | matches | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_offsets` | `(N+1,)` | int32 or int64 | contiguous | none |
| return | `(B, T, H, V)` | `v.dtype` | fresh contiguous (`at::zeros`) | — |

## Math (per chunk × (b, h))

Let `c` be a chunk with token range `[chunk_start, chunk_end)`,
`BT_eff = chunk_end − chunk_start ≤ BT`, and `r = H // Hg`.

```python
kh = h_idx // r

q_block = q[chunk_start:chunk_end, kh]         # (BT_eff, K)
k_block = k[chunk_start:chunk_end, kh]         # (BT_eff, K)
v_block = v[chunk_start:chunk_end, h_idx]      # (BT_eff, V)
h_chunk = h[chunk_idx_in_h, h_idx]              # (V, K)
g_block = g[chunk_start:chunk_end, h_idx]      # (BT_eff,)

# History contribution: q · h_chunk^T → per-token V-vector
o_history = q_block @ h_chunk.T                 # (BT_eff, V)
o_history *= exp(g_block).unsqueeze(-1)

# In-chunk attention scores
A = q_block @ k_block.T                          # (BT_eff, BT_eff)
A *= exp(g_block.unsqueeze(-1) - g_block.unsqueeze(0))

# Causal mask, INCLUDING the diagonal (j ≤ t).
A[i, j] = A[i, j] if i >= j else 0

# In-chunk contribution
o_in_chunk = A @ v_block                        # (BT_eff, V)

o[chunk_start:chunk_end, h_idx] = (o_history + o_in_chunk) * scale   # cast to v.dtype
```

The `j == i` (diagonal) inclusion is the only structural difference from
`chunk_scaled_dot_kkt_fwd`, which used `j < i`. Both flow from the
chunked-recurrent algebra; in `chunk_fwd_o` the diagonal entry
contributes `q[t] · k[t] · v[t] * scale` to the output, which is the
"current token attends to itself" term of the standard attention sum.

All matmuls accumulate in fp32; output cast to `v.dtype` on store.

## Output buffer initialization

`o` is zero-initialised so that valid token positions get the actual
computed values and invalid positions (rows outside any sequence in
varlen, or rows beyond `T` in the last partial chunk) are
deterministically zero.

## Parallelization

- **Outer parallel axis:** `(chunk × head)` pairs are independent.
- **Inside a chunk × head:** three GEMMs of sizes `(BT, K) × (V, K)`,
  `(BT, K) × (K, BT)`, and `(BT, BT) × (BT, V)` via `at::matmul`. With
  `BT = 64`, `K = V = 128`, all three fit comfortably in L1.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `g = None` (skip decay) | none | not supported (`g` is required) |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_offsets=None` (recompute internally) | none | not supported |
| `scale=None` (default `K^-0.5`) | none | not supported (caller passes scale explicitly) |

## Tolerances

The two GEMMs and the decay-reweighting accumulate in fp32 over a
`K = 128` reduction inside each chunk. Project default tolerances
suffice at bf16 input (`max(input_tol, fp32_tol)` slack).

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_chunk_fwd_o`.
Returns an empty tensor of shape `(B, T, H, V)` in **`v.dtype`**
(matching the cpp's `at::zeros(..., v.options())`). H is read from `v`
(the V-head count), not from `q` (which has `Hg = num_k_heads`).
Required by Inductor and FakeTensorMode for AOT graph capture; also
used by the `make_fallback` Inductor lowering that routes calls through
the dispatcher to the cpp impl.
