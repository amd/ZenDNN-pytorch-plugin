# `chunk_gated_delta_rule_fwd_h`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd_h` |
| C++ symbol | `zentorch::zentorch_gdn_chunk_gated_delta_rule_fwd_h` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/ChunkGatedDeltaRuleFwdH.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_chunk_gated_delta_rule_fwd_h.py` |
| Profiler span | `zentorch::gdn::chunk_gated_delta_rule_fwd_h` |
| Backends | `CPU`, `Meta` |

## What this op does

`chunk_gated_delta_rule_fwd_h` is the fifth of the six FLA sub-kernels
that compose `chunk_gated_delta_rule`. It performs the chunk-recurrent
update of the hidden state `h: (V, K)` per `(b, h)`. The state evolves
chunk by chunk:

```
for i_t in 0 .. NT-1:
    1. snapshot:   h_out[i_t] = h
    2. correct:    v_new[t] = u[t] - sum_k w[t, k] * h[v, k]                     ∀ t in chunk
                   ≡ v_new = u - w @ h.T                                          (BT_eff, V)
    3. save:       v_new_buffer[chunk_t] = v_new                                  (pre-decay)
    4. decay-back: v_decayed[t] = v_new[t] * exp(g_last - g[t])                  ∀ t
                   (per-token "rewind" so all tokens align to the chunk's last decay level)
    5. state-decay: h ← h * exp(g_last)                                           (V, K)
    6. update:     h ← h + v_decayed.T @ k_chunk                                  (V, K) += (V, K)
```

The "value-correction" step (`u - w @ h.T`) un-applies the recursion that
the WY representation `(w, u)` had baked into `u`: at the start of the
chunk, `u` was constructed from `v` (`recompute_w_u_fwd`) using the
*expected* intra-chunk state evolution, but as we now actually advance
the state we need to subtract back the contribution that came from the
pre-chunk state `h`. This is what makes the algorithm *exactly
equivalent* to a naive token-by-token recurrence while only ever doing
chunk-sized matmuls.

`v_new` (the "actual `v`" buffer the next kernel wants) is also saved
out: `chunk_fwd_o` consumes `v_new` (not `u`) when computing the chunk's
output.

## How `chunk_gated_delta_rule_fwd_h` is wired into Qwen3.5 / Qwen3-Next GDN

In `chunk_gated_delta_rule`:

```python
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=k,                            # (1, T_total, Hg, K)
    w=w,                            # (1, T_total, H, K)   ← recompute_w_u_fwd
    u=u,                            # (1, T_total, H, V)   ← recompute_w_u_fwd
    g=g,                            # (1, T_total, H)      ← chunk_local_cumsum
    initial_state=initial_state,    # (N=1, H, V, K) fp32 (or None for cold-start)
    output_final_state=output_final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,    # (N+1,)
    NT_total=int(chunk_offsets[-1]),  # = total chunks across all sequences
)
# h:           (1, NT_total, H, V, K)  ← per-chunk snapshot of state
# v_new:       (1, T_total, H, V)      ← value-corrected u (pre-decay)
# final_state: (N, H, V, K) fp32       ← state at end of each sequence
```

`NT_total` is the total chunk count across all sequences (the last value
of `chunk_offsets`).

## Schema

```
zentorch::gdn_chunk_gated_delta_rule_fwd_h(
    Tensor k, Tensor w, Tensor u, Tensor g, Tensor? initial_state,
    bool output_final_state, int chunk_size, bool save_new_value,
    Tensor cu_seqlens, Tensor chunk_offsets, int NT_total,
    *, str zentorch_op_name='zentorch::gdn_chunk_gated_delta_rule_fwd_h'
) -> (Tensor, Tensor, Tensor)
```

`NT_total` is required: it is the total number of chunks across all
sequences, i.e. `int(chunk_offsets[-1])`. The op cannot derive it from
tensor shapes alone (it depends on the *values* in `chunk_offsets`,
which are unknown to `FakeTensorMode` / `torch.compile`). Passing it
as an explicit int makes the output `h_out` shape known at trace time.

Schema arity is fixed at 3. When `output_final_state=False` or
`save_new_value=False`, the corresponding return is a 0-element tensor
(cleaner than `Tensor?` in the schema).

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `k` | `(B, T, Hg, K)` with `B == 1` | fp32 or bf16 | inner dim unit-stride | none |
| `w` | `(B, T, H, K)` | same as `k` | inner dim unit-stride | none |
| `u` | `(B, T, H, V)` | same as `k` | inner dim unit-stride | none |
| `g` | `(B, T, H)` | floating (typically fp32) | matches | none |
| `initial_state` | `(N, H, V, K)` or empty | fp32 | contiguous | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_offsets` | `(N+1,)` | int32 or int64 | contiguous | none |
| return `h` | `(B, NT_total, H, V, K)` | `k.dtype` | fresh contiguous (`at::zeros`) | — |
| return `v_new` | `(B, T, H, V)` if `save_new_value` else `(0,)` | `u.dtype` | fresh contiguous (`at::zeros` or `at::empty`) | — |
| return `final_state` | `(N, H, V, K)` if `output_final_state` else `(0,)` | **fp32** (always) | fresh contiguous | — |

## Math (per sequence × head)

Let `r = H // Hg` (the GQA ratio) and `bos`, `T_n` be the sequence's
start and length in the flat varlen layout. State stays in fp32 across
chunks within a sequence; cast to `k.dtype` only at snapshot stores.

```python
state = (
    initial_state[n, h_idx].clone().float()  # (V, K)
    if initial_state is not None
    else torch.zeros(V, K, dtype=fp32)
)

NT = ceil(T_n / BT)
for i_t in range(NT):
    chunk_start = bos + i_t * BT
    chunk_end   = min(chunk_start + BT, bos + T_n)
    BT_eff      = chunk_end - chunk_start

    # 1. Snapshot — written before any update for this chunk.
    h_out[boh + i_t, h_idx] = state.to(k.dtype)

    # 2. Value correction.
    w_block = w[chunk_start:chunk_end, h_idx]           # (BT_eff, K)
    u_block = u[chunk_start:chunk_end, h_idx]           # (BT_eff, V)
    v_corr  = u_block - w_block @ state.T                # (BT_eff, V)

    # 3. Save pre-decay v_new.
    v_new[chunk_start:chunk_end, h_idx] = v_corr.to(u.dtype)

    # 4 & 5. Per-token decay on v_corr; bulk decay on state.
    g_block = g[chunk_start:chunk_end, h_idx]           # (BT_eff,)
    g_last  = g_block[-1]                               # scalar
    v_corr  = v_corr * exp(g_last - g_block).unsqueeze(-1)
    state   = state * exp(g_last)

    # 6. State update with the GQA-shared k.
    k_block = k[chunk_start:chunk_end, h_idx // r]      # (BT_eff, K)
    state   = state + v_corr.T @ k_block                 # (V, K)

# Final state.
if output_final_state:
    final_state[n, h_idx] = state                        # fp32
```

The `exp(g_last - g[t])` factor is the kernel's "decay-back" that makes
the chunk-recurrent form *exactly equivalent* to a per-token recurrence;
it pre-rewinds each token's contribution by the residual decay it would
have accumulated up to the chunk's last position.

## Parallelization

- **Outer parallel axis:** `(sequence × head)` pairs. Sequential within;
  parallel across.
- **Within a `(sequence × head)`:** the chunk loop is *strictly serial*
  (state carries forward).
- **Within one chunk:** two `(BT, K) × (V, K)` GEMMs (the value-correction
  and the state update) plus a length-`BT` decay computation. With
  `BT = 64`, both GEMMs are tiny and L1-resident.

## Output buffer initialization

`h`, `v_new`, and `final_state` are zero-initialised so that valid
positions get the actual computed values and invalid positions (rows
outside any sequence in varlen, or rows beyond `T` in the last partial
chunk) are deterministically zero. Downstream `chunk_fwd_o` only reads
valid positions anyway.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `gk` (per-K KDA-style gating) | none | not supported |
| `g = None` (skip decay) | none | not supported (`g` is required) |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_offsets=None` (recompute internally) | none | not supported |
| `chunk_size` not a power of 2 | none | not supported |

## Tolerances

The recurrence walks up to `NT` chunks per sequence (4–50+ for typical
prefill lengths). Each chunk does an fp32 matmul of width `BT = 64`,
applies an `exp`, and a state-update matmul of the same width. The
accumulated bf16 round-trip on `state ← state * exp(g_last)` between
chunks is the dominant precision floor; project default `bf16`
tolerances apply with `max(input_tol, fp32_tol)` slack.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_chunk_gated_delta_rule_fwd_h`. Returns the fixed 3-tuple
`(h, v_new, final_state)`:

- `h` shape `(B, NT_total, H, V, K)` in **`k.dtype`** (matching the
  cpp's `at::zeros(..., k.options())`).
- `v_new` shape `(B, T, H, V)` in **`u.dtype`** if `save_new_value` else
  shape `(0,)`.
- `final_state` shape `(N, H, V, K)` always **fp32** if
  `output_final_state` else shape `(0,)`.

`NT_total` is the explicit `int` argument passed by the caller (equal
to `int(chunk_offsets[-1])`). The meta uses it directly for `h`'s
second dimension because the true value cannot be derived from tensor
shapes alone — it depends on the *values* in `chunk_offsets`, which a
meta function in `FakeTensorMode` cannot read.
