# `chunk_scaled_dot_kkt_fwd`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_chunk_scaled_dot_kkt_fwd` |
| C++ symbol | `zentorch::zentorch_gdn_chunk_scaled_dot_kkt_fwd` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/ChunkScaledDotKktFwd.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_chunk_scaled_dot_kkt_fwd.py` |
| Profiler span | `zentorch::gdn::chunk_scaled_dot_kkt_fwd` |
| Backends | `CPU`, `Meta` |

## What this op does

`chunk_scaled_dot_kkt_fwd` is the second of the six FLA sub-kernels that
compose `chunk_gated_delta_rule`. It computes the lower-triangular
Gram-like matrix that the WY representation builds on:

```
A[t, j] = beta[t] * <k[t], k[j]> * exp(g[t] - g[j])    if t > j (within chunk)
        = 0                                            otherwise
```

per chunk of length `BT`, per head, per batch. The result is stored in a
*per-token* output of shape `(B, T, H, BT)`: the row index is the global
token index `t`, the column index `j` is the *within-chunk* index of the
source token (`0..BT-1`).

## How `chunk_scaled_dot_kkt_fwd` is wired into Qwen3.5 / Qwen3-Next GDN

In `chunk_gated_delta_rule`:

```python
A = chunk_scaled_dot_kkt_fwd(
    k=k,                            # (1, T_total, Hg=num_k_heads, K)
    beta=beta,                      # (1, T_total, H=num_v_heads)
    g=g,                            # (1, T_total, H)  — already chunk-local cumsummed
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    output_dtype=torch.float32,
)
# A: (1, T_total, H, BT=64)
```

The output `A` then feeds `solve_tril`, which inverts each chunk's
strictly-lower-triangular block plus an identity to obtain the WY
representation.

## GQA layout

| Symbol | Meaning | Qwen3.5 / Qwen3-Next |
|---|---|---|
| `Hg` | number of K-heads (= shape of `k`'s head dim) | 16 |
| `H` | number of V-heads (= shape of `beta`'s head dim) | 32 |
| `r = H // Hg` | GQA ratio (V-heads sharing a K-head) | 2 |

Each output V-head `h` reads `k[..., h // r, :]` (the K-head it belongs to)
and uses its own `beta[..., h]` and `g[..., h]`.

## Schema

```
zentorch::gdn_chunk_scaled_dot_kkt_fwd(
    Tensor k, Tensor g, Tensor beta,
    Tensor cu_seqlens, Tensor chunk_indices, int chunk_size,
    *, str zentorch_op_name='zentorch::gdn_chunk_scaled_dot_kkt_fwd'
) -> Tensor
```

Returns a fresh fp32 tensor of shape `(B, T_total, H, BT)`. Required
`g`, `cu_seqlens`, `chunk_indices` (no `Tensor?` Optional) — production
always provides them. The upstream `g=None` skip-decay branch is not
supported.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `k` | `(B, T, Hg, K)` with `B == 1` | fp32 or bf16 | inner dim unit-stride | none |
| `beta` | `(B, T, H)` with `B == 1` | floating | matches `k` outer dims | none |
| `g` | `(B, T, H)` with `B == 1` | floating (typically fp32) | same shape as `beta` | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_indices` | `(NT, 2)` | int32 | contiguous | none |
| return | `(B, T, H, BT)` | fp32 | fresh contiguous (`at::zeros`) | — |

## Math (per chunk, per (b, h))

Let `c` be a chunk with token range `[chunk_start, chunk_end)`
(`chunk_end <= chunk_start + BT`, `chunk_len = chunk_end - chunk_start`).

```python
k_chunk    = k[b, chunk_start:chunk_end, h // r]            # (chunk_len, K)
beta_chunk = beta[b, chunk_start:chunk_end, h]              # (chunk_len,)
g_chunk    = g[b, chunk_start:chunk_end, h]                 # (chunk_len,)

KB = k_chunk * beta_chunk.unsqueeze(-1)                     # (chunk_len, K)
A_chunk = KB @ k_chunk.T                                    # (chunk_len, chunk_len)
decay = exp(g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(0))   # (chunk_len, chunk_len)
A_chunk = A_chunk * decay

# Strict lower triangle only.
A_chunk = where(i > j, A_chunk, 0)

# Write back: row = global token index, column = within-chunk source index.
# Columns >= chunk_len stay at zero (their source tokens don't exist).
A[b, chunk_start:chunk_end, h, :chunk_len] = A_chunk
```

The reduction is in fp32 (`tl.dot` accumulator is fp32) regardless of
input dtype; output is fp32.

## Output buffer initialization

The Triton kernel uses `torch.empty(...)` for `A` and stores via masked
`tl.store`, so token positions outside any sequence (e.g. the tail of a
varlen-padded last chunk where rows >= T) are left **uninitialised**.
The cpp op uses `at::zeros(...)` so the output is deterministically `0`
at those positions and the test comparisons stay meaningful — downstream
consumers should never read uninitialised rows anyway, so this is a safe
normalisation.

## Parallelization

- **Outer parallel axis:** `chunk_indices` rows × V-head index `h`. All
  chunk × head combinations are independent.
- **Inside a chunk × head:** a `(chunk_len, K) @ (K, chunk_len)` GEMM
  via `at::matmul`, plus a pointwise decay multiply and a `at::where`
  mask.
- For Qwen3.5 / Qwen3-Next at `chunk_size=64` and `K=128`, the inner
  matmul is small enough to fit in L1; ATen's matmul handles the per-(b,
  h) thread assignment internally.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `g = None` (skip-decay branch) | none | not supported (`g` is required) |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_indices=None` (recompute internally) | none | not supported |
| `output_dtype != fp32` | none | not supported (always fp32) |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply, with a `max(input_tol, fp32_tol)` slack since the op upcasts inputs
to fp32 before the matmul. The inner reduction is `K = 128` long; well
within bf16 precision for the typical `beta * k` magnitudes.

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_chunk_scaled_dot_kkt_fwd`.
Returns an empty fp32 tensor of shape `(B, T_total, H, BT)`, allocated from
`k.new_empty(...)` to mirror the cpp's `k.options().dtype(fp32)`. `B` and
`T_total` are also read from `k` (not `beta`) to follow the cpp's
source-of-truth precisely — the schema enforces `(B, T)` agreement, but
deriving from the same tensor as the cpp keeps FakeTensorMode in sync
under torch.compile if any caller ever passes tensors that differ in
device or layout. Required by Inductor and FakeTensorMode for AOT graph
capture; also used by the `make_fallback` Inductor lowering that routes
calls through the dispatcher to the cpp impl.
