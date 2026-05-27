# `solve_tril`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_solve_tril` |
| C++ symbol | `zentorch::zentorch_gdn_solve_tril` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/SolveTril.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/solve_tril.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_solve_tril.py` |
| Profiler span | `zentorch::gdn::solve_tril` |
| Backends | `CPU`, `Meta` |

## What this op does

`solve_tril` is the third of the six FLA sub-kernels that compose
`chunk_gated_delta_rule`. It takes `A`, the `(B, T, H, BT)` per-token
store of strictly-lower-triangular chunks produced by
`chunk_scaled_dot_kkt_fwd`, and computes
`Ai[t, h, :] = (I + A[chunk(t), h, :, :])^{-1}[t mod BT, :]` — the
chunk-local inverse of `(I + A)`, written back in the same per-token
`(B, T, H, BT)` layout.

The matrix `(I + A)` is **unit lower triangular** (1s on the diagonal,
strictly-lower-triangular below), so its inverse is also unit lower
triangular and well-defined for any `A`. Upstream FLA picks one of
three Triton kernels (`solve_tril_16x16_kernel`,
`merge_16x16_to_32x32_inverse_kernel`,
`merge_16x16_to_64x64_inverse_kernel`) depending on
`BT ∈ {16, 32, 64}`. All three compute the same math; the cpp op
collapses them into a single `at::linalg_solve_triangular` call per
chunk.

## How `solve_tril` is wired into Qwen3.5 / Qwen3-Next GDN

In `chunk_gated_delta_rule`:

```python
A = solve_tril(
    A=A,                            # (1, T_total, H=num_v_heads, BT=64)
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    output_dtype=k.dtype,           # bf16 for Qwen3.5 / Qwen3-Next
)
# A: (1, T_total, H, BT) — overwritten in place semantically
```

The output (the WY-representation matrix
`(I + tril(K K^T β decay))^{-1}`) feeds `recompute_w_u_fwd` which uses
it to construct the `w` and `u` vectors of the chunk-recurrent
delta-rule update. For GDN, `BT = FLA_CHUNK_SIZE = 64` always.

## Schema

```
zentorch::gdn_solve_tril(
    Tensor A, Tensor cu_seqlens, Tensor chunk_indices,
    *, str zentorch_op_name='zentorch::gdn_solve_tril'
) -> Tensor
```

Returns a fresh fp32 tensor with the same shape as `A`. `output_dtype`
is dropped from the schema — the cpp op always returns fp32; the wiring
layer in the fused `chunk_gated_delta_rule_fwd` op casts to `k.dtype`
if needed.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `A` | `(B, T, H, BT)` with `B == 1`, `BT ∈ {16, 32, 64}` | fp32 or bf16 | inner dim unit-stride | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_indices` | `(NT, 2)` | int32 | contiguous | none |
| return | `(B, T, H, BT)` | **fp32** (always) | fresh contiguous (`at::zeros_like`) | — |

## Math

For each chunk `c` with valid token range `[chunk_start, chunk_end)` and
effective length `BT_eff = chunk_end - chunk_start <= BT`, and for each
`(b, h)`:

```python
A_full = A[b, chunk_start:chunk_end, h, :]                # (BT_eff, BT)
# The cpp op only uses the strict lower triangle of the BT_eff × BT_eff
# top-left sub-block; everything else is 0 (chunk_scaled_dot_kkt_fwd's
# output is zero-initialised elsewhere).
A_strict = tril(A_full[:BT_eff, :BT_eff], diagonal=-1)    # (BT_eff, BT_eff)

L = I_{BT_eff} + A_strict                                  # unit lower triangular
M = solve_triangular(L, I_{BT_eff}, upper=False, unitriangular=True)
Ai[b, chunk_start:chunk_end, h, :BT_eff] = M
# Cols [BT_eff, BT) of valid rows stay at zero (the source columns don't exist).
```

For the strictly-lower-triangular `A`, `(I + A)^{-1}` admits the
closed-form **Neumann series**
`M = sum_{k=0}^{BT_eff-1} (-A)^k`, which terminates exactly because
strictly-lower-triangular matrices are nilpotent of order `BT_eff`. The
oracle uses this series; the cpp op uses `at::linalg_solve_triangular`.

## Output buffer initialization and the partial-chunk corner case

The cpp op uses `at::zeros_like(A, dtype=fp32)` to initialise `Ai`. For
chunks with `BT_eff < BT` (partial last chunk in a sequence):

- The valid sub-block `Ai[:BT_eff, :BT_eff]` is the actual inverse
  `(I + A_strict[:BT_eff, :BT_eff])^{-1}`.
- All other positions stay at zero.

Mathematically, `(I + A_full_BT×BT)^{-1}` would have `I` on the
bottom-right diagonal block (`I_{BT-BT_eff}`) for the "padded" rows, but
the cpp op does not write those cells (boundary-checked via narrow), so
they remain zero. This matches the kernel's output and is what the
downstream consumer (`recompute_w_u_fwd`) expects, since that op also
iterates only over valid tokens.

## Parallelization

- **Outer parallel axis:** `(chunk × head)` pairs. Independent.
- **Inside a chunk × head:** one `at::linalg_solve_triangular` call on a
  `(BT_eff, BT_eff)` matrix, `BT_eff ≤ 64`. The triangular solve is
  L1-resident at this size, so a single ATen call per chunk × head is
  sufficient.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `output_dtype != fp32` | none | not supported (always fp32; wiring layer casts if needed) |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_indices=None` (recompute internally) | none | not supported |
| `BT ∉ {16, 32, 64}` | none | not supported (matches upstream FLA) |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply, with `max(input_tol, fp32_tol)` slack since the math runs in
fp32. The Neumann series in the oracle has up to `BT_eff = 64` terms,
but each strictly nilpotent `(-A)^k` is bounded by `||A||_∞^k`, which
decays fast for the typical `||A||_∞ ~ 0.5` we see at inference time.
No per-kernel loosening.

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_solve_tril`. Returns
an empty fp32 tensor of the same shape as `A` (the cpp op always
returns fp32 regardless of `A.dtype`). Required by Inductor and
FakeTensorMode for AOT graph capture; also used by the `make_fallback`
Inductor lowering that routes calls through the dispatcher to the cpp
impl.
