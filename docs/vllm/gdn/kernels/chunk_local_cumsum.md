# `chunk_local_cumsum`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_chunk_local_cumsum` |
| C++ symbol | `zentorch::zentorch_gdn_chunk_local_cumsum` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/ChunkLocalCumsum.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/cumsum.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_chunk_local_cumsum.py` |
| Profiler span | `zentorch::gdn::chunk_local_cumsum` |
| Backends | `CPU`, `Meta` |

## What this op does

`chunk_local_cumsum` is the first of the six FLA sub-kernels that
together compose `chunk_gated_delta_rule`. Given a 3-D scalar tensor
`g.shape = (1, T_total, HV)`, it splits the time axis `T_total` into
chunks of length `BT = chunk_size` according to the supplied
`chunk_indices` table and computes a cumulative sum *within* each
chunk. There is no carry between chunks — hence "local" cumsum.

In the GDN code path, this runs once on the gating tensor `g` produced
by `fused_post_conv_prep`, turning it into the per-chunk cumulative
log-decay used by all the downstream chunked-recurrent steps
(`chunk_scaled_dot_kkt_fwd`, `chunk_gated_delta_rule_fwd_h`,
`chunk_fwd_o`, …) — `decay_mask = (g[i] - g[j]).exp()` for `i >= j`
within a chunk.

## Schema

```
zentorch::gdn_chunk_local_cumsum(
    Tensor g,
    int chunk_size,
    Tensor cu_seqlens,
    Tensor chunk_indices,
    *,
    str zentorch_op_name='zentorch::gdn_chunk_local_cumsum'
) -> Tensor
```

Returns a fresh fp32 tensor of the same shape as `g`. Required
`cu_seqlens` and `chunk_indices` (no `Tensor?` Optional) — in
production both are always provided. Non-varlen mode and missing
`chunk_indices` are not supported by the C++ op.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `g` | `(1, T_total, HV)` | fp32 or bf16 (typically fp32) | inner dim unit-stride (`stride(-1) == 1`) | none |
| `cu_seqlens` | `(N+1,)` | int32 | contiguous | none |
| `chunk_indices` | `(NT, 2)` | int32 | contiguous | none |
| return | `(1, T_total, HV)` | fp32 | fresh contiguous tensor (`at::empty`) | — |

## Inferred dimensions

- `B = g.size(0)` — must equal 1 (varlen requirement).
- `T_total = g.size(1)`.
- `HV = g.size(2)`.
- `N = cu_seqlens.size(0) - 1` — number of sequences in the varlen pack.
- `NT = chunk_indices.size(0)` — total number of chunks across all sequences.

## Math

For each row `r` in `chunk_indices` (parallel-safe):

```
seq_idx, chunk_idx = chunk_indices[r]
bos = cu_seqlens[seq_idx]
eos = cu_seqlens[seq_idx + 1]
cs_start = bos + chunk_idx * chunk_size
cs_end   = min(cs_start + chunk_size, eos)

for h in [0, HV):
    acc = 0.0f
    for t in [cs_start, cs_end):
        acc += static_cast<float>(g[0, t, h])
        out[0, t, h] = acc
```

Empty chunks (`cs_start >= cs_end`) are no-ops. The outer iteration
order over `chunk_indices` rows is unobservable — each row writes to a
disjoint `[cs_start, cs_end)` × `[0, HV)` rectangle of the output. The
accumulator is fp32 regardless of `g.dtype`, and the output is always
fp32 (matching the Triton kernel which does
`tl.load(...).to(tl.float32)` then `tl.cumsum`).

## Parallelization

- **Outer parallel axis:** rows of `chunk_indices` (one row = one chunk).
  Chunks are independent.
- **Within a chunk:** loop over `H` lanes serially; each lane is a
  short serial scan over `BT ≤ 64` tokens, fast enough in registers.
- **Tail handling:** the last chunk of each varlen sequence is shorter
  than `BT`; the inner loop naturally stops at `eos`.

## Production-narrowing rationale

The Python ref (and the upstream Triton kernel) supports a wider
surface than this cpp op:

| Variant | Production caller? | Cpp op support |
|---|---|---|
| 4-D vector input `(B, T, H, S)` | none | not supported |
| `head_first=True` (i.e. `(B, H, T)` layout) | none | not supported |
| `reverse=True` | none | not supported |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_indices=None` (recompute internally) | none | not supported |
| `output_dtype != fp32` | none | not supported (always fp32) |

Each of these would be a one-liner to add later, but adding them now
trades C++ surface for no perf benefit — no caller exercises them.

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply. The reduction is at most `chunk_size = 64` items long, well
within bf16 precision; no per-kernel loosening.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_chunk_local_cumsum`. Returns an empty tensor of the same
shape as `g` with `dtype=torch.float32`. Required by Inductor and
FakeTensorMode for AOT graph capture; also used by the
`make_fallback` Inductor lowering that routes calls through the
dispatcher to the cpp impl.
