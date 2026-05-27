# `chunk_gated_delta_rule_fwd` (fused)

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd` |
| C++ symbol | `zentorch::zentorch_gdn_chunk_gated_delta_rule_fwd` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/ChunkGatedDeltaRuleFwd.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton wrapper | `vllm/model_executor/layers/fla/ops/chunk.py` (`chunk_gated_delta_rule`) |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_chunk_gated_delta_rule_fwd.py` |
| Profiler span | `zentorch::gdn::chunk_gated_delta_rule_fwd` |
| Backends | `CPU`, `Meta` |

## What this op does

The **fused production op** for the GDN prefill path. Internally it
performs the work of six sub-kernels in a single C++ entry point,
sharing intermediate buffers so the prefill path makes one cpp dispatch
instead of six:

```
g_cum  ← chunk_local_cumsum(g)                       (per-chunk cumulative gate)
                ↓
A_block ← chunk_scaled_dot_kkt_fwd(k, β, g_cum, …)   (WY Gram, per-head batched)
                ↓
A_solved ← solve_tril(A_block, …)                    ((I + A)^{-1} per chunk)
                ↓
(w, u)  ← recompute_w_u_fwd(k, v, β, g_cum, A_solved, …)  (WY representation)
                ↓
(h, v_new, final_state) ← chunk_gated_delta_rule_fwd_h(k, w, u, g_cum, …)
                                                     (chunk-recurrent state)
                ↓
o      ← chunk_fwd_o(q, k, v_new, h, g_cum, …)       (per-token output)
```

Each phase is implemented inline in this binding using the same math as
the standalone ops registered in commits 1–7, but with shared per-head
batched intermediates (`(H, BT, K)` etc.) routed through
`zentorch_bmm` instead of `at::matmul` for better small-batch GEMM
throughput on AMD CPUs.

This is the op that the production CPU forward path invokes via
`forward_cpu_zen` for prefill batches. The standalone ops 1–7 in this
stack are exposed only for unit-testing; production never calls them
individually.

## How `chunk_gated_delta_rule_fwd` is wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py`'s prefill path:

```python
core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
    q=query_non_spec,                       # (1, T_total, Hg=num_k_heads, K)
    k=key_non_spec,                         # (1, T_total, Hg, K)
    v=value_non_spec,                       # (1, T_total, H=num_v_heads, V)
    g=g_non_spec,                           # (1, T_total, H)  ← from fused_post_conv_prep
    beta=beta_non_spec,                     # (1, T_total, H)  ← from fused_post_conv_prep
    initial_state=initial_state,            # (N, H, V, K)
    output_final_state=True,
    cu_seqlens=non_spec_query_start_loc,
    chunk_indices=attn_metadata.chunk_indices,
    chunk_offsets=attn_metadata.chunk_offsets,
)
```

`forward_cpu_zen` substitutes the cpp op directly:

```python
o, final_state = torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale, initial_state, True, 64,
    cu_seqlens, chunk_indices, chunk_offsets,
)
```

## Schema

```
zentorch::gdn_chunk_gated_delta_rule_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    float scale, Tensor? initial_state, bool output_final_state,
    int chunk_size, Tensor cu_seqlens, Tensor chunk_indices,
    Tensor chunk_offsets,
    *, str zentorch_op_name='zentorch::gdn_chunk_gated_delta_rule_fwd'
) -> (Tensor, Tensor)
```

Returns `(o, final_state)`. When `output_final_state=False`,
`final_state` is a 0-element tensor.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `(B, T, Hg, K)` with `B == 1` | fp32 or bf16 | K-head queries |
| `k` | `(B, T, Hg, K)` | same as `q` | K-head keys |
| `v` | `(B, T, H, V)` | same as `q` | V-head values |
| `g` | `(B, T, H)` | floating | raw per-(token, V-head) log-decay (NOT yet chunk-cumsummed) |
| `beta` | `(B, T, H)` | floating | per-(token, V-head) gate |
| `initial_state` | `(N, H, V, K)` or empty | fp32 | per-sequence seed state |
| `cu_seqlens` | `(N+1,)` | int32 | varlen offsets |
| `chunk_indices` | `(NT, 2)` | int32 | chunk-walk schedule |
| `chunk_offsets` | `(N+1,)` | int32 or int64 | per-sequence chunk offsets |
| return `o` | `(B, T, H, V)` | `v.dtype` | fresh contiguous |
| return `final_state` | `(N, H, V, K)` if `output_final_state` else `(0,)` | **fp32** | fresh contiguous |

`H` (the V-head count) is read from `v.size(2)`, NOT from `q.size(2)`
(which is `Hg`, the K-head count). This distinction matters in the
meta function — see below.

## GQA layout

Same convention as the standalone building blocks:

| Symbol | Meaning | Qwen3.5 / Qwen3-Next |
|---|---|---|
| `Hg` | number of K-heads (head dim of `q` and `k`) | 16 |
| `H` | number of V-heads (head dim of `v`, `o`, `g`, `beta`) | 32 |
| `r = H / Hg` | GQA ratio | 2 |

## Algorithm (high-level — see source for per-phase implementation)

The cpp op runs **four phases** sequentially, each operating on shared
fp32 working buffers (`g_cum`, `w_f`, `u_f`, `h_out_f`, `v_new_f`):

1. **`g_cum` build** (`run_g_cumsum`): per-chunk cumulative sum of `g`
   with `at::parallel_for` over `chunk_indices` rows. Output `g_cum`
   in fp32.
2. **WY representation** (`run_recompute_w_u_fused`): for each chunk
   row, compute the per-head batched KKT block, run a batched
   `linalg_solve_triangular` to produce `A_solved`, then derive `w` and
   `u` via two more `zentorch_bmm` calls. All matmuls are per-head
   batched `(H, BT_eff, *)` to amortise the BMM dispatch.
3. **Chunk-recurrent state** (`run_chunk_recurrent_state`): per-`(seq,
   head)`, walk chunks serially. Per chunk: snapshot state, compute
   value-correction, save `v_new`, apply per-token rewind + bulk decay,
   update state. State stays in fp32 across chunks.
4. **Output projection** (`run_chunk_output`): per-chunk × head, three
   `zentorch_bmm` calls (history, attention scores, in-chunk weighted
   sum) plus the diagonal-inclusive causal mask and scale. Cast to
   `v.dtype` on store.

## Output buffer initialization

`o` and `final_state` are allocated via `at::detail::empty_strided_cpu`
(fresh contiguous, no zero-init). Every byte is overwritten by the
algorithm except in the empty-input edge case (`NT == 0 || H == 0 ||
T == 0`), which explicitly zeros and returns early.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `use_qk_l2norm_in_kernel=True` | none (call site uses `False`) | not supported (call `gdn_l2norm_fwd` separately if needed) |
| Non-varlen mode (`cu_seqlens=None`) | none | not supported |
| `chunk_indices=None` / `chunk_offsets=None` (recompute internally) | none | not supported |
| `chunk_size` not in `{16, 32, 64}` | none | not supported |
| `scale=None` (default `K^-0.5`) | none | not supported (caller passes scale explicitly) |

## Tolerances

The chunk-recurrent state walks up to `NT` chunks per sequence,
accumulating bf16 round-trip noise linearly. Tests scale `atol`/`rtol`
proportionally to `NT`. For Qwen3.5 / Qwen3-Next at production shapes
(`H=32`, `K=V=128`, GQA ratio 2), the test additionally scales `atol`
to the oracle's output magnitude to avoid per-element trips on small
values that pick up compounded noise from large-magnitude rows.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_chunk_gated_delta_rule_fwd`. Returns the 2-tuple
`(o, final_state)`:

- `o` shape `(B, T, H, V)` in **`v.dtype`**, where `H = v.size(2)`
  (NOT `q.size(2)`, which would give `Hg`).
- `final_state` shape `(N, H, V, K)` always **fp32** if
  `output_final_state` else shape `(0,)`.

**This is the most-traced GDN op under `torch.compile`** — the production
prefill path goes through it. Unlike the standalone building-block ops
(commits 1–7) where the meta is best-effort, this meta MUST match the
cpp output exactly for compile mode to work correctly.
