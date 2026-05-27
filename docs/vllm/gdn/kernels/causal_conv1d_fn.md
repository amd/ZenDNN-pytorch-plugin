# `causal_conv1d_fn`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_causal_conv1d_fn` |
| C++ symbol | `zentorch::zentorch_gdn_causal_conv1d_fn` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/CausalConv1dFn.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_causal_conv1d_fn.py` |
| Profiler span | `zentorch::gdn::causal_conv1d_fn` |
| Backends | `CPU`, `Meta` |

## What this op does

`causal_conv1d_fn` is the **prefill-path** causal 1-D depthwise
convolution that runs at the entrance of the GDN attention block. For
each token of a varlen continuous batch it computes:

```
out[c, t] = sum_{w=0}^{width-1} padded[c, t + w] * weight[c, w] + bias[c]
```

where `padded` is the per-sequence concatenation of the previous
`state_len = width - 1` columns of the conv-state cache and the current
sequence's tokens, applied as a *depthwise* (per-channel) conv. After
the conv, `silu` activation is applied (when `activation = "silu"` /
`"swish"`).

This op also **updates the conv-state cache in place**: after each
sequence, the last `state_len` columns of its `padded` tensor are
written back to `conv_states[slot, :, :state_len]`, so the next prefill
or decode step can resume from there.

The `pad_slot_id` sentinel (typically `-1`) marks padded entries in
`cache_indices` that should be skipped (no compute, no cache write).

## How `causal_conv1d_fn` is wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (prefill path):

```python
mixed_qkv_non_spec = causal_conv1d_fn(
    x=mixed_qkv_non_spec,                         # (conv_dim, T_total)
    weight=self.conv1d.weight.squeeze(1),         # (conv_dim, width)
    bias=self.conv1d.bias,                        # (conv_dim,) or None
    conv_states=conv_state,                       # (num_cache_lines, conv_dim, state_len)
    query_start_loc=non_spec_query_start_loc,     # (N+1,)
    cache_indices=cache_indices_non_spec,         # (N,)
    has_initial_state=has_initial_state,          # (N,) bool
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,                      # -1
)
```

`forward_cpu_zen` substitutes the cpp op directly. The output then
feeds `fused_post_conv_prep` which splits it into the
`(q, k, v, g, beta)` tensors consumed by the chunked attention path.

## Schema

```
zentorch::gdn_causal_conv1d_fn(
    Tensor x, Tensor weight, Tensor? bias,
    Tensor(a!) conv_states,
    Tensor query_start_loc, Tensor cache_indices,
    Tensor has_initial_state,
    str activation, int pad_slot_id,
    *, str zentorch_op_name='zentorch::gdn_causal_conv1d_fn'
) -> Tensor
```

The `Tensor(a!)` annotation on `conv_states` declares it as
**mutated in-place** by the op. Returns the conv output `out` shaped
`(dim, cu_seqlen)` in `x.dtype`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `x` | `(dim, cu_seqlen)` | fp32 or bf16 | inner dim unit-stride | none |
| `weight` | `(dim, width)` | same as `x` | inner dim unit-stride | none |
| `bias` | `(dim,)` or `None` | same as `x` | contiguous | none |
| `conv_states` | `(num_cache_lines, dim, state_len ≥ width-1)` | fp32 (typically) | — | **in-place write to `[slot, :, :state_len]` per non-padded sequence** |
| `query_start_loc` | `(N+1,)` | int32 | contiguous | none |
| `cache_indices` | `(N,)` | int32 | — | none |
| `has_initial_state` | `(N,)` | bool (or int) | — | none |
| return | `(dim, cu_seqlen)` | `x.dtype` | fresh contiguous (`x.clone()` so pad-slot positions retain `x`) | — |

## Math (per-sequence)

For each sequence `b in [0, N)` with `start = query_start_loc[b]`,
`end = query_start_loc[b+1]`, `T_b = end - start`:

```python
if cache_indices[b] == pad_slot_id:
    # Output for the skipped range is left equal to x[:, start:end]
    # (the op pre-clones x into out). Mirrors gdn_causal_conv1d_update,
    # which clones x_3d into out_3d before its masked overwrite.
    continue
slot = cache_indices[b]

if has_initial_state[b]:
    state = conv_states[slot, :, -state_len:]   # (dim, state_len)
else:
    state = zeros(dim, state_len)

padded = cat([state, x[:, start:end]], dim=-1)   # (dim, state_len + T_b)
conv_out = depthwise_conv1d(padded, weight, bias, groups=dim)  # (dim, T_b)
if activation in ("silu", "swish"):
    conv_out = silu(conv_out)

out[:, start:end] = conv_out
conv_states[slot, :, :state_len] = padded[:, -state_len:]   # cache update
```

All compute runs in `conv_states.dtype` (typically fp32); the output is
cast back to `x.dtype` on store.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| Chunked-prefill (state at end of chunk, not end of sequence) | none currently | not supported |
| `activation` other than `silu` / `swish` / `""` | none | not supported |
| `bias = None` | yes | supported (`Tensor?` Optional) |
| `has_initial_state = None` | none (always provided) | not supported |

## Tolerances

The reduction is `width = 4` long at production shapes — well within
bf16 precision. Tolerance picks `max(x_tol, state_tol)` since the
compute runs in `conv_states.dtype` and noise compounds at that dtype.

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_causal_conv1d_fn`.
Returns `x.new_empty(x.size())` (matching the cpp's `at::empty_like(x)`).
Required by Inductor and FakeTensorMode for AOT graph capture; also
used by the `make_fallback` Inductor lowering. The `conv_states`
in-place mutation is declared via the `Tensor(a!)` schema annotation,
which `make_fallback` honours through the dispatcher.
