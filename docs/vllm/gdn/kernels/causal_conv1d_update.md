# `causal_conv1d_update`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_causal_conv1d_update` |
| C++ symbol | `zentorch::zentorch_gdn_causal_conv1d_update` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/CausalConv1dUpdate.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_causal_conv1d_update.py` |
| Profiler span | `zentorch::gdn::causal_conv1d_update` |
| Backends | `CPU`, `Meta` |

## What this op does

`causal_conv1d_update` is the **decode-path** causal 1-D depthwise
convolution. Whereas `causal_conv1d_fn` runs over varlen prefill
sequences, this op runs over a flat decode batch where each row owns
one cache slot:

```
for b in [0, batch):
    if cache_state_indices[b] in {null_block_id, pad_slot_id}:
        continue
    slot = cache_state_indices[b]
    state = conv_state[slot, :, -state_len:]            # (dim, state_len)
    padded = cat([state, x[b]], dim=-1)                  # (dim, state_len + seqlen)
    out[b] = depthwise_conv1d(padded, weight, bias, groups=dim)  # (dim, seqlen)
    if silu/swish: out[b] = silu(out[b])
    conv_state[slot, :, -state_len:] = padded[:, -state_len:]   # in-place state update
```

The op handles both 2-D (`(batch, dim)`, the typical decode shape) and
3-D (`(batch, dim, seqlen)`, the spec-decode shape) input layouts. For
sentinel slots (`null_block_id` or `pad_slot_id`), the output row is
left equal to the input — so downstream code that consumes the full
output tensor sees consistent shapes — and the cache slot is not
touched.

## How `causal_conv1d_update` is wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (decode and spec paths):

```python
mixed_qkv_decode = causal_conv1d_update(
    x=mixed_qkv_decode,                       # (batch, conv_dim) or (batch, conv_dim, seqlen)
    conv_state=conv_state,                    # (num_cache_lines, conv_dim, state_len)
    weight=self.conv1d.weight.squeeze(1),     # (conv_dim, width)
    bias=self.conv1d.bias,                    # (conv_dim,) or None
    activation="silu",
    conv_state_indices=cache_indices_decode,
    null_block_id=NULL_BLOCK_ID,              # 0
    pad_slot_id=PAD_SLOT_ID,                  # -1
)
```

`forward_cpu_zen` dispatches to the cpp op directly.

## Schema

```
zentorch::gdn_causal_conv1d_update(
    Tensor x, Tensor(a!) conv_state, Tensor weight, Tensor? bias,
    str activation, Tensor conv_state_indices,
    int null_block_id, int pad_slot_id,
    *, str zentorch_op_name='zentorch::gdn_causal_conv1d_update'
) -> Tensor
```

The `Tensor(a!)` annotation on `conv_state` declares it as **mutated
in-place** by the op. Returns the conv output `out` with the same shape
and dtype as `x`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `x` | `(batch, dim)` or `(batch, dim, seqlen)` | fp32 or bf16 | — | none |
| `weight` | `(dim, width)` | same as `x` | inner dim unit-stride | none |
| `bias` | `(dim,)` or `None` | same as `x` | contiguous | none |
| `conv_state` | `(num_cache_lines, dim, state_len ≥ width-1)` | fp32 (typically) | — | **in-place write to `[slot, :, -state_len:]` per non-sentinel sequence** |
| `conv_state_indices` | `(batch,)` | int32 or int64 | — | none |
| return | same as `x` | `x.dtype` | fresh (`x_3d.to(compute_dtype).clone()` then `squeeze(-1).to(out_dtype)`) | — |

`compute_dtype = conv_state.scalar_type()`. All math is done in
`compute_dtype`; the output is cast back to `x.dtype` on store. The
op tolerates `conv_state` having extra trailing columns beyond
`state_len = width - 1` (e.g. for spec-decode buffering); only the
last `state_len` columns are read and written.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `activation` other than `silu` / `swish` / `""` | none | not supported |
| `bias = None` | yes | supported (`Tensor?` Optional) |
| Per-sequence different `seqlen` | none (decode uses fixed `seqlen=1` per batch) | not supported (batched-uniform `seqlen`) |

## Tolerances

The reduction is `width = 4` long at production shapes — well within
bf16 precision. Tolerance picks `max(x_tol, state_tol)` since the
compute runs in `conv_state.dtype` and noise compounds at that dtype.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_causal_conv1d_update`. Returns `x.new_empty(x.size())`
(matching the cpp's output shape and dtype). Required by Inductor and
FakeTensorMode for AOT graph capture; also used by the `make_fallback`
Inductor lowering. The `conv_state` in-place mutation is declared via
the `Tensor(a!)` schema annotation, which `make_fallback` honours
through the dispatcher.
