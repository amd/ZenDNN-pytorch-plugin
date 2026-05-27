# `rms_norm_gated`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_rms_norm_gated` |
| C++ symbol | `zentorch::zentorch_gdn_rms_norm_gated` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/RmsNormGated.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/layernorm_guard.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_rms_norm_gated.py` |
| Profiler span | `zentorch::gdn::rms_norm_gated` |
| Backends | `CPU`, `Meta` |

## What this op does

`rms_norm_gated` is the output norm of the GDN attention block. It runs
once per forward pass on the per-token `(M, V)` output of the chunked
attention path, applying both an RMSNorm and a multiplicative gate
derived from a separate input `z`:

```
y_norm = RMSNorm(x, weight, eps)         # standard RMS norm of x
gate   = silu(z)  or  sigmoid(z)         # depending on `activation`
out    = y_norm * gate
```

The cpp op implements the **production parametrisation**: `z` is
required, `group_size = None` (one group over the last dim),
`norm_before_gate = True` (gate is applied AFTER the norm, not folded
into the reduction). The reference oracle additionally supports the
non-production variants (no `z`, `group_size != None`,
`norm_before_gate = False`) so tests can verify the cpp's narrowed
behavior matches the broader contract.

## How `rms_norm_gated` is wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (post-attention pre-out_proj):

```python
core_attn_out = self.norm(core_attn_out, gate)  # RMSNormGated
```

where `core_attn_out` and `gate` are shaped `(M, V)` after the per-token
flattening of the attention output. The `weight` is the learned
RMSNorm weight and `eps` is the model's `rms_norm_eps` (typically
`1e-6`).

`forward_cpu_zen` substitutes this with:

```python
out = torch.ops.zentorch.gdn_rms_norm_gated(
    core_attn_out, weight, gate, eps, "silu",
)
```

## Schema

```
zentorch::gdn_rms_norm_gated(
    Tensor x, Tensor weight, Tensor z, float eps, str activation,
    *, str zentorch_op_name='zentorch::gdn_rms_norm_gated'
) -> Tensor
```

Returns a fresh tensor of the same shape and dtype as `x`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `x` | `(M, V)` | fp32 or bf16 | (made contiguous internally) | none |
| `weight` | `(V,)` | same as `x` | contiguous | none |
| `z` | `(M, V)` | same as `x` | (made contiguous internally) | none |
| `activation` | scalar | string | `"silu"`, `"swish"` (alias), or `"sigmoid"` | — |
| return | `(M, V)` | `x.dtype` | fresh contiguous | — |

`activation = "swish"` is an alias for `"silu"` (they compute the same
function, `z * sigmoid(z)`). `"sigmoid"` returns `sigmoid(z)` only (no
extra `* z` factor).

## Math

```python
y_norm = zentorch_rms_norm(x.float(), weight.float(), eps)   # fp32 internally
gate   = sigmoid(z.float())               if activation == "sigmoid"
       = silu(z.float()) = z * sigmoid(z) if activation in ("silu", "swish")
out    = (y_norm * gate).to(x.dtype)
```

All intermediate math is in fp32; the final cast back to `x.dtype`
happens on the output store. The cpp op reuses the existing
`zentorch_rms_norm` primitive (defined in `RMS_norm.cpp`, linked into
the same shared library) for the RMS reduction.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `z = None` (no gate) | none | not supported (`z` is required) |
| `group_size != None` (grouped RMS) | none | not supported (always one group) |
| `norm_before_gate = False` (gate folded into reduction) | none | not supported (always norm-then-gate) |
| Higher rank than 2-D | none | not supported (caller flattens to `(M, V)`) |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply. The reduction is `V = 128` long at production shapes — well
within bf16 precision.

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_rms_norm_gated`.
Returns an empty tensor of the same shape and dtype as `x` (matching
the cpp's `(y_f * gate_f).to(x.scalar_type())`). Required by Inductor
and FakeTensorMode for AOT graph capture; also used by the
`make_fallback` Inductor lowering that routes calls through the
dispatcher to the cpp impl.
