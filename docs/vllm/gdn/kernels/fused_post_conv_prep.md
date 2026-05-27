# `fused_post_conv_prep`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_fused_post_conv_prep` |
| C++ symbol | `zentorch::zentorch_gdn_fused_post_conv_prep` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/FusedPostConvPrep.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_fused_post_conv_prep.py` |
| Profiler span | `zentorch::gdn::fused_post_conv_prep` |
| Backends | `CPU`, `Meta` |

## What this op does

`fused_post_conv_prep` runs on the prefill path immediately after
`causal_conv1d_fn` and bridges the conv block to the chunked attention
block. Given the per-token conv output `(L, qkv_dim)` and the linear
projections `a` and `b` of the same per-token batch, it:

1. **Splits** `conv_output` into `(q, k, v)` slices: shapes `(L, H, K)`,
   `(L, H, K)`, `(L, HV, V)` respectively where
   `qkv_dim = 2*H*K + HV*V`.
2. Optionally **L2-normalises** `q` and `k` along their last dim (the
   `apply_l2norm=True` path; the head-internal norm).
3. Computes the **gate / decay tensors** `g` and `beta`:
   ```
   g    = -exp(A_log) * softplus(a + dt_bias, threshold=20)
   g    = exp(g)                       if output_g_exp else g
   beta = sigmoid(b)
   ```

Returns the 5-tuple `(q, k, v, g, beta)` that feeds the chunked
attention path (`gdn_chunk_gated_delta_rule_fwd`).

## How `fused_post_conv_prep` is wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (prefill path, after the causal conv):

```python
q, k, v, g, beta = fused_post_conv_prep(
    conv_output=mixed_qkv_non_spec.T,    # (L_total, qkv_dim)
    a=a_non_spec,                        # (L_total, HV)
    b=b_non_spec,                        # (L_total, HV)
    A_log=self.A_log,                    # (HV,)
    dt_bias=self.dt_bias,                # (HV,)
    num_k_heads=self.num_k_heads,
    head_k_dim=self.head_k_dim,
    head_v_dim=self.head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
)
```

`forward_cpu_zen` substitutes the cpp op directly.

## Schema

```
zentorch::gdn_fused_post_conv_prep(
    Tensor conv_output, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias,
    int num_k_heads, int head_k_dim, int head_v_dim,
    bool apply_l2norm, bool output_g_exp,
    *, str zentorch_op_name='zentorch::gdn_fused_post_conv_prep'
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

Returns `(q, k, v, g, beta)` — fixed 5-tuple, never optional.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `conv_output` | `(L, qkv_dim)` where `qkv_dim = 2*H*K + HV*V` | fp32 or bf16 | inner dim unit-stride | none |
| `a` | `(L, HV)` | same as `conv_output` | inner dim unit-stride | none |
| `b` | `(L, HV)` | same as `conv_output` | inner dim unit-stride | none |
| `A_log` | `(HV,)` | floating | contiguous | none |
| `dt_bias` | `(HV,)` | floating | contiguous | none |
| return `q` | `(L, H, K)` | model dtype | fresh contiguous | — |
| return `k` | `(L, H, K)` | model dtype | fresh contiguous | — |
| return `v` | `(L, HV, V)` | model dtype | fresh contiguous (view + `.contiguous()`) | — |
| return `g` | `(L, HV)` | **fp32** | fresh contiguous | — |
| return `beta` | `(L, HV)` | **fp32** | fresh contiguous | — |

`g` and `beta` are always fp32 regardless of input dtype.

## Math (per token)

```python
# Split conv_output along the feature dim.
q = conv_output[:, :H*K].view(L, H, K)
k = conv_output[:, H*K:2*H*K].view(L, H, K)
v = conv_output[:, 2*H*K:].view(L, HV, V)

# Optional L2 norm on q and k.
if apply_l2norm:
    q_f = q.float()
    k_f = k.float()
    q = (q_f * rsqrt(sum(q_f**2, dim=-1, keepdim=True) + 1e-6)).to(dtype)
    k = (k_f * rsqrt(sum(k_f**2, dim=-1, keepdim=True) + 1e-6)).to(dtype)

# Gate / decay computation (all in fp32, output stays fp32).
sp   = softplus(a.float() + dt_bias.float(), threshold=20.0)
g    = -exp(A_log.float()) * sp
if output_g_exp:
    g = exp(g)
beta = sigmoid(b.float())
```

The `softplus(..., threshold=20)` is the numerically-stable
"`softplus(x) ≈ x` for `x > 20`" branch — at `x = 25` the naive
`log(1 + exp(x))` would overflow fp32, so the kernel switches to
identity for large inputs.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `apply_l2norm = False` | none | supported (runtime arg) |
| `output_g_exp = True` | none | supported (runtime arg) |
| Higher-rank `conv_output` | none | not supported (caller flattens to 2-D) |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply. The longest reduction is the L2-norm (`K = 128` at production
shapes) — well within bf16 precision.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_fused_post_conv_prep`. Returns the fixed 5-tuple
`(q, k, v, g, beta)` with the shapes and dtypes documented above (q/k/v
in `conv_output.dtype`, g/beta in fp32). Required by Inductor and
FakeTensorMode for AOT graph capture; also used by the `make_fallback`
Inductor lowering.
