# `l2norm_fwd`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_l2norm_fwd` |
| C++ symbol | `zentorch::zentorch_gdn_l2norm_fwd` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/L2NormFwd.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/l2norm.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_l2norm_fwd.py` |
| Profiler span | `zentorch::gdn::l2norm_fwd` |
| Backends | `CPU`, `Meta` |

## What this op does

For every row of `x` along the last dim, normalise to unit L2 length:

```
y[..., i, :] = x[..., i, :] / sqrt( sum(x[..., i, :]^2) + eps )
```

The upstream Triton kernel has three variants (`l2norm_fwd_kernel`,
`l2norm_fwd_kernel1`, `l2norm_fwd_kernel2`) that differ in tile shapes
for different `D` ranges on the GPU. All three compute the *same* math;
the C++ op collapses them into one implementation since the tile-shape
choice doesn't apply to the CPU path.

## How `l2norm_fwd` is wired into Qwen3.5 / Qwen3-Next GDN

In the `use_qk_l2norm_in_kernel=True` branch of `chunk_gated_delta_rule`:

```python
if use_qk_l2norm_in_kernel:
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
```

This is the only call site. Both call sites use the default `eps=1e-6`
and `output_dtype=None` (return same dtype as input).

## Schema

```
zentorch::gdn_l2norm_fwd(
    Tensor x,
    float eps,
    *,
    str zentorch_op_name='zentorch::gdn_l2norm_fwd'
) -> Tensor
```

Returns a fresh tensor with the same shape and dtype as `x`. `eps` is
required (no default in the schema) so the call site is explicit.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Contiguity | Mutation |
|---|---|---|---|---|
| `x` | any-rank, `D = x.size(-1) >= 1` | fp32 or bf16 | `stride(-1) == 1` (last dim unit-stride) | none |
| return | same as `x` | same as `x.dtype` | fresh contiguous tensor | — |

The op does not require full `is_contiguous()` on the outer dims —
production passes `q.unsqueeze(0)` whose size-1 dim has a non-default
stride. Outer offsets are computed via the standard "unravel + sum
strides" pattern.

## Math

For each row `r in [0, prod(x.shape[:-1]))` (parallel-safe):

```
sum_sq = 0.0f                           # fp32
for k in [0, D):
    v = static_cast<float>(x[..., k])
    sum_sq += v * v
inv_norm = 1.0f / sqrtf(sum_sq + eps)
for k in [0, D):
    out[..., k] = static_cast<x.dtype>(static_cast<float>(x[..., k]) * inv_norm)
```

All reduction in fp32 (matches the kernel's `tl.load(...).to(tl.float32)`
upcast); writes round to `x.dtype`. Output rows are disjoint, so the
outer iteration order is unobservable.

## Why the kernel form is *not* `F.normalize`

`torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)` computes:

```
y = x / max( ||x||_2 , eps )
```

i.e. it clamps the **denominator** to `eps`, not the squared sum.
For all-zero rows this gives `y = 0 / eps = 0` (with eps "outside" the
sqrt). The kernel's `sqrt(sum_sq + eps)` form gives `y = 0 / sqrt(eps) = 0`
for the all-zero row but a *different* result for tiny-but-nonzero rows.

Specifically: for `||x|| << sqrt(eps)`, the kernel form does almost no
normalisation (denominator is dominated by `sqrt(eps)`), while
`F.normalize` clamps to a fixed denominator. The two forms agree
whenever `||x||^2 >> eps`, which is the typical case at inference time.
The cpp op uses the kernel form. The oracle uses a different
*implementation* of the same math (`torch.linalg.vector_norm` then
square + add eps + rsqrt) to cross-check.

## Parallelization

- **Outer parallel axis:** rows along the flattened `M = prod(x.shape[:-1])` dim.
- **Inside a row:** one length-`D` fp32 reduction (`sum_sq`), one
  `rsqrt(sum_sq + eps)`, one length-`D` pointwise multiply.
- For Qwen3.5 / Qwen3-Next at `D=128`: grain size 16 keeps per-task work
  meaningful while letting ATen auto-tune within the chosen grain.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `output_dtype != None` (return a different dtype than input) | none | not supported (always returns input dtype) |
| `eps != 1e-6` | none (always 1e-6 in `chunk_gated_delta_rule`) | supported (just a runtime arg) |

## Tolerances

Project defaults from `test/zentorch_test_utils.py::default_tolerance(*dtypes)`
apply. The reduction is at most `D` items long (`D ≤ 128` at
Qwen3.5 / Qwen3-Next), well within bf16 precision.

## Meta function

Registered in `_meta_registrations.py` as `meta_gdn_l2norm_fwd`. Returns
an empty tensor of the same shape and dtype as `x`. Required by Inductor
and FakeTensorMode for AOT graph capture; also used by the
`make_fallback` Inductor lowering that routes calls through the
dispatcher to the cpp impl.
