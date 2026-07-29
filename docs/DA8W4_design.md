(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch DA8W4 (W4A8) — Dynamic INT8 Activation × Symmetric INT4 Weight Linear

## 1. Overview

DA8W4 (W4A8) is a linear op that multiplies **bf16 activations** by **symmetric
int4 weights** and returns a bf16 output. The activations are quantized to int8
at runtime and the int4 weights are widened to int8, so the matmul runs as
int8 × int8 on ZenDNN's AOCL-DLP backend.

**Single op for DA8W8 and DA8W4, mode inferred from the weight.** Both
dynamic-quantization modes share the existing `zentorch_dynamic_qlinear`
operator, with **no mode selector**. The kernel infers the mode from the weight
dtype and its K-dimension relative to the activation's K:

- `int8` weight, `dim1 == K`   → **DA8W8** (s8 weight)
- `int8` weight, `dim1 == K/2` → **DA8W4** (packed s4, 2 nibbles/byte)
- `int32` weight, `dim1 == K/8` → **DA8W4** (packed s4, 8 int4/int32)

So the op signature is **unchanged** from the original DA8W8 op: a call with an
`int8 [N, K]` weight behaves exactly as before, and a packed-s4 weight selects
DA8W4. No new op or parameter is introduced and none is deprecated.

The checkpoint format stays **W4A16** (compressed-tensors, symmetric, no
act-ordering), and the **same checkpoint runs as either W4A16 or W4A8** — the
choice is a runtime toggle, not a re-quantization. W4A8 is an *inference-time
execution mode*: the int4 weights are reused as-is and only the activation is
dynamically quantized to s8 at runtime. This mirrors Intel's "enable W4A8 for
all WNA16 methods" approach (vLLM PR #43841). W4A8 is **enabled by default**;
set `VLLM_CPU_INT4_W4A8=0` to run the same checkpoint as W4A16 instead.

> **Note:**
> - DA8W4 is **symmetric only** (no zero-points). Asymmetric (`uint4`) or
>   activation-reordered (`g_idx`/`desc_act`) checkpoints are not supported and
>   fall back to the W4A16 path.
> - Source scale granularity is fixed to per-token `[M, 1]`.

## 2. Motivation

In W4A16 the int4 weights are de-quantized to bf16 on the fly and the matmul
runs as bf16×bf16, which becomes compute-bound at larger batch. DA8W4 instead
quantizes the activations to s8 **dynamically at runtime** (per-token, no
calibration) and runs the matmul as s8×s8 against the same int4 weights (widened
to s8) — trading a small per-step quant overhead for higher GEMM throughput that
pays off as batch grows.

## 3. API

### 3.1 Linear op

```
torch.ops.zentorch.zentorch_dynamic_qlinear(
    input,          # torch.bfloat16, shape [M, K] or [*, K]
    weight,         # DA8W4: packed s4, int8 [N, K/2] (or int32 [N, K/8])
    weight_scales,  # torch.float32 / torch.bfloat16, shape {G, N} (per-group)
    bias,           # torch.float32 / torch.bfloat16 or None
    *, zentorch_op_name='zentorch::zentorch_dynamic_qlinear'
) -> Tensor         # torch.bfloat16, shape [*, N]
```

There is **no mode selector**: DA8W4 is chosen when `weight` is a packed s4
tensor (int8 `[N, K/2]` or int32 `[N, K/8]`); an int8 `[N, K]` weight runs DA8W8.
This document covers the DA8W4 (packed-s4) path.

An **out variant** is also registered for buffer reuse / Inductor+AOTI memory
planning; it writes the same result into a caller-provided `out` tensor. Per the
repo convention, `out` is the first argument (mutable, `Tensor(a!)`) and the op
returns nothing (`-> ()`):

```
torch.ops.zentorch.zentorch_dynamic_qlinear.out(
    out,            # torch.bfloat16, contiguous, shape [*, N] (mutated in place)
    input, weight, weight_scales, bias=None, *,
    zentorch_op_name='zentorch::zentorch_dynamic_qlinear',
) -> ()             # no return; result written into `out`
```

### 3.2 Weight packing

No DA8W4-specific packing op is added. The existing WOQ repack is reused and its
int32 output reinterpreted as int8 (see §6.2):

```
torch.ops.zentorch.zentorch_woq_repack_weight(
    unpacked_weight   # torch.int8, shape [N, K], signed values in [-8, 7]
).view(torch.int8)    # torch.int8, shape [N, K/2] (2 nibbles per byte)
```

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor (bf16) | Activation tensor of shape `[*, K]`; dynamically quantized to s8 at runtime. FP32 is rejected. |
| `weight` | Tensor (int32) | Symmetric s4 weight, 8 int4 packed per int32, shape `[N, K/8]` (`[out, in]` orientation). |
| `weight_scales` | Tensor (f32/bf16) | Per-group weight scales of shape `{G, N}`, where `G = K / group_size` |
| `bias` | Tensor? (f32/bf16) | Optional bias vector of shape `[N]` |
| `zentorch_op_name` | str | Operator name for profiling/tracing |

The mode (DA8W8 vs DA8W4) is not a parameter — it is inferred from `weight`'s
dtype and K-dimension (see §1).

## 5. Constraints and Validation

| Constraint | Condition |
|-----------|-----------|
| Input dtype | Must be `torch.bfloat16` (FP32 rejected by the DA8W4 kernel) |
| Weight dtype | `torch.int8` (packed s4 `[N, K/2]`) or `torch.int32` (packed s4 `[N, K/8]`) |
| Weight shape | 2D; dim 1 must equal `K/2` (int8) or `K/8` (int32) |
| Weight scales dtype | `torch.float32` or `torch.bfloat16` |
| Weight scales shape | Per-group 2D `{G, N}` |
| Group size | `K % G == 0` and `(K / G) % 4 == 0` (AOCL sym_quant constraint) |
| Zero points | Not supported (symmetric only) |
| Output dtype | `torch.bfloat16` |
| Bias dtype (if provided) | `torch.float32` or `torch.bfloat16` |
| Input last dim | Must equal `K` = `pack_factor × weight dim 1` (2 for int8, 8 for int32) |
| Backend | AOCL-DLP (auto-routed by ZenDNN for W4A8) |

Mode inference / weight dtype-shape check (`check_weight_and_infer_is_da8w4`)
always runs (needed to dispatch). All other checks — dtypes, dims, contiguity,
and the out variant's `out` tensor — are **gated behind `ZENTORCH_ENABLE_CHECKS`**
(off by default); contiguity/shape are otherwise guaranteed by the replacement /
lowering.

## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_dynamic_qlinear(packed s4 weight)   # Public API entry point (DA8W4 inferred)
  ├─ check_weight_and_infer_is_da8w4(): infer mode from pack_factor (K / dim1)
  │   and validate the weight dtype (int8 for K or K/2, int32 for K/8)
  ├─ validate input/weight_scales/bias dtypes (both modes; DA8W4 is bf16-only)
  │   and bias size
  ├─ Reshape input/output to 2D, allocate bf16 output
  └─ zentorch_dynamic_qlinear_impl()
            ├─ Configure matmul_data_types (src=bf16, wei=s4, dst=bf16, compute=s8)
            ├─ Set params.dynamic_quant = true
            ├─ Keep params.lowoha_algo at its default (ZenDNN selects at runtime)
            ├─ Set src_scale.buff = nullptr, dims = {M, 1}  (per-token, runtime)
            ├─ Set wei_scale from weight_scales, dims = {G, N}  (per-group)
            └─ Call matmul_direct() with transB=true, ldb=K, is_weights_const=true
```

### 6.2 Weight Packing (reuses `zentorch_woq_repack_weight`)

The DA8W4 kernel reads the weight as raw **s4, 2 nibbles per byte** (low nibble =
first element, signed two's-complement `[-8, 7]`, no `+8` offset). This is
byte-for-byte identical to the existing WOQ repack output: `zentorch_woq_repack_weight`
packs 8 int4 per int32 as `(val_i & 0xF) << (4*i)`, which on little-endian
(x86/Zen) is the same byte stream as the s4 layout. So its int32 `[N, K/8]`
output needs no separate packing op — and since the bytes are identical, the
kernel accepts either the int32 `[N, K/8]` weight or its int8 `[N, K/2]` view
(`.view(torch.int8)`); the vLLM path passes the int8 view. No transpose is
needed — the `[N, K-dim]` orientation is consumed directly with `transB=true`.

### 6.3 ZenDNN LowOHA Integration

The operator uses the `matmul_direct` API with:
- `transB = true`, `ldb = K` (ldb counted in s4 elements/nibbles) since the
  weight is `[N, K]` packed s4
- `dynamic_quant = true` to enable runtime source quantization
- `lowoha_algo` ZenDNN auto-detects W4A8 (dynamic s8 × s4) and
  routes it to AOCL-DLP, honoring any runtime algo override
- `src_scale.buff = nullptr`, `dims = {M, 1}` — per-token scales computed at
  runtime
- `wei_scale.dims = {G, N}` — per-group weight scales
- `compute = data_type_t::s8` — s4 weights are widened to s8 and the matmul runs
  as s8×s8→bf16

### 6.4 Weight Caching

`is_weights_const = true` is passed to `matmul_direct`, so the LowOHA backend
caches the s4→s8 widened + reordered weight after the first call (dedicated
DA8W4 cache).

## 7. vLLM Integration

DA8W4 is wired into vLLM through `ZentorchWNA16LinearKernel`, selected by
`choose_mp_linear_kernel` ahead of the generic `CPUWNA16LinearKernel`. The W4A8
vs W4A16 path is chosen at runtime. DA8W4 is **enabled by default**; set
**`VLLM_CPU_INT4_W4A8=0`** to disable it and force the W4A16 path. Even when
enabled, only DA8W4-eligible layers take the path — everything else falls back
to W4A16 automatically (see below).

- **`process_weights_after_loading`**: when DA8W4-eligible (DA8W4 not disabled,
  symmetric `uint4b8`, no `g_idx`, bf16 activations, `group_size % 4 == 0`),
  unpack the checkpoint int4 → repack to s4 `[N, K/2]` via
  `zentorch_woq_repack_weight(...).view(torch.int8)`, and transpose scales
  `[N, G] → {G, N}` (bf16). Otherwise fall back to the W4A16 WOQ path (or
  `super()`).
- **`apply_weights`**: cast activation to bf16 and call
  `zentorch_dynamic_qlinear(...)` with the packed s4 weight (DA8W4 inferred).

## 8. Comparison with Existing Operators

| Feature | `zentorch_dynamic_qlinear` (DA8W8, int8 `[N, K]`) | `zentorch_dynamic_qlinear` (DA8W4, packed s4) |
|---------|---------------------------------|----------------------------------|
| Input dtype | bf16 or f32 | bf16 only (f32 rejected) |
| Weight dtype / layout | int8, `[N, K]` | packed s4: int8 `[N, K/2]` or int32 `[N, K/8]` |
| Weight scales | per-channel `{1, N}` | per-group `{G, N}` |
| Source quantization | dynamic per-token s8 | dynamic per-token s8 |
| Compute | s8×s8 | s8×s8 (s4 widened to s8) |
| Zero points | Not needed (symmetric) | Not needed (symmetric) |

## 9. Reference

This operator is based on **Example 8** (DA8W4: dynamic BF16→s8 activation ×
symmetric s4 weight) from the ZenDNN LowOHA MatMul Operator documentation, and
uses the `matmul_direct` API from ZenDNN's LowOHA backend.
