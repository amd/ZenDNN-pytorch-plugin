(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch_group_matmul — Parallel Group MatMul with MoE Post-Ops (Out Variant)

## 1. Overview

`zentorch_group_matmul.out` is a parallel group matrix multiplication operator that executes multiple independent GEMMs in a single call using the ZenDNN LowOHA `group_matmul_direct` backend. It follows the **out variant** pattern — the caller pre-allocates all output tensors and the operator writes results directly into them.

It is designed for Mixture of Experts (MoE) inference and supports three optional post-ops that can be composed together:

These post-ops execute in the following order when combined:

1. **Gated activation** — applied after the gate+up GEMM; fuses SiLU/GELU/SwigluOAI activation
2. **Fused down projection (w2)** — applied after activation; fuses the down projection GEMM
3. **MoE weighted-reduce** — applied last; blends expert outputs per token using routing weights

When all three are combined, the entire MoE FFN block (gate+up → activation → down → weighted-reduce) executes in a single API call.

> **Note:**
> - Only **parallel mode** is supported. Sequential mode (chained matmuls) is not implemented.
> - Weight tensors follow `nn.Linear` layout `[N, K]` and **must be contiguous**. For packed-s4 weights the requirement is load-bearing in either container, since the packed metadata takes `ldb`(`K`) from the activation's contraction dim rather than `stride(0)`.
> - Weights may be `bf16`/`f32`, full-width `int8` (dynamic-A8W8, per-channel scale), or **packed `s4` (DA8W4)** with per-group scales, in either of two byte-identical containers: `int32 [N, K/8]` (8 nibbles per int32) or `int8 [N, K/2]` (2 nibbles per byte). The DA8W8 and DA8W4 paths share the dynamic per-token activation-quant wiring; see [§6.4](#64-dynamic-quantization--da8w8-and-da8w4).
> - `int8` therefore serves both quantized regimes and is disambiguated by its last dim against the unpacked `K`: full width is DA8W8, half width is packed s4.

## 2. Motivation

In MoE models (Mixtral, DeepSeek, etc.), a router assigns each token to its top-k experts. Each expert runs an independent FFN (gate, up, and down projections) on its assigned tokens. Naive execution calls `torch.nn.functional.linear` in a loop — one call per expert per projection — incurring repeated function-call overhead and poor thread utilization.

`zentorch_group_matmul.out` addresses this by:
- **Batching all expert GEMMs** into a single `group_matmul_direct` call
- **Gated activation fusion** (`activation`): fuses SiLU/GELU/SwigluOAI activation (strings `"silu"`, `"gelu"`, `"gelu_tanh"`, `"swigluoai"`) with the gate+up projection, avoiding a separate activation kernel and memory round-trip
- **Fused down projection** (`w2_weights`, `w2_bias`): chains the down projection GEMM into the same call, eliminating a second kernel launch between the activated intermediate and the down projection. ZenDNN manages the output buffers internally by reusing the input buffers
- **MoE weighted-reduce fusion** (`moe_output`, `topk_weights`, `row_ptrs`): blends expert outputs into per-token results using router weights, avoiding a separate reduce kernel and an extra read over all expert output buffers
- **Out variant pattern**: the caller controls output memory allocation and can reuse buffers across inference steps

## 3. API

### Signature

```python
torch.ops.zentorch.zentorch_group_matmul.out(
    gemm_outputs,           # List[Tensor], pre-allocated [M_i, N] per expert (or [] for internal alloc)
    inputs,                 # List[Tensor], one [M_i, K] per expert (bf16/f32/fp16)
    w13_weights,            # List[Tensor], one [N, K] per expert (w13: gate+up; bf16/f32/fp16,
                            #   int8 DA8W8, or packed s4: int32 [N, K/8] / int8 [N, K/2])
    w2_weights,             # List[Optional[Tensor]], one [K_out, D] per expert ([] when unused)
    moe_output,             # Optional[Tensor], [num_tokens, hidden_dim] (MoE reduce result)
    topk_weights,           # Optional[Tensor], [num_tokens, topk] routing weights (f32)
    row_ptrs,               # Optional[Tensor], [num_tokens * topk]
    activation,             # str: 'none', 'silu', 'gelu', 'gelu_tanh', 'swigluoai'
    w13_bias,               # List[Optional[Tensor]], one [N] or None per expert
    w2_bias,                # List[Optional[Tensor]], one [K_out] or None per expert ([] when unused)
    w13_scales,      # List[Optional[Tensor]], per-expert scales ([] for fp32/bf16, required for DA8W8 / DA8W4 s4)
    w2_scales,       # List[Optional[Tensor]], per-expert scales for DA8W8 / DA8W4 s4 w2 ([] for fp32/bf16)
    *, zentorch_op_name='zentorch::zentorch_group_matmul.out'
) -> None
```

> **Note:** All positional parameters are required. Pass `[]` for unused list parameters
> (`w2_weights`, `w2_bias`, `w13_bias`, `w13_scales`, `w2_scales`) and `None` for
> unused optional parameters (`moe_output`, `topk_weights`, `row_ptrs`).
> `zentorch_op_name` is keyword-only with a default.

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `gemm_outputs` | `List[Tensor]` (bf16/f32/fp16) | Pre-allocated Op1 output tensors, one per expert, shape `[M_i, N]`. Pass `[]` when intermediate GEMM results are not needed — ZenDNN allocates and manages dst buffers internally. |
| `inputs` | `List[Tensor]` (bf16/f32/fp16) | One input tensor per expert, shape `[M_i, K]`. |
| `w13_weights` | `List[Tensor]` (bf16/f32/fp16/int8/int32) | Op1 weight matrices (w13: gate+up), shape `[N, K]` (nn.Linear layout). For **DA8W4** the tensor is packed s4 — `[N, K/8]` int32 or `[N, K/2]` int8 — and the logical `K` is the activation's last dim (`size(1) * pack_factor`). |
| `w2_weights` | `List[Optional[Tensor]]` (bf16/f32/fp16/int8/int32) | Down projection weights, one `[K_out, D]` per expert. `D = N/2` after gated activation, `D = N` without. For **DA8W4** the tensor is packed s4 in the same container as `w13` — `[K_out, D/8]` int32 or `[K_out, D/2]` int8 — and `D` is taken from `w13`, not from `size(1)`. Pass `[]` when unused. |
| `moe_output` | `Optional[Tensor]` (bf16/f32/fp16) | Pre-allocated `[num_tokens, hidden_dim]` for weighted-reduce result. |
| `topk_weights` | `Optional[Tensor]` (f32) | Routing weights `[num_tokens, topk]`. |
| `row_ptrs` | `Optional[Tensor]` (int64) | Pre-built pointer table `[num_tokens * topk]` into the final expert output buffers. When fused w2 is active, must point into the input buffers (ZenDNN reuses them for w2 output). |
 | `activation` | `str` | `'none'`, `'silu'`, `'gelu'`, `'gelu_tanh'`, or `'swigluoai'`. Maps to `silu_and_mul`, `gelu_and_mul`, `swiglu_oai_mul` enums internally (`gelu_tanh` aliases `gelu_and_mul`; `gelu`/`gelu_tanh` use tanh-approx GELU in fused kernels). Gated activations require `N = 2*D` (even). || `w13_bias` | `List[Optional[Tensor]]` | Op1 bias, one `[N]` or `None` per expert. |
| `w2_bias` | `List[Optional[Tensor]]` | Down projection bias, one `[K_out]` or `None` per expert. Pass `[]` when unused. |
| `w13_scales` | `List[Optional[Tensor]]` (f32/bf16) | Per-expert weight scales for dynamic A8W8 / A8W4 w13. Dynamic-A8W8: `[N]` (per-channel, normalized to `{1, N}`). **Dynamic-A8W4**: `{G, N}` per-group, passed through unchanged (same for either container). Pass `[]` for fp32/bf16/fp16 weights. |
| `w2_scales` | `List[Optional[Tensor]]` (f32/bf16) | Per-expert weight scales for dynamic A8W8 / A8W4 w2. Dynamic-A8W8: `[K_out]` (per-channel). **Dynamic-A8W4**: `{G, K_out}` per-group, passed through unchanged. Pass `[]` for fp32/bf16/fp16 w2 weights. |

> **DA8W4 detection.** The regime is inferred from `w13_weights[0]` by the shared classifier `check_weight_and_infer_is_da8w4` (`DynamicQLinear.hpp`), with no extra schema arg: it divides the unpacked `K` (`inputs[0].size(1)`) by the weight's last dim and reads the pack factor — `1` → DA8W8 (full-width int8), `2` → DA8W4 int8 container, `8` → DA8W4 int32 container. Any other ratio, or a pack factor paired with the wrong dtype, is rejected; a floating-point weight is classified as unquantized and its `K` checked separately.
>
> `w2_weights` inherit the regime rather than re-deriving it: they are packed s4 exactly when `w13` is and their dtype matches. Their own packed density is **not** checked, because the down-projection input dim (`N/2` with gated act, `N` without) need not be a multiple of the pack factor even for a validly packed weight. `ldb_down` still comes from that dim, so a packed `w2` must be contiguous. See [§6.4](#64-dynamic-quantization--da8w8-and-da8w4).

> **Note:** `w2_outputs` is not required — ZenDNN manages the down projection output buffers internally by reusing the input buffers.

## 5. Constraints

| Constraint | Condition |
|-----------|-----------|
| Execution mode | Parallel only (`len(inputs) > 1`) |
| Input dtype | `torch.bfloat16`, `torch.float32`, or `torch.float16` (fp16 requires AVX-512 FP16 hardware support; DA8W8 and DA8W4 are **bf16-only** — see the rows below) |
| Weight dtype | Must match input dtype, `torch.int8` at full `K` (dynamic A8W8), or packed s4 (`torch.int32` at `K/8`, `torch.int8` at `K/2`) |
| Dynamic A8W8 | When `w13_weights[i]` is full-width int8, `w13_scales[i]` is required and the activation **must be `torch.bfloat16`**. Kernel quantizes activations at runtime (`dynamic_quant=true`, `dtypes.compute=s8`), per-channel `{1, N}` weight scale |
| Dynamic A8W4 | When `w13_weights[i]` is packed s4 (`int32` at `K/8` or `int8` at `K/2`), `w13_scales[i]` is required (per-group `{G, N}`), and the activation **must be `torch.bfloat16`** (the ZenDNN DA8W4 kernel rejects `float32` and `float16`). The wrapper sets `dtypes.wei=s4`, `dynamic_quant=true`, `dtypes.compute=s8`; ZenDNN backend then selects the s4 kernel from `dtypes.wei=s4`. Logical `K` is the input's last dim (`ldb = K`, in nibble units, identical for both containers). See [§6.4](#64-dynamic-quantization--da8w8-and-da8w4) |
| Weight regime consistency | All `w13_weights` must share one dtype (the op infers the fp/DA8W8/DA8W4 regime from `w13_weights[0]`); all `w2_weights` must share one dtype, which must equal the `w13` dtype — so an int32-packed `w13` cannot pair with an int8-packed `w2`, and vice versa |
| Contiguity | All `w13` and `w2` weights must be **contiguous**. For packed s4 (either container) this is load-bearing: `ldb`/`ldb_down` derive from the activation's contraction dim, not `stride(0)` |
| Dtype consistency (fp) | For fp32/bf16/fp16 weights: inputs, w13_weights, w13_bias must share dtype per expert |
| Weight shape | `[N, K]` (nn.Linear layout); DA8W4 packed weights are `[N, K/8]` int32 or `[N, K/2]` int8, with logical `K = size(1) * pack_factor` |
| gemm_outputs | Either empty `[]` (ZenDNN allocates internally) or `len(gemm_outputs) == len(w13_weights)` |
| Gated activation | Requires `N` to be even (`N = 2 * D`) |
| MoE params | When `topk_weights` is provided, `row_ptrs` and `moe_output` must also be provided |
| Fused w2 params | `w2_weights` and `w2_bias` must both be provided or both be `[]` |
| w2 inner dim | For fp and DA8W8: `w2_weights[i].size(1)` must equal `N/2` (with gated act) or `N` (without). For DA8W4 this is **not** checked — the logical inner dim is taken from `w13` and `w2`'s packed last dim is left alone |
| w2 list lengths | Must equal `len(w13_weights)` (one per expert) |
| w2 dtype | `w2_weights[i]` must match input dtype, be full-width int8 (with `w2_scales`), or be packed s4 in `w13`'s container (with per-group `w2_scales`). `w2_bias[i]` must match input dtype |
| Buffer reuse (K==K_out) | When fused w2 is active, `K_out` must equal `K` for the kernel to safely write w2 output back into input buffers |

## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_group_matmul_out_impl()
  ├─ Parse activation string → gated_act enum, compute use_gated_act
  │     Supported: "none", "silu", "gelu", "gelu_tanh", "swigluoai"
  │     (mapped to silu_and_mul, gelu_and_mul, swiglu_oai_mul enums)
  ├─ validate_all_inputs() [gated by EnvReader::getEnvVariableAsInt("ZENTORCH_ENABLE_CHECKS")]:
  │     ├─ validate_dtypes_and_shapes (inputs, w13_weights, w13_bias, w13_scales)
  │     ├─ validate_gemm_outputs (if non-empty, size must match weights)
  │     ├─ validate_w2_params (if non-empty: list sizes, per-expert shapes, dtypes)
  │     └─ validate_moe_params (topk_weights → row_ptrs + moe_output required)
  ├─ Single-pass loop: extract dimensions, pointers, dtypes for Op1
  │     ├─ If gemm_outputs empty: dst_ptrs stays nullptr (ZenDNN allocates internally)
  │     ├─ If weight is packed s4 (DA8W4, int32 [N,K/8] or int8 [N,K/2]):
  │     │     dtypes.wei=s4, K=ldb=input's last dim (pack factor validated by
  │     │     check_weight_and_infer_is_da8w4; else: dtypes.wei from tensor dtype,
  │     │     K=size(1), ldb=stride(0))
  │     └─ If weight is full-width int8 OR DA8W4: set dynamic_quant=true, compute=s8,
  │           populate quant_params
  ├─ Configure gated activation post-op
  ├─ If MoE: populate group_matmul_moe_postop_params
  ├─ If fused w2: populate grp_matmul_fused_moe_params
  │     ├─ DA8W4: ldb_down = down-proj input dim K_down (taken from w13, not validated
  │     │     against w2's last dim); else ldb_down = stride(0)
  │     └─ If w2_scales non-empty: populate fused_moe.down_scale (Op2 weight scale)
  └─ Call group_matmul_direct(... moe_params, gated_act, fused_moe)
       ├─ Op1: Parallel expert GEMMs (gate+up) → gemm_outputs (or internal buffers)
       ├─ Gated activation (if enabled) → first D columns
       ├─ Op2: Down projection (if fused) → reuses input buffers
       └─ MoE weighted-reduce (if enabled) → moe_output
```

### 6.2 Fused MoE Pipeline

When all post-ops are combined, the kernel executes the full MoE FFN in one call:

```
For each expert:
  Step 1 (Op1):  input[M,K] @ w13[2D,K].T  →  [M, 2D]     (gate+up GEMM)
  Step 2 (Act):  SiLU(gate) × up            →  [M, D]       (gated activation)
  Step 3 (Op2):  activated[M,D] @ w2[K_out,D].T → [M, K_out] (down projection)

After all experts complete:
  Step 4 (MoE):  weighted-reduce across experts → [num_tokens, K_out]
```

### 6.3 ZenDNN LowOHA Integration

Hardcoded defaults for Op1:

| Parameter | Value |
|-----------|-------|
| `layout` | `'r'` (row-major) |
| `transA` | `false` |
| `transB` | `true` (nn.Linear `[N, K]`) |
| `alpha` | `1.0` |
| `beta` | `0.0` |
| `is_weights_const` | `true` |

### 6.4 Dynamic quantization — DA8W8 and DA8W4

Two weight types drive the dynamic-quant path, where the bf16/fp32 activation is quantized to `s8` **per token at runtime** (`dynamic_quant = true`, `dtypes.compute = s8`) rather than offline. They are detected from the Op1 weight and share most of the wiring; the differences are summarized below.

| Aspect | Dynamic-A8W8 (`s8` weight) | Dynamic-A8W4 (packed `s4` weight) |
|--------|----------------------------|-----------------------------------|
| Detection | pack factor `unpacked_K / size(1) == 1` with `kChar` | pack factor `2` with `kChar` (int8 container) or `8` with `kInt` (int32 container) |
| `dtypes.wei` | `s8` (from tensor dtype) | `s4` — set **explicitly** (`get_zendnnl_dtype` would return `s32` / `s8` for the packed buffer) |
| Weight buffer | `[N, K]` int8, `ldb = stride(0)` | `[N, K/8]` int32 or `[N, K/2]` int8 — the same nibble stream either way; logical `K` = input's last dim, `ldb = K` (the unpacked K nibble-stream leading dim), `transB = true`, contiguous required |
| Weight scale | `[N]` → normalized to per-channel `{1, N}` | per-group `{G, N}` — **passed through unchanged** (the ZenDNN's sym-quant kernel derives the source group size from the `G` scale rows) |
| Source scale | `{M, 1}` per-token (the wrapper always allocates `src_scale.dims = {M, 1}`) | `{M, 1}` per-token; ZenDNN's DA8W4 path broadcasts it across the `G` weight-scale groups internally |
| Kernel / algo | LowOHA default | LowOHA default |

Because the compute-side sizing is all in nibble units, the container affects nothing past detection: `K`, `ldb`, `dtypes.wei`, and the scales are identical for an int32-packed and an int8-packed weight built from the same quantized values. The int8 container only relaxes the shape requirement, needing `K % 2 == 0` instead of `K % 8 == 0`.

**Op2 (fused w2).** When `w2_weights` are packed s4, `fused_moe.ldb_down[i]` is set to the unpacked `K_down` — the down-projection input dim (`N/2` with gated act, `N` without), derived from `w13_weights[i]`. Unlike the W13 metadata this is **not** cross-checked against `w2_weights[i].size(1)`: `K_down` need not be a multiple of the pack factor, so the regime is inherited from `w13` (same dtype ⇒ same container) instead. Op2 inherits `dynamic_quant`, `dtypes.compute`, `dtypes.wei`, and `src_scale.dims` from `params[i]`; only its per-group weight scale (`fused_moe.down_scale`) is per-pass.

The int4 weight layout and per-group scales handed in here are identical to the single-matmul WOQ path (`zentorch_woq_linear_impl`); only the dispatch (grouped vs. single matmul) differs. This is the path the DA8W4 regime of the fused-MoE op relies on — see [zentorch_fused_moe.md](./zentorch_fused_moe.md) §9.

## 7. Test Plan

Tests live in `test/unittests/op_tests/test_group_matmul.py`. The class `Test_GroupMatmul` extends `GroupMatmulTestCase`.

### 7.1 Hypothesis strategy

Tests are **Hypothesis-based** decorated with
`@GroupMatmulTestCase.hypothesis_params_group_matmul_itr(...)` (defined in
`test/unittests/unittest_utils.py`), which wraps the composite strategy
`tensor_group_matmul_strategy`. For every generated example the strategy draws a random
dtype plus a full set of dimensions and records a `tensor_seed`; the per-example seed is
applied via `torch.manual_seed(tensor_seed_val)` (and numpy/random) so any failing example
is fully reproducible from the seed printed in the failure decorator.

Dimensions are drawn from the constants in `zentorch_test_utils.py`:

| Dimension | Source constant |
|-----------|-----------------|
| `num_experts` | `GROUP_MATMUL_NUM_EXPERTS` |
| `M` | `GROUP_MATMUL_M_VALUES` |
| `K` | `GROUP_MATMUL_K_VALUES` |
| `N` | `GROUP_MATMUL_N_VALUES` |
| `D` | `GROUP_MATMUL_D_VALUES` |
| `K_out` | `GROUP_MATMUL_K_OUT_VALUES`|
| `topk` | `GROUP_MATMUL_TOPK_VALUES`|
| `num_tokens` | `GROUP_MATMUL_NUM_TOKENS_VALUES`|

`K` is drawn with `st.sampled_from(k_list)`; the DA8W8 tests override `k_list` with
`GROUP_MATMUL_INT8_K_VALUES = [4, 8]` or `GROUP_MATMUL_INT8_GATED_K_VALUES = [8, 16]` to satisfy
their tighter shape constraints. Dtype is supplied via `dtype_list=supported_dtypes`
(`"float32"`, plus `"bfloat16"` when BF16 is supported, plus `"float16"` when AVX-512 FP16 is
supported; the int8 tests use `supported_dtypes_int8`, which excludes `"float16"`).
`GroupMatmulTestCase` sets `max_example_per_test = 5` and
`time_out = 10000` ms — fewer examples and a longer deadline than the default because each
example builds full per-expert w13/w2 weight, bias, and scale tensors.

The DA8W4 tests reuse `num_experts`, `M`, `topk` and `num_tokens` from the table above
and draw only their own contraction dims, since int32-packed s4 needs every K dim to be
a multiple of 8 and of the group size (>= 2 groups). These draws default to empty and are
skipped entirely unless a test supplies them, so only `Test_FusedMoEDA8W4`
(`test/unittests/op_tests/test_fused_moe_da8w4.py`) pays for them — it passes:

| Draw | Value supplied by the test |
|------|----------------------------|
| `hidden` | `hidden_list=GROUP_MATMUL_DA8W4_HIDDEN_VALUES` |
| `inter` | `inter_list=GROUP_MATMUL_DA8W4_INTER_VALUES` |
| `group_size` | `group_size_list=GROUP_MATMUL_DA8W4_GROUP_SIZE_VALUES` |

When those dims are drawn, the example's packed weights are built once by
`build_quant_moe_data`; otherwise `group_matmul_quant_moe_data` is `None`.


### 7.2 Test matrix for `zentorch_group_matmul.out`

| Test | Post-ops | Notes |
|------|----------|-------|
| `test_plain_gemm` | None (bare GEMM) | Parallel expert GEMMs only |
| `test_moe_weighted_reduce` | MoE weighted-reduce | GEMM + per-token reduce |
| `test_gated_activations` | Gated activation (silu, gelu, swigluoai) | Loops over all activations per example |
| `test_int8_w13` | Dynamic A8W8 w13 (bare GEMM) | `k_list = [4, 8]`; needs AVX512 + bf16 |
| `test_int8_w13_and_w2_single_pass` | Dynamic A8W8 w13 + w2, 3 sub-tests (see below) | `k_list = [4, 8]`, `K == K_out == N` |
| `test_int8_w13_and_w2_two_pass` | DA8W8 w13 + silu + w2 + MoE reduce via `zentorch_fused_moe` | `k_list = [8, 16]`, `K == K_out`; exercises the `ZENTORCH_TWO_PASS` split path when run with `ZENTORCH_TWO_PASS=1`|
| `test_unsupported_activation` | Invalid activation strings | Expects `RuntimeError` |
| `test_int8_missing_scales` | DA8W8 weights with None scales (negative) | Requires `ZENTORCH_ENABLE_CHECKS=1` set before process start (`@unittest.skipUnless`); `k_list = [4, 8]` |
| `test_empty_gemm_outputs_fused_w2` | Fused w2 with gemm_outputs=[] | Backend allocates dst internally |
| `test_fused_moe_pipeline` | Full pipeline: w13 → act → w2 → MoE reduce | Verifies both `zentorch_group_matmul.out` and `zentorch_fused_moe` paths |

#### `test_int8_w13_and_w2_single_pass` sub-tests

| Sub-test | API | Post-ops | Detail |
|----------|-----|----------|--------|
| 1 | `zentorch_group_matmul.out` | No activation, no MoE reduce | Per-expert DA8W8 w13 + DA8W8 w2. Kernel writes w2 output back into inputs. |
| 2 | `zentorch_group_matmul.out` | silu activation + MoE weighted reduce | Uses `row_ptrs` into the input buffers for fused w2 buffer reuse |
| 3 | `zentorch_fused_moe` | No activation + MoE weighted reduce | High-level fused_moe op with DA8W8 weights + scales |

All sub-tests use `K == K_out == N` (buffer reuse constraint).


#### DA8W4 coverage

The packed-s4 (DA8W4) grouped path has no dedicated `zentorch_group_matmul` test; it
is covered end-to-end through the DA8W4 regime of the fused-MoE op in
`test/unittests/op_tests/test_fused_moe_da8w4.py`
(see [zentorch_fused_moe.md](./zentorch_fused_moe.md) §9).

| Test | Post-ops | Covers |
|------|----------|--------|
| `test_fused_moe_da8w4_accuracy` | silu + fused w2 + MoE reduce (optional bias) | Op1 and Op2 DA8W4 branches: bf16 activations × packed s4 with per-group scales |
| `test_fused_moe_da8w4_requires_bf16` | — | DA8W4 rejects non-bf16 activations |


### 7.3 Known limitations

| Limitation | Detail |
|------------|--------|
| Mixed DA8W8 / DA8W4 across Op1/Op2 | Unsupported — Op2 inherits Op1’s quantization configuration, so `w13` and `w2` must use the same quantization regime |
| Mixed packed-s4 containers across Op1/Op2 | Unsupported — the dtype-equality check rejects an int32-packed `w13` with an int8-packed `w2` (and vice versa), even though the two byte streams are identical |

## 8. Reference

This operator wraps the `group_matmul_direct` API from the [LowOHA Group MatMul Operator](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_matmul_operator.md), using:
- Parallel execution mode
- Optional `group_matmul_moe_postop_params` for weighted-reduce
- Optional `grp_matmul_gated_act_params` for gated activations
- Optional `grp_matmul_fused_moe_params` for fused down projection (Op2)

Related operators:
- Fused MoE consumer (bf16 / DA8W8 / DA8W4): [zentorch_fused_moe.md](./zentorch_fused_moe.md) (DA8W4 details in §9).
- Single-matmul WOQ / DA8W4 linear op sharing the packed-s4 layout(dynamic BF16→s8 activation × symmetric s4 weight).
