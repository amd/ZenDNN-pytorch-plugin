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
> - Weight tensors follow `nn.Linear` layout `[N, K]` and need not be contiguous.

## 2. Motivation

In MoE models (Mixtral, DeepSeek, etc.), a router assigns each token to its top-k experts. Each expert runs an independent FFN (gate, up, and down projections) on its assigned tokens. Naive execution calls `torch.nn.functional.linear` in a loop — one call per expert per projection — incurring repeated function-call overhead and poor thread utilization.

`zentorch_group_matmul.out` addresses this by:
- **Batching all expert GEMMs** into a single `group_matmul_direct` call
- **Gated activation fusion** (`activation`): fuses SiLU/GELU/SwigluOAI activation (strings `"silu"`, `"gelu"`, `"swigluoai"`) with the gate+up projection, avoiding a separate activation kernel and memory round-trip
- **Fused down projection** (`w2_weights`, `w2_bias`): chains the down projection GEMM into the same call, eliminating a second kernel launch between the activated intermediate and the down projection. ZenDNN manages the output buffers internally by reusing the input buffers
- **MoE weighted-reduce fusion** (`moe_output`, `topk_weights`, `row_ptrs`): blends expert outputs into per-token results using router weights, avoiding a separate reduce kernel and an extra read over all expert output buffers
- **Out variant pattern**: the caller controls output memory allocation and can reuse buffers across inference steps

## 3. API

### Signature

```python
torch.ops.zentorch.zentorch_group_matmul.out(
    gemm_outputs,           # List[Tensor], pre-allocated [M_i, N] per expert (or [] for internal alloc)
    inputs,                 # List[Tensor], one [M_i, K] per expert (bf16/f32)
    w13_weights,            # List[Tensor], one [N, K] per expert (w13: gate+up weights)
    w2_weights,             # List[Optional[Tensor]], one [K_out, D] per expert ([] when unused)
    moe_output,             # Optional[Tensor], [num_tokens, hidden_dim] (MoE reduce result)
    topk_weights,           # Optional[Tensor], [num_tokens, topk] routing weights (f32)
    row_ptrs,               # Optional[Tensor], [num_tokens * topk]
    activation,             # str: 'none', 'silu', 'gelu', 'swigluoai'
    w13_bias,               # List[Optional[Tensor]], one [N] or None per expert
    w2_bias,                # List[Optional[Tensor]], one [K_out] or None per expert ([] when unused)
    w13_scales,      # List[Optional[Tensor]], per-expert scales ([] for fp32/bf16, required for int8)
    w2_scales,       # List[Optional[Tensor]], per-expert scales for int8 w2 ([] for fp32/bf16)
    *, zentorch_op_name='zentorch::zentorch_group_matmul.out'
) -> None
```

> **Note:** All parameters are required positional parameters. Pass `[]` for unused list parameters
> (`w2_weights`, `w2_bias`, `w13_bias`, `w13_scales`, `w2_scales`) and `None` for
> unused optional parameters (`moe_output`, `topk_weights`, `row_ptrs`).

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `gemm_outputs` | `List[Tensor]` (bf16/f32) | Pre-allocated Op1 output tensors, one per expert, shape `[M_i, N]`. Pass `[]` when intermediate GEMM results are not needed — ZenDNN allocates and manages dst buffers internally. |
| `inputs` | `List[Tensor]` (bf16/f32) | One input tensor per expert, shape `[M_i, K]`. |
| `w13_weights` | `List[Tensor]` (bf16/f32/int8) | Op1 weight matrices (w13: gate+up), shape `[N, K]` (nn.Linear layout). |
| `w2_weights` | `List[Optional[Tensor]]` (bf16/f32/int8) | Down projection weights, one `[K_out, D]` per expert. `D = N/2` after gated activation, `D = N` without. Pass `[]` when unused. |
| `moe_output` | `Optional[Tensor]` (bf16/f32) | Pre-allocated `[num_tokens, hidden_dim]` for weighted-reduce result. |
| `topk_weights` | `Optional[Tensor]` (f32) | Routing weights `[num_tokens, topk]`. |
| `row_ptrs` | `Optional[Tensor]` (int64) | Pre-built pointer table `[num_tokens * topk]` into the final expert output buffers. When fused w2 is active, must point into the input buffers (ZenDNN reuses them for w2 output). |
| `activation` | `str` | `'none'`, `'silu'`, `'gelu'`, or `'swigluoai'`. Maps to `silu_and_mul`, `gelu_and_mul`, `swiglu_oai_mul` enums internally. Gated activations require `N = 2*D` (even). |
| `w13_bias` | `List[Optional[Tensor]]` | Op1 bias, one `[N]` or `None` per expert. |
| `w2_bias` | `List[Optional[Tensor]]` | Down projection bias, one `[K_out]` or `None` per expert. Pass `[]` when unused. |
| `w13_scales` | `List[Optional[Tensor]]` (f32/bf16) | Per-expert quantization scales for dynamic int8 w13. Shape `[N]` (per-channel, normalized to `{1,N}`) or `{G, N}` (per-group). Pass `[]` for fp32/bf16 weights. |
| `w2_scales` | `List[Optional[Tensor]]` (f32/bf16) | Per-expert quantization scales for dynamic int8 w2 weights. Shape `[K_out]` (per-channel) or `{G, K_out}` (per-group). Pass `[]` for fp32/bf16 w2 weights. |

> **Note:** `w2_outputs` is not required — ZenDNN manages the down projection output buffers internally by reusing the input buffers.

## 5. Constraints

| Constraint | Condition |
|-----------|-----------|
| Execution mode | Parallel only (`len(inputs) > 1`) |
| Input dtype | `torch.bfloat16` or `torch.float32` |
| Weight dtype | Must match input dtype, or `torch.int8` (dynamic int8 quantization) |
| Dynamic int8 | When `w13_weights[i]` is int8, `w13_scales[i]` is required. Kernel quantizes activations at runtime (`dynamic_quant=true`, `dtypes.compute=s8`) |
| Dtype consistency (fp) | For fp32/bf16 weights: inputs, w13_weights, w13_bias must share dtype per expert |
| Weight shape | `[N, K]` (nn.Linear layout) |
| gemm_outputs | Either empty `[]` (ZenDNN allocates internally) or `len(gemm_outputs) == len(w13_weights)` |
| Gated activation | Requires `N` to be even (`N = 2 * D`) |
| MoE params | When `topk_weights` is provided, `row_ptrs` and `moe_output` must also be provided |
| Fused w2 params | `w2_weights` and `w2_bias` must both be provided or both be `[]` |
| w2 inner dim | `w2_weights[i].size(1)` must equal `N/2` (with gated act) or `N` (without) |
| w2 list lengths | Must equal `len(w13_weights)` (one per expert) |
| w2 dtype | `w2_weights[i]` must match input dtype or be int8 (with `w2_scales`). `w2_bias[i]` must match input dtype |
| Buffer reuse (K==K_out) | When fused w2 is active, `K_out` must equal `K` for the kernel to safely write w2 output back into input buffers |

## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_group_matmul_out_impl()
  ├─ Parse activation string → gated_act enum, compute use_gated_act
  │     Supported: "none", "silu", "gelu", "swigluoai"
  │     (mapped to silu_and_mul, gelu_and_mul, swiglu_oai_mul enums)
  ├─ validate_all_inputs() [gated by EnvReader::getEnvVariableAsInt("ZENTORCH_ENABLE_CHECKS")]:
  │     ├─ validate_dtypes_and_shapes (inputs, w13_weights, w13_bias, w13_scales)
  │     ├─ validate_gemm_outputs (if non-empty, size must match weights)
  │     ├─ validate_w2_params (if non-empty: list sizes, per-expert shapes, dtypes)
  │     └─ validate_moe_params (topk_weights → row_ptrs + moe_output required)
  ├─ Single-pass loop: extract dimensions, pointers, dtypes for Op1
  │     ├─ If gemm_outputs empty: dst_ptrs stays nullptr (ZenDNN allocates internally)
  │     └─ If weight is int8: set dynamic_quant=true, compute=s8, populate quant_params
  ├─ Configure gated activation post-op
  ├─ If MoE: populate group_matmul_moe_postop_params
  ├─ If fused w2: populate grp_matmul_fused_moe_params
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

`K` is drawn with `st.sampled_from(k_list)`; the int8 tests override `k_list` with
`GROUP_MATMUL_INT8_K_VALUES = [4, 8]` or `GROUP_MATMUL_INT8_GATED_K_VALUES = [8, 16]` to satisfy
their tighter shape constraints. Dtype is supplied via `dtype_list=supported_dtypes`
(`"float32"`, plus `"bfloat16"` when BF16 is supported). `GroupMatmulTestCase` sets `max_example_per_test = 5` and
`time_out = 10000` ms — fewer examples and a longer deadline than the default because each
example builds full per-expert w13/w2 weight, bias, and scale tensors.


### 7.2 Test matrix for `zentorch_group_matmul.out`

| Test | Post-ops | Notes |
|------|----------|-------|
| `test_plain_gemm` | None (bare GEMM) | Parallel expert GEMMs only |
| `test_moe_weighted_reduce` | MoE weighted-reduce | GEMM + per-token reduce |
| `test_gated_activations` | Gated activation (silu, gelu, swigluoai) | Loops over all activations per example |
| `test_int8_w13` | Dynamic int8 w13 (bare GEMM) | `k_list = [4, 8]`; needs AVX512 + bf16 |
| `test_int8_w13_and_w2_single_pass` | Dynamic int8 w13 + w2, 3 sub-tests (see below) | `k_list = [4, 8]`, `K == K_out == N` |
| `test_int8_w13_and_w2_two_pass` | int8 w13 + silu + w2 + MoE reduce via `zentorch_fused_moe` | `k_list = [8, 16]`, `K == K_out`; exercises the `ZENTORCH_TWO_PASS` split path when run with `ZENTORCH_TWO_PASS=1`|
| `test_unsupported_activation` | Invalid activation strings | Expects `RuntimeError` |
| `test_int8_missing_scales` | int8 weights with None scales (negative) | Requires `ZENTORCH_ENABLE_CHECKS=1` set before process start (`@unittest.skipUnless`); `k_list = [4, 8]` |
| `test_empty_gemm_outputs_fused_w2` | Fused w2 with gemm_outputs=[] | Backend allocates dst internally |
| `test_fused_moe_pipeline` | Full pipeline: w13 → act → w2 → MoE reduce | Verifies both `zentorch_group_matmul.out` and `zentorch_fused_moe` paths |

#### `test_int8_w13_and_w2_single_pass` sub-tests

| Sub-test | API | Post-ops | Detail |
|----------|-----|----------|--------|
| 1 | `zentorch_group_matmul.out` | No activation, no MoE reduce | Per-expert int8 w13 + int8 w2. Kernel writes w2 output back into inputs. |
| 2 | `zentorch_group_matmul.out` | silu activation + MoE weighted reduce | Uses `row_ptrs` into the input buffers for fused w2 buffer reuse |
| 3 | `zentorch_fused_moe` | No activation + MoE weighted reduce | High-level fused_moe op with int8 weights + scales |

All sub-tests use `K == K_out == N` (buffer reuse constraint).


### 7.3 Known limitations

| Limitation | Detail |
|------------|--------|
| Mixed bf16-Op1 / int8-Op2 | Unsupported — LowOHA enforces one quant scheme for both passes |

## 8. Reference

This operator wraps the `group_matmul_direct` API from the [LowOHA Group MatMul Operator](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_matmul_operator.md), using:
- Parallel execution mode
- Optional `group_matmul_moe_postop_params` for weighted-reduce
- Optional `grp_matmul_gated_act_params` for gated activations
- Optional `grp_matmul_fused_moe_params` for fused down projection (Op2)
