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

Tests live in `test/unittests/op_tests/test_group_matmul.py`. The class `Test_GroupMatmul` extends `Zentorch_TestCase`.

### 7.1 Parameterization strategy

Tests use `@parameterized.expand(product(...))` with dimension configs from `GROUP_MATMUL_CONFIGS`:
```python
GROUP_MATMUL_CONFIGS = {
    "num_experts": [4], "M": [8], "K": [64], "N": [32],
    "D": [16, 32], "K_out": [32, 64], "topk": [2], "num_tokens": [8],
}
```

All tests use `torch.manual_seed(42)` for determinism. Dtype is parameterized via `supported_dtypes` (`["float32", "bfloat16"]`).


### 7.2 Test matrix for `zentorch_group_matmul.out`

| Test | Post-ops | dtype | Parameterized |
|------|----------|-------|---------------|
| `test_plain_gemm` | None (bare GEMM) | f32 + bf16 | dtypes × num_experts × M × K × N |
| `test_moe_weighted_reduce` | MoE weighted-reduce | f32 + bf16 | dtypes × num_experts × K × N × topk × num_tokens |
| `test_gated_activations` | Gated activation (silu, gelu, swigluoai) | f32 + bf16 | dtypes × num_experts × M × K × D |
| `test_int8_w13` | Dynamic int8 w13 (bare GEMM) | f32 + bf16 | dtypes × num_experts × M × K × N |
| `test_int8_w13_and_w2` | Dynamic int8 w13 + w2, five sub-tests (see below) | f32 only | `["float32"]` × num_experts × K × topk × num_tokens |
| `test_unsupported_activation` | Invalid activation strings | f32 + bf16 | dtypes |
| `test_int8_missing_scales` | int8 weights with None scales (negative) | f32 | Not parameterized. Requires `ZENTORCH_ENABLE_CHECKS=1` set before process start (`@unittest.skipUnless`) |
| `test_empty_gemm_outputs_fused_w2` | Fused w2 with gemm_outputs=[] | f32 + bf16 | dtypes × num_experts × M × K × D × K_out |
| `test_fused_moe_pipeline` | Full pipeline: w13 → act → w2 → MoE reduce | f32 + bf16 | dtypes × num_experts × K × D × topk × num_tokens |

#### `test_int8_w13_and_w2` sub-tests

| Sub-test | API | Post-ops | Detail |
|----------|-----|----------|--------|
| 1 | `zentorch_group_matmul.out` | No activation, no MoE reduce | Per-expert int8 w13 + int8 w2 outputs. Kernel writes w2 output back into inputs. |
| 2 | `zentorch_group_matmul.out` | No activation, with MoE weighted reduce | Uses `row_ptrs_into_inputs` for fused w2 buffer reuse |
| 3 | `zentorch_fused_moe` | No activation + MoE weighted reduce | High-level fused_moe op with int8 weights + scales |

All sub-tests use `K == K_out == N` (buffer reuse constraint). Parameterized with `["float32"]` only.


### 7.3 Known limitations

| Limitation | Detail |
|------------|--------|
| Mixed bf16-Op1 / int8-Op2 | Unsupported — LowOHA enforces one quant scheme for both passes |
| Buffer reuse constraint | Fused w2 requires `K == K_out` (kernel writes w2 output back into input buffers) |
| int8 + gated activation + fused w2 | Produces incorrect results — ZenDNN's `group_matmul_direct` does not correctly propagate `src_scale` buffers when dynamic int8 quantization is combined with gated activation + fused w2 in the MoE pipeline. Standalone int8 group GEMM without gated activation works correctly. |
| Validation gating | Input validation checks (`validate_all_inputs`) are gated by `ZENTORCH_ENABLE_CHECKS` env var, read via `EnvReader::getEnvVariableAsInt()`. Because `EnvReader` caches values at initialization time (`std::call_once`), the env var must be set **before process start** — setting it after the process has begun has no effect. Default: `0` (disabled). |

## 8. Reference

This operator wraps the `group_matmul_direct` API from the [LowOHA Group MatMul Operator](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_matmul_operator.md), using:
- Parallel execution mode
- Optional `group_matmul_moe_postop_params` for weighted-reduce
- Optional `grp_matmul_gated_act_params` for gated activations
- Optional `grp_matmul_fused_moe_params` for fused down projection (Op2)
