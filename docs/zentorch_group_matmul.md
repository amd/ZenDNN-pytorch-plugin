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
- **Gated activation fusion** (`activation`): fuses SiLU/GELU/SwigluOAI activation with the gate+up projection, avoiding a separate activation kernel and memory round-trip
- **Fused down projection** (`w2_weights`, `w2_bias`): chains the down projection GEMM into the same call, eliminating a second kernel launch between the activated intermediate and the down projection. ZenDNN manages the output buffers internally by reusing the input buffers
- **MoE weighted-reduce fusion** (`moe_output`, `topk_weights`, `row_ptrs`): blends expert outputs into per-token results using router weights, avoiding a separate reduce kernel and an extra read over all expert output buffers
- **Out variant pattern**: the caller controls output memory allocation and can reuse buffers across inference steps

## 3. API

### Signature

```python
torch.ops.zentorch.zentorch_group_matmul.out(
    gemm_outputs,       # List[Tensor], pre-allocated [M_i, N] per expert (or [] for internal alloc)
    inputs,             # List[Tensor], one [M_i, K] per expert (bf16/f32)
    weights,            # List[Tensor], one [N, K] per expert (w13: gate+up weights)
    bias,               # List[Optional[Tensor]], one [N] or None per expert
    activation,         # str: 'none', 'silu', 'gelu', 'swigluoai'
    w2_weights,         # List[Optional[Tensor]], one [K_out, D] per expert ([] when unused)
    w2_bias,            # List[Optional[Tensor]], one [K_out] or None per expert ([] when unused)
    moe_output=None,    # Optional[Tensor], [num_tokens, hidden_dim] (MoE reduce result)
    topk_weights=None,  # Optional[Tensor], [num_tokens, topk] routing weights (f32)
    row_ptrs=None,      # Optional[Tensor], [num_tokens * topk]
    *, zentorch_op_name='zentorch::zentorch_group_matmul.out'
) -> None
```

> **Note:** `bias`, `w2_weights`, and `w2_bias` are required positional parameters
> (PyTorch's schema does not support default empty lists for `Tensor[]` types). Pass `[]` when unused.
> Only `moe_output`, `topk_weights`, and `row_ptrs` have defaults (`=None`) and can be omitted.

## 4. Parameters

### Required parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `gemm_outputs` | `List[Tensor]` (bf16/f32) | Pre-allocated Op1 output tensors, one per expert, shape `[M_i, N]`. Pass `[]` when intermediate GEMM results are not needed — ZenDNN allocates and manages dst buffers internally. |
| `inputs` | `List[Tensor]` (bf16/f32) | One input tensor per expert, shape `[M_i, K]`. |
| `weights` | `List[Tensor]` (bf16/f32) | Op1 weight matrices (w13: gate+up), shape `[N, K]` (nn.Linear layout). |
| `bias` | `List[Optional[Tensor]]` | Op1 bias, one `[N]` or `None` per expert. |
| `activation` | `str` | `'none'`, `'silu'`, `'gelu'`, or `'swigluoai'`. Gated activations require `N = 2*D` (even). |

### Fused down projection parameters (pass `[]` when unused)

| Parameter | Type | Description |
|-----------|------|-------------|
| `w2_weights` | `List[Optional[Tensor]]` (bf16/f32) | Down projection weights, one `[K_out, D]` per expert. `D = N/2` after gated activation. |
| `w2_bias` | `List[Optional[Tensor]]` | Down projection bias, one `[K_out]` or `None` per expert. |

> **Note:** `w2_outputs` is not required — ZenDNN manages the down projection output buffers internally by reusing the input buffers.

### MoE weighted-reduce parameters (optional, defaults to `None`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `moe_output` | `Optional[Tensor]` (bf16/f32) | Pre-allocated `[num_tokens, hidden_dim]` for weighted-reduce result. |
| `topk_weights` | `Optional[Tensor]` (f32) | Routing weights `[num_tokens, topk]`. |
| `row_ptrs` | `Optional[Tensor]` | Pre-built pointer table `[num_tokens * topk]` into the final expert output buffers. |

## 5. Constraints

| Constraint | Condition |
|-----------|-----------|
| Execution mode | Parallel only (`len(inputs) > 1`) |
| Input dtype | `torch.bfloat16` or `torch.float32` |
| Dtype consistency | inputs, weights, bias must share dtype per expert |
| Weight shape | `[N, K]` (nn.Linear layout) |
| gemm_outputs | Either empty `[]` (ZenDNN allocates internally) or `len(gemm_outputs) == len(weights)` |
| Gated activation | Requires `N` to be even (`N = 2 * D`) |
| MoE params | When `topk_weights` is provided, `row_ptrs` and `moe_output` must also be provided |
| Fused w2 params | `w2_weights` and `w2_bias` must both be provided or both be `[]` |
| w2 inner dim | `w2_weights[i].size(1)` must equal `N/2` (with gated act) or `N` (without) |
| w2 list lengths | Must equal `len(weights)` (one per expert) |
| w2 dtype | `w2_weights[i]` and `w2_bias[i]` must match input dtype |
| `row_ptrs` | `Optional[Tensor]` (int64) | Pre-built pointer table `[num_tokens * topk]`. When fused w2 is active, `row_ptrs` must point into the input buffers (ZenDNN reuses them for w2 output). |

## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_group_matmul_out_impl()
  ├─ Parse activation string → gated_act enum, compute use_gated_act
  ├─ validate_all_inputs():
  │     ├─ validate_dtypes_and_shapes (inputs, weights, bias)
  │     ├─ validate_gemm_outputs (if non-empty, size must match weights)
  │     ├─ validate_w2_params (if non-empty: list sizes, per-expert shapes, dtypes)
  │     └─ validate_moe_params (topk_weights → row_ptrs + moe_output required)
  ├─ Single-pass loop: extract dimensions, pointers, dtypes for Op1
  │     └─ If gemm_outputs empty: dst_ptrs stays nullptr (ZenDNN allocates internally)
  ├─ Configure gated activation post-op
  ├─ If MoE: populate group_matmul_moe_postop_params
  ├─ If fused w2: validate and populate grp_matmul_fused_moe_params
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

## 7. Reference

This operator wraps the `group_matmul_direct` API from the [LowOHA Group MatMul Operator](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_matmul_operator.md), using:
- Parallel execution mode
- Optional `group_matmul_moe_postop_params` for weighted-reduce
- Optional `grp_matmul_gated_act_params` for gated activations
- Optional `grp_matmul_fused_moe_params` for fused down projection (Op2)
