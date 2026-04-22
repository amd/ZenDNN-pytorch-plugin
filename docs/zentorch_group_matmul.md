(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch_group_matmul — Parallel Group MatMul with MoE Weighted-Reduce (Out Variant)

## 1. Overview

`zentorch_group_matmul.out` is a parallel group matrix multiplication operator that executes multiple independent GEMMs in a single call using the ZenDNN LowOHA `group_matmul_direct` backend. It follows the **out variant** pattern — the caller pre-allocates all output tensors and the operator writes results directly into them.

It is designed for Mixture of Experts (MoE) inference, where each expert is an `nn.Linear` layer processing a subset of tokens. An optional **MoE weighted-reduce post-op** can be fused into the same call, blending expert outputs into a single per-token result without a separate kernel launch.

> **Note:**
> - Only **parallel mode** is supported. Sequential mode (chained matmuls) is not implemented.
> - Weight tensors follow `nn.Linear` layout `[N, K]` and need not be contiguous.

## 2. Motivation

In MoE models (Mixtral, DeepSeek, etc.), a router assigns each token to its top-k experts. Each expert runs an independent linear layer on its assigned tokens. Naive execution calls `torch.nn.functional.linear` in a loop — one call per expert — incurring repeated function-call overhead and poor thread utilization.

`zentorch_group_matmul.out` addresses this by:
- Batching all expert GEMMs into a single `group_matmul_direct` call.
- Optionally fusing the weighted-reduce post-op, avoiding a separate memory round-trip to blend expert outputs
- Using the out variant pattern so the caller controls output memory allocation and can reuse buffers across inference steps


## 3. API

### Signature

```python
torch.ops.zentorch.zentorch_group_matmul.out(
    gemm_outputs,    # List[Tensor], pre-allocated [M_i, N] per expert (written in-place)
    inputs,          # List[Tensor], one [M_i, K] per expert (bf16/f32)
    weights,         # List[Tensor], one [N, K] per expert (nn.Linear layout)
    bias,            # List[Optional[Tensor]], one [N] or None per expert
    moe_output,      # Optional[Tensor], pre-allocated [num_tokens, N] (written in-place), or None
    topk_weights,    # Optional[Tensor], [num_tokens, topk] routing weights (f32)
    row_ptrs,        # Optional[Tensor], [num_tokens * topk] int64 pointers into gemm_outputs
    *, zentorch_op_name='zentorch::zentorch_group_matmul.out'
) -> None
```

### Two output parameters

| Output | Type | When needed | Description |
|--------|------|-------------|-------------|
| `gemm_outputs` | `List[Tensor]` (mutated) | Always | Pre-allocated expert GEMM results. One `[M_i, N]` tensor per expert. The kernel writes into these buffers. |
| `moe_output` | `Optional[Tensor]` (mutated) | MoE post-op only | Pre-allocated `[num_tokens, N]` tensor for the fused weighted-reduce result. Pass `None` when MoE post-op is not needed. |

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `gemm_outputs` | `List[Tensor]` (bf16/f32) | Pre-allocated output tensors, one per expert. Shape `[M_i, N]`. Modified in-place by the kernel. |
| `moe_output` | `Optional[Tensor]` (bf16/f32) | Pre-allocated MoE output tensor, shape `[num_tokens, N]`. Pass `None` to disable MoE post-op. Modified in-place when provided. |
| `inputs` | `List[Tensor]` (bf16/f32) | One input tensor per expert, shape `[M_i, K]`. `M_i` can vary per expert (depends on routing). |
| `weights` | `List[Tensor]` (bf16/f32) | One weight matrix per expert, shape `[N, K]` (nn.Linear layout). Need not be contiguous. |
| `bias` | `List[Optional[Tensor]]` (bf16/f32) | One optional bias vector per expert, shape `[N]`. Pass `None` for experts without bias. |
| `topk_weights` | `Optional[Tensor]` (f32) | Routing weights, shape `[num_tokens, topk]`. Required for MoE post-op. For plain gather-sum, pass all `1.0`. |
| `row_ptrs` | `Optional[Tensor]` (int64) | Pre-built pointer table, shape `[num_tokens * topk]`. Each element is a raw data pointer (as int64) into a `gemm_outputs` tensor. Built by the caller during token-to-expert scatter. Required for MoE post-op. |
| `zentorch_op_name` | `str` | Operator name for profiling/tracing (default: `'zentorch::zentorch_group_matmul.out'`). |

## 5. Constraints and Validation

| Constraint | Condition |
|-----------|-----------|
| Execution mode | Parallel only (`len(inputs) > 1`). Sequential mode is not supported. |
| Input dtype | Each `inputs[i]` must be `torch.bfloat16` or `torch.float32` |
| Dtype consistency | `inputs[i]`, `weights[i]`, and `bias[i]` (if provided) must all share the same dtype |
| Input shape | Each `inputs[i]` must be 2D `[M_i, K]` |
| Weight shape | Each `weights[i]` must be 2D `[N, K]` (nn.Linear layout) |
| K compatibility | `inputs[i].size(1)` must equal `weights[i].size(1)` for each expert |
| Bias shape (if provided) | Must be 1D with `bias[i].size(0) == weights[i].size(0)` (= N) |
| List lengths | `len(inputs)`, `len(weights)`, `len(bias)`, and `len(gemm_outputs)` must all be equal |
| gemm_outputs | Must be pre-allocated with correct shapes `[M_i, N]` |
| moe_output | Must be pre-allocated `[num_tokens, N]` when MoE is active, or `None` |
| topk_weights shape | Must be 2D `[num_tokens, topk]` when provided |
| row_ptrs | Must be 1D int64, length `num_tokens * topk`, with valid pointers into `gemm_outputs` |


## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_group_matmul_out()                  # Public API entry point
  ├─ Validate dtypes and shapes
  ├─ Single-pass loop over num_ops:
  │     ├─ Extract M, K, N from tensor shapes
  │     ├─ Extract data pointers and leading dimensions
  │     ├─ Use caller's gemm_outputs[i] as dst buffers
  │     └─ Configure matmul_params (dtypes, plugin_op)
  ├─ If MoE post-op requested (topk_weights + row_ptrs + moe_output):
  │     ├─ Derive num_tokens, topk from topk_weights shape
  │     ├─ Convert row_ptrs int64 tensor to const void** array
  │     └─ Populate group_matmul_moe_postop_params
  └─ Call group_matmul_direct()
       ├─ Parallel expert GEMMs → writes into gemm_outputs
       └─ Fused weighted-reduce → writes into moe_output (if enabled)
```

### 6.2 ZenDNN LowOHA Integration

The operator calls `group_matmul_direct` with the following hardcoded defaults:

| LowOHA Parameter | Value | Rationale |
|-----------------|-------|-----------|
| `layout` | `'r'` (row-major) | Standard for PyTorch tensors |
| `transA` | `false` | Input is `[M, K]`, no transpose needed |
| `transB` | `true` | Weight is `[N, K]` (nn.Linear layout), transposed by the kernel |
| `alpha` | `1.0` | No scaling on A×B |
| `beta` | `0.0` | No accumulation into C |
| `is_weights_const` | `true` | Enables weight caching in the LowOHA backend |


### 6.3 MoE Weighted-Reduce Post-Op

When `topk_weights`, `row_ptrs`, and `moe_output` are all provided, the operator enables the fused weighted-reduce post-op.

**Reduction formula:**

```
moe_output[t, d] = Σ_{k=0}^{topk-1} topk_weights[t, k] × row_ptrs[t * topk + k][d]
```

### 6.4 Weight Caching

Since `is_weights_const = true` is always passed to `group_matmul_direct`, the LowOHA backend automatically caches reordered weight tensors.


## 7. Comparison With Direct LowOHA API

| Aspect | `zentorch_group_matmul.out` | `group_matmul_direct` (LowOHA) |
|--------|---------------------------|-------------------------------|
| Input format | PyTorch tensors | Raw `void*` pointers + scalar dimensions |
| Output ownership | Caller pre-allocates (out variant) | Caller provides raw dst pointers |
| Parameters skipped | `transA`, `transB`, `alpha`, `beta`, `is_weights_const`, `M`, `N`, `K` | All must be provided explicitly |
| Weight layout | Always `[N, K]` (nn.Linear), `transB=true` | Configurable per op |
| Execution mode | Parallel only | Sequential or parallel |
| `row_ptrs` | Caller builds and passes as int64 tensor | Caller builds as `const void**` |
| MoE output | Separate `moe_output` parameter | Part of `group_matmul_moe_postop_params` struct |
| Meta registration | Provided for `torch.compile` (out variant) | N/A |

## 8. Reference

This operator wraps the `group_matmul_direct` API from the [LowOHA Group MatMul Operator](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_gemm_operator.md) and 
uses the parallel execution mode with optional MoE weighted-reduce post-op.
