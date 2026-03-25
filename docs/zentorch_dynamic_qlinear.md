(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# zentorch_dynamic_qlinear — INT8 Per-Group Symmetric Quantization with Dynamic Source Quantization

## 1. Overview

`zentorch_dynamic_qlinear` is a quantized linear operator that performs **INT8 symmetric quantization with dynamic source quantization**. It accepts BF16 or FP32 activations and pre-quantized s8 weights with quantization scales, and produces output in the same dtype as the input. The source is dynamically quantized to s8 inside the kernel at runtime, eliminating the need for the caller to pre-quantize the activations or provide source scales. The current implementation uses per-token (per-row) source scales with fixed granularity `[M, 1]`.

> **Note:**
> - Other source scale granularities (such as per-tensor or per-group) are not yet implemented and may be added in future versions.


## 2. Motivation

Static quantization relies on calibration data collected offline to fix activation scales before deployment. While this can yield good throughput, the fixed scales may not represent the true range of activations seen at inference time, leading to clipping or under-utilization of the quantized range and, consequently, accuracy degradation.

Dynamic quantization addresses this by computing activation scales at runtime, adapting to the actual data distribution of each input. This consistently preserves more of the original model's accuracy without requiring a representative calibration dataset, making it both simpler to integrate and more robust across diverse inputs.

`zentorch_dynamic_qlinear` applies this approach by accepting full-precision activations (BF16/FP32) alongside pre-quantized INT8 weights. At runtime the kernel quantizes the activations, computes the matmul in INT8, and dequantizes the result back to the original dtype — all within a single `matmul_direct` call to the ZenDNN LowOHA backend.

## 3. API

### Signature
```
torch.ops.zentorch.zentorch_dynamic_qlinear(
    input,          # torch.bfloat16 or torch.float32, shape [M, K] or [*, K]
    weight,         # torch.int8, shape [N, K] (nn.Linear layout)
    weight_scales,  # torch.float32 / torch.bfloat16, shape [N] or [1, N] (per-channel)
    bias,           # torch.float32 / torch.bfloat16 or None
    *, zentorch_op_name='zentorch::zentorch_dynamic_qlinear'
) -> Tensor         # Same dtype as input [*, N]
```

## 4. Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor (bf16/f32) | Activation tensor of shape `[*, K]` where `*` denotes any number of batch dimensions |
| `weight` | Tensor (s8) | Pre-quantized weight tensor of shape `[N, K]` (nn.Linear layout: `[out_features, in_features]`) |
| `weight_scales` | Tensor (f32/bf16) | Per-channel weight quantization scales of shape `[N]` or `{1, N}` (1D is treated as a row: `{1, N}`) |
| `bias` | Tensor? (f32/bf16) | Optional bias vector of shape `[N]` |
| `zentorch_op_name` | str | Operator name for profiling/tracing (default: `'zentorch::zentorch_dynamic_qlinear'`) |

## 5. Constraints and Validation

| Constraint | Condition |
|-----------|-----------|
| Input dtype | Must be `torch.bfloat16` or `torch.float32` |
| Weight dtype | Must be `torch.int8` |
| Weight shape | Must be 2D `[N, K]` |
| Weight scales dtype | Must be `torch.float32`, `torch.bfloat16`|
| Weight scales shape | Per-channel `[N]` or `{1, N}` (1D is internally normalized to `{1, N}`) |
| Output dtype | Same as input dtype |
| Bias dtype (if provided) | `torch.float32` or `torch.bfloat16` |
| Input last dim | Must equal `K` (weight dim 1) |
| Hardware | Optimized for AVX512-capable CPUs; current implementation does not perform an explicit runtime capability check |

## 6. Implementation Details

### 6.1 Execution Flow

```
zentorch_dynamic_qlinear()            # Public API entry point
  ├─ Allocate output tensor (same dtype as input)
  ├─ Validate dtypes and shapes
  ├─ Reshape input/output to 2D
  └─ zentorch_dynamic_qlinear_impl()  # Core implementation
       ├─ Configure matmul_data_types (src=input dtype, wei=s8, dst=input dtype, compute=s8)
       ├─ Set params.dynamic_quant = true
       ├─ Set src_scale.buff = nullptr (dynamic)
       ├─ Set wei_scale from weight_scales tensor
       └─ Call matmul_direct() with transB=true
```

### 6.2 ZenDNN LowOHA Integration

The operator uses the `matmul_direct` API with:
- `transB = true` since weight is stored as `[N, K]` (nn.Linear layout)
- `dynamic_quant = true` to enable runtime source quantization
- `src_scale.buff = nullptr` so the kernel computes per-token source scales from the data
- `src_scale.dims = [M, 1]` for per-token granularity
- `compute = data_type_t::s8` to indicate symmetric INT8 computation

### 6.3 Weight Caching

Since `is_weights_const = true` is passed to `matmul_direct`, the LowOHA backend automatically caches the reordered weight tensor.

## 7. Comparison with Existing Operators

| Feature | `zentorch_qlinear` | `zentorch_dynamic_qlinear` |
|---------|-------------------|---------------------------|
| Input dtype | f32/bf16/u8/s8 | bf16 or f32 |
| Weight layout | `[K, N]` | `[N, K]` (nn.Linear) |
| Source quantization | Static (caller provides scales) | Dynamic per-token (computed at runtime) |
| Quantization granularity | Per-tensor / per-channel | Per-token (source) / per-channel (weight) |
| Zero points | Supported (asymmetric) | Not needed (symmetric) |
| Weight scales shape | 1D `[1]` or `[N]` | 2D `{1, N}` |
| Output dtype | f32/bf16/u8/s8 | Same as input (bf16 or f32) |
| Post-ops (relu, etc.) | Supported | Not yet supported |

## 8. Reference

This operator is based on **Example 7** from the [LowOHA MatMul Operator documentation](https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_matmul_operator.md#example-7-int8-per-group-symmetric-quantization-with-dynamic-source-quantization) (INT8 Per-Group Symmetric Quantization with Dynamic Source Quantization) and uses the `matmul_direct` API from ZenDNN's LowOHA backend.
