/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

// Workaround for PyTorch C++ wrapper codegen using Python-style booleans
#ifndef True
#define True 1
#endif
#ifndef False
#define False 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_linear_unary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear_mul_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle mul_input,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    const char *zentorch_op_name);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    const char *zentorch_op_name);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_linear_unary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *post_op_1,
    const char *post_op_2, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_linear_binary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input_1,
    AtenTensorHandle binary_input_2, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op_1, const char *post_op_2,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

// ============================================================================
// Quantized embedding bag (single + horizontally-fused group). These ops have
// `Tensor[]`, `Tensor?[]`, `int[]` and `str` schema args, none of which are
// representable via StableIValue, so without these shims cpp_wrapper falls
// back to the slow `custom_op_wrapper` Python path (~+25us/call).
// ============================================================================

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_quant_embedding_bag(
    AtenTensorHandle weight, AtenTensorHandle indices, AtenTensorHandle offsets,
    int64_t num_bits_per_weight, int32_t output_dtype, bool scale_grad_by_freq,
    int64_t mode, bool sparse, AtenTensorHandle *per_sample_weights,
    bool include_last_offset, int64_t padding_idx, const char *zentorch_op_name,
    AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_zentorch_quant_embedding_bag_out(
    AtenTensorHandle output, AtenTensorHandle weight, AtenTensorHandle indices,
    AtenTensorHandle offsets, int64_t num_bits_per_weight, int32_t output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    AtenTensorHandle *per_sample_weights, bool include_last_offset,
    int64_t padding_idx, const char *zentorch_op_name);

// `zentorch_horizontal_quant_embedding_bag_group.default` returns
// `Tensor[]` (variable-length list of N outputs, where N == number of input
// embedding bags fused into the group call). Inductor's standard multi-output
// cpp_wrapper codegen (`generate_c_shim_fallback_kernel`) emits one
// `&ret_i_handle` per output, which can't express a variable-length list in a
// single shim signature. We instead define the shim to take an
// `(AtenTensorHandle* ret0_handles, int64_t ret0_len_)` pair and let the
// corresponding Python lowering (`_ZentorchHorizontalQuantEmbBagGroupDefault`
// in `_lowerings.py`) emit a custom codegen sequence that allocates an array
// of N handles and passes it to the shim.

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group(
    const AtenTensorHandle *weight, int64_t weight_len_,
    const AtenTensorHandle *indices, int64_t indices_len_,
    const AtenTensorHandle *offsets, int64_t offsets_len_,
    int64_t num_bits_per_weight, int32_t output_dtype,
    const int64_t *scale_grad_by_freq, int64_t scale_grad_by_freq_len_,
    const int64_t *mode, int64_t mode_len_, const int64_t *sparse,
    int64_t sparse_len_, const AtenTensorHandle **per_sample_weights,
    int64_t per_sample_weights_len_, const int64_t *include_last_offset,
    int64_t include_last_offset_len_, const int64_t *padding_idx,
    int64_t padding_idx_len_, const char *zentorch_op_name,
    AtenTensorHandle *ret0_handles, int64_t ret0_len_);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group_out(
    const AtenTensorHandle *outputs, int64_t outputs_len_,
    const AtenTensorHandle *weight, int64_t weight_len_,
    const AtenTensorHandle *indices, int64_t indices_len_,
    const AtenTensorHandle *offsets, int64_t offsets_len_,
    int64_t num_bits_per_weight, int32_t output_dtype,
    const int64_t *scale_grad_by_freq, int64_t scale_grad_by_freq_len_,
    const int64_t *mode, int64_t mode_len_, const int64_t *sparse,
    int64_t sparse_len_, const AtenTensorHandle **per_sample_weights,
    int64_t per_sample_weights_len_, const int64_t *include_last_offset,
    int64_t include_last_offset_len_, const int64_t *padding_idx,
    int64_t padding_idx_len_, const char *zentorch_op_name);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_tanh(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_erf(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle add_input,
    AtenTensorHandle *B, const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_mul_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle mul_input,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle add_input,
    AtenTensorHandle add_input_2, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0);

#ifdef __cplusplus
} // extern "C"
#endif
