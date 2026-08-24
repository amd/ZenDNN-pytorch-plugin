/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#include "shim_cpu_zentorch.hpp"
#include "DynamicQLinear.hpp"
#include "Embedding.hpp"
#include "FusedMoE.hpp"
#include "Linear.hpp"
#include "QLinear.hpp"
#include "QuantEmbedBag.hpp"
#include "RMS_norm.hpp"
#include "Utils.hpp"
#include "WOQ_Linear.hpp"
#include <ATen/ops/native_layer_norm.h>

#include <c10/util/Optional.h>
#include <optional>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <vector>

using namespace torch::aot_inductor;

namespace {

// Declared ahead of the list builders below, which are defined in terms of
// them.
// Required Tensor shim args arrive as AtenTensorHandle by value
// (AtenTensorOpaque*, non-null). Optional Tensor? args arrive as
// AtenTensorHandle* (nullptr => std::nullopt); dereference before bridging.
inline torch::stable::Tensor stable_from_handle(AtenTensorHandle orig_handle);
inline std::optional<torch::stable::Tensor>
stable_optional_from_handle(AtenTensorHandle *handle);

// Build a `std::vector<torch::stable::Tensor>` from a contiguous array of
// non-null AtenTensorHandles -- the C ABI representation of `Tensor[]`.
inline std::vector<torch::stable::Tensor>
build_stable_tensor_vector(const AtenTensorHandle *handles, int64_t len) {
  std::vector<torch::stable::Tensor> out;
  out.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    out.emplace_back(stable_from_handle(handles[i]));
  }
  return out;
}

// Build a `std::vector<std::optional<torch::stable::Tensor>>` from a
// (potentially-null) array of (potentially-null) AtenTensorHandle pointers.
// The C ABI representation of `Tensor?[]` is a contiguous array where each
// entry is either a non-null handle or nullptr (== std::nullopt).
inline std::vector<std::optional<torch::stable::Tensor>>
build_stable_optional_tensor_vector(const AtenTensorHandle **handles,
                                    int64_t len) {
  std::vector<std::optional<torch::stable::Tensor>> out;
  out.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    if (handles && handles[i]) {
      out.emplace_back(stable_from_handle(*handles[i]));
    } else {
      out.emplace_back(std::nullopt);
    }
  }
  return out;
}

// The C ABI representation of a schema `int[]` arg. An empty list may arrive
// as (nullptr, 0), so the pointer range is only formed once there is data.
inline std::vector<int64_t> build_int_vector(const int64_t *values,
                                             int64_t len) {
  if (!values || len <= 0) {
    return {};
  }
  return std::vector<int64_t>(values, values + len);
}

// Bridge an AOTI caller-owned handle to torch::stable::Tensor without stealing
// ownership. aoti_torch_new_tensor_handle creates a new handle referencing the
// same TensorImpl (refcount bump only, no tensor data copy). Required because
// stable::Tensor(AtenTensorHandle) takes ownership of its handle.
// Use this for required Tensor args (X, W, out, ...) passed as AtenTensorHandle
// by value. Do not pass AtenTensorHandle* here; that is for optional args only.
inline torch::stable::Tensor stable_from_handle(AtenTensorHandle orig_handle) {
  AtenTensorHandle new_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_tensor_handle(orig_handle, &new_handle));
  return torch::stable::Tensor(new_handle);
}

// Bridge an optional Tensor? arg (C ABI: potentially-null handle pointer).
// AtenTensorHandle is already a pointer type, so optional args arrive as
// AtenTensorHandle* (null -> std::nullopt, otherwise
// stable_from_handle(*handle)).
inline std::optional<torch::stable::Tensor>
stable_optional_from_handle(AtenTensorHandle *handle) {
  if (handle && *handle) {
    return stable_from_handle(*handle);
  }
  return std::nullopt;
}

// Reverse bridge: return a new caller-owned AtenTensorHandle from a stable
// tensor after a stable op completes. Same refcount-only semantics as above.
inline AtenTensorHandle
handle_from_stable(const torch::stable::Tensor &stable_tensor) {
  AtenTensorHandle orig_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_tensor_handle(stable_tensor.get(), &orig_handle));
  return orig_handle;
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_cpu_zentorch_linear_unary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    auto tmp_result = zentorch::zentorch_linear_unary(
        input, weight, bias, is_weight_prepacked, post_op, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_unary_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *post_op,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor out_stable = stable_from_handle(out);
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    zentorch::zentorch_linear_unary_out_impl(input, weight, bias,
                                             is_weight_prepacked, post_op,
                                             zentorch_op_name, out_stable);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, bool is_weight_prepacked,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::POST_OP_NONE>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(X_scales), stable_from_handle(X_zero_points),
            stable_from_handle(W_scales), stable_from_handle(W_zero_points),
            stable_optional_from_handle(B),
            stable_optional_from_handle(output_scales),
            stable_optional_from_handle(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            is_weight_prepacked, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, bool is_weight_prepacked,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::RELU>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(X_scales), stable_from_handle(X_zero_points),
            stable_from_handle(W_scales), stable_from_handle(W_zero_points),
            stable_optional_from_handle(B),
            stable_optional_from_handle(output_scales),
            stable_optional_from_handle(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            is_weight_prepacked, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, bool is_weight_prepacked,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::SIGMOID>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(X_scales), stable_from_handle(X_zero_points),
            stable_from_handle(W_scales), stable_from_handle(W_zero_points),
            stable_optional_from_handle(B),
            stable_optional_from_handle(output_scales),
            stable_optional_from_handle(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            is_weight_prepacked, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_mul_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle mul_input,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, bool is_weight_prepacked,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_binary_binary<zentorch::BINARY_POST_OP::MUL,
                                                 zentorch::BINARY_POST_OP::ADD>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(X_scales), stable_from_handle(X_zero_points),
            stable_from_handle(W_scales), stable_from_handle(W_zero_points),
            stable_from_handle(mul_input), stable_from_handle(add_input),
            stable_optional_from_handle(B),
            stable_optional_from_handle(output_scales),
            stable_optional_from_handle(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            is_weight_prepacked, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    bool is_weight_prepacked, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_qlinear_out_unary<zentorch::UNARY_POST_OP::POST_OP_NONE>(
        result, stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(X_scales), stable_from_handle(X_zero_points),
        stable_from_handle(W_scales), stable_from_handle(W_zero_points),
        stable_optional_from_handle(B),
        stable_optional_from_handle(output_scales),
        stable_optional_from_handle(output_zero_points),
        pointer_to_optional<c10::ScalarType>(output_dtype), is_weight_prepacked,
        zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    bool is_weight_prepacked, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_qlinear_out_unary<zentorch::UNARY_POST_OP::RELU>(
        result, stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(X_scales), stable_from_handle(X_zero_points),
        stable_from_handle(W_scales), stable_from_handle(W_zero_points),
        stable_optional_from_handle(B),
        stable_optional_from_handle(output_scales),
        stable_optional_from_handle(output_zero_points),
        pointer_to_optional<c10::ScalarType>(output_dtype), is_weight_prepacked,
        zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_unary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *post_op_1,
    const char *post_op_2, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const torch::stable::Tensor binary_input_stable =
        stable_from_handle(binary_input);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    auto tmp_result = zentorch::zentorch_linear_unary_binary(
        input, weight, binary_input_stable, bias, is_weight_prepacked,
        post_op_1, post_op_2, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_unary_binary_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle binary_input, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op_1, const char *post_op_2,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor out_stable = stable_from_handle(out);
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const torch::stable::Tensor binary_input_stable =
        stable_from_handle(binary_input);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    zentorch::zentorch_linear_unary_binary_out_impl(
        input, weight, binary_input_stable, bias, is_weight_prepacked,
        post_op_1, post_op_2, zentorch_op_name, out_stable);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_binary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input_1,
    AtenTensorHandle binary_input_2, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op_1, const char *post_op_2,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const torch::stable::Tensor binary_input_1_stable =
        stable_from_handle(binary_input_1);
    const torch::stable::Tensor binary_input_2_stable =
        stable_from_handle(binary_input_2);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    auto tmp_result = zentorch::zentorch_linear_binary_binary(
        input, weight, binary_input_1_stable, binary_input_2_stable, bias,
        is_weight_prepacked, post_op_1, post_op_2, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_binary_binary_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle binary_input_1, AtenTensorHandle binary_input_2,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *post_op_1,
    const char *post_op_2, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor out_stable = stable_from_handle(out);
    const torch::stable::Tensor input = stable_from_handle(X);
    const torch::stable::Tensor weight = stable_from_handle(W);
    const torch::stable::Tensor binary_input_1_stable =
        stable_from_handle(binary_input_1);
    const torch::stable::Tensor binary_input_2_stable =
        stable_from_handle(binary_input_2);
    const std::optional<torch::stable::Tensor> bias =
        stable_optional_from_handle(B);
    zentorch::zentorch_linear_binary_binary_out_impl(
        input, weight, binary_input_1_stable, binary_input_2_stable, bias,
        is_weight_prepacked, post_op_1, post_op_2, zentorch_op_name,
        out_stable);
  });
}

// ============================================================================
// Quantized embedding bag shims. Calling these directly from cpp_wrapper
// avoids the `custom_op_wrapper` Python fallback path, which was empirically
// measured at ~+25us/call vs the dispatcher path for the single-tensor
// variant.
// ============================================================================

AOTITorchError aoti_torch_cpu_zentorch_quant_embedding_bag(
    AtenTensorHandle weight, AtenTensorHandle indices, AtenTensorHandle offsets,
    int64_t num_bits_per_weight, int32_t output_dtype, bool scale_grad_by_freq,
    int64_t mode, bool sparse, AtenTensorHandle *per_sample_weights,
    bool include_last_offset, int64_t padding_idx, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zendnnl_quant_embedding_bag(
        stable_from_handle(weight), stable_from_handle(indices),
        stable_from_handle(offsets), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype), scale_grad_by_freq, mode,
        sparse, stable_optional_from_handle(per_sample_weights),
        include_last_offset, padding_idx, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_quant_embedding_bag_out(
    AtenTensorHandle output, AtenTensorHandle weight, AtenTensorHandle indices,
    AtenTensorHandle offsets, int64_t num_bits_per_weight, int32_t output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    AtenTensorHandle *per_sample_weights, bool include_last_offset,
    int64_t padding_idx, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto output_stable = stable_from_handle(output);
    zentorch::zendnnl_quant_embedding_bag_out(
        output_stable, stable_from_handle(weight), stable_from_handle(indices),
        stable_from_handle(offsets), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype), scale_grad_by_freq, mode,
        sparse, stable_optional_from_handle(per_sample_weights),
        include_last_offset, padding_idx, zentorch_op_name);
  });
}

// The `.default` overload returns `Tensor[]`. Inductor's standard
// multi-output codegen would emit one `&handle` per output, which can't
// express a variable-length list in a single shim signature. The matching
// Python lowering in `_lowerings.py` overrides codegen to emit
// `(handle_array_pointer, length)` instead, matching the signature below.
AOTITorchError aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group(
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
    AtenTensorHandle *ret0_handles, int64_t ret0_len_) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto outs = zentorch::zendnnl_horizontal_quant_embedding_bag_group_impl(
        build_stable_tensor_vector(weight, weight_len_),
        build_stable_tensor_vector(indices, indices_len_),
        build_stable_tensor_vector(offsets, offsets_len_), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype),
        build_int_vector(scale_grad_by_freq, scale_grad_by_freq_len_),
        build_int_vector(mode, mode_len_),
        build_int_vector(sparse, sparse_len_),
        build_stable_optional_tensor_vector(per_sample_weights,
                                            per_sample_weights_len_),
        build_int_vector(include_last_offset, include_last_offset_len_),
        build_int_vector(padding_idx, padding_idx_len_), zentorch_op_name);
    TORCH_CHECK(static_cast<int64_t>(outs.size()) == ret0_len_,
                "horizontal_quant_embedding_bag_group: kernel returned ",
                outs.size(), " tensors but caller asked for ", ret0_len_);
    for (int64_t i = 0; i < ret0_len_; ++i) {
      ret0_handles[i] = handle_from_stable(outs[i]);
    }
  });
}

AOTITorchError aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group_out(
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
    int64_t padding_idx_len_, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // The kernel mutates the user-provided `outputs` buffers in place. The
    // stable tensors built here reference the same storage, so the writes
    // land in the caller's buffers.
    zentorch::zendnnl_horizontal_quant_embedding_bag_group_out(
        build_stable_tensor_vector(outputs, outputs_len_),
        build_stable_tensor_vector(weight, weight_len_),
        build_stable_tensor_vector(indices, indices_len_),
        build_stable_tensor_vector(offsets, offsets_len_), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype),
        build_int_vector(scale_grad_by_freq, scale_grad_by_freq_len_),
        build_int_vector(mode, mode_len_),
        build_int_vector(sparse, sparse_len_),
        build_stable_optional_tensor_vector(per_sample_weights,
                                            per_sample_weights_len_),
        build_int_vector(include_last_offset, include_last_offset_len_),
        build_int_vector(padding_idx, padding_idx_len_), zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_unary<
        zentorch::UNARY_POST_OP::POST_OP_NONE>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::RELU>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(weight_scales),
            stable_optional_from_handle(weight_zero_points),
            stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::SIGMOID>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(weight_scales),
            stable_optional_from_handle(weight_zero_points),
            stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_tanh(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::GELU_TANH>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(weight_scales),
            stable_optional_from_handle(weight_zero_points),
            stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_erf(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::GELU_ERF>(
            stable_from_handle(X), stable_from_handle(W),
            stable_from_handle(weight_scales),
            stable_optional_from_handle(weight_zero_points),
            stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle add_input,
    AtenTensorHandle *B, const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_unary_binary<
        zentorch::UNARY_POST_OP::POST_OP_NONE, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(add_input), stable_optional_from_handle(B),
        zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_mul_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle mul_input,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_binary_binary<
        zentorch::BINARY_POST_OP::MUL, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(mul_input), stable_from_handle(add_input),
        stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle add_input,
    AtenTensorHandle add_input_2, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_binary_binary<
        zentorch::BINARY_POST_OP::ADD, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(add_input), stable_from_handle(add_input_2),
        stable_optional_from_handle(B), zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

// Out variants.
AOTITorchError aoti_torch_cpu_zentorch_woq_linear_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_out<
        zentorch::UNARY_POST_OP::POST_OP_NONE>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_relu_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_out<zentorch::UNARY_POST_OP::RELU>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_sigmoid_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_out<zentorch::UNARY_POST_OP::SIGMOID>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_tanh_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_out<zentorch::UNARY_POST_OP::GELU_TANH>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_erf_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_out<zentorch::UNARY_POST_OP::GELU_ERF>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_unary_binary_out<
        zentorch::UNARY_POST_OP::POST_OP_NONE, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(add_input), stable_optional_from_handle(B),
        zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_mul_add_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle mul_input, AtenTensorHandle add_input, AtenTensorHandle *B,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_binary_binary_out<
        zentorch::BINARY_POST_OP::MUL, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(mul_input), stable_from_handle(add_input),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add_add_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *weight_zero_points,
    AtenTensorHandle add_input, AtenTensorHandle add_input_2,
    AtenTensorHandle *B, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    torch::stable::Tensor result = stable_from_handle(out);
    zentorch::zentorch_woq_linear_binary_binary_out<
        zentorch::BINARY_POST_OP::ADD, zentorch::BINARY_POST_OP::ADD>(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales),
        stable_optional_from_handle(weight_zero_points),
        stable_from_handle(add_input), stable_from_handle(add_input_2),
        stable_optional_from_handle(B), zentorch_op_name, result);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_dynamic_qlinear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_dynamic_qlinear(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales), stable_optional_from_handle(B),
        is_weight_prepacked, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

// Out variant: writes into the caller-owned `out` handle (first arg, per the
// *_out shim convention); no return handle.
AOTITorchError aoti_torch_cpu_zentorch_dynamic_qlinear_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle weight_scales, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto out_stable = stable_from_handle(out);
    zentorch::zentorch_dynamic_qlinear_out(
        stable_from_handle(X), stable_from_handle(W),
        stable_from_handle(weight_scales), stable_optional_from_handle(B),
        is_weight_prepacked, zentorch_op_name, out_stable);
  });
}

// Void-returning, output-mutating op: `output` (Tensor(a!)) is written in
// place, no return handle. `act` and `zentorch_op_name` arrive as const char*
// (std::string_view / std::string construct from them implicitly).
AOTITorchError aoti_torch_cpu_zentorch_fused_moe(
    AtenTensorHandle output, AtenTensorHandle input, AtenTensorHandle w13,
    AtenTensorHandle w2, AtenTensorHandle *w13_bias, AtenTensorHandle *w2_bias,
    AtenTensorHandle topk_weights, AtenTensorHandle topk_id, bool skip_weighted,
    const char *act, AtenTensorHandle *w13_scales, AtenTensorHandle *w2_scales,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // Named local: the kernel takes a non-const torch::stable::Tensor & for the
    // mutated output, which cannot bind to the temporary stable_from_handle
    // returns. Writes still land in the caller's buffer since the bumped handle
    // references the same storage.
    auto output_stable = stable_from_handle(output);
    zentorch::zentorch_fused_moe(
        output_stable, stable_from_handle(input), stable_from_handle(w13),
        stable_from_handle(w2), stable_optional_from_handle(w13_bias),
        stable_optional_from_handle(w2_bias), stable_from_handle(topk_weights),
        stable_from_handle(topk_id), skip_weighted, act,
        stable_optional_from_handle(w13_scales),
        stable_optional_from_handle(w2_scales), zentorch_op_name);
  });
}

// RMS norm: tensor-returning. `epsilon` arrives as a C++ double literal from
// the schema `float` arg; `zentorch_op_name` as const char* (std::string
// constructs from it implicitly).
AOTITorchError aoti_torch_cpu_zentorch_rms_norm(AtenTensorHandle input,
                                                AtenTensorHandle weight,
                                                double epsilon,
                                                const char *zentorch_op_name,
                                                AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto input_stable = stable_from_handle(input);
    auto weight_stable = stable_from_handle(weight);
    auto tmp_result = zentorch::zentorch_rms_norm(input_stable, weight_stable,
                                                  epsilon, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

// Embedding lookup: tensor-returning. `padding_idx` arrives as int64_t, the
// two flags as bool, and `zentorch_op_name` as const char* (std::string
// constructs from it implicitly).
AOTITorchError aoti_torch_cpu_zentorch_embedding(
    AtenTensorHandle weight, AtenTensorHandle indices, int64_t padding_idx,
    bool scale_grad_by_freq, bool sparse, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_embedding(
        stable_from_handle(weight), stable_from_handle(indices), padding_idx,
        scale_grad_by_freq, sparse, zentorch_op_name);
    *ret0 = handle_from_stable(tmp_result);
  });
}

// Out variant: writes into the Inductor-allocated `out` buffer.
AOTITorchError aoti_torch_cpu_zentorch_embedding_out(
    AtenTensorHandle out, AtenTensorHandle weight, AtenTensorHandle indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto out_stable = stable_from_handle(out);
    zentorch::zendnnl_embedding_impl(
        stable_from_handle(weight), stable_from_handle(indices), padding_idx,
        scale_grad_by_freq, sparse, zentorch_op_name, out_stable);
  });
}

// Void-returning, output-mutating op: `input` (Tensor(a!)) and `residual`
// (Tensor(b!)) are written in place, no return handle.
AOTITorchError aoti_torch_cpu_zentorch_add_rms_norm_(
    AtenTensorHandle input, AtenTensorHandle weight, AtenTensorHandle residual,
    double epsilon, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto input_stable = stable_from_handle(input);
    auto weight_stable = stable_from_handle(weight);
    auto residual_stable = stable_from_handle(residual);
    zentorch::zentorch_add_rms_norm_(input_stable, weight_stable,
                                     residual_stable, epsilon,
                                     zentorch_op_name);
  });
}

} // extern "C"
