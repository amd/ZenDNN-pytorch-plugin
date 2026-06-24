/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#include "shim_cpu_zentorch.hpp"
#include "DynamicQLinear.hpp"
#include "Linear.hpp"
#include "QLinear.hpp"
#include "QuantEmbedBag.hpp"
#include "Utils.hpp"
#include "WOQ_Linear.hpp"
#include <ATen/ops/native_layer_norm.h>

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

using namespace torch::aot_inductor;

namespace {

// Build a `c10::List<c10::optional<at::Tensor>>` from a (potentially-null)
// array of (potentially-null) AtenTensorHandle pointers. The C ABI
// representation of `Tensor?[]` is a contiguous array where each entry is
// either a non-null handle or nullptr (== std::nullopt).
inline c10::List<c10::optional<at::Tensor>>
build_optional_tensor_list(const AtenTensorHandle **handles, int64_t len) {
  c10::List<c10::optional<at::Tensor>> out;
  out.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    if (handles && handles[i]) {
      out.push_back(*tensor_handle_to_tensor_pointer(*handles[i]));
    } else {
      out.push_back(c10::nullopt);
    }
  }
  return out;
}

// Build a `std::vector<at::Tensor>` from a contiguous array of non-null
// AtenTensorHandles -- the C ABI representation of `Tensor[]`.
inline std::vector<at::Tensor>
build_tensor_vector(const AtenTensorHandle *handles, int64_t len) {
  std::vector<at::Tensor> out;
  out.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    out.emplace_back(*tensor_handle_to_tensor_pointer(handles[i]));
  }
  return out;
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_cpu_zentorch_linear_unary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_linear_unary_impl(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W), pointer_to_optional<at::Tensor>(B),
        is_weight_prepacked, post_op, zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::POST_OP_NONE>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(X_scales),
            *tensor_handle_to_tensor_pointer(X_zero_points),
            *tensor_handle_to_tensor_pointer(W_scales),
            *tensor_handle_to_tensor_pointer(W_zero_points),
            pointer_to_optional<at::Tensor>(B),
            pointer_to_optional<at::Tensor>(output_scales),
            pointer_to_optional<at::Tensor>(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::RELU>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(X_scales),
            *tensor_handle_to_tensor_pointer(X_zero_points),
            *tensor_handle_to_tensor_pointer(W_scales),
            *tensor_handle_to_tensor_pointer(W_zero_points),
            pointer_to_optional<at::Tensor>(B),
            pointer_to_optional<at::Tensor>(output_scales),
            pointer_to_optional<at::Tensor>(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_unary<zentorch::UNARY_POST_OP::SIGMOID>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(X_scales),
            *tensor_handle_to_tensor_pointer(X_zero_points),
            *tensor_handle_to_tensor_pointer(W_scales),
            *tensor_handle_to_tensor_pointer(W_zero_points),
            pointer_to_optional<at::Tensor>(B),
            pointer_to_optional<at::Tensor>(output_scales),
            pointer_to_optional<at::Tensor>(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_mul_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle X_scales,
    AtenTensorHandle X_zero_points, AtenTensorHandle W_scales,
    AtenTensorHandle W_zero_points, AtenTensorHandle mul_input,
    AtenTensorHandle add_input, AtenTensorHandle *B,
    AtenTensorHandle *output_scales, AtenTensorHandle *output_zero_points,
    const int32_t *output_dtype, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_qlinear_binary_binary<zentorch::BINARY_POST_OP::MUL,
                                                 zentorch::BINARY_POST_OP::ADD>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(X_scales),
            *tensor_handle_to_tensor_pointer(X_zero_points),
            *tensor_handle_to_tensor_pointer(W_scales),
            *tensor_handle_to_tensor_pointer(W_zero_points),
            *tensor_handle_to_tensor_pointer(mul_input),
            *tensor_handle_to_tensor_pointer(add_input),
            pointer_to_optional<at::Tensor>(B),
            pointer_to_optional<at::Tensor>(output_scales),
            pointer_to_optional<at::Tensor>(output_zero_points),
            pointer_to_optional<c10::ScalarType>(output_dtype),
            zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    zentorch::zentorch_qlinear_out_unary<zentorch::UNARY_POST_OP::POST_OP_NONE>(
        *tensor_handle_to_tensor_pointer(out),
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(X_scales),
        *tensor_handle_to_tensor_pointer(X_zero_points),
        *tensor_handle_to_tensor_pointer(W_scales),
        *tensor_handle_to_tensor_pointer(W_zero_points),
        pointer_to_optional<at::Tensor>(B),
        pointer_to_optional<at::Tensor>(output_scales),
        pointer_to_optional<at::Tensor>(output_zero_points),
        pointer_to_optional<c10::ScalarType>(output_dtype), zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_qlinear_relu_out(
    AtenTensorHandle out, AtenTensorHandle X, AtenTensorHandle W,
    AtenTensorHandle X_scales, AtenTensorHandle X_zero_points,
    AtenTensorHandle W_scales, AtenTensorHandle W_zero_points,
    AtenTensorHandle *B, AtenTensorHandle *output_scales,
    AtenTensorHandle *output_zero_points, const int32_t *output_dtype,
    const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    zentorch::zentorch_qlinear_out_unary<zentorch::UNARY_POST_OP::RELU>(
        *tensor_handle_to_tensor_pointer(out),
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(X_scales),
        *tensor_handle_to_tensor_pointer(X_zero_points),
        *tensor_handle_to_tensor_pointer(W_scales),
        *tensor_handle_to_tensor_pointer(W_zero_points),
        pointer_to_optional<at::Tensor>(B),
        pointer_to_optional<at::Tensor>(output_scales),
        pointer_to_optional<at::Tensor>(output_zero_points),
        pointer_to_optional<c10::ScalarType>(output_dtype), zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_unary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input,
    AtenTensorHandle *B, bool is_weight_prepacked, const char *post_op_1,
    const char *post_op_2, const char *zentorch_op_name,
    AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_linear_unary_binary_impl(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(binary_input),
        pointer_to_optional<at::Tensor>(B), is_weight_prepacked, post_op_1,
        post_op_2, zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_linear_binary_binary(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle binary_input_1,
    AtenTensorHandle binary_input_2, AtenTensorHandle *B,
    bool is_weight_prepacked, const char *post_op_1, const char *post_op_2,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_linear_binary_binary_impl(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(binary_input_1),
        *tensor_handle_to_tensor_pointer(binary_input_2),
        pointer_to_optional<at::Tensor>(B), is_weight_prepacked, post_op_1,
        post_op_2, zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
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
        *tensor_handle_to_tensor_pointer(weight),
        *tensor_handle_to_tensor_pointer(indices),
        *tensor_handle_to_tensor_pointer(offsets), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype), scale_grad_by_freq, mode,
        sparse, pointer_to_optional<at::Tensor>(per_sample_weights),
        include_last_offset, padding_idx, zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_quant_embedding_bag_out(
    AtenTensorHandle output, AtenTensorHandle weight, AtenTensorHandle indices,
    AtenTensorHandle offsets, int64_t num_bits_per_weight, int32_t output_dtype,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    AtenTensorHandle *per_sample_weights, bool include_last_offset,
    int64_t padding_idx, const char *zentorch_op_name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    zentorch::zendnnl_quant_embedding_bag_out(
        *tensor_handle_to_tensor_pointer(output),
        *tensor_handle_to_tensor_pointer(weight),
        *tensor_handle_to_tensor_pointer(indices),
        *tensor_handle_to_tensor_pointer(offsets), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype), scale_grad_by_freq, mode,
        sparse, pointer_to_optional<at::Tensor>(per_sample_weights),
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
        build_tensor_vector(weight, weight_len_),
        build_tensor_vector(indices, indices_len_),
        build_tensor_vector(offsets, offsets_len_), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype),
        c10::IntArrayRef(scale_grad_by_freq, scale_grad_by_freq_len_),
        c10::IntArrayRef(mode, mode_len_),
        c10::IntArrayRef(sparse, sparse_len_),
        build_optional_tensor_list(per_sample_weights, per_sample_weights_len_),
        c10::IntArrayRef(include_last_offset, include_last_offset_len_),
        c10::IntArrayRef(padding_idx, padding_idx_len_), zentorch_op_name);
    TORCH_CHECK(static_cast<int64_t>(outs.size()) == ret0_len_,
                "horizontal_quant_embedding_bag_group: kernel returned ",
                outs.size(), " tensors but caller asked for ", ret0_len_);
    for (int64_t i = 0; i < ret0_len_; ++i) {
      ret0_handles[i] = new_tensor_handle(std::move(outs[i]));
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
    // The kernel mutates the user-provided `outputs` buffers in place. We
    // materialize them as a vector of `at::Tensor` (still aliasing the
    // user's storage) and pass that as a `TensorList`.
    auto outs_vec = build_tensor_vector(outputs, outputs_len_);
    zentorch::zendnnl_horizontal_quant_embedding_bag_group_out(
        outs_vec, build_tensor_vector(weight, weight_len_),
        build_tensor_vector(indices, indices_len_),
        build_tensor_vector(offsets, offsets_len_), num_bits_per_weight,
        static_cast<c10::ScalarType>(output_dtype),
        c10::IntArrayRef(scale_grad_by_freq, scale_grad_by_freq_len_),
        c10::IntArrayRef(mode, mode_len_),
        c10::IntArrayRef(sparse, sparse_len_),
        build_optional_tensor_list(per_sample_weights, per_sample_weights_len_),
        c10::IntArrayRef(include_last_offset, include_last_offset_len_),
        c10::IntArrayRef(padding_idx, padding_idx_len_), zentorch_op_name);
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_unary<
        zentorch::UNARY_POST_OP::POST_OP_NONE>(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(weight_scales),
        pointer_to_optional<at::Tensor>(weight_zero_points),
        pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_relu(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::RELU>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(weight_scales),
            pointer_to_optional<at::Tensor>(weight_zero_points),
            pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_sigmoid(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::SIGMOID>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(weight_scales),
            pointer_to_optional<at::Tensor>(weight_zero_points),
            pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_tanh(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::GELU_TANH>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(weight_scales),
            pointer_to_optional<at::Tensor>(weight_zero_points),
            pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_gelu_erf(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle *B,
    const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result =
        zentorch::zentorch_woq_linear_unary<zentorch::UNARY_POST_OP::GELU_ERF>(
            *tensor_handle_to_tensor_pointer(X),
            *tensor_handle_to_tensor_pointer(W),
            *tensor_handle_to_tensor_pointer(weight_scales),
            pointer_to_optional<at::Tensor>(weight_zero_points),
            pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_woq_linear_add(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *weight_zero_points, AtenTensorHandle add_input,
    AtenTensorHandle *B, const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_woq_linear_unary_binary<
        zentorch::UNARY_POST_OP::POST_OP_NONE, zentorch::BINARY_POST_OP::ADD>(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(weight_scales),
        pointer_to_optional<at::Tensor>(weight_zero_points),
        *tensor_handle_to_tensor_pointer(add_input),
        pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
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
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(weight_scales),
        pointer_to_optional<at::Tensor>(weight_zero_points),
        *tensor_handle_to_tensor_pointer(mul_input),
        *tensor_handle_to_tensor_pointer(add_input),
        pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
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
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(weight_scales),
        pointer_to_optional<at::Tensor>(weight_zero_points),
        *tensor_handle_to_tensor_pointer(add_input),
        *tensor_handle_to_tensor_pointer(add_input_2),
        pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu_zentorch_dynamic_qlinear(
    AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle weight_scales,
    AtenTensorHandle *B, const char *zentorch_op_name, AtenTensorHandle *ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = zentorch::zentorch_dynamic_qlinear(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        *tensor_handle_to_tensor_pointer(weight_scales),
        pointer_to_optional<at::Tensor>(B), zentorch_op_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

} // extern "C"
