/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#include "shim_cpu_zentorch.hpp"
#include "Linear.hpp"
#include "QLinear.hpp"
#include "Utils.hpp"
#include <ATen/ops/native_layer_norm.h>

using namespace torch::aot_inductor;

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

} // extern "C"
