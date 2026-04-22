/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "MatmulUtils.hpp"
#include "Memory.hpp"

using namespace zendnnl::interface;

namespace zentorch {

static bool has_tensor(const c10::optional<at::Tensor> &opt) {
  return opt.has_value() && opt->defined();
}

static void check_valid_inputs_for_group_matmul(
    const std::vector<at::Tensor> &inputs,
    const std::vector<at::Tensor> &weights,
    const std::vector<c10::optional<at::Tensor>> &bias) {

  ZENTORCH_CHECK(inputs.size() > 1,
                 "zentorch_group_matmul: sequential mode (inputs.size() == 1) "
                 "is not supported; only parallel mode (one input per expert) "
                 "is currently implemented");

  const int num_ops = weights.size();

  ZENTORCH_CHECK(inputs.size() == bias.size() && bias.size() == weights.size(),
                 "zentorch_group_matmul: inputs.size() (", inputs.size(),
                 ") and bias.size() (", bias.size(),
                 ") must both equal weights.size() (", num_ops, ")");

  for (int expert_idx = 0; expert_idx < num_ops; ++expert_idx) {
    const auto &input = inputs[expert_idx];
    const auto &weight = weights[expert_idx];

    const auto input_dtype = input.scalar_type();
    ZENTORCH_CHECK(input_dtype == c10::kFloat || input_dtype == c10::kBFloat16,
                   "zentorch_group_matmul: input[", expert_idx,
                   "] must be float32 or bfloat16, got ", input_dtype);

    ZENTORCH_CHECK(weight.scalar_type() == input_dtype &&
                       (!has_tensor(bias[expert_idx]) ||
                        bias[expert_idx]->scalar_type() == input_dtype),
                   "zentorch_group_matmul: input[", expert_idx, "], weight[",
                   expert_idx, "], and bias[", expert_idx,
                   "] must all share the same dtype (", input_dtype, ")");

    ZENTORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
                   "zentorch_group_matmul: input[", expert_idx, "] and weight[",
                   expert_idx, "] must both be 2D, got ", input.dim(), "D and ",
                   weight.dim(), "D respectively");

    const int64_t K_input = input.size(1);
    const int64_t K_weight = weight.size(1);
    ZENTORCH_CHECK(K_input == K_weight, "zentorch_group_matmul: input[",
                   expert_idx, "] K (", K_input, ") must match weight[",
                   expert_idx, "] K (", K_weight, ")");

    if (has_tensor(bias[expert_idx])) {
      ZENTORCH_CHECK(bias[expert_idx]->dim() == 1,
                     "zentorch_group_matmul: bias[", expert_idx,
                     "] must be 1D");
      ZENTORCH_CHECK(bias[expert_idx]->size(0) == weight.size(0),
                     "zentorch_group_matmul: bias[", expert_idx, "] size (",
                     bias[expert_idx]->size(0), ") must match N (",
                     weight.size(0), ")");
    }
  }
}

void zentorch_group_matmul_out(
    std::vector<at::Tensor> gemm_outputs, const std::vector<at::Tensor> &inputs,
    const std::vector<at::Tensor> &weights,
    const std::vector<c10::optional<at::Tensor>> &bias,
    c10::optional<at::Tensor> moe_output,
    const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &row_ptrs, std::string zentorch_op_name) {

  check_valid_inputs_for_group_matmul(inputs, weights, bias);

  const int num_ops = weights.size();

  ZENTORCH_CHECK(gemm_outputs.size() == weights.size(),
                 "zentorch_group_matmul: gemm_outputs.size() (",
                 gemm_outputs.size(), ") must equal weights.size() (", num_ops,
                 ")");

  const bool use_moe = has_tensor(topk_weights);
  if (use_moe) {
    ZENTORCH_CHECK(has_tensor(row_ptrs) && has_tensor(moe_output),
                   "zentorch_group_matmul: when topk_weights is provided, "
                   "row_ptrs and moe_output must also be provided");
  } else {
    ZENTORCH_CHECK(!has_tensor(row_ptrs) && !has_tensor(moe_output),
                   "zentorch_group_matmul: when topk_weights is None, "
                   "row_ptrs and moe_output must also be None");
  }

  // Pre-allocate all vectors for group_matmul_direct
  const std::vector<char> layouts(num_ops, 'r');
  const std::vector<bool> transA_vec(num_ops, false);
  const std::vector<bool> transB_vec(num_ops, true);
  std::vector<int> M_vec(num_ops), N_vec(num_ops), K_vec(num_ops);
  const std::vector<float> alpha_vec(num_ops, 1.0f);
  const std::vector<float> beta_vec(num_ops, 0.0f);
  std::vector<const void *> src_ptrs(num_ops);
  std::vector<int> lda_vec(num_ops);
  std::vector<const void *> weight_ptrs(num_ops);
  std::vector<int> ldb_vec(num_ops);
  std::vector<const void *> bias_ptrs(num_ops, nullptr);
  std::vector<void *> dst_ptrs(num_ops);
  std::vector<int> ldc_vec(num_ops);
  const std::vector<bool> is_weights_const_vec(num_ops, true);
  std::vector<zendnnl::lowoha::matmul::matmul_params> params(num_ops);

  for (int expert_idx = 0; expert_idx < num_ops; ++expert_idx) {
    const auto &input = inputs[expert_idx];
    const auto &weight = weights[expert_idx];

    M_vec[expert_idx] = input.size(0);
    N_vec[expert_idx] = weight.size(0);
    K_vec[expert_idx] = input.size(1);

    src_ptrs[expert_idx] = input.data_ptr();
    lda_vec[expert_idx] = input.stride(0);

    weight_ptrs[expert_idx] = weight.data_ptr();
    ldb_vec[expert_idx] = weight.stride(0);

    const bool bias_defined = has_tensor(bias[expert_idx]);
    if (bias_defined) {
      bias_ptrs[expert_idx] = bias[expert_idx]->data_ptr();
    }

    dst_ptrs[expert_idx] = gemm_outputs[expert_idx].data_ptr();
    ldc_vec[expert_idx] = N_vec[expert_idx];

    params[expert_idx].dtypes.src = get_zendnnl_dtype(input);
    params[expert_idx].dtypes.wei = get_zendnnl_dtype(weight);
    params[expert_idx].dtypes.dst = get_zendnnl_dtype(gemm_outputs[expert_idx]);
    params[expert_idx].dtypes.bias =
        bias_defined ? get_zendnnl_dtype(*bias[expert_idx]) : data_type_t::none;
    params[expert_idx].plugin_op = zentorch_op_name;
  }

  zendnnl::lowoha::matmul::group_matmul_moe_postop_params moe_params{};

  if (use_moe) {
    ZENTORCH_CHECK(row_ptrs->dim() == 1 && row_ptrs->dtype() == c10::kLong,
                   "zentorch_group_matmul: row_ptrs must be a 1D int64 tensor");
    ZENTORCH_CHECK(topk_weights->dim() == 2 &&
                       topk_weights->scalar_type() == c10::kFloat,
                   "zentorch_group_matmul: topk_weights must be a 2D float32 "
                   "tensor [num_tokens, topk], got ",
                   topk_weights->dim(), "D ", topk_weights->scalar_type());

    const int64_t num_tokens = topk_weights->size(0);
    const int64_t topk = topk_weights->size(1);
    const int hidden_dim = N_vec[0];

    for (int expert_idx = 1; expert_idx < num_ops; ++expert_idx) {
      ZENTORCH_CHECK(N_vec[expert_idx] == hidden_dim,
                     "zentorch_group_matmul: all experts must have the same N "
                     "when MoE is enabled; expert[0] N=",
                     hidden_dim, " but expert[", expert_idx,
                     "] N=", N_vec[expert_idx]);
    }

    ZENTORCH_CHECK(
        moe_output->dim() == 2 && moe_output->size(0) == num_tokens &&
            moe_output->size(1) == hidden_dim,
        "zentorch_group_matmul: moe_output must be [num_tokens, N] = [",
        num_tokens, ", ", hidden_dim, "], got [", moe_output->size(0), ", ",
        moe_output->size(1), "]");

    ZENTORCH_CHECK(row_ptrs->size(0) == num_tokens * topk,
                   "zentorch_group_matmul: row_ptrs.size(0) (",
                   row_ptrs->size(0), ") must equal num_tokens * topk (",
                   num_tokens * topk, ")");

    moe_params.num_tokens = num_tokens;
    moe_params.topk = topk;
    moe_params.output = moe_output->data_ptr();
    moe_params.ldc_output = hidden_dim;
    moe_params.topk_weights = topk_weights->data_ptr<float>();
    moe_params.skip_weighted = false;
    moe_params.row_ptrs =
        reinterpret_cast<const void **>(row_ptrs->data_ptr<int64_t>());
  }

  status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
      layouts, transA_vec, transB_vec, M_vec, N_vec, K_vec, alpha_vec, src_ptrs,
      lda_vec, weight_ptrs, ldb_vec, bias_ptrs, beta_vec, dst_ptrs, ldc_vec,
      is_weights_const_vec, params, use_moe ? &moe_params : nullptr);

  ZENTORCH_CHECK(status == status_t::success,
                 "zentorch_group_matmul: group_matmul_direct execution failed");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_group_matmul.out(Tensor(a!)[] gemm_outputs, "
        "Tensor[] inputs, Tensor[] weights, "
        "Tensor?[] bias, Tensor(b!)? moe_output=None, "
        "Tensor? topk_weights=None, "
        "Tensor? row_ptrs=None, *, "
        "str zentorch_op_name='zentorch::zentorch_group_matmul.out') "
        "-> ()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_group_matmul.out", zentorch::zentorch_group_matmul_out);
}

} // namespace zentorch
