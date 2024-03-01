/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

inline void zen_embed_tensor_check(const at::Tensor &weight,
                                   const at::Tensor &indices) {
  // check if all the input tensors are on cpu device
  TORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu(),
              "ZenDNN Embedding expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  TORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                  (indices.layout() == c10::Layout::Strided),
              "ZenDNN Embedding expects dense tensor inputs!");
  // check the weight type for embedding, only supported is fp32 for now
  // (works ONLY for dtype=torch.float32)
  TORCH_CHECK(weight.scalar_type() == c10::kFloat,
              "Only fp32 type weights are supported in ZenDNN Embedding!");
}

inline void zen_eb_tensor_check(const at::Tensor &weight,
                                const at::Tensor &indices,
                                const at::Tensor &offsets) {
  // check if all the input tensors are on cpu device
  TORCH_CHECK(weight.device().is_cpu() && indices.device().is_cpu() &&
                  offsets.device().is_cpu(),
              "ZenDNN EmbeddingBag expects CPU tensor inputs!");
  // check if all the input tensors are dense format
  TORCH_CHECK((weight.layout() == c10::Layout::Strided) &&
                  (indices.layout() == c10::Layout::Strided) &&
                  (offsets.layout() == c10::Layout::Strided),
              "ZenDNN EmbeddingBag expects dense tensor inputs!");
  // check the weight type for embedding bag, only supported is fp32 for now
  // (works ONLY for dtype=torch.float32)
  TORCH_CHECK(weight.scalar_type() == c10::kFloat,
              "Only fp32 type weights are supported in ZenDNN EmbeddingBag!");
}

inline void zen_mode_to_algo(const int64_t &mode, algorithm &z_algorithm) {
  switch (mode) {
  case 0:
    z_algorithm = algorithm::embedding_bag_sum;
    break;
  case 1:
    z_algorithm = algorithm::embedding_bag_mean;
    break;
  case 2:
    z_algorithm = algorithm::embedding_bag_max;
    break;
  default:
    z_algorithm = algorithm::embedding_bag_sum;
    break;
  }
}

inline void check_valid_sizes(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(
      ((mat1.dim() <= 3 && mat2.dim() <= 3) &&  // dimensionality check
       ((mat1.dim() == 2 && mat2.dim() == 1) || // specific case for aten::mv
        (mat1.dim() == mat2.dim()))), // general check for matrix multiplication
      "zendnn_matmul:  unsupported dims for mat1 and mat2");
}

inline void check_scalar_type(const std::vector<at::Tensor> &tensor_vector) {
  bool is_float = true, is_bfloat16 = true;

  for (auto tensor : tensor_vector) {
    is_float = is_float && (tensor.scalar_type() == c10::ScalarType::Float);
    is_bfloat16 =
        is_bfloat16 && (tensor.scalar_type() == c10::ScalarType::BFloat16);
  }

  TORCH_CHECK(is_float || is_bfloat16,
              "zendnn_matmul: zendnn_matmul only supports Float and BFloat16");
}

inline bool is_zendnn_optimized_format(const at::Tensor &t) {
  if (t.is_contiguous())
    return true;
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  // check for transposed tensors
  if (t.dim() == 2) {
    return strides[0] == 1 && strides[1] == sizes[0];
  } else {
    // dim = 3
    return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 &&
           strides[2] == sizes[1];
  }
}

inline std::vector<int64_t> get_matmul_output_sizes(const at::Tensor &tensor1,
                                                    const at::Tensor &tensor2) {
  const int64_t dim = tensor1.dim();
  std::vector<int64_t> output_size(dim);
  for (auto i = 0; i < dim - 1; i++) {
    output_size[i] = tensor1.size(i);
  }
  output_size[dim - 1] = tensor2.size(dim - 1);
  return output_size;
}