/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include <ATen/cpu/vec/vec.h>

namespace zentorch {
using namespace zendnnl::interface;
void zentorch_woq_linear_impl(const at::Tensor &input, const at::Tensor &weight,
                              const at::Tensor &bias, at::Tensor &result,
                              const at::Tensor &weight_scales,
                              const at::Tensor &weight_zero_points,
                              const std::vector<int64_t> &post_op_ids,
                              const std::vector<at::Tensor> &post_op_buffers,
                              std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input dimensions: " << input.sizes();
  LOG(INFO) << "weight dimensions: " << weight.sizes();
  LOG(INFO) << "weight_scales dimensions: " << weight_scales.sizes();
  LOG(INFO) << "weight_zero_points dimensions: " << weight_zero_points.sizes();
  LOG(INFO) << "result dimensions: " << result.sizes();
  LOG(INFO) << "post_op_ids size: " << post_op_ids.size();
  LOG(INFO) << "post_op_buffers size: " << post_op_buffers.size();

  // The weight tensor must be int32 with a transposed shape of [K/8, N],
  // where each int32 packs 8 int4 values. Transposition of the weight tensor,
  // as well as arranging weight_scales and weight_zero_points contiguously,
  // is performed in op_replacements_new.py during graph passes.
  TORCH_CHECK(weight.dtype() == torch::kInt32,
              "weight must have dtype int32, got ", weight.dtype());
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D, got ", weight.dim(), "D");
  TORCH_CHECK(weight_scales.size(1) == weight.size(1), "weight_scales dim 1 (",
              weight_scales.size(1), ") must match weight dim 1 (",
              weight.size(1), ")");
  TORCH_CHECK(weight_zero_points.sizes() == weight_scales.sizes(),
              "weight_zero_points shape ", weight_zero_points.sizes(),
              " must match weight_scales shape ", weight_scales.sizes());
  constexpr int kInt4PackedPerInt32 = 8; // 8 int4 values packed per int32
  const auto unpackedK = weight.size(0) * kInt4PackedPerInt32;

  status_t status;
  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_MATMUL_DIRECT");
  const bool use_zendnnl_direct_kernel = static_cast<bool>(int_env_value);
  if (use_zendnnl_direct_kernel) {
    // Get dimensions at runtime (cannot use constexpr)
    // Weight is packed format [K/8, N] (transposed), unpacked is [K, N]
    const auto M = input.size(0);
    const auto K = input.size(1);
    const auto N = weight.size(1);

    zendnnl::lowoha::matmul::matmul_quantization_params_t quantization_params;

    // Setup per-group quantization parameters
    // weight scale
    quantization_params.wei_scale.buff = weight_scales.data_ptr();
    quantization_params.wei_scale.dt = get_zendnnl_dtype(weight_scales);
    quantization_params.wei_scale.dims = weight_scales.sizes().vec();

    // weight zero point
    quantization_params.wei_zp.buff = weight_zero_points.data_ptr();
    quantization_params.wei_zp.dt = get_zendnnl_dtype(weight_zero_points);
    quantization_params.wei_zp.dims = weight_zero_points.sizes().vec();

    // Configure data types for WOQ
    // Use u4 for unsigned int4 weights,
    // s4 for signed int4 weights
    // TODO: For symmetric quantization, pass weight_zero_points as None in the
    // replacement pattern
    //       so we can check weight_zero_points.defined() here directly instead
    //       of relying on dtype check.
    const auto weight_dtype = weight_zero_points.dtype() == torch::kFloat
                                  ? data_type_t::u4  // Asymmetric quantization
                                  : data_type_t::s4; // Symmetric quantization
    zendnnl::lowoha::matmul::matmul_data_types dtypes;
    dtypes.src = get_zendnnl_dtype(input);
    dtypes.wei = weight_dtype;
    dtypes.bias = bias.defined() ? get_zendnnl_dtype(bias) : data_type_t::none;
    dtypes.dst = get_zendnnl_dtype(result); // Match actual result tensor dtype

    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes = dtypes;
    // Add quantization params to matmul params
    params.quant_params = quantization_params;
    params.plugin_op = zentorch_op_name;
    // Batch parameters
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    // Get actual tensor strides
    // For row-major layout ('r'), leading dimension is the number of columns
    // lda = stride in first dimension for input (row-major)
    // ldb = stride in second dimension for weight (column-major packed tensor)
    // ldc = stride in first dimension for result (row-major)
    const auto lda = input.stride(0);
    const auto ldb = unpackedK;
    const auto ldc = N;

    status = zendnnl::lowoha::matmul::matmul_direct(
        'r', is_transposed(input), is_transposed(weight), M, N, K,
        1.0f /* alpha */, input.data_ptr(), lda, weight.data_ptr(), ldb,
        bias.defined() ? bias.data_ptr() : nullptr, 0.0f /* beta */,
        result.data_ptr(), ldc, true /* is_weights_const (required for WOQ) */,
        batch_params, params);

    ZENTORCH_CHECK(
        status == status_t::success,
        "matmul_direct execution failed for zentorch_woq_linear_impl");

    LOG(INFO) << "zendnnl_direct_kernel completed successfully";
    return;
  }

  using tensor_opt_ref = std::optional<std::reference_wrapper<tensor_t>>;
  tensor_t woq_input, woq_weight, woq_result, woq_weight_scales,
      woq_weight_zero_points;

  set_zendnnl_tensor_attributes(input, woq_input, "woq_input",
                                false /* is_weight_prepacked */);

  tensor_opt_ref woq_weight_scales_opt_ref = std::nullopt;
  create_zendnnl_quantized_tensor(weight_scales, woq_weight_scales,
                                  "woq_weight_scales");
  woq_weight_scales_opt_ref = tensor_opt_ref(std::ref(woq_weight_scales));

  tensor_opt_ref woq_weight_zero_points_opt_ref = std::nullopt;
  create_zendnnl_quantized_tensor(weight_zero_points, woq_weight_zero_points,
                                  "woq_weight_zero_points");
  woq_weight_zero_points_opt_ref =
      tensor_opt_ref(std::ref(woq_weight_zero_points));

  // Weight is int32 packed: each int32 contains 8 int4 values
  // Use u4 for unsigned int4 weights,
  // s4 for signed int4 weights
  const auto weight_dtype = weight_zero_points.dtype() == torch::kFloat
                                ? data_type_t::u4
                                : data_type_t::s4;
  set_zendnnl_tensor_attributes(
      weight.data_ptr(), weight_dtype, woq_weight, "woq_weight",
      false /* is_weight_prepacked */,
      {static_cast<size_t>(unpackedK),
       static_cast<size_t>(weight.size(1))} /* tensor_sizes */,
      {1UL, static_cast<size_t>(unpackedK)} /* tensor_strides */,
      {} /* tensor_aligned_sizes */,
      static_cast<int64_t>(weight.numel() * 4) /* nbytes */,
      woq_weight_scales_opt_ref, woq_weight_zero_points_opt_ref);

  set_zendnnl_tensor_attributes(result, woq_result, "woq_result",
                                false /* is_weight_prepacked */);

  auto matmul_context = matmul_context_t();
  if (bias.defined()) {
    tensor_t bias_tensor = tensor_t();
    long unsigned int bias_numel = bias.numel();
    set_zendnnl_tensor_attributes(bias, bias_tensor, "bias",
                                  false /* is_weight_prepacked */,
                                  {1, bias_numel}, {bias_numel, 1});
    set_matmul_context_attributes(matmul_context, woq_weight, post_op_ids,
                                  1.0f /* alpha */, bias_tensor);
  } else {
    set_matmul_context_attributes(matmul_context, woq_weight, post_op_ids,
                                  1.0f /* alpha */);
  }
  matmul_context.create();

  auto matmul_operator = matmul_operator_t();
  set_matmul_operator_attributes(matmul_operator, matmul_context, woq_input,
                                 woq_result, post_op_ids, post_op_buffers,
                                 zentorch_op_name);

  status = matmul_operator.execute();
  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 matmul_operator.get_name(),
                 " execution failed for zentorch_matmul_impl.");
  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";
}

template <UNARY_POST_OP fuse>
at::Tensor zentorch_woq_linear_unary(const at::Tensor &input,
                                     const at::Tensor &weight,
                                     const at::Tensor &weight_scales,
                                     const at::Tensor &weight_zero_points,
                                     const std::optional<at::Tensor> &bias,
                                     std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      get_contiguous_view(input).view(get_2d_size_for_tensor(input));
  // `result` tensor's dtype will be same as input dtype.

  // Performing this calculation here to avoid calling
  // get_matmul_and_linear_output_sizes and
  // get_matmul_and_linear_output_strides, since we know that the tensors will
  // always be 2D and calculations are trivial.
  const auto output_sz =
      std::vector<int64_t>({input_2d_view.size(0), weight.size(1)});
  const auto output_strides = std::vector<int64_t>({weight.size(1), 1});

  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;
  // Set unary post ops.
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};

  zentorch_woq_linear_impl(input_2d_view, weight, bias_t, result, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);

  return result;
}

template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline at::Tensor zentorch_woq_linear_unary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const at::Tensor &binary_input, const std::optional<at::Tensor> &bias,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      get_contiguous_view(input).view(get_2d_size_for_tensor(input));
  auto binary_input_2d_view = get_contiguous_view(binary_input)
                                  .view(get_2d_size_for_tensor(binary_input));
  // `result` tensor's dtype will be same as input dtype.

  // Performing this calculation here to avoid calling
  // get_matmul_and_linear_output_sizes and
  // get_matmul_and_linear_output_strides, since we know that the tensors will
  // always be 2D and calculations are trivial.
  const auto output_sz =
      std::vector<int64_t>({input_2d_view.size(0), weight.size(1)});
  const auto output_strides = std::vector<int64_t>({weight.size(1), 1});

  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;

  std::vector<at::Tensor> post_op_buffers = {binary_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  zentorch_woq_linear_impl(input_2d_view, weight, bias_t, result, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);
  return result;
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline at::Tensor zentorch_woq_linear_binary_binary(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &weight_scales, const at::Tensor &weight_zero_points,
    const at::Tensor &binary1_input, const at::Tensor &binary2_input,
    const std::optional<at::Tensor> &bias, std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      get_contiguous_view(input).view(get_2d_size_for_tensor(input));
  auto binary1_input_2d_view = get_contiguous_view(binary1_input)
                                   .view(get_2d_size_for_tensor(binary1_input));
  auto binary2_input_2d_view = get_contiguous_view(binary2_input)
                                   .view(get_2d_size_for_tensor(binary2_input));

  // Performing this calculation here to avoid calling
  // get_matmul_and_linear_output_sizes and
  // get_matmul_and_linear_output_strides, since we know that the tensors will
  // always be 2D and calculations are trivial.
  const auto output_sz =
      std::vector<int64_t>({input_2d_view.size(0), weight.size(1)});
  const auto output_strides = std::vector<int64_t>({weight.size(1), 1});

  at::Tensor result =
      at::detail::empty_strided_cpu(output_sz, output_strides, input.options());

  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor &bias_t = *bias_maybe_owned;

  std::vector<at::Tensor> post_op_buffers = {binary1_input_2d_view,
                                             binary2_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  zentorch_woq_linear_impl(input_2d_view, weight, bias_t, result, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);
  return result;
}

at::Tensor
zentorch_weight_from_int4pack_and_repack(const at::Tensor &unpacked_weight) {
  TORCH_CHECK(unpacked_weight.dtype() == torch::kInt8,
              "unpacked_weight must have dtype int8, got ",
              unpacked_weight.dtype());
  TORCH_CHECK(unpacked_weight.dim() == 2, "unpacked_weight must be 2D, got ",
              unpacked_weight.dim(), "D");

  int N = unpacked_weight.size(0);
  int K = unpacked_weight.size(1);
  // Pack 8 int4 columns into 1 int32, reducing column count by 8x
  constexpr int pack_num = 8;
  int K_packed = K / pack_num;

  TORCH_CHECK(K >= pack_num, "K must be at least ", pack_num, ", got ", K);
  TORCH_CHECK(K % pack_num == 0, "K must be divisible by ", pack_num, ", got ",
              K);

  int8_t *weight_data = unpacked_weight.data_ptr<int8_t>();

  // Tensor for row-wise repacked weights [N, K/8], dtype int32
  at::Tensor weight_packed_rowwise =
      torch::empty({N, K_packed}, torch::TensorOptions()
                                      .dtype(torch::kInt32)
                                      .device(unpacked_weight.device()));
  int32_t *packed_rowwise_data = weight_packed_rowwise.data_ptr<int32_t>();

  // Order map that matches zendnnl's expected byte layout
  // Original int8 packing: (even_col << 4) | odd_col
  // So within each byte: upper nibble = even col, lower nibble = odd col
  // For int32 (4 bytes), we need to swap pairs: [1,0,3,2,5,4,7,6]
  // This ensures byte 0 has (col0 << 4) | col1, byte 1 has (col2 << 4) | col3,
  // etc.
  constexpr int order_map[pack_num] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Process each row independently (parallelized)
  at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      // Get pointer to current row in unpacked data
      const int8_t *row_src = weight_data + n * K;

      // Get pointer to current row in row-wise packed data
      // Each row of K values packs into K/8 int32 values
      int32_t *row_dst = packed_rowwise_data + n * K_packed;

      // Pack groups of 8 consecutive values into one int32
      for (int c = 0; c < K_packed; c++) {
        int32_t packed = 0;
        int base_col = c * pack_num;

        // Pack 8 int4 values using the reorder map
        // Each value is shifted by (i * 4) bits
        for (int i = 0; i < pack_num; i++) {
          int8_t val = row_src[base_col + order_map[i]];
          // Mask to 4 bits and shift to correct position
          packed |= static_cast<int32_t>(val & 0x0F) << (i * 4);
        }

        row_dst[c] = packed;
      }
    }
  });

  return weight_packed_rowwise;
}

// TODO: Refactor this code to modularize the unpacking and repacking logic.
// Separate the unpack (int4 -> int8 tensor) and row-wise pack (int8 tensor ->
// int32 tensor) operations into distinct reusable functions for improved code
// modularity and readability.
at::Tensor zentorch_weight_from_int4pack_and_repack_for_opaque_tensor(
    const at::Tensor &packed_weight) {

  TORCH_CHECK(packed_weight.dtype() == torch::kUInt8,
              "packed_weight must have dtype uint8, got ",
              packed_weight.dtype());
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be 2D, got ",
              packed_weight.dim(), "D");
  // Infer original dimensions
  // Packed format: [N, K/2] where N is number of rows, K is number of columns
  int N = packed_weight.size(0);      // Number of rows
  int K_half = packed_weight.size(1); // K/2 (packed columns)
  int K = K_half * 2;                 // K (full columns after unpacking)
  constexpr int pack_num = 8;
  int K_packed = K / pack_num;
  TORCH_CHECK(K >= pack_num, "K must be at least ", pack_num, ", got ", K);
  TORCH_CHECK(K % pack_num == 0, "K must be divisible by ", pack_num, ", got ",
              K);
  // Tensor for unpacked weights [N, K], dtype int8 (one uint4 value per
  // element)
  at::Tensor weight_unpacked =
      torch::zeros({N, K}, torch::TensorOptions()
                               .dtype(torch::kInt8)
                               .device(packed_weight.device()));
  // Tensor for row-wise repacked weights [N, K/8], dtype int32 (8 uint4 per
  // int32)
  at::Tensor weight_packed_rowwise =
      torch::empty({N, K_packed}, torch::TensorOptions()
                                      .dtype(torch::kInt32)
                                      .device(packed_weight.device()));
  // Get raw pointers to tensor data
  const auto packed_strided_data =
      reinterpret_cast<const uint8_t *>(packed_weight.data_ptr<uint8_t>());
  auto weight_data = weight_unpacked.data_ptr<int8_t>();
  auto packed_rowwise_data = weight_packed_rowwise.data_ptr<int32_t>();
  // BLOCK_N is the vectorization width (typically 64 for AVX-512, 32 for AVX2)
  // TODO: Include BLOCK_N from the appropriate torch header instead of
  // hardcoding it.
  constexpr int BLOCK_N = 64;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N; // Number of blocks
  // Parallel processing over blocks of rows
  at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      // Calculate actual block size (may be smaller for last block)
      int nb_size = std::min(BLOCK_N, N - static_cast<int>(i) * BLOCK_N);
      // Calculate source pointer for this block in strided packed data
      // Each block contains K columns * BLOCK_N rows / 2 (2 values per byte)
      const uint8_t *src = packed_strided_data + i * K * BLOCK_N / 2;
      // Calculate destination pointer for this block in unpacked data
      int8_t *dst = weight_data + i * BLOCK_N * K;
      // Process each column
      for (const auto k : c10::irange(K)) {
        // Only process full blocks (partial blocks need special handling)
        if (nb_size == BLOCK_N) {
          // Process 16 iterations to handle 64 rows (16 * 4 values = 64 rows)
          for (const auto d : c10::irange(16)) {
            // Layout for each column:
            //   - Bytes [0..15]:  packed02 for d=[0..15] (contains val0, val2)
            //   - Bytes [16..31]: packed13 for d=[0..15] (contains val1, val3)
            uint8_t packed02 = src[k * 32 + d];      // Contains val0, val2
            uint8_t packed13 = src[k * 32 + 16 + d]; // Contains val1, val3

            // Extract 4-bit values, keep as unsigned (0-15)
            int8_t val0 = static_cast<int8_t>(packed02 & 0x0F);
            int8_t val2 = static_cast<int8_t>((packed02 >> 4) & 0x0F);
            int8_t val1 = static_cast<int8_t>(packed13 & 0x0F);
            int8_t val3 = static_cast<int8_t>((packed13 >> 4) & 0x0F);
            // The packing read from strided rows: (d+0), (d+16), (d+32), (d+48)
            // We restore values to these same positions in row-major layout
            dst[(d + 0) * K + k] = val0;  // Row (d+0),  column k
            dst[(d + 16) * K + k] = val1; // Row (d+16), column k
            dst[(d + 32) * K + k] = val2; // Row (d+32), column k
            dst[(d + 48) * K + k] = val3; // Row (d+48), column k
          }
        } else {
          LOG(FATAL) << "WOQ_Linear: Partial block encountered (nb_size="
                     << nb_size
                     << "), which is currently unsupported by the unpack path."
                     << "[" << __FILE__ << ": " << __LINE__ << "]";
        }
      }
    }
  });
  // Repack: 8 uint4 values into each int32, row by row (parallelized)
  // Input:  weight_data[N][K]          — one uint4 value per int8 element
  // Output: packed_rowwise_data[N][K/8] — 8 uint4 values packed per int32
  // Matches zentorch_weight_from_int4pack_and_repack format
  constexpr int order_map[pack_num] = {0, 1, 2, 3, 4, 5, 6, 7};
  at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      const int8_t *row_src = weight_data + n * K;
      int32_t *row_dst = packed_rowwise_data + n * K_packed;
      for (int c = 0; c < K_packed; c++) {
        int32_t packed = 0;
        int base_col = c * pack_num;
        for (int i = 0; i < pack_num; i++) {
          int8_t val = row_src[base_col + order_map[i]];
          packed |= static_cast<int32_t>(val & 0x0F) << (i * 4);
        }
        row_dst[c] = packed;
      }
    }
  });
  return weight_packed_rowwise;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_woq_linear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor weight_zero_points, "
        "Tensor? bias=None, "
        "*, str zentorch_op_name='zentorch::zentorch_woq_linear') -> Tensor");
  m.def(
      "zentorch_woq_linear_relu(Tensor input, Tensor weight,"
      "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias=None, *, "
      "str zentorch_op_name="
      "'zentorch::zentorch_woq_linear_relu') -> Tensor");
  m.def("zentorch_woq_linear_sigmoid(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor weight_zero_points, Tensor? bias=None, "
        "*, str "
        "zentorch_op_name='zentorch::zentorch_woq_linear_sigmoid') -> Tensor");
  m.def("zentorch_woq_linear_mul_add(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor weight_zero_points, "
        "Tensor mul_input, Tensor add_input, Tensor? bias=None, *, str "
        "zentorch_op_name="
        "'zentorch::zentorch_woq_linear_mul_add') -> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_weight_from_int4pack_and_repack(Tensor unpacked_weight) -> "
        "Tensor");
  m.def("zentorch_weight_from_int4pack_and_repack_for_opaque_tensor(Tensor "
        "packed_weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_woq_linear",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::POST_OP_NONE>);
  m.impl("zentorch_woq_linear_relu",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::RELU>);
  m.impl("zentorch_woq_linear_sigmoid",
         zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::SIGMOID>);
  m.impl("zentorch_woq_linear_mul_add",
         zentorch::zentorch_woq_linear_binary_binary<BINARY_POST_OP::MUL,
                                                     BINARY_POST_OP::ADD>);
  m.impl("zentorch_weight_from_int4pack_and_repack",
         zentorch::zentorch_weight_from_int4pack_and_repack);
  m.impl("zentorch_weight_from_int4pack_and_repack_for_opaque_tensor",
         zentorch::zentorch_weight_from_int4pack_and_repack_for_opaque_tensor);
}

} // namespace zentorch
