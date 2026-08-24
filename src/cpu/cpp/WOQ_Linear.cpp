/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "WOQ_Linear.hpp"
#include "EnvReader.hpp"
#include "MatmulUtils.hpp"
#include "Memory.hpp"
#include <ATen/Parallel.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/stable/library.h>

namespace zentorch {
using namespace zendnnl::interface;

void zentorch_woq_linear_impl(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const std::optional<torch::stable::Tensor> &bias,
    torch::stable::Tensor &result, const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const std::vector<int64_t> &post_op_ids,
    const std::vector<torch::stable::Tensor> &post_op_buffers,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;
  LOG(INFO) << "input sizes: [" << c10::Join(", ", input.sizes()) << "]";
  LOG(INFO) << "weight sizes: [" << c10::Join(", ", weight.sizes()) << "]";
  LOG(INFO) << "weight_scales sizes: ["
            << c10::Join(", ", weight_scales.sizes()) << "]";
  LOG(INFO) << "result sizes: [" << c10::Join(", ", result.sizes()) << "]";
  LOG(INFO) << "post_op_ids size: " << post_op_ids.size();
  LOG(INFO) << "post_op_buffers size: " << post_op_buffers.size();

  // The weight tensor must be int32 with a transposed shape of [K/8, N],
  // where each int32 packs 8 int4 values. Transposition of the weight tensor,
  // as well as arranging weight_scales and weight_zero_points contiguously,
  // is performed in op_replacements_new.py during graph passes.
  ZENTORCH_CHECK(weight.scalar_type() == c10::kInt,
                 "weight must have dtype int32, got ", weight.scalar_type());
  ZENTORCH_CHECK(weight.dim() == 2, "weight must be 2D, got ", weight.dim(),
                 "D");
  ZENTORCH_CHECK(weight_scales.size(1) == weight.size(1),
                 "weight_scales dim 1 (", weight_scales.size(1),
                 ") must match weight dim 1 (", weight.size(1), ")");
  constexpr int kInt4PackedPerInt32 = 8; // 8 int4 values packed per int32
  const auto unpackedK = weight.size(0) * kInt4PackedPerInt32;

  // TODO: Consider moving weight_dtype selection to graph pass level.
  // Use u4 for unsigned int4 weights (asymmetric: zero_points provided),
  // s4 for signed int4 weights (symmetric: zero_points is None)
  const auto weight_dtype = weight_zero_points.has_value()
                                ? data_type_t::u4  // Asymmetric quantization
                                : data_type_t::s4; // Symmetric quantization

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

    zendnnl::lowoha::matmul::matmul_quantization_params_t quantization_params{};

    // Setup per-group quantization parameters
    // weight scale
    quantization_params.wei_scale.buff = weight_scales.data_ptr();
    quantization_params.wei_scale.dt = get_zendnnl_dtype(weight_scales);
    quantization_params.wei_scale.dims =
        sizes_to_int64_vec(weight_scales.sizes());

    // weight zero point
    if (weight_zero_points.has_value()) {
      quantization_params.wei_zp.buff = weight_zero_points->data_ptr();
      quantization_params.wei_zp.dt = get_zendnnl_dtype(*weight_zero_points);
      quantization_params.wei_zp.dims =
          sizes_to_int64_vec(weight_zero_points->sizes());
    }

    zendnnl::lowoha::matmul::matmul_data_types dtypes;
    dtypes.src = get_zendnnl_dtype(input);
    dtypes.wei = weight_dtype;
    dtypes.bias =
        bias.has_value() ? get_zendnnl_dtype(*bias) : data_type_t::none;
    dtypes.dst = get_zendnnl_dtype(result); // Match actual result tensor dtype

    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes = dtypes;
    // Add quantization params to matmul params
    params.quant_params = quantization_params;
    params.plugin_op = zentorch_op_name;
    matmul_post_ops(params, result, post_op_ids, post_op_buffers);
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
        bias.has_value() ? bias->data_ptr() : nullptr, 0.0f /* beta */,
        result.data_ptr(), ldc, true /* is_weights_const (required for WOQ) */,
        batch_params, params);

    ZENTORCH_CHECK(
        status == status_t::success,
        "matmul_direct execution failed for zentorch_woq_linear_impl.");

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
  if (weight_zero_points.has_value()) {
    create_zendnnl_quantized_tensor(*weight_zero_points, woq_weight_zero_points,
                                    "woq_weight_zero_points");
    woq_weight_zero_points_opt_ref =
        tensor_opt_ref(std::ref(woq_weight_zero_points));
  }

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
  if (bias.has_value()) {
    tensor_t bias_tensor = tensor_t();
    unsigned long bias_numel = bias->numel();
    set_zendnnl_tensor_attributes(*bias, bias_tensor, "bias",
                                  false /* is_weight_prepacked */,
                                  {1UL, bias_numel}, {bias_numel, 1UL});
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

// Core compute for the unary variants: writes the result into `out`. Shared by
// the allocating impl and the `.out` variant.
template <UNARY_POST_OP fuse>
void zentorch_woq_linear_unary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name, torch::stable::Tensor &out) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // Validate the caller-supplied `out` (shape/contiguity) before viewing it;
  // gated behind ZENTORCH_ENABLE_CHECKS.
  check_linear_and_matmul_out_tensor(input, weight, out);

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      view_tensor(get_contiguous_view(input), get_2d_size_for_tensor(input));
  // `out` is viewed as 2d for matmul computation.
  auto out_2d = view_tensor(out, get_2d_size_for_tensor(out));

  // Set unary post ops.
  std::vector<torch::stable::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {fuse};

  zentorch_woq_linear_impl(input_2d_view, weight, bias, out_2d, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);
}

template <UNARY_POST_OP fuse>
torch::stable::Tensor zentorch_woq_linear_unary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `result` tensor's dtype will be same as input dtype.
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight);

  zentorch_woq_linear_unary_out<fuse>(input, weight, weight_scales,
                                      weight_zero_points, bias,
                                      zentorch_op_name, result);

  return result;
}

// Core compute for the unary+binary variants: writes the result into `out`.
// Shared by the allocating impl and the `.out` variant.
template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
void zentorch_woq_linear_unary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  check_linear_and_matmul_out_tensor(input, weight, out);

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      view_tensor(get_contiguous_view(input), get_2d_size_for_tensor(input));
  auto binary_input_2d_view = view_tensor(get_contiguous_view(binary_input),
                                          get_2d_size_for_tensor(binary_input));
  // `out` is viewed as 2d for matmul computation.
  auto out_2d = view_tensor(out, get_2d_size_for_tensor(out));

  std::vector<torch::stable::Tensor> post_op_buffers = {binary_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  zentorch_woq_linear_impl(input_2d_view, weight, bias, out_2d, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);
}

template <UNARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline torch::stable::Tensor zentorch_woq_linear_unary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `result` tensor's dtype will be same as input dtype.
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight);

  zentorch_woq_linear_unary_binary_out<fuse1, fuse2>(
      input, weight, weight_scales, weight_zero_points, binary_input, bias,
      zentorch_op_name, result);
  return result;
}

// Core compute for the binary+binary variants: writes the result into `out`.
// Shared by the allocating impl and the `.out` variant.
template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
void zentorch_woq_linear_binary_binary_out(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary1_input,
    const torch::stable::Tensor &binary2_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name, torch::stable::Tensor &out) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  check_linear_and_matmul_out_tensor(input, weight, out);

  // `input` is viewed as 2d for matmul computation.
  auto input_2d_view =
      view_tensor(get_contiguous_view(input), get_2d_size_for_tensor(input));
  auto binary1_input_2d_view =
      view_tensor(get_contiguous_view(binary1_input),
                  get_2d_size_for_tensor(binary1_input));
  auto binary2_input_2d_view =
      view_tensor(get_contiguous_view(binary2_input),
                  get_2d_size_for_tensor(binary2_input));
  // `out` is viewed as 2d for matmul computation.
  auto out_2d = view_tensor(out, get_2d_size_for_tensor(out));

  std::vector<torch::stable::Tensor> post_op_buffers = {binary1_input_2d_view,
                                                        binary2_input_2d_view};
  std::vector<int64_t> post_op_ids = {fuse1, fuse2};

  LOG(INFO) << "Calling  zentorch_woq_linear_impl from " << __FUNCTION__
            << "!\n";

  zentorch_woq_linear_impl(input_2d_view, weight, bias, out_2d, weight_scales,
                           weight_zero_points, post_op_ids, post_op_buffers,
                           zentorch_op_name);
}

template <BINARY_POST_OP fuse1, BINARY_POST_OP fuse2>
inline torch::stable::Tensor zentorch_woq_linear_binary_binary(
    const torch::stable::Tensor &input, const torch::stable::Tensor &weight,
    const torch::stable::Tensor &weight_scales,
    const std::optional<torch::stable::Tensor> &weight_zero_points,
    const torch::stable::Tensor &binary1_input,
    const torch::stable::Tensor &binary2_input,
    const std::optional<torch::stable::Tensor> &bias,
    std::string zentorch_op_name) {
  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  // `result` tensor's dtype will be same as input dtype.
  torch::stable::Tensor result =
      create_linear_and_matmul_output_tensor(input, weight);

  zentorch_woq_linear_binary_binary_out<fuse1, fuse2>(
      input, weight, weight_scales, weight_zero_points, binary1_input,
      binary2_input, bias, zentorch_op_name, result);
  return result;
}

torch::stable::Tensor
zentorch_woq_repack_weight(const torch::stable::Tensor &unpacked_weight) {
  ZENTORCH_CHECK(unpacked_weight.scalar_type() == c10::kChar,
                 "unpacked_weight must have dtype int8, got ",
                 unpacked_weight.scalar_type());
  ZENTORCH_CHECK(unpacked_weight.dim() == 2, "unpacked_weight must be 2D, got ",
                 unpacked_weight.dim(), "D");

  int N = unpacked_weight.size(0);
  int K = unpacked_weight.size(1);
  // Pack 8 int4 columns into 1 int32, reducing column count by 8x
  constexpr int pack_num = 8;
  int K_packed = K / pack_num;

  ZENTORCH_CHECK(K >= pack_num, "K must be at least ", pack_num, ", got ", K);
  ZENTORCH_CHECK(K % pack_num == 0, "K must be divisible by ", pack_num,
                 ", got ", K);

  int8_t *weight_data = unpacked_weight.mutable_data_ptr<int8_t>();

  // Tensor for row-wise repacked weights [N, K/8], dtype int32
  std::vector<int64_t> packed_sizes = {N, K_packed};
  torch::stable::Tensor weight_packed_rowwise = torch::stable::empty(
      packed_sizes, c10::kInt, std::nullopt, unpacked_weight.device());
  int32_t *packed_rowwise_data =
      weight_packed_rowwise.mutable_data_ptr<int32_t>();

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

// Unpacks PyTorch's int4pack layout (uint8 tensor) into a plain int8 tensor
// where each element holds one uint4 value.
torch::stable::Tensor
unpack_int4pack_to_int8(const torch::stable::Tensor &packed_weight) {
  ZENTORCH_CHECK(packed_weight.scalar_type() == c10::kByte,
                 "packed_weight must have dtype uint8, got ",
                 packed_weight.scalar_type());
  ZENTORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be 2D, got ",
                 packed_weight.dim(), "D");
  // Infer original dimensions
  // Packed format: [N, K/2] where N is number of rows, K is number of columns
  int N = packed_weight.size(0);      // Number of rows
  int K_half = packed_weight.size(1); // K/2 (packed columns)
  int K = K_half * 2;                 // K (full columns after unpacking)
  // Tensor for unpacked weights [N, K], dtype int8 (one uint4 value per
  // element)
  std::vector<int64_t> unpacked_sizes = {N, K};
  torch::stable::Tensor weight_unpacked = torch::stable::empty(
      unpacked_sizes, c10::kChar, std::nullopt, packed_weight.device());
  // Get raw pointers to tensor data
  const uint8_t *packed_strided_data = packed_weight.const_data_ptr<uint8_t>();
  int8_t *weight_data = weight_unpacked.mutable_data_ptr<int8_t>();
  // BLOCK_N = 64 is fixed by PyTorch's int4pack layout (see
  // aten/src/ATen/native/cpu/int4mm_kernel.cpp), which always groups rows
  // into blocks of 64 regardless of the CPU's SIMD width. The unpacking
  // math below (16 iterations x 4 interleaved rows, 32-byte column stride)
  // is specific to this block size. PyTorch pads N to a multiple of BLOCK_N,
  // so the source data always has full blocks. We validate this upfront.
  // TODO: Include BLOCK_N from the appropriate torch header instead of
  // hardcoding it.
  constexpr int BLOCK_N = 64;
  ZENTORCH_CHECK(N % BLOCK_N == 0, "packed_weight row count (", N,
                 ") must be a multiple of ", BLOCK_N,
                 ". The int4pack format from PyTorch pads N to BLOCK_N; "
                 "receiving an unpadded tensor indicates a packing mismatch.");
  const int NB = N / BLOCK_N;
  // Parallel processing over blocks of rows
  at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      // Calculate source pointer for this block in strided packed data
      // Each block contains K columns * BLOCK_N rows / 2 (2 values per byte)
      const uint8_t *src = packed_strided_data + i * K * BLOCK_N / 2;
      // Calculate destination pointer for this block in unpacked data
      int8_t *dst = weight_data + i * BLOCK_N * K;
      // Process each column
      for (const auto k : c10::irange(K)) {
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
      }
    }
  });
  return weight_unpacked;
}

torch::stable::Tensor
zentorch_woq_repack_from_int4pack(const torch::stable::Tensor &packed_weight) {
  torch::stable::Tensor weight_unpacked =
      unpack_int4pack_to_int8(packed_weight);
  return zentorch_woq_repack_weight(weight_unpacked);
}

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_woq_linear(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor? bias=None, "
        "*, str zentorch_op_name='zentorch::zentorch_woq_linear') -> Tensor");
  m.def("zentorch_woq_linear_relu(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "*, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_relu') -> Tensor");

  m.def("zentorch_woq_linear_sigmoid(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor? bias=None, *, str "
        "zentorch_op_name='zentorch::zentorch_woq_linear_sigmoid') -> Tensor");

  m.def("zentorch_woq_linear_gelu_tanh(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "*, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_gelu_tanh') -> Tensor");

  m.def("zentorch_woq_linear_gelu_erf(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "*, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_gelu_erf') -> Tensor");

  m.def("zentorch_woq_linear_add(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor add_input, Tensor? bias=None, *, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_add') -> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_woq_linear_mul_add(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor mul_input, Tensor add_input, Tensor? bias=None, *, str "
        "zentorch_op_name= 'zentorch::zentorch_woq_linear_mul_add') -> Tensor",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_woq_linear_add_add(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor add_input, Tensor add_input_2, Tensor? bias=None, *, str "
        "zentorch_op_name='zentorch::zentorch_woq_linear_add_add') -> Tensor",
        {at::Tag::needs_fixed_stride_order});

  // `.out` variants:
  m.def("zentorch_woq_linear.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "str zentorch_op_name='zentorch::zentorch_woq_linear_out', "
        "*, Tensor(a!) out) -> ()");
  m.def("zentorch_woq_linear_relu.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "str zentorch_op_name='zentorch::zentorch_woq_linear_relu_out', "
        "*, Tensor(a!) out) -> ()");
  m.def("zentorch_woq_linear_sigmoid.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "str zentorch_op_name='zentorch::zentorch_woq_linear_sigmoid_out', "
        "*, Tensor(a!) out) -> ()");
  m.def("zentorch_woq_linear_gelu_tanh.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "str zentorch_op_name='zentorch::zentorch_woq_linear_gelu_tanh_out', "
        "*, Tensor(a!) out) -> ()");
  m.def("zentorch_woq_linear_gelu_erf.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, Tensor? bias=None, "
        "str zentorch_op_name='zentorch::zentorch_woq_linear_gelu_erf_out', "
        "*, Tensor(a!) out) -> ()");

  m.def("zentorch_woq_linear_add.out(Tensor input, Tensor weight, "
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor add_input, Tensor? bias=None, str zentorch_op_name="
        "'zentorch::zentorch_woq_linear_add_out', *, Tensor(a!) out) -> ()",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_woq_linear_mul_add.out(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor mul_input, Tensor add_input, Tensor? bias=None, str "
        "zentorch_op_name='zentorch::zentorch_woq_linear_mul_add_out', "
        "*, Tensor(a!) out) -> ()",
        {at::Tag::needs_fixed_stride_order});
  m.def("zentorch_woq_linear_add_add.out(Tensor input, Tensor weight,"
        "Tensor weight_scales, Tensor? weight_zero_points, "
        "Tensor add_input, Tensor add_input_2, Tensor? bias=None, str "
        "zentorch_op_name='zentorch::zentorch_woq_linear_add_add_out', "
        "*, Tensor(a!) out) -> ()",
        {at::Tag::needs_fixed_stride_order});

  m.def("zentorch_woq_repack_weight(Tensor unpacked_weight) -> Tensor");

  m.def("zentorch_woq_repack_from_int4pack(Tensor "
        "packed_weight) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl(
      "zentorch_woq_linear",
      TORCH_BOX(
          (&zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::POST_OP_NONE>)));

  m.impl(
      "zentorch_woq_linear_relu",
      TORCH_BOX((&zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::RELU>)));

  m.impl("zentorch_woq_linear_sigmoid",
         TORCH_BOX(
             (&zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::SIGMOID>)));

  m.impl("zentorch_woq_linear_add",
         TORCH_BOX((&zentorch::zentorch_woq_linear_unary_binary<
                    UNARY_POST_OP::POST_OP_NONE, BINARY_POST_OP::ADD>)));

  m.impl("zentorch_woq_linear_gelu_tanh",
         TORCH_BOX(
             (&zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::GELU_TANH>)));

  m.impl("zentorch_woq_linear_gelu_erf",
         TORCH_BOX(
             (&zentorch::zentorch_woq_linear_unary<UNARY_POST_OP::GELU_ERF>)));

  m.impl("zentorch_woq_linear_mul_add",
         TORCH_BOX((&zentorch::zentorch_woq_linear_binary_binary<
                    BINARY_POST_OP::MUL, BINARY_POST_OP::ADD>)));
  m.impl("zentorch_woq_linear_add_add",
         TORCH_BOX((&zentorch::zentorch_woq_linear_binary_binary<
                    BINARY_POST_OP::ADD, BINARY_POST_OP::ADD>)));

  m.impl("zentorch_woq_linear.out",
         TORCH_BOX((&zentorch::zentorch_woq_linear_unary_out<
                    UNARY_POST_OP::POST_OP_NONE>)));
  m.impl("zentorch_woq_linear_relu.out",
         TORCH_BOX(
             (&zentorch::zentorch_woq_linear_unary_out<UNARY_POST_OP::RELU>)));
  m.impl(
      "zentorch_woq_linear_sigmoid.out",
      TORCH_BOX(
          (&zentorch::zentorch_woq_linear_unary_out<UNARY_POST_OP::SIGMOID>)));
  m.impl(
      "zentorch_woq_linear_gelu_tanh.out",
      TORCH_BOX((
          &zentorch::zentorch_woq_linear_unary_out<UNARY_POST_OP::GELU_TANH>)));
  m.impl(
      "zentorch_woq_linear_gelu_erf.out",
      TORCH_BOX(
          (&zentorch::zentorch_woq_linear_unary_out<UNARY_POST_OP::GELU_ERF>)));

  m.impl("zentorch_woq_linear_add.out",
         TORCH_BOX((&zentorch::zentorch_woq_linear_unary_binary_out<
                    UNARY_POST_OP::POST_OP_NONE, BINARY_POST_OP::ADD>)));
  m.impl("zentorch_woq_linear_mul_add.out",
         TORCH_BOX((&zentorch::zentorch_woq_linear_binary_binary_out<
                    BINARY_POST_OP::MUL, BINARY_POST_OP::ADD>)));
  m.impl("zentorch_woq_linear_add_add.out",
         TORCH_BOX((&zentorch::zentorch_woq_linear_binary_binary_out<
                    BINARY_POST_OP::ADD, BINARY_POST_OP::ADD>)));

  m.impl("zentorch_woq_repack_weight",
         TORCH_BOX(&zentorch::zentorch_woq_repack_weight));
  m.impl("zentorch_woq_repack_from_int4pack",
         TORCH_BOX(&zentorch::zentorch_woq_repack_from_int4pack));
}

} // namespace zentorch
