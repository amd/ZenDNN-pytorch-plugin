/***************************************************************************************************************************
Modifications Copyright(c) 2025 Advanced Micro Devices, Inc.
All rights reserved.

Was sourced from
https: //
github.com/intel/intel-extension-for-pytorch/blob/main/csrc/cpu/aten/kernels/MaskedMultiHeadAttentionKrnl.cpp
***************************************************************************************************************************/

#include "Utils.hpp"
#include <torch/all.h>

namespace zentorch {

template <typename T>
inline void
attention_mask_2d_to_4d(const T *attention_mask_ptr, T *causal_4d_mask_ptr,
                        const at::Tensor &finfo_min, int64_t batch_size,
                        int64_t seq_length, int64_t src_length,
                        int64_t past_key_value_length, int64_t length,
                        int64_t diagonal) {
  T finfo_min_val = finfo_min.item<T>();

#pragma omp parallel for collapse(3)
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t l = 0; l < seq_length; ++l) {
      for (int64_t c = 0; c < length; ++c) {
        int64_t idx = b * seq_length * length + l * length + c;
        T value = finfo_min_val;
        if (l + diagonal <= c && l + past_key_value_length >= c) {
          value = 0;
        }
        if (c < src_length) {
          T inverted_mask_value = 1.0 - attention_mask_ptr[b * src_length + c];
          if (inverted_mask_value != 0) {
            value = finfo_min_val;
          }
        }
        causal_4d_mask_ptr[idx] = value;
      }
    }
  }
}

at::Tensor prepare_4d_causal_attention_mask_kernel_impl(
    const at::Tensor &attention_mask, const at::Tensor &inputs_embeds,
    int64_t past_key_value_length, const at::Tensor &finfo_min,
    int64_t sliding_window) {

  auto dtype = inputs_embeds.scalar_type();
  int64_t batch_size = inputs_embeds.size(0);
  int64_t seq_length = inputs_embeds.size(1);
  int64_t src_length = attention_mask.size(-1);
  int64_t length = seq_length + past_key_value_length;
  int64_t diagonal = past_key_value_length - sliding_window;

  at::Tensor causal_4d_mask = torch::empty({batch_size, 1, seq_length, length},
                                           inputs_embeds.options());
  if (dtype == at::kFloat) {
    float *attention_mask_ptr = attention_mask.data_ptr<float>();
    float *causal_4d_mask_ptr = causal_4d_mask.data_ptr<float>();
    attention_mask_2d_to_4d<float>(
        attention_mask_ptr, causal_4d_mask_ptr, finfo_min, batch_size,
        seq_length, src_length, past_key_value_length, length, diagonal);
  } else if (dtype == at::kBFloat16) {
    at::BFloat16 *attention_mask_ptr = attention_mask.data_ptr<at::BFloat16>();
    at::BFloat16 *causal_4d_mask_ptr = causal_4d_mask.data_ptr<at::BFloat16>();
    attention_mask_2d_to_4d<at::BFloat16>(
        attention_mask_ptr, causal_4d_mask_ptr, finfo_min, batch_size,
        seq_length, src_length, past_key_value_length, length, diagonal);
  } else {
    ZENTORCH_CHECK(false, "zentorch::prepare_4d_causal_attention_mask_"
                          "kernel_impl supports only float and bfloat16 "
                          "datatypes");
  }

  return causal_4d_mask;
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("prepare_4d_causal_attention_mask(Tensor attention_mask, "
        "Tensor inputs_embeds, int past_key_value_length, Tensor "
        "finfo_min, int "
        "sliding_window)-> (Tensor)");
}
TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("prepare_4d_causal_attention_mask",
         prepare_4d_causal_attention_mask_kernel_impl);
}

} // namespace zentorch
