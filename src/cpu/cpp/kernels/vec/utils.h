/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Was sourced from
 * https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/csrc/cpu/vec/ref
 * IPEX commit ID: d3c52443e
 ******************************************************************************/

// zentorch:: APIs that are suffix with 256 will be invoked on AVX256 m/c

// TODO:: Observing seg-falut issues when enabling omp pragmas in AVX256
// implementations will be fixed in later patches
#pragma once
#define ZENTORCH_FORCE_INLINE inline __attribute__((always_inline))

namespace zentorch {
template <typename dst_type, typename src_type>
ZENTORCH_FORCE_INLINE void move_ker_ref(dst_type *inout, const src_type *in,
                                        int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(inout + i) = *(in + i);
  }
}

template <typename dst_type, typename src_type>
ZENTORCH_FORCE_INLINE void move_ker(dst_type *inout, const src_type *in,
                                    int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(inout + i) = *(in + i);
  }
}

template <typename T>
ZENTORCH_FORCE_INLINE void zero_ker_ref(T *out, int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(out + i) = 0;
  }
}

template <typename T> ZENTORCH_FORCE_INLINE void zero_ker(T *out, int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(out + i) = 0;
  }
}

template <typename dst_type, typename src_type>
ZENTORCH_FORCE_INLINE void add_ker(dst_type *inout, const src_type *in,
                                   int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(inout + i) += *(in + i);
  }
}

template <typename dst_type, typename src_type>
ZENTORCH_FORCE_INLINE void add_ker_ref(dst_type *inout, const src_type *in,
                                       int64_t len) {
#pragma omp simd
  for (int64_t i = 0; i < len; i++) {
    *(inout + i) += *(in + i);
  }
}

} // namespace zentorch