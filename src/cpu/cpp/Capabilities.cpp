/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

// The ISA probes are exposed as ops, mirroring the cache flush in
// WeightReorder.cpp, so they are reachable from both libraries. The same checks
// are available through the _C pybind module (Bindings.cpp), but _C links
// libtorch_python.so and libzentorch.so and so cannot be imported when the
// portable library is the one that loaded. Callers that must know what the
// machine supports - the test suite selecting dtypes, a serving stack choosing
// a precision - would otherwise have no answer in that mode.
//
// Keyed on CompositeExplicitAutograd rather than CPU because these schemas take
// no tensors, so the dispatcher computes an empty key set and needs a
// backend-agnostic kernel.

#include "Utils.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

namespace zentorch {

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_is_avx512_supported() -> bool");
  m.def("zentorch_is_bf16_supported() -> bool");
  m.def("zentorch_is_fp16_supported() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CompositeExplicitAutograd, m) {
  m.impl("zentorch_is_avx512_supported",
         TORCH_BOX(&zentorch::is_avx512_supported));
  m.impl("zentorch_is_bf16_supported",
         TORCH_BOX(&zentorch::zendnn_bf16_device_check));
  m.impl("zentorch_is_fp16_supported",
         TORCH_BOX(&zentorch::zendnn_fp16_device_check));
}

} // namespace zentorch
