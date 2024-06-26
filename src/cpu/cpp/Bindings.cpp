/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

/*
        TORCH_LIBRARY is used for all ops which will replace ATen/prims ops in
   fx based graph optimizations. Few guidelines for prototypes.
                - If there is simlar op in ATen of PyTorch, please check
   "aten/src/ATen/native/native_functions.yaml" in PyTorch repo.
                - Our op arguments should be superset of the corresponding
   arguments in ATen op.
                - Our op arguments should match the arguments of corresponding
   op in both order of the arguments and type.
                - Our op specific arguments should be at the end of the list.
                - All ops should have prefix "zendnn_", for example
   zendnn_<corresponding op name>.
*/

#include "ZenTorchOps.hpp"

TORCH_LIBRARY(zentorch, m) {
  m.def("zendnn_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("zendnn_embedding(Tensor weight, Tensor indices, "
        "int padding_idx=-1, bool scale_grad_by_freq=False, "
        "bool sparse=False) -> Tensor");

  /*
    The fuse variable is introduced to set the post-ops or fusion-ops;
    by default, fuse = 0,
    fuse = 1 for relu op,
    fuse = 2 for gelu approximate (tanh)
    fuse = 3 for gelu exact (erf)
  */

  m.def("zendnn_mm(Tensor self, Tensor mat2, *, int fuse=0) -> Tensor");
  m.def("zendnn_bmm(Tensor self, Tensor mat2) -> Tensor");
  m.def("zendnn_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, "
        "Scalar alpha=1, int fuse=0) -> Tensor");
  // for 1d bias
  m.def("zendnn_addmm_1dbias(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, int fuse=0) -> Tensor");
  m.def("zendnn_baddbmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, Scalar alpha=1) -> Tensor");
  m.def("zendnn_horizontal_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx) -> Tensor[]");
  m.def("zendnn_horizontal_embedding_group(Tensor[] weight, Tensor[] indices, "
        "int[] padding_idx, int[] scale_grad_by_freq, "
        "int[] sparse) -> Tensor[]");
  m.def("zendnn_vertical_mlp_group(Tensor[] self, Tensor inputs, "
        "Tensor[] weight, float[] betas, float[] alphas, "
        "int[] fuse) -> Tensor");
  m.def("zendnn_attn_horizontal_mlp_group(Tensor[] self, Tensor[] inputs, "
        "Tensor[] weights, "
        "float[] betas, float[] alphas, int[] fuse, int[] is_zendnnmm) -> "
        "Tensor[]");
  m.def(
      "zendnn_fused_eb_mlp(Tensor[] eb_weight, Tensor[] eb_indices, "
      "Tensor[] eb_offsets, int[] eb_scale_grad_by_freq, int[] eb_mode, int[] "
      "eb_sparse, Tensor?[] eb_per_sample_weights, "
      "int[] eb_include_last_offset, int[] eb_padding_idx, Tensor[] mlp_self, "
      "Tensor mlp_inputs, Tensor[] mlp_weight, float[] mlp_betas, "
      "float[] mlp_alphas, int[] mlp_fuse) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zendnn_embedding_bag", zentorch::zendnn_embedding_bag_impl);
  m.impl("zendnn_embedding", zentorch::zendnn_embedding_impl);
  m.impl("zendnn_mm", zentorch::zendnn_mm);
  m.impl("zendnn_bmm", zentorch::zendnn_bmm);
  m.impl("zendnn_addmm", zentorch::zendnn_addmm);
  m.impl("zendnn_addmm_1dbias", zentorch::zendnn_addmm_1dbias);
  m.impl("zendnn_baddbmm", zentorch::zendnn_baddbmm);
  m.impl("zendnn_horizontal_embedding_bag_group",
         zentorch::zendnn_horizontal_embedding_bag_group);
  m.impl("zendnn_horizontal_embedding_group",
         zentorch::zendnn_horizontal_embedding_group);
  m.impl("zendnn_vertical_mlp_group", zentorch::zendnn_vertical_mlp_group);
  m.impl("zendnn_attn_horizontal_mlp_group",
         zentorch::zendnn_attn_horizontal_mlp_group);
  m.impl("zendnn_fused_eb_mlp", zentorch::zendnn_fused_eb_mlp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("show_config", zentorch::show_config);

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check);

  m.def("zendnn_matmul_impl", zentorch::zendnn_matmul_impl, py::arg("mat1"),
        py::arg("mat2"), py::arg("bias"), py::arg("self_or_result"),
        py::arg("beta") = 0.0f, py::arg("alpha") = 1.0f, py::arg("fuse") = 0);
}
