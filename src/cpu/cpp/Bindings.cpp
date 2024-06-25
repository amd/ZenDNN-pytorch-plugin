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
                - All ops should have prefix "zentorch_", for example
   zentorch_<corresponding op name>.
*/

#include "Ops.hpp"
#include "kernels/zen_cpukernels_ops.hpp"

TORCH_LIBRARY(zentorch, m) {
  m.def("zentorch_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1, str "
        "zentorch_op_name='zentorch::zentorch_embedding_bag') -> "
        "(Tensor, Tensor, Tensor, Tensor)");
  m.def("zentorch_embedding(Tensor weight, Tensor indices, "
        "int padding_idx=-1, bool scale_grad_by_freq=False, "
        "bool sparse=False, str "
        "zentorch_op_name='zentorch::zentorch_embedding') -> "
        "Tensor");
  m.def("zentorch_mm(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm') -> Tensor");
  m.def("zentorch_mm_relu(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_relu') -> Tensor");
  m.def("zentorch_mm_gelu_tanh(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_gelu_tanh') -> Tensor");
  m.def("zentorch_mm_gelu_erf(Tensor self, Tensor mat2, *, str "
        "zentorch_op_name='zentorch::zentorch_mm_gelu_erf') -> Tensor");
  m.def("zentorch_bmm(Tensor self, Tensor mat2, str "
        "zentorch_op_name='zentorch::zentorch_bmm') -> Tensor");
  m.def(
      "zentorch_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, "
      "Scalar alpha=1, str zentorch_op_name='zentorch::zentorch_addmm') "
      "-> Tensor");
  m.def("zentorch_addmm_relu(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_relu') -> Tensor");
  m.def("zentorch_addmm_gelu_tanh(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_gelu_tanh') -> "
        "Tensor");
  m.def("zentorch_addmm_gelu_erf(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, "
        "Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_gelu_erf') -> "
        "Tensor");
  // for 1d bias
  m.def("zentorch_addmm_1dbias(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_relu') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_relu(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_relu') -> "
        "Tensor");
  m.def("zentorch_addmm_1dbias_gelu_tanh(Tensor self, Tensor mat1, Tensor mat2,"
        " *, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_gelu_tanh') "
        "-> Tensor");
  m.def("zentorch_addmm_1dbias_gelu_erf(Tensor self, Tensor mat1, Tensor mat2,"
        " *, Scalar beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_addmm_1dbias_gelu_erf') "
        "-> Tensor");
  m.def("zentorch_baddbmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
        "beta=1, Scalar alpha=1, str "
        "zentorch_op_name='zentorch::zentorch_baddbmm') -> "
        "Tensor");
  m.def("zentorch_horizontal_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_embedding_bag_group') -> Tensor[]");
  m.def(
      "zentorch_horizontal_embedding_group(Tensor[] weight, Tensor[] indices, "
      "int[] padding_idx, int[] scale_grad_by_freq, "
      "int[] sparse, str zentorch_op_name = "
      "'zentorch::zentorch_horizontal_embedding_group') -> Tensor[]");
  m.def("zentorch_vertical_mlp_group(Tensor[] self, Tensor inputs, "
        "Tensor[] weight, float[] betas, float[] alphas, "
        "int[] fuse, str zentorch_op_name = "
        "'zentorch::zentorch_vertical_mlp_group') -> "
        "Tensor");
  m.def("zentorch_attn_horizontal_mlp_group(Tensor[] self, Tensor[] inputs, "
        "Tensor[] weights, "
        "float[] betas, float[] alphas, int[] fuse, int[] is_zentorchmm, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_attn_horizontal_mlp_group') -> "
        "Tensor[]");
  m.def(
      "zentorch_fused_eb_mlp(Tensor[] eb_weight, Tensor[] eb_indices, "
      "Tensor[] eb_offsets, int[] eb_scale_grad_by_freq, int[] eb_mode, int[] "
      "eb_sparse, Tensor?[] eb_per_sample_weights, "
      "int[] eb_include_last_offset, int[] eb_padding_idx, Tensor[] mlp_self, "
      "Tensor mlp_inputs, Tensor[] mlp_weight, float[] mlp_betas, "
      "float[] mlp_alphas, int[] mlp_fuse, str zentorch_op_name = "
      "'zentorch::zentorch_fused_eb_mlp') -> Tensor[]");
  m.def("zentorch_rope(Tensor t_in, Tensor t_emb_pos, Tensor t_pos, int N, int "
        "H, int offset, int rotary_dim, str zentorch_op_name = "
        "'zentorch::zentorch_rope') -> (Tensor, Tensor, Tensor)");
  m.def("zentorch_masked_multihead_self_attention(Tensor query, Tensor key, "
        "Tensor value, Tensor key_cache, "
        "Tensor value_cache, Tensor beam_idx, Tensor seq_info, float "
        "scale_attn, int max_positions, "
        "Tensor? head_mask, Tensor? attention_mask, bool? "
        "add_casual_mask=None, str zentorch_op_name = "
        "'zentorch::zentorch_masked_multihead_self_attention')-> (Tensor, "
        "Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag", zentorch::zentorch_embedding_bag_impl);
  m.impl("zentorch_embedding", zentorch::zentorch_embedding_impl);
  m.impl("zentorch_mm", zentorch::zentorch_mm<0>);
  m.impl("zentorch_mm_relu", zentorch::zentorch_mm<1>);
  m.impl("zentorch_mm_gelu_tanh", zentorch::zentorch_mm<2>);
  m.impl("zentorch_mm_gelu_erf", zentorch::zentorch_mm<3>);
  m.impl("zentorch_bmm", zentorch::zentorch_bmm);
  m.impl("zentorch_addmm", zentorch::zentorch_addmm<0>);
  m.impl("zentorch_addmm_relu", zentorch::zentorch_addmm<1>);
  m.impl("zentorch_addmm_gelu_tanh", zentorch::zentorch_addmm<2>);
  m.impl("zentorch_addmm_gelu_erf", zentorch::zentorch_addmm<3>);
  m.impl("zentorch_addmm_1dbias", zentorch::zentorch_addmm_1dbias<0>);
  m.impl("zentorch_addmm_1dbias_relu", zentorch::zentorch_addmm_1dbias<1>);
  m.impl("zentorch_addmm_1dbias_gelu_tanh", zentorch::zentorch_addmm_1dbias<2>);
  m.impl("zentorch_addmm_1dbias_gelu_erf", zentorch::zentorch_addmm_1dbias<3>);
  m.impl("zentorch_baddbmm", zentorch::zentorch_baddbmm);
  m.impl("zentorch_horizontal_embedding_bag_group",
         zentorch::zentorch_horizontal_embedding_bag_group);
  m.impl("zentorch_horizontal_embedding_group",
         zentorch::zentorch_horizontal_embedding_group);
  m.impl("zentorch_vertical_mlp_group", zentorch::zentorch_vertical_mlp_group);
  m.impl("zentorch_attn_horizontal_mlp_group",
         zentorch::zentorch_attn_horizontal_mlp_group);
  m.impl("zentorch_fused_eb_mlp", zentorch::zentorch_fused_eb_mlp);
  m.impl("zentorch_rope", zentorch::zentorch_rope_impl);
  m.impl("zentorch_masked_multihead_self_attention",
         zentorch::zentorch_masked_multihead_self_attention_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("show_config", zentorch::show_config);

  m.def("is_bf16_supported", zendnn::utils::zendnn_bf16_device_check);

  m.def("zentorch_matmul_impl", zentorch::zentorch_matmul_impl, py::arg("mat1"),
        py::arg("mat2"), py::arg("bias"), py::arg("self_or_result"),
        py::arg("beta") = 0.0f, py::arg("alpha") = 1.0f, py::arg("fuse") = 0,
        py::arg("zentorch_op_name") = "zentorch::zendnn_matmul_impl");
}
