# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
import torch
import torch.nn as nn

mode_dict = {
    "sum": 0,
    "mean": 1,
    "max": 2,
}


# This is a custom ZenTorchWOQEmbeddingBag module to support woq EmbeddingBag
# modules through ZenTorch optimization and execution flow.
class ZenTorchWOQEmbeddingBag(nn.Module):
    def __init__(
        self,
        mod,
        packed_weight,
        weight_scales,
        weight_zero_points=False,
        group_size=None,
        weight_bits=4,
        compute_dtype="bfloat16",
        scale_dtype="float",
        quant_dtype="uint4",
    ):
        r"""Create a ZenTorchWOQEmbeddingBag module from a
        float module and int4 packed_weight.
        Weight is dequantized at runtime for computation.
        Args:
            mod (Module): A float nn.EmbeddingBag module provided by the user.
            packed_weight: Tensor that is fused tensor of weight,scales and bias.
            weight_scales: weight_scales for packed_weight.
            weight_zero_points : Zero points for packed_weight.
            group_size : Group size for weight quantization.
            weight_bits : Number of bits for weight quantization.
            compute_dtype : Dtype of the module computation.
            scale_dtype : Dtype of the scale which help to determine output dtype.
            quant_dtype : Dtype of quant scheme.
        """

        float_module = torch.nn.EmbeddingBag
        assert type(mod) is not type(
            float_module
        ), "mod and float_module must be of the same type"

        if hasattr(mod, "mode"):
            self.mode = mode_dict[mod.mode]
        else:
            self.mode = mode_dict["mean"]  # Default mode
        if hasattr(mod, "sparse"):
            self.sparse = mod.sparse
        else:
            self.sparse = False  # Default value
        if hasattr(mod, "include_last_offset"):
            self.include_last_offset = mod.include_last_offset
        else:
            self.include_last_offset = False  # Default value
        if hasattr(mod, "padding_idx"):
            if mod.padding_idx is not None:
                self.padding_idx = mod.padding_idx
            else:
                self.padding_idx = -1
        else:
            self.padding_idx = -1  # Default value

        if hasattr(mod, "scale_grad_by_freq"):
            self.scale_grad_by_freq = mod.scale_grad_by_freq
        else:
            self.scale_grad_by_freq = False  # Default value
        self.num_embeddings = mod.num_embeddings
        self.embedding_dim = mod.embedding_dim
        self.packed_weight = packed_weight
        self.weight_bits = weight_bits
        self.compute_dtype = compute_dtype
        self.group_size = group_size
        self.dtype = self._get_torch_type(compute_dtype)
        super(ZenTorchWOQEmbeddingBag, self).__init__()

    def _get_torch_type(self, str_type):
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        assert str_type in dtypes, "Unsupported dtype for WOQ embedding bag compute!"
        return dtypes[str_type]

    def _get_name(self):
        return "ZenTorchWOQEmbeddingBag"

    def extra_repr(self):
        extra_repr_str = "num_embeddings={}, embedding_dim={}, dtype={}".format(
            self.num_embeddings, self.embedding_dim, self.compute_dtype
        )
        extra_repr_str += ", scale_grad_by_freq={}".format(self.scale_grad_by_freq)
        extra_repr_str += ", mode={}".format(self.mode)
        extra_repr_str += ", sparse={}".format(self.sparse)
        extra_repr_str += ", include_last_offset={}".format(self.include_last_offset)
        extra_repr_str += ", weight_bits={}".format(self.weight_bits)
        extra_repr_str += ", group_size={}".format(self.group_size)
        return extra_repr_str

    def forward(self, input, offset, per_sample_weights=None):
        return torch.ops.zentorch.zentorch_quant_embedding_bag(
            # Tensor weight
            self.packed_weight,
            # Tensor indices
            input,
            # Tensor offsets
            offset,
            # int num_bits_per_weight
            4,
            self.dtype,
            # bool scale_grad_by_freq
            self.scale_grad_by_freq,
            # int mode
            self.mode,
            # bool sparse
            self.sparse,
            # Tensor? per_sample_weights
            None,
            # bool include_last_offset
            self.include_last_offset,
            # SymInt padding_idx
            self.padding_idx,
        )
