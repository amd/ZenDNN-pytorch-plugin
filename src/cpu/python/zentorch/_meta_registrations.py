# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch._inductor.lowering import make_fallback
import functools


@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library("zentorch", "IMPL", "Meta")


def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        get_meta_lib().impl(
            getattr(getattr(torch.ops.zentorch, op_name), overload_name), fn
        )
        return fn

    return wrapper


# During graph compilation, inductor runs the operators with
# faketensor data to get the output shapes. And it looks for
# operators with {op}.default implementations. Hence, ops are registered
# in this file with zentorch library. The reason every op is registered
# with Meta dispatch key is that to align with pytorch framework.

# More details can be found from below link
# https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html


@register_meta("zendnn_addmm")
def meta_zendnn_addmm(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
    fuse=0,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zendnn_addmm_1dbias")
def meta_zendnn_addmm_1dbias(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
    fuse=0,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zendnn_mm")
def meta_zendnn_mm(
    input,
    weight,
    fuse=0,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zendnn_bmm")
def meta_zendnn_bmm(input, weight):
    return input.new_empty((input.shape[0], input.shape[1], weight.shape[-1]))


@register_meta("zendnn_baddbmm")
def meta_zendnn_baddbmm(bias, input, weight, alpha=1, beta=1):
    return input.new_empty((input.shape[0], input.shape[1], weight.shape[-1]))


@register_meta("zendnn_embedding_bag")
def meta_zendnn_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    num_bags = offsets.size(0)

    output = weight.new_empty(num_bags, weight.size(1))
    bag_size = indices.new_empty(offsets.size())
    offset2bag = offsets.new_empty(0)
    max_indices = offsets.new_empty(bag_size.size())
    return output, offset2bag, bag_size, max_indices


@register_meta("zendnn_embedding")
def meta_zendnn_embedding(
    weight,
    indices,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    dim_embedding = weight.size(1)
    num_bags = indices.size(0)

    output = weight.new_empty(num_bags, dim_embedding)
    return output


@register_meta("zendnn_horizontal_embedding_bag_group")
def meta_zendnn_horizontal_embedding_bag_group(
    weight,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    output_list = []

    for i in range(len(weight)):
        num_bags = offsets[i].size(0)
        output = weight[i].new_empty(num_bags, weight[i].size(1))
        bag_size = indices[i].new_empty(offsets[i].size())
        offset2bag = offsets[i].new_empty(0)
        max_indices = offsets[i].new_empty(bag_size.size())

        output_list.extend([output, offset2bag, bag_size, max_indices])

    return output_list


@register_meta("zendnn_horizontal_embedding_group")
def meta_zendnn_horizontal_embedding_group(
    weight, indices, padding_idx, scale_grad_by_freq, sparse
):
    output_list = []

    for i in range(len(weight)):
        dim_embedding = weight[i].size(1)
        num_bags = indices[i].size(0)
        output = weight[i].new_empty(num_bags, dim_embedding)

        output_list.append(output)

    return output_list


@register_meta("zendnn_vertical_mlp_group")
def meta_zendnn_vertical_mlp_group(self, inputs, weight, betas, alphas, fuse):
    # For the functionality of GroupMLP op, the outputs of one MLP will act as
    # the input for the next MLP. That is why, overwriting the same variable
    # instead of creating multiple variables, and finally returning an empty
    # tensor of the final shape.
    for idx in range(len(weight)):
        inputs = meta_zendnn_addmm(
            self[idx], inputs, weight[idx], betas[idx], alphas[idx], fuse[idx]
        )

    return inputs.new_empty(inputs.size())


@register_meta("zendnn_attn_horizontal_mlp_group")
def meta_zendnn_attn_horizontal_mlp_group(
    self, inputs, weights, betas, alphas, fuse, is_zendnnmm
):
    output_list = []
    for idx in range(len(inputs)):
        output = meta_zendnn_addmm(
            self[idx], inputs[idx], weights[idx], betas[idx], alphas[idx], fuse[idx]
        )
        output_list.append(output)
    return output_list


@register_meta("zendnn_fused_eb_mlp")
def meta_zendnn_fused_eb_mlp(
    eb_weight,
    eb_indices,
    eb_offsets,
    eb_scale_grad_by_freq,
    eb_mode,
    eb_sparse,
    eb_per_sample_weights_opt,
    eb_include_last_offset,
    eb_padding_idx,
    mlp_self,
    mlp_inputs,
    mlp_weight,
    mlp_betas,
    mlp_alphas,
    mlp_fuse,
):

    output = meta_zendnn_horizontal_embedding_bag_group(
        eb_weight,
        eb_indices,
        eb_offsets,
        eb_scale_grad_by_freq,
        eb_mode,
        eb_sparse,
        eb_per_sample_weights_opt,
        eb_include_last_offset,
        eb_padding_idx,
    )

    output.append(
        meta_zendnn_vertical_mlp_group(
            mlp_self, mlp_inputs, mlp_weight, mlp_betas, mlp_alphas, mlp_fuse
        )
    )

    return output


make_fallback(torch.ops.zentorch.zendnn_addmm)
make_fallback(torch.ops.zentorch.zendnn_addmm_1dbias)
make_fallback(torch.ops.zentorch.zendnn_embedding_bag)
make_fallback(torch.ops.zentorch.zendnn_embedding)
make_fallback(torch.ops.zentorch.zendnn_bmm)
make_fallback(torch.ops.zentorch.zendnn_baddbmm)
make_fallback(torch.ops.zentorch.zendnn_mm)
make_fallback(torch.ops.zentorch.zendnn_horizontal_embedding_bag_group)
make_fallback(torch.ops.zentorch.zendnn_horizontal_embedding_group)
make_fallback(torch.ops.zentorch.zendnn_vertical_mlp_group)
make_fallback(torch.ops.zentorch.zendnn_attn_horizontal_mlp_group)
make_fallback(torch.ops.zentorch.zendnn_fused_eb_mlp)
