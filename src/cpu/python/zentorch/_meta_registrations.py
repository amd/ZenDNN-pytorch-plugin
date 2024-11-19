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


@register_meta("zentorch_addmm")
def meta_zentorch_addmm(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_relu")
def meta_zentorch_addmm_relu(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_silu")
def meta_zentorch_addmm_silu(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_gelu_tanh")
def meta_zentorch_addmm_gelu_tanh(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_gelu_erf")
def meta_zentorch_addmm_gelu_erf(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_1dbias")
def meta_zentorch_addmm_1dbias(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_1dbias_add")
def meta_zentorch_addmm_1dbias_add(
    bias,
    input,
    weight,
    add_input,
    alpha=1,
    beta=1,
):
    return add_input.new_empty(add_input.size())


@register_meta("zentorch_addmm_1dbias_add_add")
def meta_zentorch_addmm_1dbias_add_add(
    bias,
    input,
    weight,
    add1_input,
    add2_input,
    alpha=1,
    beta=1,
):
    return add2_input.new_empty(add2_input.size())


@register_meta("zentorch_addmm_1dbias_relu")
def meta_zentorch_addmm_1dbias_relu(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_1dbias_silu")
def meta_zentorch_addmm_1dbias_silu(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_1dbias_gelu_tanh")
def meta_zentorch_addmm_1dbias_gelu_tanh(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_addmm_1dbias_gelu_erf")
def meta_zentorch_addmm_1dbias_gelu_erf(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_mm")
def meta_zentorch_mm(
    input,
    weight,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_mm_relu")
def meta_zentorch_mm_relu(
    input,
    weight,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_mm_silu")
def meta_zentorch_mm_silu(
    input,
    weight,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_mm_gelu_tanh")
def meta_zentorch_mm_gelu_tanh(
    input,
    weight,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_mm_gelu_erf")
def meta_zentorch_mm_gelu_erf(
    input,
    weight,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zentorch_bmm")
def meta_zentorch_bmm(input, weight):
    return input.new_empty((input.shape[0], input.shape[1], weight.shape[-1]))


@register_meta("zentorch_baddbmm")
def meta_zentorch_baddbmm(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
):
    return bias.new_empty(input.shape[0], input.shape[1], weight.shape[-1])


@register_meta("zentorch_mm_silu_mul")
def meta_zentorch_mm_silu_mul(input, weight, mul_tensor):
    return mul_tensor.new_empty(mul_tensor.size())


@register_meta("zentorch_addmm_silu_mul")
def meta_zentorch_addmm_silu_mul(bias, input, weight, mul_tensor, alpha=1, beta=1):
    return mul_tensor.new_empty(mul_tensor.size())


@register_meta("zentorch_addmm_1dbias_silu_mul")
def meta_zentorch_addmm_1dbias_silu_mul(
    bias, input, weight, mul_tensor, alpha=1, beta=1
):
    return mul_tensor.new_empty(mul_tensor.size())


@register_meta("zentorch_embedding_bag")
def meta_zentorch_embedding_bag(
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
    if include_last_offset:
        num_bags = num_bags - 1
    output = weight.new_empty(num_bags, weight.size(1))
    bag_size = indices.new_empty(offsets.size())
    offset2bag = offsets.new_empty(0)
    max_indices = offsets.new_empty(bag_size.size())
    return output, offset2bag, bag_size, max_indices


@register_meta("zentorch_embedding")
def meta_zentorch_embedding(
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


@register_meta("zentorch_horizontal_embedding_bag_group")
def meta_zentorch_horizontal_embedding_bag_group(
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
        if include_last_offset[i]:
            num_bags = num_bags - 1
        output = weight[i].new_empty(num_bags, weight[i].size(1))
        bag_size = indices[i].new_empty(offsets[i].size())
        offset2bag = offsets[i].new_empty(0)
        max_indices = offsets[i].new_empty(bag_size.size())

        output_list.extend([output, offset2bag, bag_size, max_indices])

    return output_list


@register_meta("zentorch_horizontal_embedding_group")
def meta_zentorch_horizontal_embedding_group(
    weight, indices, padding_idx, scale_grad_by_freq, sparse
):
    output_list = []

    for i in range(len(weight)):
        dim_embedding = weight[i].size(1)
        num_bags = indices[i].size(0)
        output = weight[i].new_empty(num_bags, dim_embedding)

        output_list.append(output)

    return output_list


zentorch_addmm_mappings = {
    0: meta_zentorch_addmm,
    1: meta_zentorch_addmm_relu,
    2: meta_zentorch_addmm_gelu_tanh,
    3: meta_zentorch_addmm_gelu_erf,
    4: meta_zentorch_addmm_silu,
}

zentorch_addmm_1dbias_mappings = {
    0: meta_zentorch_addmm_1dbias,
    1: meta_zentorch_addmm_1dbias_relu,
    2: meta_zentorch_addmm_1dbias_gelu_tanh,
    3: meta_zentorch_addmm_1dbias_gelu_erf,
    4: meta_zentorch_addmm_1dbias_silu,
}

zentorch_mm_mappings = {
    0: meta_zentorch_mm,
    1: meta_zentorch_mm_relu,
    2: meta_zentorch_mm_gelu_tanh,
    3: meta_zentorch_mm_gelu_erf,
    4: meta_zentorch_mm_silu,
}


#  TODO: return data type has to be taken care appropriately
@register_meta("zentorch_vertical_mlp_group")
def meta_zentorch_vertical_mlp_group(self, inputs, weight, betas, alphas, fuse):
    # For the functionality of GroupMLP op, the outputs of one MLP will act as
    # the input for the next MLP. That is why, overwriting the same variable
    # instead of creating multiple variables, and finally returning an empty
    # tensor of the final shape.
    for idx in range(len(weight)):
        inputs = zentorch_addmm_mappings[fuse[idx]](
            self[idx], inputs, weight[idx], betas[idx], alphas[idx]
        )

    return inputs.new_empty(inputs.size())


@register_meta("zentorch_attn_qkv_fusion")
def meta_zentorch_attn_qkv_fusion(
    self, inputs, weights, betas, alphas, fuse, is_zentorch_mm
):
    output_list = []
    if is_zentorch_mm is False:
        for idx in range(len(inputs)):
            output = zentorch_addmm_1dbias_mappings[fuse[idx]](
                self[idx], inputs[idx], weights[idx], betas[idx], alphas[idx]
            )
            output_list.append(output)
    else:
        for idx in range(len(inputs)):
            output = zentorch_mm_mappings[fuse[idx]](inputs[idx], weights[idx])
            output_list.append(output)

    return output_list


@register_meta("zentorch_fused_eb_mlp")
def meta_zentorch_fused_eb_mlp(
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

    output = meta_zentorch_horizontal_embedding_bag_group(
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
        meta_zentorch_vertical_mlp_group(
            mlp_self, mlp_inputs, mlp_weight, mlp_betas, mlp_alphas, mlp_fuse
        )
    )

    return output


# meta registration for RoPE
@register_meta("zentorch_rope")
def meta_zentorch_rope(
    t_in,
    t_emb_pos,
    t_pos,
    N,
    H,
    offset,
    rotary_ndims,
):
    ndims = t_in.dim()
    stride_s = t_in.stride(1)
    batch = t_in.shape[0]
    seq_len = t_in.shape[1]
    concat_qkv = False
    if ndims == 3 and stride_s > N * H:
        concat_qkv = True
        kv_head = (t_in.shape[2] - N * H) // (2 * H)
    if not concat_qkv:
        return (
            t_in.new_empty(t_in.shape).contiguous(),
            None,
            None,
        )
    else:
        return (
            t_in.new_empty(batch, seq_len, N, H),
            t_in.new_empty(batch, seq_len, kv_head, H),
            t_in.new_empty(batch, seq_len, kv_head, H),
        )


@register_meta("zentorch_masked_multihead_self_attention")
def meta_masked_multihead_self_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    beam_idx,
    seq_info,
    scale_attn,
    max_positions,
    head_mask,
    attention_mask,
    add_casual_mask=None,
):
    attn_output = query.new_empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3])
    )
    if query.dtype == torch.bfloat16:
        attn_output.as_strided_(
            attn_output.shape,
            (
                query.shape[1] * query.shape[2] * query.shape[3],
                query.shape[3],
                query.shape[2] * query.shape[3],
                1,
            ),
        )
    attn_weights = None
    key_cache_out = query.new_empty(
        (key_cache.shape[0], key_cache.shape[1], key.shape[2], key.shape[3])
    )
    value_cache_out = query.new_empty(
        (value_cache.shape[0], value_cache.shape[1], value.shape[2], value.shape[3])
    )
    beam_idx_out = query.new_empty(beam_idx.shape)
    return (attn_output, attn_weights, key_cache_out, value_cache_out, beam_idx_out)


@register_meta("zentorch_woq_linear")
def meta_zentorch_woq_linear(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    out_dim = list(input.size())
    unpacking_ratio = 1
    if qweight.dtype == torch.int32:
        bits_in_1_byte = 8
        total_bits = qweight.element_size() * bits_in_1_byte
        unpacking_ratio = total_bits // weight_bits
    out_dim[-1] = qweight.size(1) * unpacking_ratio
    return input.new_empty(out_dim)


@register_meta("zentorch_woq_linear_relu")
def meta_zentorch_woq_linear_relu(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return meta_zentorch_woq_linear(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    )


@register_meta("zentorch_woq_linear_silu")
def meta_zentorch_woq_linear_silu(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return meta_zentorch_woq_linear(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    )


@register_meta("zentorch_woq_linear_gelu_erf")
def meta_zentorch_woq_linear_gelu_erf(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return meta_zentorch_woq_linear(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    )


@register_meta("zentorch_woq_linear_gelu_tanh")
def meta_zentorch_woq_linear_gelu_tanh(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return meta_zentorch_woq_linear(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    )


@register_meta("zentorch_woq_linear_add")
def meta_zentorch_woq_linear_add(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    add_input,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return add_input.new_empty((add_input.size()))


@register_meta("zentorch_woq_linear_add_add")
def meta_zentorch_woq_linear_add_add(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    add1_input,
    add2_input,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return add2_input.new_empty((add2_input.size()))


@register_meta("zentorch_woq_linear_silu_mul")
def meta_zentorch_woq_linear_silu_mul(
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    mul_input,
    group_size,
    weight_bits=4,
    compute_dtype="bfloat16",
):
    return mul_input.new_empty((mul_input.size()))


make_fallback(torch.ops.zentorch.zentorch_addmm)
make_fallback(torch.ops.zentorch.zentorch_addmm_relu)
make_fallback(torch.ops.zentorch.zentorch_addmm_silu)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_relu)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_silu)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_add)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_add_add)
make_fallback(torch.ops.zentorch.zentorch_mm_silu_mul)
make_fallback(torch.ops.zentorch.zentorch_addmm_silu_mul)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_silu_mul)
make_fallback(torch.ops.zentorch.zentorch_embedding_bag)
make_fallback(torch.ops.zentorch.zentorch_embedding)
make_fallback(torch.ops.zentorch.zentorch_bmm)
make_fallback(torch.ops.zentorch.zentorch_baddbmm)
make_fallback(torch.ops.zentorch.zentorch_mm)
make_fallback(torch.ops.zentorch.zentorch_mm_relu)
make_fallback(torch.ops.zentorch.zentorch_mm_silu)
make_fallback(torch.ops.zentorch.zentorch_mm_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_mm_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_horizontal_embedding_bag_group)
make_fallback(torch.ops.zentorch.zentorch_horizontal_embedding_group)
make_fallback(torch.ops.zentorch.zentorch_vertical_mlp_group)
make_fallback(torch.ops.zentorch.zentorch_attn_qkv_fusion)
make_fallback(torch.ops.zentorch.zentorch_fused_eb_mlp)
make_fallback(torch.ops.zentorch.zentorch_rope)
make_fallback(torch.ops.zentorch.zentorch_masked_multihead_self_attention)
make_fallback(torch.ops.zentorch.zentorch_woq_linear)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_relu)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_silu)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_add)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_add_add)
