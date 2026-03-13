# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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


@register_meta("zentorch_linear_unary")
def meta_zentorch_linear_unary(
    input,
    weight,
    bias=None,
    is_weight_prepacked=False,
    post_op="none",
    zentorch_op_name="zentorch::zentorch_linear_unary",
):
    out_dim = list(input.size())
    out_dim[-1] = weight.size(0)
    return input.new_empty(out_dim)


@register_meta("zentorch_linear_binary_binary")
def meta_zendnn_linear_binary_binary(
    input,
    weight,
    binary_input_1,
    binary_input_2,
    bias=None,
    is_weight_prepacked=False,
    post_op_1="none",
    post_op_2="none",
    zentorch_op_name="zentorch::zentorch_linear_binary_binary",
):
    return binary_input_2.new_empty(binary_input_2.shape)


@register_meta("zentorch_linear_unary_binary")
def meta_zendnn_linear_unary_binary(
    input,
    weight,
    binary_input,
    bias=None,
    is_weight_prepacked=False,
    post_op_1="none",
    post_op_2="none",
    zentorch_op_name="zentorch::zentorch_linear_unary_binary",
):
    return binary_input.new_empty(binary_input.shape)


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


@register_meta("zentorch_addmm_1dbias_mul_add")
def meta_zentorch_addmm_1dbias_mul_add(
    bias,
    input,
    weight,
    mul_input,
    add_input,
    alpha=1,
    beta=1,
):
    return add_input.new_empty(add_input.size())


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


if hasattr(torch.ops.zentorch, "zentorch_sdpa"):

    @register_meta("zentorch_sdpa")
    def meta_zentorch_sdpa(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        attn_mask=None,
        scale=None,
    ):
        batch_size = query.size(0)
        num_heads = query.size(1)
        max_seqlen_batch_q = query.size(2)

        attention = torch.empty_like(query)
        logsumexp = query.new_empty(
            (
                batch_size,
                max_seqlen_batch_q,
                num_heads,
            ),
            dtype=torch.float,
        ).transpose(1, 2)
        return (
            attention,
            logsumexp,
        )


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

    return output


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

        output_list.append(output)

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


@register_meta("zentorch_quant_embedding_bag", "out")
def meta_zentorch_quant_embedding_bag_out(
    output,
    weight,
    indices,
    offsets,
    num_bits_per_weight,
    output_dtype,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    return


@register_meta("zentorch_quant_embedding_bag")
def meta_zentorch_quant_embedding_bag(
    weight,
    indices,
    offsets,
    num_bits_per_weight,
    output_dtype,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    # TODO Remove this assumption
    # Currently the scale is packed as float16
    # zero point is also packed as float16
    # so scale+zp takes 32bits which is equal
    # to one element of packed embedding bag vector
    scale_tensor = torch.empty(0, dtype=torch.bfloat16)
    zp_tensor = torch.empty(0, dtype=torch.bfloat16)
    num_scale_zp_dim = (
        scale_tensor.element_size() + zp_tensor.element_size()
    ) / weight.element_size()
    embedding_dim = weight.size(1) - num_scale_zp_dim
    bits_in_1_byte = 8
    num_bits_per_packed_weight = weight.element_size() * bits_in_1_byte
    output_embedding_dim = int(
        embedding_dim * (num_bits_per_packed_weight / num_bits_per_weight)
    )
    num_bags = offsets.size(0)
    if include_last_offset:
        num_bags = num_bags - 1
    output = torch.empty(
        num_bags, output_embedding_dim, dtype=output_dtype, device="meta"
    )

    return output


@register_meta("zentorch_horizontal_quant_embedding_bag_group")
def meta_zentorch_horizontal_quant_embedding_bag_group(
    weight,
    indices,
    offsets,
    num_bits_per_weight,
    output_dtype,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    output_list = []

    for i in range(len(weight)):
        output = meta_zentorch_quant_embedding_bag(
            weight[i],
            indices[i],
            offsets[i],
            num_bits_per_weight,
            output_dtype,
            include_last_offset=include_last_offset[i],
        )

        output_list.append(output)

    return output_list


@register_meta("zentorch_horizontal_quant_embedding_bag_group", "out")
def meta_zentorch_horizontal_quant_embedding_bag_group_out(
    outputs,
    weight,
    indices,
    offsets,
    num_bits_per_weight,
    output_dtype,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return


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


@torch.library.register_fake("zentorch::zentorch_masked_multihead_self_attention")
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
    add_causal_mask=None,
):
    attn_output = query.new_empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3])
    )

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
    ctx = torch._custom_ops.get_ctx()
    # Key_cache_out shape is dependent on input data
    # Hence needs to be dynamic
    max_positions = ctx.new_dynamic_size()
    key_cache_out = query.new_empty(
        (max_positions, beam_idx.shape[1], key.shape[2], key.shape[3])
    )
    value_cache_out = query.new_empty(
        (max_positions, beam_idx.shape[1], value.shape[2], value.shape[3])
    )
    num_to_keep = ctx.new_dynamic_size()
    beam_idx_out = query.new_empty((num_to_keep, beam_idx.shape[1]))
    return (attn_output, attn_weights, key_cache_out, value_cache_out, beam_idx_out)


@register_meta("zentorch_weight_prepack_for_linear")
def meta_zentorch_weight_prepack_for_linear(weight):
    return weight.new_empty(weight.size())


@register_meta("zentorch_qlinear", "out")
def meta_zentorch_qlinear_out(
    out,
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    bias,
    output_scales,
    output_zero_points,
    output_dtype,
):
    return


@register_meta("zentorch_qlinear_relu", "out")
def meta_zentorch_qlinear_relu_out(
    out,
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    bias,
    output_scales,
    output_zero_points,
    output_dtype,
):
    return


@register_meta("zentorch_qlinear")
def meta_zentorch_qlinear(
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    bias,
    output_scales,
    output_zero_points,
    output_dtype=None,
):
    if output_dtype is None:
        output_dtype = torch.float32
    out_dim = list(input.size())
    out_dim[-1] = weight.size(0)
    return input.new_empty(out_dim, dtype=output_dtype)


@register_meta("zentorch_qlinear_relu")
def meta_zentorch_qlinear_relu(
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    bias,
    output_scales,
    output_zero_points,
    output_dtype=None,
):
    return meta_zentorch_qlinear(
        input,
        weight,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        bias,
        output_scales,
        output_zero_points,
        output_dtype,
    )


@register_meta("zentorch_qlinear_sigmoid")
def meta_zentorch_qlinear_sigmoid(
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    bias,
    output_scales,
    output_zero_points,
    output_dtype=None,
):
    return meta_zentorch_qlinear(
        input,
        weight,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        bias,
        output_scales,
        output_zero_points,
        output_dtype,
    )


@register_meta("zentorch_qlinear_mul_add")
def meta_zentorch_qlinear_mul_add(
    input,
    weight,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
    bias,
    output_scales,
    output_zero_points,
    output_dtype=None,
):
    if output_dtype is None:
        output_dtype = torch.float32
    return add_input.new_empty((add_input.size()), dtype=output_dtype)


@register_meta("zentorch_woq_linear")
def meta_zentorch_woq_linear(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear",
):
    out_dim = list(input.size())
    out_dim[-1] = weight.shape[1]
    return input.new_empty(out_dim, dtype=input.dtype)


@register_meta("zentorch_woq_linear_relu")
def meta_zentorch_woq_linear_relu(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_relu",
):
    return meta_zentorch_woq_linear(
        input, weight, weight_scales, weight_zero_points, bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_sigmoid")
def meta_zentorch_woq_linear_sigmoid(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_sigmoid",
):
    return meta_zentorch_woq_linear(
        input, weight, weight_scales, weight_zero_points, bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_mul_add")
def meta_zentorch_woq_linear_mul_add(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_mul_add",
):
    # The output shape matches add_input, dtype follows
    return add_input.new_empty((add_input.size()), dtype=input.dtype)


@register_meta("zentorch_weight_from_int4pack_and_repack")
def meta_zentorch_weight_from_int4pack_and_repack(unpacked_weight):
    # Returns a packed weight tensor of shape [N, K/8]
    K = unpacked_weight.size(1)
    K_packed = K // 8
    return unpacked_weight.new_empty((unpacked_weight.size(0), K_packed))


make_fallback(torch.ops.zentorch.zentorch_addmm)
make_fallback(torch.ops.zentorch.zentorch_addmm_relu)
make_fallback(torch.ops.zentorch.zentorch_addmm_silu)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_linear_binary_binary)
make_fallback(torch.ops.zentorch.zentorch_linear_unary_binary)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_relu)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_silu)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_add)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_add_add)
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_mul_add)
make_fallback(torch.ops.zentorch.zentorch_linear_unary)
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
make_fallback(torch.ops.zentorch.zentorch_quant_embedding_bag)
make_fallback(torch.ops.zentorch.zentorch_quant_embedding_bag.out)
make_fallback(torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group)
make_fallback(torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.out)
make_fallback(torch.ops.zentorch.zentorch_rope)
make_fallback(torch.ops.zentorch.zentorch_masked_multihead_self_attention)
make_fallback(torch.ops.zentorch.zentorch_weight_prepack_for_linear)
make_fallback(torch.ops.zentorch.zentorch_qlinear)
make_fallback(torch.ops.zentorch.zentorch_qlinear_relu)
make_fallback(torch.ops.zentorch.zentorch_qlinear_sigmoid)
make_fallback(torch.ops.zentorch.zentorch_qlinear_mul_add)
make_fallback(torch.ops.zentorch.zentorch_woq_linear)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_relu)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_sigmoid)
make_fallback(torch.ops.zentorch.zentorch_woq_linear_mul_add)
make_fallback(torch.ops.zentorch.zentorch_weight_from_int4pack_and_repack)
if hasattr(torch.ops.zentorch, "zentorch_sdpa"):
    make_fallback(torch.ops.zentorch.zentorch_sdpa)
