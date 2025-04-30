# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

if hasattr(torch.ops.zentorch, "zentorch_attention_reshape_and_cache"):

    @register_meta("zentorch_attention_reshape_and_cache")
    def meta_zentorch_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping
    ):
        return None


if hasattr(torch.ops.zentorch, "zentorch_attention_single_query_cached_kv_attention"):

    @register_meta("zentorch_attention_single_query_cached_kv_attention")
    def meta_zentorch_single_query_cached_kv_attention(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    ):
        return out.new_empty(out.size())


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


@register_meta("zentorch_quant_group_eb_mlp_concat_zendnn")
def meta_zentorch_quant_group_eb_mlp_concat_zendnn(
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
    cat_dim,
    other_arguments_position,
    other_arguments,
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

    for idx, pos in enumerate(other_arguments_position):
        output_list.insert(pos, other_arguments[idx])

    cat_output = torch.cat(output_list, dim=cat_dim)

    return cat_output


@register_meta("zentorch_quant_group_eb_mlp_concat_fbgemm")
def meta_zentorch_quant_group_eb_mlp_concat_fbgemm(
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
    cat_dim,
    other_arguments_position,
    other_arguments,
):
    return meta_zentorch_quant_group_eb_mlp_concat_zendnn(
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
        cat_dim,
        other_arguments_position,
        other_arguments,
    )


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


@register_meta("zentorch_convolution")
def meta_zentorch_convolution(
    input,
    weight,
    bias_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    input_size = input.size()
    weight_size = weight.size()
    has_dilation = len(dilation) > 0
    dim = len(input_size)

    output_size = [0] * dim
    output_size[0] = input_size[0]  # Batch size
    output_size[1] = weight_size[0]  # Number of output channels

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[
            d - 2
        ] + 1

    output = input.new_empty(output_size)
    if input.is_contiguous(memory_format=torch.channels_last):
        output = output.to(memory_format=torch.channels_last)
    return output


@register_meta("zentorch_qlinear", "out")
def meta_zentorch_qlinear_out(
    out,
    output_stride,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
):
    return


@register_meta("zentorch_qlinear_relu", "out")
def meta_zentorch_qlinear_relu_out(
    out,
    output_stride,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
):
    return


@register_meta("zentorch_qlinear")
def meta_zentorch_qlinear(
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
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
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
):
    return meta_zentorch_qlinear(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype,
        output_scales,
        output_zero_points,
    )


@register_meta("zentorch_qlinear_sigmoid")
def meta_zentorch_qlinear_sigmoid(
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
):
    return meta_zentorch_qlinear(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype,
        output_scales,
        output_zero_points,
    )


@register_meta("zentorch_qlinear_mul_add")
def meta_zentorch_qlinear_mul_add(
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
    output_dtype=None,
    output_scales=None,
    output_zero_points=None,
):
    if output_dtype is None:
        output_dtype = torch.float32
    return add_input.new_empty((add_input.size()), dtype=output_dtype)


@register_meta("zentorch_weight_reorder_for_matmul")
def meta_zentorch_weight_reorder_for_matmul(
    weight,
    is_weight_oc_x_ic=True,
):
    return weight.new_empty(weight.size())


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
make_fallback(torch.ops.zentorch.zentorch_addmm_1dbias_mul_add)
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
make_fallback(torch.ops.zentorch.zentorch_woq_linear_silu_mul)
make_fallback(torch.ops.zentorch.zentorch_convolution)
make_fallback(torch.ops.zentorch.zentorch_qlinear)
make_fallback(torch.ops.zentorch.zentorch_qlinear_relu)
make_fallback(torch.ops.zentorch.zentorch_qlinear_sigmoid)
make_fallback(torch.ops.zentorch.zentorch_qlinear_mul_add)
make_fallback(torch.ops.zentorch.zentorch_quant_embedding_bag)
make_fallback(torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group)
make_fallback(torch.ops.zentorch.zentorch_quant_group_eb_mlp_concat_zendnn)
make_fallback(torch.ops.zentorch.zentorch_quant_group_eb_mlp_concat_fbgemm)
make_fallback(torch.ops.zentorch.zentorch_weight_reorder_for_matmul)
if hasattr(torch.ops.zentorch, "zentorch_sdpa"):
    make_fallback(torch.ops.zentorch.zentorch_sdpa)
if hasattr(torch.ops.zentorch, "zentorch_attention_reshape_and_cache"):
    make_fallback(torch.ops.zentorch.zentorch_attention_reshape_and_cache)
if hasattr(torch.ops.zentorch, "zentorch_attention_single_query_cached_kv_attention"):
    make_fallback(
        torch.ops.zentorch.zentorch_attention_single_query_cached_kv_attention
    )
