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


if hasattr(torch.ops.zentorch, "zentorch_attention_flash_attn_varlen"):

    @register_meta("zentorch_attention_flash_attn_varlen")
    def meta_zentorch_flash_attn_varlen(
        out,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        block_table,
        alibi_slopes,
        window_size_left,
        window_size_right,
        kv_cache_dtype,
        k_scale,
        v_scale,
        softcap,
        zentorch_op_name="",
    ):
        return None


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
    input, weight, bias=None, is_weight_prepacked=False, post_op="none"
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


make_fallback(torch.ops.zentorch.zentorch_addmm)
make_fallback(torch.ops.zentorch.zentorch_addmm_relu)
make_fallback(torch.ops.zentorch.zentorch_addmm_silu)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_addmm_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_linear_unary)
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
make_fallback(torch.ops.zentorch.zentorch_rope)
make_fallback(torch.ops.zentorch.zentorch_masked_multihead_self_attention)
make_fallback(torch.ops.zentorch.zentorch_weight_prepack_for_linear)
make_fallback(torch.ops.zentorch.zentorch_qlinear)
make_fallback(torch.ops.zentorch.zentorch_qlinear_relu)
make_fallback(torch.ops.zentorch.zentorch_qlinear_sigmoid)
make_fallback(torch.ops.zentorch.zentorch_qlinear_mul_add)
if hasattr(torch.ops.zentorch, "zentorch_sdpa"):
    make_fallback(torch.ops.zentorch.zentorch_sdpa)
if hasattr(torch.ops.zentorch, "zentorch_attention_reshape_and_cache"):
    make_fallback(torch.ops.zentorch.zentorch_attention_reshape_and_cache)
if hasattr(torch.ops.zentorch, "zentorch_attention_single_query_cached_kv_attention"):
    make_fallback(
        torch.ops.zentorch.zentorch_attention_single_query_cached_kv_attention
    )
