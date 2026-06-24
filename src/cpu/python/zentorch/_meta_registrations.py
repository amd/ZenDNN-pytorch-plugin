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


@register_meta("zentorch_bmm", "out")
def meta_zentorch_bmm_out(input, weight, *, out):
    return


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


@register_meta("zentorch_add_rms_norm_")
def meta_zentorch_add_rms_norm_(
    input,
    weight,
    residual,
    epsilon,
    zentorch_op_name="zentorch::zentorch_add_rms_norm",
):
    return None


@register_meta("zentorch_rms_norm")
def meta_zentorch_rms_norm(
    input,
    weight,
    epsilon,
    zentorch_op_name="zentorch::zentorch_rms_norm",
):
    return input.new_empty(input.size())


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
    weight_zero_points=None,
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
    weight_zero_points=None,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_relu",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_sigmoid")
def meta_zentorch_woq_linear_sigmoid(
    input,
    weight,
    weight_scales,
    weight_zero_points=None,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_sigmoid",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_gelu_tanh")
def meta_zentorch_woq_linear_gelu_tanh(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_gelu_tanh",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_gelu_erf")
def meta_zentorch_woq_linear_gelu_erf(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_gelu_erf",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_add")
def meta_zentorch_woq_linear_add(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    add_input,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_add",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
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
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_woq_linear_add_add")
def meta_zentorch_woq_linear_add_add(
    input,
    weight,
    weight_scales,
    weight_zero_points,
    add_input,
    add_input_2,
    bias=None,
    zentorch_op_name="zentorch::zentorch_woq_linear_add_add",
):
    return meta_zentorch_woq_linear(
        input,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        zentorch_op_name,
    )


@register_meta("zentorch_dynamic_qlinear")
def meta_zentorch_dynamic_qlinear(
    input,
    weight,
    weight_scales,
    bias=None,
    zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
):
    out_dim = list(input.size())
    out_dim[-1] = weight.size(0)
    return input.new_empty(out_dim)


@register_meta("zentorch_group_matmul", "out")
def meta_zentorch_group_matmul_out(
    gemm_outputs,
    inputs,
    w13_weights,
    w2_weights,
    moe_output,
    topk_weights,
    row_ptrs,
    activation,
    w13_bias,
    w2_bias,
    w13_scales,
    w2_scales,
    zentorch_op_name="zentorch::zentorch_group_matmul.out",
):
    return


@register_meta("zentorch_fused_moe")
def meta_zentorch_fused_moe(
    output,
    input,
    w13,
    w2,
    w13_bias,
    w2_bias,
    topk_weights,
    topk_id,
    skip_weighted,
    act,
    w13_scales=None,
    w2_scales=None,
    zentorch_op_name="zentorch::zentorch_fused_moe",
):
    return


@register_meta("zentorch_woq_repack_weight")
def meta_zentorch_woq_repack_weight(unpacked_weight):
    # Returns a packed weight tensor of shape [N, K/8]
    K = unpacked_weight.size(1)
    K_packed = K // 8
    return unpacked_weight.new_empty((unpacked_weight.size(0), K_packed))


@register_meta("zentorch_woq_repack_from_int4pack")
def meta_zentorch_woq_repack_from_int4pack(
    packed_weight,
):
    N = packed_weight.size(0)
    # TODO: Use dtype to calculate number of bits packed instead of hardcoding "* 2"
    # For uint4-packed input, each byte stores 2 values (4 bits each)
    K = packed_weight.size(1) * 2
    K_packed = K // 8
    return packed_weight.new_empty((N, K_packed), dtype=torch.int32)


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
make_fallback(torch.ops.zentorch.zentorch_bmm.out)
make_fallback(torch.ops.zentorch.zentorch_baddbmm)
make_fallback(torch.ops.zentorch.zentorch_mm)
make_fallback(torch.ops.zentorch.zentorch_mm_relu)
make_fallback(torch.ops.zentorch.zentorch_mm_silu)
make_fallback(torch.ops.zentorch.zentorch_mm_gelu_tanh)
make_fallback(torch.ops.zentorch.zentorch_mm_gelu_erf)
make_fallback(torch.ops.zentorch.zentorch_horizontal_embedding_bag_group)
make_fallback(torch.ops.zentorch.zentorch_horizontal_embedding_group)
make_fallback(torch.ops.zentorch.zentorch_add_rms_norm_)
make_fallback(torch.ops.zentorch.zentorch_rms_norm)
# All four overloads of the quantized embedding-bag op family
# (`zentorch_quant_embedding_bag.{default,out}` and
# `zentorch_horizontal_quant_embedding_bag_group.{default,out}`) are routed
# through dedicated AOTI shims via `register_lowering` in `_lowerings.py`,
# so none of them go through `make_fallback`. The `.default` group overload
# returns `Tensor[]` (variable-length); its lowering
# (`_ZentorchHorizontalQuantEmbBagGroupDefault`) overrides codegen to emit
# `(handle_array, N)` to the shim instead of Inductor's default
# `&handle_0, ..., &handle_{N-1}`.
make_fallback(torch.ops.zentorch.zentorch_weight_prepack_for_linear)
# zentorch_dynamic_qlinear is routed through a dedicated AOTI shim
# (aoti_torch_cpu_zentorch_dynamic_qlinear) via register_lowering in
# _lowerings.py, so it must NOT go through make_fallback (which would force
# the slow custom_op_wrapper Python path under cpp_wrapper).
make_fallback(torch.ops.zentorch.zentorch_group_matmul.out)
make_fallback(torch.ops.zentorch.zentorch_fused_moe)
make_fallback(torch.ops.zentorch.zentorch_woq_repack_weight)
make_fallback(
    torch.ops.zentorch.zentorch_woq_repack_from_int4pack
)
if hasattr(torch.ops.zentorch, "zentorch_sdpa"):
    make_fallback(torch.ops.zentorch.zentorch_sdpa)


# GatedDeltaNet (GDN) ops for Qwen3.5 / Qwen3-Next CPU attention.


@register_meta("gdn_chunk_local_cumsum")
def meta_gdn_chunk_local_cumsum(
    g,
    chunk_size,
    cu_seqlens,
    chunk_indices,
    zentorch_op_name="zentorch::gdn_chunk_local_cumsum",
):
    return g.new_empty(g.size(), dtype=torch.float32)


@register_meta("gdn_l2norm_fwd")
def meta_gdn_l2norm_fwd(
    x,
    eps,
    zentorch_op_name="zentorch::gdn_l2norm_fwd",
):
    return x.new_empty(x.size())


@register_meta("gdn_chunk_scaled_dot_kkt_fwd")
def meta_gdn_chunk_scaled_dot_kkt_fwd(
    k,
    g,
    beta,
    cu_seqlens,
    chunk_indices,
    chunk_size,
    zentorch_op_name="zentorch::gdn_chunk_scaled_dot_kkt_fwd",
):
    B = k.size(0)
    T = k.size(1)
    H = beta.size(2)
    return k.new_empty((B, T, H, chunk_size), dtype=torch.float32)


@register_meta("gdn_solve_tril")
def meta_gdn_solve_tril(
    A,
    cu_seqlens,
    chunk_indices,
    zentorch_op_name="zentorch::gdn_solve_tril",
):
    return A.new_empty(A.size(), dtype=torch.float32)


@register_meta("gdn_recompute_w_u_fwd")
def meta_gdn_recompute_w_u_fwd(
    k,
    v,
    beta,
    g_cumsum,
    A,
    cu_seqlens,
    chunk_indices,
    zentorch_op_name="zentorch::gdn_recompute_w_u_fwd",
):
    B = k.size(0)
    T = k.size(1)
    H = v.size(2)
    K_dim = k.size(3)
    V_dim = v.size(3)
    w = k.new_empty((B, T, H, K_dim))
    u = v.new_empty((B, T, H, V_dim))
    return w, u


@register_meta("gdn_chunk_gated_delta_rule_fwd_h")
def meta_gdn_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g,
    initial_state,
    output_final_state,
    chunk_size,
    save_new_value,
    cu_seqlens,
    chunk_offsets,
    NT_total,
    zentorch_op_name="zentorch::gdn_chunk_gated_delta_rule_fwd_h",
):
    B = k.size(0)
    T = k.size(1)
    H = u.size(2)
    K_dim = k.size(3)
    V_dim = u.size(3)
    N = cu_seqlens.size(0) - 1
    h_out = k.new_empty((B, NT_total, H, V_dim, K_dim))
    v_new = (
        u.new_empty((B, T, H, V_dim))
        if save_new_value else u.new_empty((0,))
    )
    final_state = (
        k.new_empty((N, H, V_dim, K_dim), dtype=torch.float32)
        if output_final_state else k.new_empty((0,), dtype=torch.float32)
    )
    return h_out, v_new, final_state


@register_meta("gdn_chunk_fwd_o")
def meta_gdn_chunk_fwd_o(
    q,
    k,
    v,
    h,
    g,
    scale,
    cu_seqlens,
    chunk_offsets,
    chunk_size,
    zentorch_op_name="zentorch::gdn_chunk_fwd_o",
):
    B = q.size(0)
    T = q.size(1)
    H = v.size(2)
    V_dim = v.size(3)
    return v.new_empty((B, T, H, V_dim))


@register_meta("gdn_fused_recurrent_gated_delta_rule_packed_decode")
def meta_gdn_fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel,
    zentorch_op_name=(
        "zentorch::gdn_fused_recurrent_gated_delta_rule_packed_decode"
    ),
):
    return None


@register_meta("gdn_fused_sigmoid_gating_delta_rule_update")
def meta_gdn_fused_sigmoid_gating_delta_rule_update(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    beta_temp,
    threshold,
    scale,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
    zentorch_op_name="zentorch::gdn_fused_sigmoid_gating_delta_rule_update",
):
    return q.new_empty(v.size())


@register_meta("gdn_fused_post_conv_prep")
def meta_gdn_fused_post_conv_prep(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm,
    output_g_exp,
    zentorch_op_name="zentorch::gdn_fused_post_conv_prep",
):
    L = conv_output.size(0)
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.size(0)
    q = conv_output.new_empty((L, H, K))
    k = conv_output.new_empty((L, H, K))
    v = conv_output.new_empty((L, HV, V))
    g = conv_output.new_empty((L, HV), dtype=torch.float32)
    beta = conv_output.new_empty((L, HV), dtype=torch.float32)
    return q, k, v, g, beta


@register_meta("gdn_causal_conv1d_update")
def meta_gdn_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    null_block_id,
    pad_slot_id,
    zentorch_op_name="zentorch::gdn_causal_conv1d_update",
):
    return x.new_empty(x.size())


@register_meta("gdn_causal_conv1d_fn")
def meta_gdn_causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    activation,
    pad_slot_id,
    zentorch_op_name="zentorch::gdn_causal_conv1d_fn",
):
    return x.new_empty(x.size())


@register_meta("gdn_rms_norm_gated")
def meta_gdn_rms_norm_gated(
    x,
    weight,
    z,
    eps,
    activation,
    zentorch_op_name="zentorch::gdn_rms_norm_gated",
):
    return x.new_empty(x.size())


@register_meta("gdn_chunk_gated_delta_rule_fwd")
def meta_gdn_chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    chunk_size,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    zentorch_op_name="zentorch::gdn_chunk_gated_delta_rule_fwd",
):
    B = q.size(0)
    T = q.size(1)
    H = v.size(2)
    K_dim = q.size(3)
    V_dim = v.size(3)
    N = cu_seqlens.size(0) - 1
    o = v.new_empty((B, T, H, V_dim))
    if output_final_state:
        final_state = k.new_empty((N, H, V_dim, K_dim), dtype=torch.float32)
    else:
        final_state = k.new_empty((0,), dtype=torch.float32)
    return o, final_state


make_fallback(torch.ops.zentorch.gdn_chunk_local_cumsum)
make_fallback(torch.ops.zentorch.gdn_l2norm_fwd)
make_fallback(torch.ops.zentorch.gdn_chunk_scaled_dot_kkt_fwd)
make_fallback(torch.ops.zentorch.gdn_solve_tril)
make_fallback(torch.ops.zentorch.gdn_recompute_w_u_fwd)
make_fallback(torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd_h)
make_fallback(torch.ops.zentorch.gdn_chunk_fwd_o)
make_fallback(torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd)
make_fallback(torch.ops.zentorch.gdn_rms_norm_gated)
make_fallback(torch.ops.zentorch.gdn_causal_conv1d_fn)
make_fallback(torch.ops.zentorch.gdn_causal_conv1d_update)
make_fallback(torch.ops.zentorch.gdn_fused_post_conv_prep)
make_fallback(torch.ops.zentorch.gdn_fused_sigmoid_gating_delta_rule_update)
make_fallback(torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode)
