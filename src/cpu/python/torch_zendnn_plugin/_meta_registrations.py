# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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


@register_meta("zendnn_mm")
def meta_zendnn_mm(
    bias,
    input,
    weight,
    alpha=1,
    beta=1,
    fuse=0,
):
    return input.new_empty((input.shape[0], weight.shape[-1]))


@register_meta("zendnn_bmm")
def meta_zendnn_bmm(
    input,
    weight
):
    return input.new_empty((input.shape[0], input.shape[1], weight.shape[-1]))


@register_meta("zendnn_baddbmm")
def meta_zendnn_baddbmm(
    bias,
    input,
    weight,
    alpha=1,
    beta=1
):
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


make_fallback(torch.ops.zentorch.zendnn_addmm)
make_fallback(torch.ops.zentorch.zendnn_embedding_bag)
make_fallback(torch.ops.zentorch.zendnn_bmm)
make_fallback(torch.ops.zentorch.zendnn_baddbmm)
make_fallback(torch.ops.zentorch.zendnn_mm)
