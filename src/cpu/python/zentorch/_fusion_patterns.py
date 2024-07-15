# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._compile_backend import torch_version
from ._utils import counters
import functools
from functools import partial


at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

# add patterns below


def _mm_silu_mul_pattern(arg_0, arg_1, arg_2):
    # arg_0: input rmsnorm
    # arg_1: input permute
    # arg_2: input mm
    shape_0 = arg_0.size()  # [bs, seq_len, n]
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])
    mm_0 = zt_ops.zentorch_mm_silu.default(view_0, arg_1)
    view_1 = at_ops.view.default(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])
    view_2 = at_ops.view.default(arg_2, [shape_0[0], shape_0[1], shape_1[-1]])
    mul_0 = at_ops.mul.Tensor(view_1, view_2)
    return (mul_0,)


def _mm_silu_mul_replacement(arg_0, arg_1, arg_2):
    counters["zentorch"]["pattern_matcher_mm_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])
    mul_0 = zt_ops.zentorch_mm_silu_mul.default(view_0, arg_1, arg_2)
    view_1 = at_ops.view.default(mul_0, [shape_0[0], shape_0[1], shape_1[-1]])
    return (view_1,)


def _mm_silu_mul_pattern_bf16(arg_0_, arg_1_, arg_2_):
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mm_0 = zt_ops.zentorch_mm_silu.default(view_0, arg_1_)
    view_1 = at_ops.view.default(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])
    view_2 = at_ops.view.default(arg_2_, [shape_0[0], shape_0[1], shape_1[-1]])
    mul_0 = at_ops.mul.Tensor(view_1, view_2)
    return (mul_0,)


def _mm_silu_mul_replacement_bf16(arg_0_, arg_1_, arg_2_):
    counters["zentorch"]["pattern_matcher_mm_silu_mul"] += 1
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mul_0 = zt_ops.zentorch_mm_silu_mul.default(view_0, arg_1_, arg_2_)
    view_1 = at_ops.view.default(mul_0, [shape_0[0], shape_0[1], shape_1[-1]])
    return (view_1,)


def _addmm_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0):
    # arg_0 > 2, 16, 256
    # arg_1 > 256, 688
    # arg_2 > 32, 688
    # bias_0 > 32, 688
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(
        arg_0, [shape_0[0] * shape_0[1], shape_0[2]]
    )  # 32, 256
    mm_0 = zt_ops.zentorch_addmm_silu.default(bias_0, view_0, arg_1)  # 32, 688
    view_1 = at_ops.view.default(
        mm_0, [shape_0[0], shape_0[1], shape_1[-1]]
    )  # 2, 16, 688
    view_2 = at_ops.view.default(
        arg_2, [shape_0[0], shape_0[1], shape_1[-1]]
    )  # 2, 16, 688
    mul_0 = at_ops.mul.Tensor(view_1, view_2)  # 2, 16, 688
    return (mul_0,)


def _addmm_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(
        arg_0, [shape_0[0] * shape_0[1], shape_0[2]]
    )  # 32, 256
    mul_0 = zt_ops.zentorch_addmm_silu_mul.default(
        bias_0, view_0, arg_1, arg_2
    )  # 32, 688
    view_1 = at_ops.view.default(
        mul_0, [shape_0[0], shape_0[1], shape_1[-1]]
    )  # 2, 16, 688
    return (view_1,)


def _addmm_silu_mul_pattern_bf16(arg_0_, arg_1_, arg_2_, bias_0_):
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mm_0 = zt_ops.zentorch_addmm_silu.default(bias_0_, view_0, arg_1_)
    view_1 = at_ops.view.default(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])
    view_2 = at_ops.view.default(arg_2_, [shape_0[0], shape_0[1], shape_1[-1]])
    mul_0 = at_ops.mul.Tensor(view_1, view_2)
    return (mul_0,)


def _addmm_silu_mul_replacement_bf16(arg_0_, arg_1_, arg_2_, bias_0_):
    counters["zentorch"]["pattern_matcher_addmm_silu_mul"] += 1
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mul_0 = zt_ops.zentorch_addmm_silu_mul.default(bias_0_, view_0, arg_1_, arg_2_)
    view_1 = at_ops.view.default(mul_0, [shape_0[0], shape_0[1], shape_1[-1]])
    return (view_1,)


def _addmm_1dbias_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0):
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])
    mm_0 = zt_ops.zentorch_addmm_1dbias_silu.default(bias_0, view_0, arg_1)
    view_1 = at_ops.view.default(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])
    view_2 = at_ops.view.default(arg_2, [shape_0[0], shape_0[1], shape_1[-1]])
    mul_0 = at_ops.mul.Tensor(view_1, view_2)
    return (mul_0,)


def _addmm_1dbias_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view.default(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])
    mul_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(bias_0, view_0, arg_1, arg_2)
    view_1 = at_ops.view.default(mul_0, [shape_0[0], shape_0[1], shape_1[-1]])
    return (view_1,)


def _addmm_1dbias_silu_mul_pattern_bf16(arg_0_, arg_1_, arg_2_, bias_0_):
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mm_0 = zt_ops.zentorch_addmm_1dbias_silu.default(bias_0_, view_0, arg_1_)
    view_1 = at_ops.view.default(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])
    view_2 = at_ops.view.default(arg_2_, [shape_0[0], shape_0[1], shape_1[-1]])
    mul_0 = at_ops.mul.Tensor(view_1, view_2)
    return (mul_0,)


def _addmm_1dbias_silu_mul_replacement_bf16(arg_0_, arg_1_, arg_2_, bias_0_):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"] += 1
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    view_0 = at_ops.view.default(arg_0_, [shape_0[0] * shape_0[1], shape_0[2]])
    mul_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
        bias_0_, view_0, arg_1_, arg_2_
    )
    view_1 = at_ops.view.default(mul_0, [shape_0[0], shape_0[1], shape_1[-1]])
    return (view_1,)


# adding patterns completed #


def _dummy_extra_check(match):
    return True


def _get_pattern_with_replacement():
    # get the matcher_pass to register with
    from ._fusion_matcher import matcher_pass

    arg_1 = partial(torch.empty, (2, 16, 256), device="cpu", requires_grad=True)
    a_1 = partial(arg_1, dtype=torch.float)
    a_1_bf16 = partial(arg_1, dtype=torch.bfloat16)
    arg_2 = partial(torch.empty, (256, 688), device="cpu", requires_grad=True)
    a_2 = partial(arg_2, dtype=torch.float)
    a_2_bf16 = partial(arg_2, dtype=torch.bfloat16)
    arg_3 = partial(torch.empty, (32, 688), device="cpu", requires_grad=True)
    a_3 = partial(arg_3, dtype=torch.float)
    a_3_bf16 = partial(arg_3, dtype=torch.bfloat16)
    arg_4 = partial(torch.empty, (1, 688), device="cpu", requires_grad=True)
    a_4 = partial(arg_4, dtype=torch.float)
    a_4_bf16 = partial(arg_4, dtype=torch.bfloat16)

    candidates = [
        (
            _mm_silu_mul_pattern,
            _mm_silu_mul_replacement,
            [a_1(), a_2(), a_3()],  # used to pass arguments
            {},  # this can be used to pass kwargs
            _dummy_extra_check,  # fake extra check, cannot be skipped
        ),
        (
            _mm_silu_mul_pattern_bf16,
            _mm_silu_mul_replacement_bf16,
            [a_1_bf16(), a_2_bf16(), a_3_bf16()],
            {},
            _dummy_extra_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [a_1(), a_2(), a_3(), a_3()],
            {},
            _dummy_extra_check,
        ),
        (
            _addmm_silu_mul_pattern_bf16,
            _addmm_silu_mul_replacement_bf16,
            [a_1_bf16(), a_2_bf16(), a_3_bf16(), a_3_bf16()],
            {},
            _dummy_extra_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [a_1(), a_2(), a_3(), a_4()],
            {},
            _dummy_extra_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern_bf16,
            _addmm_1dbias_silu_mul_replacement_bf16,
            [a_1_bf16(), a_2_bf16(), a_3_bf16(), a_4_bf16()],
            {},
            _dummy_extra_check,
        ),
    ]
    for pattern, replacement, args, workaround, extra_check in candidates:
        assert isinstance(workaround, dict)
        name = pattern.__name__
        inference_name = name + "_inference"
        # pre 2.2 PT versions use a different name for fwd-tracer
        # remove the if block when deprecating support for PT <= 2.1.x
        if torch_version < "2.2":
            from torch._inductor.pattern_matcher import inference_graph

            yield inference_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": inference_graph,
                "pass_dict": matcher_pass,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
            }
        else:
            from torch._inductor.pattern_matcher import fwd_only

            yield inference_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": fwd_only,  # tracer for forward function
                "pass_dicts": matcher_pass,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
            }


@functools.lru_cache(None)
def _replace_init():
    from torch._inductor.pattern_matcher import register_replacement

    # loop for _get_pattern_with_replacement
    for _key, register_replacement_kwargs in _get_pattern_with_replacement():
        # TODO: try to use use gen_register_replacement for generating the patterns
        # for PT >= 2.4, do the following
        # gen_register_replacement(key, **register_replacement_kwargs)
        register_replacement(**register_replacement_kwargs)
