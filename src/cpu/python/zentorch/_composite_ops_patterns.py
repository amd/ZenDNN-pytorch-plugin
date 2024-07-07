# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import functools
from ._compile_backend import torch_version
from functools import partial
from ._counters import counters

# Brief steps:
# 1. define both patterns
# 2. call pattermatcher pass and loop over dtypes for those
#    and create functools.patrial versions
# 3. use yield keyword as in pytorch
# 4. register both patterns
# Note : Keep args different for F32 and BF16 patterns.
# Take a look at gen_attention_patterns.py file in PT repo as well

aten = torch.ops.aten


# adding gelu pattern here, find a way to generate patterns
def _gelu_tanh_pattern(arg_0):
    mul_0 = aten.mul.Tensor(arg_0, 0.5)
    pow_0 = aten.pow.Tensor_Scalar(arg_0, 3.0)
    mul_1 = aten.mul.Tensor(pow_0, 0.044715)
    add_0 = aten.add.Tensor(arg_0, mul_1)
    mul_2 = aten.mul.Tensor(add_0, 0.7978845608028654)
    tanh_0 = aten.tanh.default(mul_2)
    add_1 = aten.add.Tensor(tanh_0, 1.0)
    mul_3 = aten.mul.Tensor(mul_0, add_1)
    return (mul_3,)


def _gelu_tanh_replacement(arg_0):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = aten.gelu.default(arg_0, approximate="tanh")
    return (gelu_0,)


def _gelu_tanh_pattern_bf16(arg_0_):
    mul_0 = aten.mul.Tensor(arg_0_, 0.5)
    pow_0 = aten.pow.Tensor_Scalar(arg_0_, 3.0)
    mul_1 = aten.mul.Tensor(pow_0, 0.044715)
    add_0 = aten.add.Tensor(arg_0_, mul_1)
    mul_2 = aten.mul.Tensor(add_0, 0.7978845608028654)
    tanh_0 = aten.tanh.default(mul_2)
    add_1 = aten.add.Tensor(tanh_0, 1.0)
    mul_3 = aten.mul.Tensor(mul_0, add_1)
    return (mul_3,)


def _gelu_tanh_replacement_bf16(arg_0_):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = aten.gelu.default(arg_0_, approximate="tanh")
    return (gelu_0,)


def _gelu_erf_pattern(arg_0):
    mul_0 = aten.mul.Tensor(arg_0, 0.5)
    mul_1 = aten.mul.Tensor(arg_0, 0.7071067811865476)
    erf_0 = aten.erf.default(mul_1)
    add_0 = aten.add.Tensor(erf_0, 1.0)
    mul_2 = aten.mul.Tensor(mul_0, add_0)
    return (mul_2,)


def _gelu_erf_replacement(arg_0):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = aten.gelu.default(arg_0)
    return (gelu_0,)


def _gelu_erf_pattern_bf16(arg_0_):
    mul_0 = aten.mul.Tensor(arg_0_, 0.5)
    mul_1 = aten.mul.Tensor(arg_0_, 0.7071067811865476)
    erf_0 = aten.erf.default(mul_1)
    add_0 = aten.add.Tensor(erf_0, 1.0)
    mul_2 = aten.mul.Tensor(mul_0, add_0)
    return (mul_2,)


def _gelu_erf_replacement_bf16(arg_0_):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = aten.gelu.default(arg_0_)
    return (gelu_0,)


# Eliminate the copy overhead with first token
# for ChatGLM3 with zentorch.llm.optimize.
# AOT Autograd was doing an expand for the weights to
# accomodate 3D matmul as inputs are in 3D which is
# causing a huge copy overhead.
# Rewriting the pattern to remove the expand for weights
# and squeezing the inputs to 2D to enable mm instead of bmm.
# Pattern 1 : 2 Expands, 1 from input and 1 from weight
# will be inputs to bmm node.
def _bmm_to_mm_pattern_1_bf16(arg_0_, arg_1_):
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    exp_0 = aten.expand.default(arg_0_, [shape_0[0], 1, shape_0[-1]])
    exp_1 = aten.expand.default(arg_1_, [shape_0[0], shape_1[0], shape_1[1]])
    bmm_0 = aten.bmm.default(exp_0, exp_1)
    return (bmm_0,)


def _bmm_to_mm_replacement_1_bf16(arg_0_, arg_1_):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = aten.squeeze.dim(arg_0_, 1)
    mm_0 = aten.mm.default(squeeze_0, arg_1_)
    unsqueeze_0 = aten.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


def _bmm_to_mm_pattern_1(arg_0, arg_1):
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    exp_0 = aten.expand.default(arg_0, [shape_0[0], 1, shape_0[-1]])
    exp_1 = aten.expand.default(arg_1, [shape_0[0], shape_1[0], shape_1[1]])
    bmm_0 = aten.bmm.default(exp_0, exp_1)
    return (bmm_0,)


def _bmm_to_mm_replacement_1(arg_0, arg_1):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = aten.squeeze.dim(arg_0, 1)
    mm_0 = aten.mm.default(squeeze_0, arg_1)
    unsqueeze_0 = aten.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


# Pattern for subsequent token for ChatGLM3
# Expand followed by a view for the input,
# Expand for the weight will be args to bmm.
# TODO: Validate and remove the pattern after
# experimenting with 'remove_redundant_view'
def _bmm_to_mm_pattern_2_bf16(arg_0_, arg_1_):
    shape_0 = arg_0_.size()
    shape_1 = arg_1_.size()
    exp_0 = aten.expand.default(arg_0_, arg_0_.size())
    view_0 = aten.view.default(exp_0, arg_0_.size())
    exp_1 = aten.expand.default(arg_1_, [shape_0[0], shape_1[0], shape_1[1]])
    bmm_0 = aten.bmm.default(view_0, exp_1)
    return (bmm_0,)


def _bmm_to_mm_replacement_2_bf16(arg_0_, arg_1_):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = aten.squeeze.dim(arg_0_, 0)
    mm_0 = aten.mm.default(squeeze_0, arg_1_)
    unsqueeze_0 = aten.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


def _bmm_to_mm_pattern_2(arg_0, arg_1):
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    exp_0 = aten.expand.default(arg_0, arg_0.size())
    view_0 = aten.view.default(exp_0, arg_0.size())
    exp_1 = aten.expand.default(arg_1, [shape_0[0], shape_1[0], shape_1[1]])
    bmm_0 = aten.bmm.default(view_0, exp_1)
    return (bmm_0,)


def _bmm_to_mm_replacement_2(arg_0, arg_1):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = aten.squeeze.dim(arg_0, 0)
    mm_0 = aten.mm.default(squeeze_0, arg_1)
    unsqueeze_0 = aten.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


# adding patterns completed #


def _dummy_extra_check(match):
    return True


# Checks for ChatGLM pattern
# Check if the number of users for the expand and view nodes involved
# have only 1 user.
# Check the shape of the input to be squeezable in the 2nd dimension.
def _bmm_to_mm_check_1(match):
    arg_key = "arg_0" if "arg_0" in match.kwargs else "arg_0_"
    if match.kwargs[arg_key].meta["val"].shape[1] != 1:
        return False
    return True


def _bmm_to_mm_check_2(match):
    arg_key = "arg_0" if "arg_0" in match.kwargs else "arg_0_"
    if match.kwargs[arg_key].meta["val"].shape[0] != 1:
        return False
    return True


def _get_pattern_with_replacement():
    # get the matcher_pass to register with
    from ._composite_ops_matcher import matcher_pass

    arg_1 = partial(torch.empty, (4096, 4096), device="cpu", requires_grad=True)
    a_1 = partial(arg_1, dtype=torch.float)
    a_1_bf16 = partial(arg_1, dtype=torch.bfloat16)
    arg_2 = partial(torch.empty, (512, 1, 4096), device="cpu", requires_grad=True)
    a_2 = partial(arg_2, dtype=torch.float)
    a_2_bf16 = partial(arg_2, dtype=torch.bfloat16)

    candidates = [
        (
            _gelu_tanh_pattern,
            _gelu_tanh_replacement,
            [a_1()],  # used to pass arguments
            {},  # this can be used to pass kwargs
            _dummy_extra_check,  # fake extra check, cannot be skipped
        ),
        (
            _gelu_tanh_pattern_bf16,
            _gelu_tanh_replacement_bf16,
            [a_1_bf16()],
            {},
            _dummy_extra_check,
        ),
        (
            _gelu_erf_pattern,
            _gelu_erf_replacement,
            [a_1()],
            {},
            _dummy_extra_check,
        ),
        (
            _gelu_erf_pattern_bf16,
            _gelu_erf_replacement_bf16,
            [a_1_bf16()],
            {},
            _dummy_extra_check,
        ),
        (
            _bmm_to_mm_pattern_1,
            _bmm_to_mm_replacement_1,
            [a_2(), a_1()],
            {},
            _bmm_to_mm_check_1,
        ),
        (
            _bmm_to_mm_pattern_1_bf16,
            _bmm_to_mm_replacement_1_bf16,
            [a_2_bf16(), a_1_bf16()],
            {},
            _bmm_to_mm_check_1,
        ),
        (
            _bmm_to_mm_pattern_2,
            _bmm_to_mm_replacement_2,
            [a_2(), a_1()],
            {},
            _bmm_to_mm_check_2,
        ),
        (
            _bmm_to_mm_pattern_2_bf16,
            _bmm_to_mm_replacement_2_bf16,
            [a_2_bf16(), a_1_bf16()],
            {},
            _bmm_to_mm_check_2,
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
