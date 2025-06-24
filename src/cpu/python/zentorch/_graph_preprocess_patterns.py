# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._utils import counters, is_version_compatible_import
from ._fusion_patterns import _matmul_dtypes_check
import functools
from functools import partial

# Brief steps:
# 1. define both patterns
# 2. call pattermatcher pass
# 3. use yield keyword as in pytorch
# 4. register both patterns
# Note : Keep args different for F32 and BF16 patterns.
# Take a look at gen_attention_patterns.py file in PT repo as well

at_ops = torch.ops.aten


# adding gelu pattern here, find a way to generate patterns
def _gelu_tanh_pattern(arg_0):
    mul_0 = at_ops.mul.Tensor(arg_0, 0.5)
    pow_0 = at_ops.pow.Tensor_Scalar(arg_0, 3.0)
    mul_1 = at_ops.mul.Tensor(pow_0, 0.044715)
    add_0 = at_ops.add.Tensor(arg_0, mul_1)
    mul_2 = at_ops.mul.Tensor(add_0, 0.7978845608028654)
    tanh_0 = at_ops.tanh.default(mul_2)
    add_1 = at_ops.add.Tensor(tanh_0, 1.0)
    mul_3 = at_ops.mul.Tensor(mul_0, add_1)
    return (mul_3,)


def _gelu_tanh_replacement(arg_0):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = at_ops.gelu.default(arg_0, approximate="tanh")
    return (gelu_0,)


def _gelu_erf_pattern(arg_0):
    mul_0 = at_ops.mul.Tensor(arg_0, 0.5)
    mul_1 = at_ops.mul.Tensor(arg_0, 0.7071067811865476)
    erf_0 = at_ops.erf.default(mul_1)
    add_0 = at_ops.add.Tensor(erf_0, 1.0)
    mul_2 = at_ops.mul.Tensor(mul_0, add_0)
    return (mul_2,)


def _gelu_erf_replacement(arg_0):
    counters["zentorch"]["pattern_matcher_gelu"] += 1
    gelu_0 = at_ops.gelu.default(arg_0)
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
# TODO : check _bmm_to_mm_pattern_2
# pattern for both args as 3D tensor case. based on that "args_2D",
# "args_3D" and req_shape_func helpers should be made global and
# necessary changes needs to added to _bmm_to_mm_pattern_2 pattern as well


def _bmm_to_mm_pattern_1(arg_0, arg_1):
    def args_2D(batch_size, tensor_shape):
        return [batch_size, tensor_shape[0], tensor_shape[-1]]

    def args_3D(batch_size, tensor_shape):
        return [batch_size, tensor_shape[1], tensor_shape[-1]]

    def req_shape_func(tensor_dim):
        get_shape_func = {"2": args_2D, "3": args_3D}
        return get_shape_func[str(tensor_dim)]

    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    dm0 = arg_0.dim()
    dm1 = arg_1.dim()
    # This check is required to handle 2D and 3D tensor dimensions
    batch_size = shape_0[0] if dm0 >= dm1 else shape_1[0]

    # Expand dimension for first tensor
    expand_dim_getter = req_shape_func(dm0)
    exp_0 = at_ops.expand.default(arg_0, expand_dim_getter(batch_size, shape_0))

    # Expand dimension for second tensor
    expand_dim_getter = req_shape_func(dm1)
    exp_1 = at_ops.expand.default(arg_1, expand_dim_getter(batch_size, shape_1))
    bmm_0 = at_ops.bmm.default(exp_0, exp_1)

    return (bmm_0,)


# There can be numerical differences in the output of mm and bmm.
# This pattern replacement is specifically for a model which has mm originally
# But due to view ops added by Pytorch, mm is replaced with bmm.
def _bmm_to_mm_replacement_1(arg_0, arg_1):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = at_ops.squeeze.dim(arg_0, 1)
    mm_0 = at_ops.mm.default(squeeze_0, arg_1)
    unsqueeze_0 = at_ops.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


# Pattern for subsequent token for ChatGLM3
# Expand followed by a view for the input,
# Expand for the weight will be args to bmm.
# TODO: Validate and remove the pattern after
# experimenting with 'remove_redundant_view'


def _bmm_to_mm_pattern_2(arg_0, arg_1):
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    exp_0 = at_ops.expand.default(arg_0, arg_0.size())
    view_0 = at_ops.view.default(exp_0, arg_0.size())
    exp_1 = at_ops.expand.default(arg_1, [shape_0[0], shape_1[0], shape_1[1]])
    bmm_0 = at_ops.bmm.default(view_0, exp_1)
    return (bmm_0,)


def _bmm_to_mm_replacement_2(arg_0, arg_1):
    counters["zentorch"]["pattern_matcher_bmm_to_mm"] += 1
    squeeze_0 = at_ops.squeeze.dim(arg_0, 0)
    mm_0 = at_ops.mm.default(squeeze_0, arg_1)
    unsqueeze_0 = at_ops.unsqueeze.default(mm_0, 1)
    return (unsqueeze_0,)


# adding patterns completed #


def _dummy_extra_check(match):
    return True


# Checks for ChatGLM pattern
# Check if the number of users for the expand and view nodes involved
# have only 1 user.
# Check the shape of the input to be squeezable in the 2nd dimension.
def _bmm_to_mm_check_1(match):
    dim0 = match.kwargs["arg_0"].meta["val"].dim()
    dim1 = match.kwargs["arg_1"].meta["val"].dim()
    if dim0 <= dim1:
        return False
    if match.kwargs["arg_0"].meta["val"].shape[1] != 1:
        return False
    is_dtype_same = _matmul_dtypes_check(match)
    return is_dtype_same


def _bmm_to_mm_check_2(match):
    if match.kwargs["arg_0"].meta["val"].shape[0] != 1:
        return False
    is_dtype_same = _matmul_dtypes_check(match)
    return is_dtype_same


def _get_pattern_with_replacement():
    # get the matcher_pass to register with
    from ._graph_preprocess_matcher import matcher_pass

    arg_1 = partial(
        torch.empty, (64, 32), device="cpu", requires_grad=True, dtype=torch.float
    )
    arg_2 = partial(
        torch.empty, (512, 1, 64), device="cpu", requires_grad=True, dtype=torch.float
    )

    candidates = [
        (
            _gelu_tanh_pattern,
            _gelu_tanh_replacement,
            [arg_1()],  # used to pass arguments
            {},  # this can be used to pass kwargs
            _dummy_extra_check,  # fake extra check, cannot be skipped
        ),
        (
            _gelu_erf_pattern,
            _gelu_erf_replacement,
            [arg_1()],
            {},
            _dummy_extra_check,
        ),
        (
            _bmm_to_mm_pattern_1,
            _bmm_to_mm_replacement_1,
            [arg_2(), arg_1()],
            {},
            _bmm_to_mm_check_1,
        ),
        (
            _bmm_to_mm_pattern_2,
            _bmm_to_mm_replacement_2,
            [arg_2(), arg_1()],
            {},
            _bmm_to_mm_check_2,
        ),
    ]
    for pattern, replacement, args, workaround, extra_check in candidates:
        assert isinstance(workaround, dict)
        name = pattern.__name__
        inference_name = name + "_inference"
        # pre 2.2 PT versions use a different name for fwd-tracer
        # remove the else block when deprecating support for PT <= 2.1.x
        if is_version_compatible_import(["_inductor", "pattern_matcher"], ["fwd_only"]):
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
        else:
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


@functools.lru_cache(None)
def _replace_init():
    from torch._inductor.pattern_matcher import register_replacement

    # loop for _get_pattern_with_replacement
    for _key, register_replacement_kwargs in _get_pattern_with_replacement():
        # TODO: try to use use gen_register_replacement for generating the patterns
        # for PT >= 2.4, do the following
        # gen_register_replacement(key, **register_replacement_kwargs)
        register_replacement(**register_replacement_kwargs)
