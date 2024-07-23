# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import collections
import functools

# Brief steps:
# 1. define both patterns
# 2. call pattermatcher pass and loop over dtypes for those
#    and create functools.patrial versions
# 3. use yield keyword as in pytorch
# 4. register both patterns
# Take a look at gen_attention_patterns.py file in PT repo as well

aten = torch.ops.aten
counters = collections.defaultdict(collections.Counter)


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


# adding patterns completed #


def _dummy_extra_check(match):
    return True


def _get_pattern_with_replacement():
    # get the matcher_pass to register with
    from ._composite_ops_matcher import matcher_pass
    from torch._inductor.pattern_matcher import fwd_only

    arg_1 = functools.partial(torch.empty, (3, 15), device="cpu", requires_grad=True)
    a_1 = functools.partial(arg_1, dtype=torch.float)
    a_1_bf16 = functools.partial(arg_1, dtype=torch.bfloat16)
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
    ]
    for pattern, replacement, args, workaround, extra_check in candidates:
        assert isinstance(workaround, dict)
        name = pattern.__name__
        inference_name = name + "_inference"
        yield inference_name, {
            "search_fn": pattern,
            "replace_fn": replacement,
            "example_inputs": args,
            "trace_fn": fwd_only,
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
        register_replacement(**register_replacement_kwargs, search_fn_pattern=None)
