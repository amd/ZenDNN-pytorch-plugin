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
    # arg_0 -> 2D (n x k)
    # arg_1 -> 2D (k x m)
    # arg_2 can be N-dimensional (N >= 2), below example is for 3D case
    # arg_2 -> 3D (n/b x b*a x m/a) => (n x m)
    mm_0 = zt_ops.zentorch_mm_silu.default(arg_0, arg_1)  # (n x m)
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(mm_0, arg_2.size())  # (n/b x a*b x m/a)
        mul_0 = at_ops.mul.Tensor(view_0, arg_2)  # (n/b x a*b x m/a)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, arg_2)
    return (mul_0,)


def _mm_silu_mul_replacement(arg_0, arg_1, arg_2):
    counters["zentorch"]["pattern_matcher_mm_silu_mul"] += 1
    shape_0 = arg_0.size()  # (n x k)
    shape_1 = arg_1.size()  # (k x m)
    shape_2 = arg_2.size()  # (n/b x a*b x m/a) => (n x m)
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[-1]])  # (n x m)
        mul_0 = zt_ops.zentorch_mm_silu_mul.default(arg_0, arg_1, view_0)  # (n x m)
        out_0 = at_ops.view.default(mul_0, shape_2)  # (n/b x a*b x m/a)
    else:
        out_0 = zt_ops.zentorch_mm_silu_mul.default(arg_0, arg_1, arg_2)
    return (out_0,)


def _addmm_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0):
    mm_0 = zt_ops.zentorch_addmm_silu.default(bias_0, arg_0, arg_1)
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(mm_0, arg_2.size())
        mul_0 = at_ops.mul.Tensor(view_0, arg_2)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, arg_2)
    return (mul_0,)


def _addmm_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_silu_mul.default(bias_0, arg_0, arg_1, view_0)
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_silu_mul.default(bias_0, arg_0, arg_1, arg_2)
    return (out_0,)


def _addmm_1dbias_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0):
    mm_0 = zt_ops.zentorch_addmm_1dbias_silu.default(bias_0, arg_0, arg_1)
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(mm_0, arg_2.size())
        mul_0 = at_ops.mul.Tensor(view_0, arg_2)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, arg_2)
    return (mul_0,)


def _addmm_1dbias_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, view_0
        )
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, arg_2
        )
    return (out_0,)


def _addmm_1dbias_add_pattern(arg_0, arg_1, add1, bias_0):
    # bias_0: bias
    # arg_0_: mat1
    # arg_1_: mat2
    # add1: add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1)
    if add1.dim() != 2:
        view = at_ops.view.default(addmm, add1.size())
        add_res = at_ops.add(view, add1)
    else:
        add_res = at_ops.add(addmm, add1)
    return add_res


def _addmm_1dbias_add_replacement(arg_0, arg_1, add1, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add1.size()
    if add1.dim() != 2:
        view_0 = at_ops.view.default(add1, [shape_0[0], shape_1[1]])
        add_0 = zt_ops.zentorch_addmm_1dbias_add.default(bias_0, arg_0, arg_1, view_0)
        out_0 = at_ops.view.default(add_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add.default(bias_0, arg_0, arg_1, add1)
    return (out_0,)


def _addmm_1dbias_view_add_add_pattern(arg_0, arg_1, add1, add2, bias_0):
    # bias_0: bias
    # arg_0: mat1
    # arg_1: mat2
    # add1: add
    # add2: 2nd add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1)
    if add1.dim() != 2:
        view = at_ops.view.default(addmm, add1.size())
        add_res = at_ops.add(view, add1)
    else:
        add_res = at_ops.add(addmm, add1)
    add_res_2 = at_ops.add(add_res, add2)
    return add_res_2


def _addmm_1dbias_view_add_add_replacement(arg_0, arg_1, add1, add2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add1.size()
    # set of conditions is possible for this (for now, we have just 2)
    if add1.dim() != 2 and add2.dim() != 2:
        view_0 = at_ops.view.default(add1, [shape_0[0], shape_1[1]])
        view_1 = at_ops.view.default(add2, [shape_0[0], shape_1[1]])
        linear_add = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, view_0, view_1
        )
        out_0 = at_ops.view.default(linear_add, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, add1, add2
        )
    return out_0


def _mm_add_pattern(arg_0, arg_1, add1):
    mm = zt_ops.zentorch_mm(arg_0, arg_1)
    if add1.dim() != 2:
        view = at_ops.view.default(mm, add1.size())
        add_res = at_ops.add(view, add1)
    else:
        add_res = at_ops.add(mm, add1)
    return add_res


def _mm_add_replacement(arg_0, arg_1, add1):
    counters["zentorch"]["pattern_matcher_mm_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add1.size()

    if add1.dim() != 2:
        view_0 = at_ops.view.default(add1, [shape_0[0], shape_1[1]])
        addmm = zt_ops.zentorch_addmm.default(view_0, arg_0, arg_1)
        out_0 = at_ops.view.default(addmm, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm.default(add1, arg_0, arg_1)
    return (out_0,)


# adding patterns completed #


def _same_dtypes_check(match):
    is_bf16 = True
    is_fp32 = True
    for _, v in match.kwargs.items():
        if not torch.is_tensor(v.meta["val"]):
            continue
        is_bf16 = is_bf16 and (v.meta["val"].dtype == torch.bfloat16)
        is_fp32 = is_fp32 and (v.meta["val"].dtype == torch.float)
    return is_bf16 ^ is_fp32


def _matmul_dtypes_check(match):
    is_bf16 = True
    is_fp32 = True
    for k, v in match.kwargs.items():
        if not torch.is_tensor(v.meta["val"]) or k in ("add1", "add2"):
            continue
        is_bf16 = is_bf16 and (v.meta["val"].dtype == torch.bfloat16)
        is_fp32 = is_fp32 and (v.meta["val"].dtype == torch.float)
    return is_bf16 ^ is_fp32


def _dim_check(match):
    if match.kwargs["add1"].meta["val"].shape != match.kwargs["add2"].meta["val"].shape:
        return False
    is_dtype_same = _matmul_dtypes_check(match)
    return is_dtype_same


def _get_pattern_with_replacement():
    # get the matcher_pass to register with
    from ._fusion_matcher import matcher_pass

    # mat 1
    arg_1 = partial(
        torch.empty, (256, 32), device="cpu", requires_grad=True, dtype=torch.float
    )
    # mat2
    arg_2 = partial(
        torch.empty, (32, 512), device="cpu", requires_grad=True, dtype=torch.float
    )
    arg_3 = partial(
        torch.empty, (64, 128, 16), device="cpu", requires_grad=True, dtype=torch.float
    )
    # bias
    arg_4 = partial(
        torch.empty, (256, 512), device="cpu", requires_grad=True, dtype=torch.float
    )
    # 1d-bias
    arg_5 = partial(
        torch.empty, (1, 512), device="cpu", requires_grad=True, dtype=torch.float
    )
    # add
    arg_6 = partial(
        torch.empty, (4, 64, 512), device="cpu", requires_grad=True, dtype=torch.float
    )

    # TODO: Add kwargs later to the patterns when removing
    # support for PT 2.1
    # kwarg_beta_alpha = {"beta": 7.8, "alpha": -9.6}

    candidates = [
        (
            _mm_silu_mul_pattern,
            _mm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3()],
            {},
            _same_dtypes_check,
        ),
        (
            _mm_silu_mul_pattern,
            _mm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4()],
            {},
            _same_dtypes_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3(), arg_4()],
            {},
            _same_dtypes_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_4()],
            {},
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3(), arg_5()],
            {},
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_5()],
            {},
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_add_pattern,
            _addmm_1dbias_add_replacement,
            [arg_1(), arg_2(), arg_3(), arg_5()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_1dbias_add_pattern,
            _addmm_1dbias_add_replacement,
            [arg_1(), arg_2(), arg_4(), arg_5()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_1dbias_view_add_add_pattern,
            _addmm_1dbias_view_add_add_replacement,
            [arg_1(), arg_2(), arg_3(), arg_3(), arg_5()],
            {},
            _dim_check,
        ),
        (
            _addmm_1dbias_view_add_add_pattern,
            _addmm_1dbias_view_add_add_replacement,
            [arg_1(), arg_2(), arg_4(), arg_4(), arg_5()],
            {},
            _dim_check,
        ),
        (
            _mm_add_pattern,
            _mm_add_replacement,
            [arg_1(), arg_2(), arg_6()],
            {},
            _matmul_dtypes_check,
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
