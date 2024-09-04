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


def _addmm_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    mm_0 = zt_ops.zentorch_addmm_silu.default(
        bias_0, arg_0, arg_1, beta=beta, alpha=alpha
    )
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(mm_0, arg_2.size())
        mul_0 = at_ops.mul.Tensor(view_0, arg_2)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, arg_2)
    return (mul_0,)


def _addmm_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    counters["zentorch"]["pattern_matcher_addmm_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_silu_mul.default(
            bias_0, arg_0, arg_1, view_0, beta=beta, alpha=alpha
        )
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_silu_mul.default(
            bias_0, arg_0, arg_1, arg_2, beta=beta, alpha=alpha
        )
    return (out_0,)


def _addmm_1dbias_silu_mul_pattern(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    mm_0 = zt_ops.zentorch_addmm_1dbias_silu.default(
        bias_0, arg_0, arg_1, beta=beta, alpha=alpha
    )
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(mm_0, arg_2.size())
        mul_0 = at_ops.mul.Tensor(view_0, arg_2)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, arg_2)
    return (mul_0,)


def _addmm_1dbias_silu_mul_replacement(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, view_0, beta=beta, alpha=alpha
        )
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, arg_2, beta=beta, alpha=alpha
        )
    return (out_0,)


def _addmm_1dbias_add_pattern(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    # bias_0: bias
    # arg_0_: mat1
    # arg_1_: mat2
    # arg_2_: add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1, beta=beta, alpha=alpha)
    if arg_2.dim() != 2:
        view = at_ops.view.default(addmm, arg_2.size())
        add_res = at_ops.add(view, arg_2)
    else:
        add_res = at_ops.add(addmm, arg_2)
    return add_res


def _addmm_1dbias_add_replacement(arg_0, arg_1, arg_2, bias_0, beta, alpha):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    if arg_2.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[1]])
        add_0 = zt_ops.zentorch_addmm_1dbias_add.default(
            bias_0, arg_0, arg_1, view_0, beta=beta, alpha=alpha
        )
        out_0 = at_ops.view.default(add_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add.default(
            bias_0, arg_0, arg_1, arg_2, beta=beta, alpha=alpha
        )
    return (out_0,)


def _addmm_1dbias_view_add_add_pattern(arg_0, arg_1, arg_2, arg_3, bias_0, beta, alpha):
    # bias_0: bias
    # arg_0: mat1
    # arg_1: mat2
    # arg_2: add
    # arg_3: 2nd add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1, beta=beta, alpha=alpha)
    if arg_2.dim() != 2:
        view = at_ops.view.default(addmm, arg_2.size())
        add_res = at_ops.add(view, arg_2)
    else:
        add_res = at_ops.add(addmm, arg_2)
    add_res_2 = at_ops.add(add_res, arg_3)
    return add_res_2


def _addmm_1dbias_view_add_add_replacement(
    arg_0, arg_1, arg_2, arg_3, bias_0, beta, alpha
):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = arg_2.size()
    # set of conditions is possible for this (for now, we have just 2)
    if arg_2.dim() != 2 and arg_3.dim() != 2:
        view_0 = at_ops.view.default(arg_2, [shape_0[0], shape_1[1]])
        view_1 = at_ops.view.default(arg_3, [shape_0[0], shape_1[1]])
        linear_add = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, view_0, view_1, beta=beta, alpha=alpha
        )
        out_0 = at_ops.view.default(linear_add, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, arg_2, view_1, beta=beta, alpha=alpha
        )
    return out_0


# adding patterns completed #

# add required extra checks for patterns #


def _same_dtypes_check(match):
    is_bf16 = True
    is_fp32 = True
    for _, v in match.kwargs.items():
        is_bf16 = is_bf16 and (v.meta["val"].dtype == torch.bfloat16)
        is_fp32 = is_fp32 and (v.meta["val"].dtype == torch.float)
    return is_bf16 or is_fp32


def _dim_check(match):
    if (
        match.kwargs["arg_2"].meta["val"].shape
        != match.kwargs["arg_3"].meta["val"].shape
    ):
        return False
    is_dtype_same = _same_dtypes_check(match)
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

    kwarg_beta_alpha = {"beta": 7.8, "alpha": -9.6}
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
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_4()],
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3(), arg_5()],
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_5()],
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_add_pattern,
            _addmm_1dbias_add_replacement,
            [arg_1(), arg_2(), arg_3(), arg_5()],
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_add_pattern,
            _addmm_1dbias_add_replacement,
            [arg_1(), arg_2(), arg_4(), arg_5()],
            kwarg_beta_alpha,
            _same_dtypes_check,
        ),
        (
            _addmm_1dbias_view_add_add_pattern,
            _addmm_1dbias_view_add_add_replacement,
            [arg_1(), arg_2(), arg_3(), arg_3(), arg_5()],
            kwarg_beta_alpha,
            _dim_check,
        ),
        (
            _addmm_1dbias_view_add_add_pattern,
            _addmm_1dbias_view_add_add_replacement,
            [arg_1(), arg_2(), arg_4(), arg_4(), arg_5()],
            kwarg_beta_alpha,
            _dim_check,
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
