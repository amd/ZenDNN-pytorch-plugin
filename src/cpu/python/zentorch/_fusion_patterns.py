# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._compile_backend import torch_version
from ._utils import counters
import functools
from functools import partial
import operator
from ._utils import add_version_suffix

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

# add patterns below


def _mm_silu_mul_pattern(arg_0, arg_1, mul_1):
    # arg_0 -> 2D (n x k)
    # arg_1 -> 2D (k x m)
    # mul_1 can be N-dimensional (N >= 2), below example is for 3D case
    # mul_1 -> 3D (n/b x b*a x m/a) => (n x m)
    mm_0 = zt_ops.zentorch_mm_silu.default(arg_0, arg_1)  # (n x m)
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mm_0, mul_1.size())  # (n/b x a*b x m/a)
        mul_0 = at_ops.mul.Tensor(view_0, mul_1)  # (n/b x a*b x m/a)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, mul_1)
    return (mul_0,)


def _mm_silu_mul_replacement(arg_0, arg_1, mul_1):
    counters["zentorch"]["pattern_matcher_mm_silu_mul"] += 1
    shape_0 = arg_0.size()  # (n x k)
    shape_1 = arg_1.size()  # (k x m)
    shape_2 = mul_1.size()  # (n/b x a*b x m/a) => (n x m)
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mul_1, [shape_0[0], shape_1[-1]])  # (n x m)
        mul_0 = zt_ops.zentorch_mm_silu_mul.default(arg_0, arg_1, view_0)  # (n x m)
        out_0 = at_ops.view.default(mul_0, shape_2)  # (n/b x a*b x m/a)
    else:
        out_0 = zt_ops.zentorch_mm_silu_mul.default(arg_0, arg_1, mul_1)
    return (out_0,)


def _addmm_silu_mul_pattern(arg_0, arg_1, mul_1, bias_0):
    mm_0 = zt_ops.zentorch_addmm_silu.default(bias_0, arg_0, arg_1)
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mm_0, mul_1.size())
        mul_0 = at_ops.mul.Tensor(view_0, mul_1)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, mul_1)
    return (mul_0,)


def _addmm_silu_mul_replacement(arg_0, arg_1, mul_1, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = mul_1.size()
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mul_1, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_silu_mul.default(bias_0, arg_0, arg_1, view_0)
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_silu_mul.default(bias_0, arg_0, arg_1, mul_1)
    return (out_0,)


def _addmm_1dbias_silu_mul_pattern(arg_0, arg_1, mul_1, bias_0):
    mm_0 = zt_ops.zentorch_addmm_1dbias_silu.default(bias_0, arg_0, arg_1)
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mm_0, mul_1.size())
        mul_0 = at_ops.mul.Tensor(view_0, mul_1)
    else:
        mul_0 = at_ops.mul.Tensor(mm_0, mul_1)
    return (mul_0,)


def _addmm_1dbias_silu_mul_replacement(arg_0, arg_1, mul_1, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = mul_1.size()
    if mul_1.dim() != 2:
        view_0 = at_ops.view.default(mul_1, [shape_0[0], shape_1[-1]])
        mul_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, view_0
        )
        out_0 = at_ops.view.default(mul_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_silu_mul.default(
            bias_0, arg_0, arg_1, mul_1
        )
    return (out_0,)


def _addmm_1dbias_add_pattern(arg_0, arg_1, add_1, bias_0):
    # bias_0: bias
    # arg_0_: mat1
    # arg_1_: mat2
    # add_1: add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1)
    if add_1.dim() != 2:
        view = at_ops.view.default(addmm, add_1.size())
        add_res = at_ops.add(view, add_1)
    else:
        add_res = at_ops.add(addmm, add_1)
    return add_res


def _addmm_1dbias_add_replacement(arg_0, arg_1, add_1, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add_1.size()
    if add_1.dim() != 2:
        view_0 = at_ops.view.default(add_1, [shape_0[0], shape_1[1]])
        add_0 = zt_ops.zentorch_addmm_1dbias_add.default(bias_0, arg_0, arg_1, view_0)
        out_0 = at_ops.view.default(add_0, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add.default(bias_0, arg_0, arg_1, add_1)
    return (out_0,)


def _addmm_1dbias_view_add_add_pattern(arg_0, arg_1, add_1, add_2, bias_0):
    # bias_0: bias
    # arg_0: mat1
    # arg_1: mat2
    # add_1: add
    # add_2: 2nd add
    addmm = zt_ops.zentorch_addmm_1dbias(bias_0, arg_0, arg_1)
    if add_1.dim() != 2:
        view = at_ops.view.default(addmm, add_1.size())
        add_res = at_ops.add(view, add_1)
    else:
        add_res = at_ops.add(addmm, add_1)
    add_res_2 = at_ops.add(add_res, add_2)
    return add_res_2


def _addmm_1dbias_view_add_add_replacement(arg_0, arg_1, add_1, add_2, bias_0):
    counters["zentorch"]["pattern_matcher_addmm_1dbias_add_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add_1.size()
    # set of conditions is possible for this (for now, we have just 2)
    if add_1.dim() != 2 and add_2.dim() != 2:
        view_0 = at_ops.view.default(add_1, [shape_0[0], shape_1[1]])
        view_1 = at_ops.view.default(add_2, [shape_0[0], shape_1[1]])
        linear_add = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, view_0, view_1
        )
        out_0 = at_ops.view.default(linear_add, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm_1dbias_add_add.default(
            bias_0, arg_0, arg_1, add_1, add_2
        )
    return out_0


def _mm_add_pattern(arg_0, arg_1, add_1):
    mm = zt_ops.zentorch_mm(arg_0, arg_1)
    if add_1.dim() != 2:
        view = at_ops.view.default(mm, add_1.size())
        add_res = at_ops.add(view, add_1)
    else:
        add_res = at_ops.add(mm, add_1)
    return add_res


def _mm_add_replacement(arg_0, arg_1, add_1):
    counters["zentorch"]["pattern_matcher_mm_add"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    shape_2 = add_1.size()

    if add_1.dim() != 2:
        view_0 = at_ops.view.default(add_1, [shape_0[0], shape_1[1]])
        addmm = zt_ops.zentorch_addmm.default(view_0, arg_0, arg_1)
        out_0 = at_ops.view.default(addmm, shape_2)
    else:
        out_0 = zt_ops.zentorch_addmm.default(add_1, arg_0, arg_1)
    return (out_0,)


# Adding 2 Isometric patterns for linear+add fusion
def _woq_linear_add_pattern_1(arg_0, arg_1, arg_2, arg_3, bias_0, add_1):
    # arg_0 : input_tensor (n-d)
    # arg_1 : qweight
    # arg_2 : weight_scales
    # arg_3 : weight_zero_point
    # bias_0 : bias
    # add_1 : add_tensor
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add_res = at_ops.add(add_1, woq_linear_out)
    return add_res


def _woq_linear_add_pattern_2(arg_0, arg_1, arg_2, arg_3, bias_0, add_1):
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add_res = at_ops.add(woq_linear_out, add_1)
    return add_res


def _woq_linear_add_replacement(arg_0, arg_1, arg_2, arg_3, bias_0, add_1):
    counters["zentorch"]["pattern_matcher_woq_add"] += 1
    out_0 = zt_ops.zentorch_woq_linear_add.default(
        arg_0, arg_1, arg_2, arg_3, bias_0, add_1
    )
    return (out_0,)


# Adding 4 Isometric pattern for linear+add+add fusion
def _woq_linear_add_add_pattern_1(arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2):
    # arg_0 : input_tensor (n-d)
    # arg_1 : qweight
    # arg_2 : weight_scales
    # arg_3 : weight_zero_point
    # bias_0 : bias
    # add_1 : add_tensor
    # add_2 : add_tensor
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add1_res = at_ops.add(woq_linear_out, add_1)
    add_res = at_ops.add(add1_res, add_2)
    return add_res


def _woq_linear_add_add_pattern_2(arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2):
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add1_res = at_ops.add(add_1, woq_linear_out)
    add_res = at_ops.add(add1_res, add_2)
    return add_res


def _woq_linear_add_add_pattern_3(arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2):
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add1_res = at_ops.add(woq_linear_out, add_1)
    add_res = at_ops.add(add_2, add1_res)
    return add_res


def _woq_linear_add_add_pattern_4(arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2):
    woq_linear_out = zt_ops.zentorch_woq_linear(arg_0, arg_1, arg_2, arg_3, bias_0)
    add1_res = at_ops.add(add_1, woq_linear_out)
    add_res = at_ops.add(add_2, add1_res)
    return add_res


def _woq_linear_add_add_replacement(arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2):
    counters["zentorch"]["pattern_matcher_woq_add_add"] += 1
    out_0 = zt_ops.zentorch_woq_linear_add_add.default(
        arg_0, arg_1, arg_2, arg_3, bias_0, add_1, add_2
    )
    return out_0


# Adding 2 Isometric patterns for linear+silu+mul fusion
def _woq_linear_silu_mul_pattern_1(arg_0, arg_1, arg_2, arg_3, bias_0, mul_1):
    woq_linear_out = zt_ops.zentorch_woq_linear_silu(arg_0, arg_1, arg_2, arg_3, bias_0)
    mul_1_res = at_ops.mul(woq_linear_out, mul_1)
    return mul_1_res


def _woq_linear_silu_mul_pattern_2(arg_0, arg_1, arg_2, arg_3, bias_0, mul_1):
    woq_linear_out = zt_ops.zentorch_woq_linear_silu(arg_0, arg_1, arg_2, arg_3, bias_0)
    mul_1_res = at_ops.mul(mul_1, woq_linear_out)
    return mul_1_res


def _woq_linear_silu_mul_replacement(arg_0, arg_1, arg_2, arg_3, bias_0, mul_1):
    counters["zentorch"]["pattern_matcher_woq_silu_mul"] += 1
    out_0 = zt_ops.zentorch_woq_linear_silu_mul(
        arg_0, arg_1, arg_2, arg_3, bias_0, mul_1
    )
    return out_0


# add a split mm op for phi-3 model
def _split_mm_pattern(arg_0, arg_1):
    # view -> mm -> view -> split ------>\
    #                        |           mul
    #                        |-> silu -->/
    # arg_0: input to the input view to mm, shape => [bs, sl, m]
    # arg_1: second input argument to mm, shape => [m, k]
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])  # [bs*sl, m]
    mm_0 = zt_ops.zentorch_mm(view_0, arg_1)  # [bs*sl, k]
    view_1 = at_ops.view(mm_0, [shape_0[0], shape_0[1], shape_1[-1]])  # [bs, sl, k]
    split_0 = at_ops.split(view_1, shape_1[-1] // 2, -1)
    getitem_0 = operator.getitem(split_0, 0)  # [bs, sl, k//2]
    getitem_1 = operator.getitem(split_0, 1)
    silu_0 = at_ops.silu(getitem_0)  # [bs, sl, k//2]
    mul_0 = at_ops.mul(getitem_1, silu_0)  # [bs, sl, k//2]
    return mul_0


def _split_mm_replacement(arg_0, arg_1):
    # we will split the mm itself instead of its output
    # view ------------> mm --|
    #   |   split ------>/    |
    #   |     |-------->\     V
    #   |-----------> mm_silu_mul
    counters["zentorch"]["pattern_matcher_split_mm"] += 1
    shape_0 = arg_0.size()
    shape_1 = arg_1.size()
    view_0 = at_ops.view(arg_0, [shape_0[0] * shape_0[1], shape_0[2]])  # [bs*sl, m]
    split_0 = at_ops.split(arg_1, shape_1[-1] // 2, -1)
    mat_0_0 = operator.getitem(split_0, 0)  # [m, k//2]
    mat_0_1 = operator.getitem(split_0, 1)  # [m, k//2]
    mm_0 = zt_ops.zentorch_mm(view_0, mat_0_1)  # [bs*sl, k//2]
    mul_0 = zt_ops.zentorch_mm_silu_mul(view_0, mat_0_0, mm_0)  # [bs*sl, k//2]
    view_1 = at_ops.view(
        mul_0, [shape_0[0], shape_0[1], shape_1[-1] // 2]
    )  # [bs, sl, k//2]
    return view_1


# adding patterns completed #


def _matmul_dtypes_check(match):
    # This check is for handling the datatypes of the matmul parameters and
    # post op buffers.
    # The cases are as follows:
    #     -> If the matmul parameters are fp32, then the post op buffers must
    #        be fp32.
    #     -> If the matmul parameters are bf16, then the post op buffers must
    #        be bf16.

    is_fp32 = True
    is_bf16 = True
    do_post_ops_exist = False
    post_op_dtypes_fp32 = True
    post_op_dtypes_bf16 = True

    for k, v in match.kwargs.items():
        if not torch.is_tensor(v.meta["val"]):
            continue
        if k in ("add_1", "add_2", "mul_1"):
            post_op_dtypes_bf16 = post_op_dtypes_bf16 and (
                v.meta["val"].dtype == torch.bfloat16
            )
            post_op_dtypes_fp32 = post_op_dtypes_fp32 and (
                v.meta["val"].dtype == torch.float
            )
            do_post_ops_exist = True
        else:
            is_bf16 = is_bf16 and (v.meta["val"].dtype == torch.bfloat16)
            is_fp32 = is_fp32 and (v.meta["val"].dtype == torch.float)

    if do_post_ops_exist:
        if is_fp32 and not is_bf16:
            return post_op_dtypes_fp32 and not post_op_dtypes_bf16
        elif is_bf16 and not is_fp32:
            return post_op_dtypes_bf16 and not post_op_dtypes_fp32
    else:
        return is_bf16 ^ is_fp32


def _woq_check(match):
    expected_dtype = {
        "arg_0": torch.bfloat16,
        "arg_1": torch.int32,
        "arg_2": torch.float32,
        "arg_3": torch.int32,
        "bias_0": torch.bfloat16,
    }

    post_op_dtypes_bf16 = True

    for k, v in match.kwargs.items():
        if k in ("add_1", "add_2", "mul_1"):
            post_op_dtypes_bf16 = post_op_dtypes_bf16 and (
                v.meta["val"].dtype == torch.bfloat16
            )
        if not (torch.is_tensor(v.meta["val"]) and k in expected_dtype.keys()):
            continue
        if expected_dtype[str(k)] != v.meta["val"].dtype:
            return False

    return post_op_dtypes_bf16


def _dim_check(match):
    if (
        match.kwargs["add_1"].meta["val"].shape
        != match.kwargs["add_2"].meta["val"].shape
    ):
        return False
    is_dtype_same = _matmul_dtypes_check(match)
    return is_dtype_same


def _dummy_check_(match):
    return True


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
    arg_7 = partial(
        torch.empty, (512, 128), device="cpu", requires_grad=True, dtype=torch.float
    )
    inp = partial(
        torch.empty,
        (4, 32, 32),
        device="cpu",
        requires_grad=False,
        dtype=torch.bfloat16,
    )
    qweight = partial(
        torch.empty, (32, 4), device="cpu", requires_grad=False, dtype=torch.int32
    )
    woq_scales = partial(
        torch.empty, (1, 32), device="cpu", requires_grad=False, dtype=torch.float32
    )
    woq_qzeros = partial(
        torch.empty, (1, 4), device="cpu", requires_grad=False, dtype=torch.int32
    )
    woq_bias = partial(
        torch.empty, (32), device="cpu", requires_grad=False, dtype=torch.bfloat16
    )
    woq_binary = partial(
        torch.empty,
        (4, 32, 32),
        device="cpu",
        requires_grad=False,
        dtype=torch.bfloat16,
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
            _matmul_dtypes_check,
        ),
        (
            _mm_silu_mul_pattern,
            _mm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3(), arg_4()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_silu_mul_pattern,
            _addmm_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_4()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_3(), arg_5()],
            {},
            _matmul_dtypes_check,
        ),
        (
            _addmm_1dbias_silu_mul_pattern,
            _addmm_1dbias_silu_mul_replacement,
            [arg_1(), arg_2(), arg_4(), arg_5()],
            {},
            _matmul_dtypes_check,
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
        # TODO : check the appropriate order required for different fusions.
        (
            _woq_linear_add_add_pattern_1,
            _woq_linear_add_add_replacement,
            [
                inp(),
                qweight(),
                woq_scales(),
                woq_qzeros(),
                woq_bias(),
                woq_binary(),
                woq_binary(),
            ],
            {},
            _woq_check,
        ),
        (
            _woq_linear_add_add_pattern_2,
            _woq_linear_add_add_replacement,
            [
                inp(),
                qweight(),
                woq_scales(),
                woq_qzeros(),
                woq_bias(),
                woq_binary(),
                woq_binary(),
            ],
            {},
            _woq_check,
        ),
        (
            _woq_linear_add_add_pattern_3,
            _woq_linear_add_add_replacement,
            [
                inp(),
                qweight(),
                woq_scales(),
                woq_qzeros(),
                woq_bias(),
                woq_binary(),
                woq_binary(),
            ],
            {},
            _woq_check,
        ),
        (
            _woq_linear_add_add_pattern_4,
            _woq_linear_add_add_replacement,
            [
                inp(),
                qweight(),
                woq_scales(),
                woq_qzeros(),
                woq_bias(),
                woq_binary(),
                woq_binary(),
            ],
            {},
            _woq_check,
        ),
        (
            _woq_linear_add_pattern_1,
            _woq_linear_add_replacement,
            [inp(), qweight(), woq_scales(), woq_qzeros(), woq_bias(), woq_binary()],
            {},
            _woq_check,
        ),
        (
            _woq_linear_add_pattern_2,
            _woq_linear_add_replacement,
            [inp(), qweight(), woq_scales(), woq_qzeros(), woq_bias(), woq_binary()],
            {},
            _woq_check,
        ),
        (
            _woq_linear_silu_mul_pattern_1,
            _woq_linear_silu_mul_replacement,
            [inp(), qweight(), woq_scales(), woq_qzeros(), woq_bias(), woq_binary()],
            {},
            _woq_check,
        ),
        (
            _woq_linear_silu_mul_pattern_2,
            _woq_linear_silu_mul_replacement,
            [inp(), qweight(), woq_scales(), woq_qzeros(), woq_bias(), woq_binary()],
            {},
            _woq_check,
        ),
        (
            _split_mm_pattern,
            _split_mm_replacement,
            [arg_6(), arg_7()],
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
        if torch_version < add_version_suffix("2", "2"):
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
