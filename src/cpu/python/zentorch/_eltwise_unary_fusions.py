# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import functools
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    KeywordArg,
    Arg,
    Match,
)
from ._logging import get_logger

from ._utils import counters

logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

pass_pattern = PatternMatcherPass()


# register patterns below with replacements
# mm-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
)
def mm_relu_replacement(match: Match, arg_0, arg_1):
    def repl(arg_0, arg_1):
        counters["zentorch"]["zentorch_mm_relu"] += 1
        # TODO: remove fusion counters once testcases have been improved to check for op-name
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_mm_relu(arg_0, arg_1)

    match.replace_by_example(repl, [arg_0, arg_1])


# mm-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
)
def mm_silu_replacement(match, arg_0, arg_1):
    def repl(arg_0, arg_1):
        counters["zentorch"]["zentorch_mm_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_mm_silu(arg_0, arg_1)

    match.replace_by_example(repl, [arg_0, arg_1])


# gelu-erf extra check
def gelu_erf_check(match):
    if (
        match.kwargs.get("approximate") == "none"
        or match.kwargs.get("approximate") is None
    ):
        return True
    return False


# mm-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def mm_gelu_erf_replacement(match, arg_0, arg_1, approximate):
    def repl(arg_0, arg_1, approximate):
        counters["zentorch"]["zentorch_mm_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_mm_gelu_erf(arg_0, arg_1)

    match.replace_by_example(repl, [arg_0, arg_1, approximate])


# gelu-tanh extra check
def gelu_tanh_check(match):
    if match.kwargs.get("approximate") == "tanh":
        return True
    return False


# mm-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def mm_gelu_tanh_replacement(match, arg_0, arg_1, approximate):
    def repl(arg_0, arg_1, approximate):
        counters["zentorch"]["zentorch_mm_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_mm_gelu_tanh(arg_0, arg_1)

    match.replace_by_example(repl, [arg_0, arg_1, approximate])


# addmm-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_relu_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_addmm_relu(inp, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# addmm-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_silu_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_addmm_silu(inp, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# addmm-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def addmm_gelu_erf_replacement(match, inp, mat_1, mat_2, *, beta, alpha, approximate):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate):
        counters["zentorch"]["zentorch_addmm_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_addmm_gelu_erf(inp, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate])


# addmm-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_addmm,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def addmm_gelu_tanh_replacement(match, inp, mat_1, mat_2, *, beta, alpha, approximate):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate):
        counters["zentorch"]["zentorch_addmm_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_addmm_gelu_tanh(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        )

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate])


# addmm_1dbias-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_1dbias_relu_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm_1dbias_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_relu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        )

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# addmm_1dbias-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_1dbias_silu_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm_1dbias_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_silu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        )

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# addmm_1dbias-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def addmm_1dbias_gelu_erf_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate):
        counters["zentorch"]["zentorch_addmm_1dbias_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_gelu_erf(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        )

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate])


# addmm_1dbias-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_addmm_1dbias,
            Arg(),
            Arg(),
            Arg(),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def addmm_1dbias_gelu_tanh_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate):
        counters["zentorch"]["zentorch_addmm_1dbias_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_gelu_tanh(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        )

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate])


# when function has 5 or more arguments, we can create a list
woq_linear_args = [Arg() for _ in range(6)]


# woq_linear-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
    ),
    pass_dict=pass_pattern,
)
def woq_linear_relu_replacement(
    match,
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    *,
    weight_bits,
    compute_dtype,
):
    def repl(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    ):
        counters["zentorch"]["zentorch_woq_linear_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_woq_linear_relu(
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits=weight_bits,
            compute_dtype=compute_dtype,
        )

    match.replace_by_example(
        repl,
        [
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits,
            compute_dtype,
        ],
    )


# woq_linear-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
    ),
    pass_dict=pass_pattern,
)
def woq_linear_silu_replacement(
    match,
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    *,
    weight_bits,
    compute_dtype,
):
    def repl(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
    ):
        counters["zentorch"]["zentorch_woq_linear_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_woq_linear_silu(
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits=weight_bits,
            compute_dtype=compute_dtype,
        )

    match.replace_by_example(
        repl,
        [
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits,
            compute_dtype,
        ],
    )


# woq_linear-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def woq_linear_gelu_erf_replacement(
    match,
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    *,
    weight_bits,
    compute_dtype,
    approximate,
):
    def repl(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
        approximate,
    ):
        counters["zentorch"]["zentorch_woq_linear_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_woq_linear_gelu_erf(
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits=weight_bits,
            compute_dtype=compute_dtype,
        )

    match.replace_by_example(
        repl,
        [
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits,
            compute_dtype,
            approximate,
        ],
    )


# woq_linear-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            zt_ops.zentorch_woq_linear,
            *woq_linear_args,
            weight_bits=KeywordArg("weight_bits"),
            compute_dtype=KeywordArg("compute_dtype"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def woq_linear_gelu_tanh_replacement(
    match,
    input,
    qweight,
    weight_scales,
    weight_zero_point,
    bias,
    group_size,
    *,
    weight_bits,
    compute_dtype,
    approximate,
):
    def repl(
        input,
        qweight,
        weight_scales,
        weight_zero_point,
        bias,
        group_size,
        weight_bits,
        compute_dtype,
        approximate,
    ):
        counters["zentorch"]["zentorch_woq_linear_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_woq_linear_gelu_tanh(
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits=weight_bits,
            compute_dtype=compute_dtype,
        )

    match.replace_by_example(
        repl,
        [
            input,
            qweight,
            weight_scales,
            weight_zero_point,
            bias,
            group_size,
            weight_bits,
            compute_dtype,
            approximate,
        ],
    )


qlinear_args = [Arg() for _ in range(7)]


# qlinear-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
            output_dtype=KeywordArg("output_dtype"),
            output_scales=KeywordArg("output_scales"),
            output_zero_points=KeywordArg("output_zero_points"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
            output_dtype=KeywordArg("output_dtype"),
            output_scales=KeywordArg("output_scales"),
            output_zero_points=KeywordArg("output_zero_points"),
        ),
    ),
    pass_dict=pass_pattern,
)
def qlinear_relu_replacement(
    match,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    *,
    output_dtype,
    output_scales,
    output_zero_points,
):
    def repl(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype=output_dtype,
        output_scales=output_scales,
        output_zero_points=output_zero_points,
    ):
        counters["zentorch"]["zentorch_qlinear_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_qlinear_relu(
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
            output_scales=output_scales,
            output_zero_points=output_zero_points,
        )

    match.replace_by_example(
        repl,
        [
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
        ],
    )


# qlinear-sigmoid
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
            output_dtype=KeywordArg("output_dtype"),
            output_scales=KeywordArg("output_scales"),
            output_zero_points=KeywordArg("output_zero_points"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid_,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
            output_dtype=KeywordArg("output_dtype"),
            output_scales=KeywordArg("output_scales"),
            output_zero_points=KeywordArg("output_zero_points"),
        ),
    ),
    pass_dict=pass_pattern,
)
def qlinear_sigmoid_replacement(
    match,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    *,
    output_dtype,
    output_scales,
    output_zero_points,
):
    def repl(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype=output_dtype,
        output_scales=output_scales,
        output_zero_points=output_zero_points,
    ):
        counters["zentorch"]["zentorch_qlinear_sigmoid"] += 1
        counters["zentorch"]["sigmoid_fusion"] += 1
        return zt_ops.zentorch_qlinear_sigmoid(
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
            output_scales=output_scales,
            output_zero_points=output_zero_points,
        )

    match.replace_by_example(
        repl,
        [
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
        ],
    )


# mm-view-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def mm_view_relu_replacement(match, arg_0, arg_1, size):
    def repl(arg_0, arg_1, size):
        counters["zentorch"]["zentorch_mm_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_mm_relu(arg_0, arg_1).view(size)

    match.replace_by_example(repl, [arg_0, arg_1, size])


# mm-view-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def mm_view_silu_replacement(match, arg_0, arg_1, size):
    def repl(arg_0, arg_1, size):
        counters["zentorch"]["zentorch_mm_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_mm_silu(arg_0, arg_1).view(size)

    match.replace_by_example(repl, [arg_0, arg_1, size])


# mm-view-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def mm_view_gelu_erf_replacement(match, arg_0, arg_1, approximate, size):
    def repl(arg_0, arg_1, approximate, size):
        counters["zentorch"]["zentorch_mm_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_mm_gelu_erf(arg_0, arg_1).view(size)

    match.replace_by_example(repl, [arg_0, arg_1, approximate, size])


# mm-view-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(zt_ops.zentorch_mm, Arg(), Arg()),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def mm_view_gelu_tanh_replacement(match, arg_0, arg_1, approximate, size):
    def repl(arg_0, arg_1, approximate, size):
        counters["zentorch"]["zentorch_mm_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_mm_gelu_tanh(arg_0, arg_1).view(size)

    match.replace_by_example(repl, [arg_0, arg_1, approximate, size])


# addmm-view-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_view_relu_replacement(match, inp, mat_1, mat_2, *, beta, alpha, size):
    def repl(inp, mat_1, mat_2, beta, alpha, size):
        counters["zentorch"]["zentorch_addmm_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_addmm_relu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, size])


# addmm-view-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_view_silu_replacement(match, inp, mat_1, mat_2, *, beta, alpha, size):
    def repl(inp, mat_1, mat_2, beta, alpha, size):
        counters["zentorch"]["zentorch_addmm_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_addmm_silu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, size])


# addmm-view-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def addmm_view_gelu_erf_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate, size
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate, size):
        counters["zentorch"]["zentorch_addmm_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_addmm_gelu_erf(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate, size])


# addmm-view-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def addmm_view_gelu_tanh_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate, size
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate, size):
        counters["zentorch"]["zentorch_addmm_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_addmm_gelu_tanh(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate, size])


# addmm_1dbias-view-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_view_1dbias_relu_replacement(match, inp, mat_1, mat_2, *, beta, alpha, size):
    def repl(inp, mat_1, mat_2, beta, alpha, size):
        counters["zentorch"]["zentorch_addmm_1dbias_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_relu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, size])


# addmm_1dbias-view-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
    ),
    pass_dict=pass_pattern,
)
def addmm_view_1dbias_silu_replacement(match, inp, mat_1, mat_2, *, beta, alpha, size):
    def repl(inp, mat_1, mat_2, beta, alpha, size):
        counters["zentorch"]["zentorch_addmm_1dbias_silu"] += 1
        counters["zentorch"]["silu_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_silu(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, size])


# addmm_1dbias-view-gelu-erf
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_erf_check,
)
def addmm_view_1dbias_gelu_erf_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate, size
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate, size):
        counters["zentorch"]["zentorch_addmm_1dbias_gelu_erf"] += 1
        counters["zentorch"]["gelu_erf_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_gelu_erf(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate, size])


# addmm_1dbias-view-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.view,
            CallFunction(
                zt_ops.zentorch_addmm_1dbias,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            KeywordArg("size"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def addmm_view_1dbias_gelu_tanh_replacement(
    match, inp, mat_1, mat_2, *, beta, alpha, approximate, size
):
    def repl(inp, mat_1, mat_2, beta, alpha, approximate, size):
        counters["zentorch"]["zentorch_addmm_1dbias_gelu_tanh"] += 1
        counters["zentorch"]["gelu_tanh_fusion"] += 1
        return zt_ops.zentorch_addmm_1dbias_gelu_tanh(
            inp, mat_1, mat_2, beta=beta, alpha=alpha
        ).view(size)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha, approximate, size])


def zentorch_eltwise_unary_fusions(gm):
    """
    zentorch_op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zentorch implementation of respective op fusions
    """
    logger.info("Fusing the zentorch unary elementwise ops in fx graph.")
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="zentorch_eltwise_unary_fusions",
    )
    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    gm.graph.lint()
    gm.recompile()
    return gm
