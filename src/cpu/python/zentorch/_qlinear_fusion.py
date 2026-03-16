# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    Arg,
    Match,
    stable_topological_sort,
)
from ._utils import counters

matcher_pass = PatternMatcherPass(pass_name="qlinear_fusion_pass")
aten = torch.ops.aten
zentorch = torch.ops.zentorch
torch_decomp = torch.ops.quantized_decomposed

qlinear_args = [Arg() for _ in range(10)]


# Replacement implementation
def _qlinear_q_dq_mul_add_replacement_impl(
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
    output_dtype,
):
    output = zentorch.zentorch_qlinear_mul_add(
        input,
        weight,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        mul_input,
        add_input,
        bias,
        None,
        None,
        output_dtype,
    )
    return (output,)


# TODO
# As soon as the performance issue for bfloat16 is fixed, this check will be removed.
def is_mul_add_fp32(match: Match, mul_input_idx: int, add_input_idx: int) -> bool:
    mul_input = match.args[mul_input_idx]
    add_input = match.args[add_input_idx]
    return mul_input.meta["val"].dtype == torch.float32 and add_input.meta["val"].dtype == torch.float32


# Pattern 1
#
# (mul_input) (QLinear)
#       \     /
#        (mul) (add_input)
#           \     /
#            (add)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        CallFunction(  # Mul
            aten.mul.Tensor,
            Arg(),  # Mul input
            CallFunction(  # QLinear
                zentorch.zentorch_qlinear.default,
                *qlinear_args,
            ),
        ),
        Arg(),  # Add input
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=0, add_input_idx=11)
)
def qlinear_mul_add_pattern_1(
    match: Match,
    mul_input,
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
    add_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
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
            output_dtype,
        ],
    )


# Pattern 2
#
#     (mul_input) (QLinear)
#           \     /
# (add_input)(mul)
#       \     /
#        (add)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        Arg(),  # Add input
        CallFunction(  # Mul
            aten.mul.Tensor,
            Arg(),  # Mul input
            CallFunction(  # QLinear
                zentorch.zentorch_qlinear.default,
                *qlinear_args,
            ),
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=1, add_input_idx=0)
)
def qlinear_mul_add_pattern_2(
    match: Match,
    add_input,
    mul_input,
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
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
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
            output_dtype,
        ],
    )


# Pattern 3
#
# (QLinear)(mul_input)
#        \     /
#         (mul) (add_input)
#             \     /
#              (add)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        CallFunction(  # Mul
            aten.mul.Tensor,
            CallFunction(  # QLinear
                zentorch.zentorch_qlinear.default,
                *qlinear_args,
            ),
            Arg(),  # Mul input
        ),
        Arg(),  # Add input
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=10, add_input_idx=11)
)
def qlinear_mul_add_pattern_3(
    match: Match,
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
    mul_input,
    add_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
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
            output_dtype,
        ],
    )


# Pattern 4
#
#    (QLinear)(mul_input)
#           \     /
# (add_input)(mul)
#       \     /
#        (add)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        Arg(),  # Add input
        CallFunction(  # Mul
            aten.mul.Tensor,
            CallFunction(  # QLinear
                zentorch.zentorch_qlinear.default,
                *qlinear_args,
            ),
            Arg(),  # Mul input
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=11, add_input_idx=0)
)
def qlinear_mul_add_pattern_4(
    match: Match,
    add_input,
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
    mul_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
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
            output_dtype,
        ],
    )


def qlinear_fusion_pass(graph):
    if config.pattern_matcher:
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="qlinear_fusion_pass",
        )
        assert graph.owning_module is not None, "Graph has no owning module"
        replacements = GraphTransformObserver(
            graph.owning_module, "qlinear_fusion_pass"
        ).apply_graph_pass(matcher_pass.apply)
        if replacements is not None:
            stable_topological_sort(graph)
            graph.lint()

    return graph
