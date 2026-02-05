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


# Replacement implementation
def _qlinear_q_dq_mul_add_replacement_impl(
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
):
    output = zentorch.zentorch_qlinear_mul_add(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        mul_input,
        add_input,
        output_dtype=input.dtype,
        output_scales=None,
        output_zero_points=None,
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
                Arg(),  # Input
                Arg(),  # Weight
                Arg(),  # Bias
                Arg(),  # Input scales
                Arg(),  # Input zero points
                Arg(),  # Weight scales
                Arg(),  # Weight zero points
            ),
        ),
        Arg(),  # Add input
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=0, add_input_idx=8)
)
def qlinear_mul_add_pattern_1(
    match: Match,
    mul_input,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    add_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            mul_input,
            add_input,
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
                Arg(),  # Input
                Arg(),  # Weight
                Arg(),  # Bias
                Arg(),  # Input scales
                Arg(),  # Input zero points
                Arg(),  # Weight scales
                Arg(),  # Weight zero points
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
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            mul_input,
            add_input,
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
                Arg(),  # Input
                Arg(),  # Weight
                Arg(),  # Bias
                Arg(),  # Input scales
                Arg(),  # Input zero points
                Arg(),  # Weight scales
                Arg(),  # Weight zero points
            ),
            Arg(),  # Mul input
        ),
        Arg(),  # Add input
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=7, add_input_idx=8)
)
def qlinear_mul_add_pattern_3(
    match: Match,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            mul_input,
            add_input,
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
                Arg(),  # Input
                Arg(),  # Weight
                Arg(),  # Bias
                Arg(),  # Input scales
                Arg(),  # Input zero points
                Arg(),  # Weight scales
                Arg(),  # Weight zero points
            ),
            Arg(),  # Mul input
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_mul_add_fp32, mul_input_idx=8, add_input_idx=0)
)
def qlinear_mul_add_pattern_4(
    match: Match,
    add_input,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
):
    counters["zentorch"]["qlinear_mul_add"] += 1
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            mul_input,
            add_input,
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
