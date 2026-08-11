# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import functools
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    Arg,
    stable_topological_sort,
)
from ._logging import get_logger

from ._utils import counters

logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

pass_pattern = PatternMatcherPass()


qlinear_args = [Arg() for _ in range(10)]


# qlinear-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
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
        ),
    ),
    pass_dict=pass_pattern,
)
def qlinear_relu_replacement(
    match,
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
    def repl(
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
        counters["zentorch"]["zentorch_qlinear_relu"] += 1
        counters["zentorch"]["relu_fusion"] += 1
        return zt_ops.zentorch_qlinear_relu(
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
        )

    match.replace_by_example(
        repl,
        [
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
        ],
    )


# qlinear-sigmoid
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid,
        CallFunction(
            zt_ops.zentorch_qlinear,
            *qlinear_args,
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
        ),
    ),
    pass_dict=pass_pattern,
)
def qlinear_sigmoid_replacement(
    match,
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
    def repl(
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
        counters["zentorch"]["zentorch_qlinear_sigmoid"] += 1
        counters["zentorch"]["sigmoid_fusion"] += 1
        return zt_ops.zentorch_qlinear_sigmoid(
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
        )

    match.replace_by_example(
        repl,
        [
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
        ],
    )


def zentorch_eltwise_unary_fusions(fx_graph):
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
        # fx_graph.owning module should return the GraphModule object that owns the graph
        assert fx_graph.owning_module is not None, "Graph has no owning module"
        GraphTransformObserver(fx_graph.owning_module, "pass_pattern").apply_graph_pass(
            pass_pattern.apply
        )
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph
