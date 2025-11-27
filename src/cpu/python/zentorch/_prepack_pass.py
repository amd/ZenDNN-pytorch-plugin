# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
from typing import Any

import torch
from ._utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from torch.fx.graph import Graph


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten
zentorch = torch.ops.zentorch


# zentorch_linear_unary replacement with weight prepacking
@register_graph_pattern(
    CallFunction(
        zentorch.zentorch_linear_unary,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_linear_replacement_without_bias(
    match: Match, mat_1: Any, mat_2: Any
) -> None:
    def repl(mat_1: Any, mat_2: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_linear"] += 1
        mat_2_prepacked = zentorch.zentorch_weight_prepack_for_linear(mat_2)
        return zentorch.zentorch_linear_unary(
            mat_1, mat_2_prepacked, is_weight_prepacked=True
        )

    match.replace_by_example(repl, [mat_1, mat_2])


@register_graph_pattern(
    CallFunction(
        zentorch.zentorch_linear_unary,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_linear_replacement_with_bias(
    match: Match, mat_1: Any, mat_2: Any, bias: Any
) -> None:
    def repl(mat_1: Any, mat_2: Any, bias: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_linear"] += 1
        mat_2_prepacked = zentorch.zentorch_weight_prepack_for_linear(mat_2)
        return zentorch.zentorch_linear_unary(
            mat_1, mat_2_prepacked, bias, is_weight_prepacked=True
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias])


def add_zentorch_weight_prepack_ops(fx_graph: Graph) -> Graph:
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="add_zentorch_weight_prepack_ops",
    )

    if config.pattern_matcher:
        GraphTransformObserver(fx_graph, "pass_pattern").apply_gm_pass(
            pass_pattern.apply
        )
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph
