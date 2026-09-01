# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
from typing import Any

import torch
from ._utils import counters, is_valid_fp16
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    Match,
    KeywordArg,
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
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        post_op=KeywordArg("post_op"),
        zentorch_op_name=KeywordArg("zentorch_op_name"),
    ),
    extra_check=functools.partial(is_valid_fp16, "zentorch_weight_prepack"),
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_linear_replacement_without_bias(
    match: Match, mat_1: Any, mat_2: Any, *, is_weight_prepacked: Any, post_op: Any, zentorch_op_name: Any
) -> None:

    def repl(mat_1: Any, mat_2: Any, is_weight_prepacked: Any, post_op: Any, zentorch_op_name: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_linear"] += 1
        mat_2_prepacked = mat_2 if is_weight_prepacked else zentorch.zentorch_weight_prepack_for_linear(mat_2)
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2_prepacked,
            is_weight_prepacked=True,
            post_op=post_op,
            zentorch_op_name=zentorch_op_name,
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked, post_op, zentorch_op_name])


@register_graph_pattern(
    CallFunction(
        zentorch.zentorch_linear_unary,
        Arg(),
        Arg(),
        Arg(),
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        post_op=KeywordArg("post_op"),
        zentorch_op_name=KeywordArg("zentorch_op_name"),
    ),
    extra_check=functools.partial(is_valid_fp16, "zentorch_weight_prepack"),
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_linear_replacement_with_bias(
    match: Match, mat_1: Any, mat_2: Any, bias: Any, *, is_weight_prepacked: Any, post_op: Any, zentorch_op_name: Any
) -> None:

    def repl(mat_1: Any, mat_2: Any, bias: Any, is_weight_prepacked: Any, post_op: Any, zentorch_op_name: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_linear"] += 1
        mat_2_prepacked = mat_2 if is_weight_prepacked else zentorch.zentorch_weight_prepack_for_linear(mat_2)
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2_prepacked,
            bias,
            is_weight_prepacked=True,
            post_op=post_op,
            zentorch_op_name=zentorch_op_name,
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked, post_op, zentorch_op_name])


def qlinear_weight_prepack_check(match: Match) -> bool:
    is_weight_prepacked = any(
        node.kwargs.get("is_weight_prepacked", False) for node in match.nodes
    )
    if is_weight_prepacked:
        return False

    weight_zero_points = match.args[5]
    return weight_zero_points is None


qlinear_args = [Arg() for _ in range(10)]
qlinear_mul_add_args = [Arg() for _ in range(12)]


# zentorch_qlinear replacement with weight prepacking
@register_graph_pattern(
    CallFunction(zentorch.zentorch_qlinear, *qlinear_args),
    extra_check=qlinear_weight_prepack_check,
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_qlinear_replacement(
    match: Match,
    input: Any,
    weight: Any,
    input_scales: Any,
    input_zero_points: Any,
    weight_scales: Any,
    weight_zero_points: Any,
    bias: Any,
    output_scales: Any,
    output_zero_points: Any,
    output_dtype: Any,
) -> None:
    # Resolve from the matched args before tracing: inside `repl`,
    # `input_zero_points` is the traced example value, not the fx arg.
    input_zero_points_defined = input_zero_points is not None

    def repl(
        input: Any,
        weight: Any,
        input_scales: Any,
        input_zero_points: Any,
        weight_scales: Any,
        weight_zero_points: Any,
        bias: Any,
        output_scales: Any,
        output_zero_points: Any,
        output_dtype: Any,
    ) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_dynamic_qlinear"] += 1
        weight_prepacked = zentorch.zentorch_weight_prepack_for_dynamic_qlinear(
            weight, input_zero_points_defined
        )
        return zentorch.zentorch_qlinear(
            input,
            weight_prepacked,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            bias,
            output_scales,
            output_zero_points,
            output_dtype,
            is_weight_prepacked=True,
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


# zentorch_qlinear_mul_add replacement with weight prepacking
@register_graph_pattern(
    CallFunction(zentorch.zentorch_qlinear_mul_add, *qlinear_mul_add_args),
    extra_check=qlinear_weight_prepack_check,
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_qlinear_mul_add_replacement(
    match: Match,
    input: Any,
    weight: Any,
    input_scales: Any,
    input_zero_points: Any,
    weight_scales: Any,
    weight_zero_points: Any,
    mul_input: Any,
    add_input: Any,
    bias: Any,
    output_scales: Any,
    output_zero_points: Any,
    output_dtype: Any,
) -> None:
    # Resolve from the matched args before tracing: inside `repl`,
    # `input_zero_points` is the traced example value, not the fx arg.
    input_zero_points_defined = input_zero_points is not None

    def repl(
        input: Any,
        weight: Any,
        input_scales: Any,
        input_zero_points: Any,
        weight_scales: Any,
        weight_zero_points: Any,
        mul_input: Any,
        add_input: Any,
        bias: Any,
        output_scales: Any,
        output_zero_points: Any,
        output_dtype: Any,
    ) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_dynamic_qlinear"] += 1
        weight_prepacked = zentorch.zentorch_weight_prepack_for_dynamic_qlinear(
            weight, input_zero_points_defined
        )
        return zentorch.zentorch_qlinear_mul_add(
            input,
            weight_prepacked,
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
            is_weight_prepacked=True,
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
            mul_input,
            add_input,
            bias,
            output_scales,
            output_zero_points,
            output_dtype,
        ],
    )


def dynamic_qlinear_weight_prepack_check(match: Match) -> bool:
    is_weight_prepacked = any(
        node.kwargs.get("is_weight_prepacked", False) for node in match.nodes
    )
    if is_weight_prepacked:
        return False

    activation_val = match.args[0].meta["val"]
    weight_val = match.args[1].meta["val"]
    if weight_val.dim() != 2:
        return False
    return weight_val.size(1) == activation_val.size(-1)


@register_graph_pattern(
    CallFunctionVarArgs(zentorch.zentorch_dynamic_qlinear),
    extra_check=dynamic_qlinear_weight_prepack_check,
    pass_dict=pass_pattern,
)
def zentorch_weight_prepack_for_dynamic_qlinear_replacement(
    match: Match, *args: Any, **kwargs: Any
) -> None:
    new_kwargs = {**kwargs, "is_weight_prepacked": True}

    def repl(*args: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_weight_prepack_for_dynamic_qlinear"] += 1
        # zentorch_dynamic_qlinear takes no input zero points; its per-token
        # activation quantization is symmetric, so the source dtype is s8.
        weight_prepacked = zentorch.zentorch_weight_prepack_for_dynamic_qlinear(
            args[1], False
        )
        return zentorch.zentorch_dynamic_qlinear(
            args[0], weight_prepacked, *args[2:], **new_kwargs
        )

    match.replace_by_example(repl, list(args))


def add_zentorch_weight_prepack_ops(fx_graph: Graph) -> Graph:
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="add_zentorch_weight_prepack_ops",
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
