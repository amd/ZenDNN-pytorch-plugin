# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
from typing import Any, Callable, Optional

import torch
from ._utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from torch.fx.graph import Graph


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten
zentorch = torch.ops.zentorch


# linear replacement
# aten.linear is present in torch.export path
@register_graph_pattern(
    CallFunction(
        aten.linear,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement(match: Match, mat_1: Any, mat_2: Any, bias: Any) -> None:
    def repl(mat_1: Any, mat_2: Any, bias: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_linear"] += 1
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=False,
            post_op="none",
            zentorch_op_name="zentorch::zentorch_linear",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias])


@register_graph_pattern(
    CallFunction(
        aten.linear,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement_no_bias(match: Match, mat_1: Any, mat_2: Any) -> None:
    def repl(mat_1: Any, mat_2: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_linear"] += 1
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2,
            is_weight_prepacked=False,
            post_op="none",
            zentorch_op_name="zentorch::zentorch_linear",
        )

    match.replace_by_example(repl, [mat_1, mat_2])


def is_bias_1d_tensor(match: Match) -> bool:
    # returns true if bias tensor is 1d
    return match.args[0].meta["val"].ndim == 1


def check_alpha_beta_bias(match: Match) -> bool:
    # check bias, beta and alpha has desired values or not
    if match.kwargs["beta"] == 1.0 and match.kwargs["alpha"] == 1.0:
        return is_bias_1d_tensor(match)
    return False


# check for weight and bias as placeholder
def is_placeholder(
    weight_idx: int, bias_idx: Optional[int] = None
) -> Callable[[Match], bool]:
    def _unwrap_bf16_convert(node):
        # Schema - prims::convert_element_type(Tensor a, ScalarType dtype) -> Tensor
        # %convert_element_type_843 : [num_users=1] =
        #   call_function[target=torch.ops.prims.convert_element_type.default]
        #   (args = (%getitem_192, torch.bfloat16), kwargs = {})

        # unwrap convert_element_type only if dtype is bf16
        if node.target != torch.ops.prims.convert_element_type.default:
            return node
        if node.args[1] != torch.bfloat16:
            return node
        return node.args[0]

    def fn(match: Match) -> bool:
        # get_attr is a corner case in export path
        weight_node = _unwrap_bf16_convert(match.args[weight_idx])
        if weight_node.op not in ("placeholder", "get_attr"):
            return False
        if bias_idx is not None:
            bias_node = _unwrap_bf16_convert(match.args[bias_idx])
            if bias_node.op not in ("placeholder", "get_attr"):
                return False
            return check_alpha_beta_bias(match)
        return True

    return fn


@register_graph_pattern(
    CallFunction(
        aten.mm,
        Arg(),
        CallFunction(aten.permute, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(1),  # weight_idx = 1, bias = None
)
def mm_linear_replacement_2d(match: Match, mat_1: Any, mat_2: Any, dims: Any) -> None:
    def repl(mat_1: Any, mat_2: Any, dims: Any) -> torch.Tensor:
        counters["zentorch"]["zentorch_linear"] += 1
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2,
            is_weight_prepacked=False,
            post_op="none",
            zentorch_op_name="zentorch::zentorch_linear",
        )

    match.replace_by_example(repl, [mat_1, mat_2, dims])


@register_graph_pattern(
    CallFunction(
        aten.view,
        CallFunction(
            aten.mm,
            CallFunction(aten.view, Arg(), Arg()),
            CallFunction(aten.permute, Arg(), Arg()),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2),  # weight_idx = 2, bias = None
)
def mm_linear_replacement_nd(
    match: Match, mat_1: Any, size: Any, mat_2: Any, dims: Any, size_1: Any
) -> None:
    def repl(mat_1: Any, size: Any, mat_2: Any, dims: Any, size_1: Any) -> torch.Tensor:
        # for n-d case, we will calculate the output size from available info
        # and insert a view op before zentorch_linear_unary, if needed
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = aten.view(mat_1, exp_inp_shape)
            counters["zentorch"]["zentorch_linear"] += 1
            return zentorch.zentorch_linear_unary(
                view_0,
                mat_2,
                is_weight_prepacked=False,
                post_op="none",
                zentorch_op_name="zentorch::zentorch_linear",
            )
        else:
            counters["zentorch"]["zentorch_linear"] += 1
            return zentorch.zentorch_linear_unary(
                mat_1,
                mat_2,
                is_weight_prepacked=False,
                post_op="none",
                zentorch_op_name="zentorch::zentorch_linear",
            )

    match.replace_by_example(repl, [mat_1, size, mat_2, dims, size_1])


@register_graph_pattern(
    CallFunction(
        aten.addmm,
        Arg(),
        Arg(),
        CallFunction(aten.permute, Arg(), Arg()),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2, 0),  # weight_idx = 2, bias_idx = 0
)
def addmm_linear_replacement_2d(
    match: Match,
    bias: Any,
    mat_1: Any,
    mat_2: Any,
    dims: Any,
    *,
    beta: float,
    alpha: float,
) -> None:
    def repl(
        bias: Any, mat_1: Any, mat_2: Any, dims: Any, beta: float, alpha: float
    ) -> torch.Tensor:
        counters["zentorch"]["zentorch_linear"] += 1
        return zentorch.zentorch_linear_unary(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=False,
            post_op="none",
            zentorch_op_name="zentorch::zentorch_linear",
        )

    match.replace_by_example(repl, [bias, mat_1, mat_2, dims, beta, alpha])


@register_graph_pattern(
    CallFunction(
        aten.view,
        CallFunction(
            aten.addmm,
            Arg(),
            CallFunction(aten.view, Arg(), Arg()),
            CallFunction(aten.permute, Arg(), Arg()),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(3, 0),  # weight_idx = 3, bias_idx = 0
)
def addmm_linear_replacement_nd(
    match: Match,
    bias: Any,
    mat_1: Any,
    size: Any,
    mat_2: Any,
    dims: Any,
    size_1: Any,
    *,
    beta: float,
    alpha: float,
) -> None:
    def repl(
        bias: Any,
        mat_1: Any,
        size: Any,
        mat_2: Any,
        dims: Any,
        size_1: Any,
        beta: float,
        alpha: float,
    ) -> torch.Tensor:
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = aten.view(mat_1, exp_inp_shape)
            counters["zentorch"]["zentorch_linear"] += 1
            return zentorch.zentorch_linear_unary(
                view_0,
                mat_2,
                bias,
                is_weight_prepacked=False,
                post_op="none",
                zentorch_op_name="zentorch::zentorch_linear",
            )
        else:
            counters["zentorch"]["zentorch_linear"] += 1
            return zentorch.zentorch_linear_unary(
                mat_1,
                mat_2,
                bias,
                is_weight_prepacked=False,
                post_op="none",
                zentorch_op_name="zentorch::zentorch_linear",
            )

    match.replace_by_example(
        repl, [bias, mat_1, size, mat_2, dims, size_1, beta, alpha]
    )


def replace_with_zentorch_ops_new(fx_graph: Graph) -> Graph:
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="replace_with_zentorch_ops_new",
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
