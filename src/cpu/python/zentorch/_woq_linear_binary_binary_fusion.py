# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
from typing import Any, Callable

import torch
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

from ._utils import counters

matcher_pass = PatternMatcherPass(pass_name="woq_linear_binary_binary_fusion_pass")


def create_woq_linear_compute() -> CallFunction:
    """WOQ linear compute pattern with bias (5 args) for binary/binary-binary fusions."""
    return CallFunction(
        zentorch.zentorch_woq_linear.default,
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
    )


def _binary_binary_output_shape_match(
    binary_idx_1: int, binary_idx_2: int
) -> Callable[[Match], bool]:
    """Both operands of the outer add must have the same shape as the output.
    Allows the inner add's operand to be another woq_linear output (we treat it
    as the addend); the fused op just needs tensors of the right shape.
    """

    def fn(match: Match) -> bool:
        out_node = match.output_node()
        if len(out_node.args) != 2:
            return False
        op0 = match.args[binary_idx_1]
        op1 = match.args[binary_idx_2]
        if not isinstance(op0, torch.fx.node.Node) or not isinstance(
            op1, torch.fx.node.Node
        ):
            return False
        out_shape = out_node.meta["val"].shape
        return op0.meta["val"].shape == out_shape and op1.meta["val"].shape == out_shape

    return fn


aten = torch.ops.aten
zentorch = torch.ops.zentorch


# -----------------------------------------------------------------------------
# Binary-binary fusion: pattern generator and registration (add-add, mul-add,
# add-mul, mul-mul). Same four DFS orderings for each (inner_op, outer_op) pair.
# -----------------------------------------------------------------------------

# Supported (inner, outer) post-op pairs. Outer is the return node.
# Only pairs whose zentorch op exists are registered (add add_mul/mul_mul in C++ when ready).
WOQ_BINARY_BINARY_OPS: list[tuple[str, str]] = [
    ("add", "add"),
    ("mul", "add"),
    # ("add", "mul"),
    # ("mul", "mul"),
]

_ATEN_BINARY_OPS: dict[str, Any] = {"add": aten.add.Tensor, "mul": aten.mul}

# All iso-metric patterns are generated for the following pattern:
#
#            Woq_linear
#                |
#      Binary_op (Woq_linear, arg)   <- arg can be another tensor, e.g. output of a second woq_linear
#                |
#      Binary_op (Woq_linear, arg)


def woq_linear_binary_binary_patterns_generator(
    aten_op_outer: Any, aten_op_inner: Any, compute_fn: CallFunction
) -> list[CallFunction]:
    """Same four DFS orderings as _binary_binary_fusions (outer, inner, woq_linear)."""
    return [
        CallFunction(
            aten_op_outer,
            CallFunction(aten_op_inner, compute_fn, Arg()),
            Arg(),
        ),
        CallFunction(
            aten_op_outer,
            CallFunction(aten_op_inner, Arg(), compute_fn),
            Arg(),
        ),
        CallFunction(
            aten_op_outer,
            Arg(),
            CallFunction(aten_op_inner, compute_fn, Arg()),
        ),
        CallFunction(
            aten_op_outer,
            Arg(),
            CallFunction(aten_op_inner, Arg(), compute_fn),
        ),
    ]


# Per-pattern (binary_idx_1, binary_idx_2) into match.args for the two
# "extra" binary operands whose shapes must match the output.
_BINARY_BINARY_EXTRA_CHECK: list[tuple[int, int]] = [
    (5, 6),  # pattern 0: outer(inner(woq(A0..A4), A5), A6)
    (0, 6),  # pattern 1: outer(inner(A0, woq(A1..A5)), A6)
    (6, 0),  # pattern 2: outer(A0, inner(woq(A1..A5), A6))
    (1, 0),  # pattern 3: outer(A0, inner(A1, woq(A2..A6)))
]


def _get_woq_binary_binary_op(inner: str, outer: str) -> Any | None:
    """Return zentorch.zentorch_woq_linear_{inner}_{outer} if it exists."""
    name = f"zentorch_woq_linear_{inner}_{outer}"
    return getattr(zentorch, name, None)


@functools.cache
def register_woq_linear_binary_binary_patterns() -> None:
    """Register binary-binary fusions for all supported (inner, outer) pairs."""
    compute = create_woq_linear_compute()

    for post_op_1, post_op_2 in WOQ_BINARY_BINARY_OPS:
        zentorch_op = _get_woq_binary_binary_op(post_op_1, post_op_2)
        if zentorch_op is None:
            continue

        aten_inner = _ATEN_BINARY_OPS[post_op_1]
        aten_outer = _ATEN_BINARY_OPS[post_op_2]
        patterns = woq_linear_binary_binary_patterns_generator(
            aten_outer, aten_inner, compute
        )
        counter_key = f"zentorch_woq_linear_{post_op_1}_{post_op_2}"

        for idx in range(len(patterns)):
            binary_idx_1, binary_idx_2 = _BINARY_BINARY_EXTRA_CHECK[idx]

            def make_replacement(
                pattern_idx: int,
                op_name_1: str,
                op_name_2: str,
                zop: Any,
                ckey: str,
            ) -> Callable[..., Any]:
                def replacement_fn(
                    match: Match,
                    arg_0: Any,
                    arg_1: Any,
                    arg_2: Any,
                    arg_3: Any,
                    arg_4: Any,
                    add_1: Any,
                    add_2: Any,
                ) -> None:
                    def repl(
                        arg_0: Any,
                        arg_1: Any,
                        arg_2: Any,
                        arg_3: Any,
                        arg_4: Any,
                        add_1: Any,
                        add_2: Any,
                    ) -> torch.Tensor:
                        counters["zentorch"][ckey] += 1
                        if pattern_idx == 0:
                            return zop(arg_0, arg_1, arg_2, arg_3, add_1, add_2, arg_4)
                        elif pattern_idx == 1:
                            return zop(arg_1, arg_2, arg_3, arg_4, arg_0, add_2, add_1)
                        elif pattern_idx == 2:
                            return zop(arg_1, arg_2, arg_3, arg_4, add_2, arg_0, add_1)
                        else:
                            return zop(arg_2, arg_3, arg_4, add_1, arg_1, arg_0, add_2)

                    match.replace_by_example(
                        repl, [arg_0, arg_1, arg_2, arg_3, arg_4, add_1, add_2]
                    )

                replacement_fn.__name__ = (
                    f"woq_linear_{op_name_1}_{op_name_2}_replacement_{pattern_idx}"
                )
                return replacement_fn

            replacement_func = make_replacement(
                idx, post_op_1, post_op_2, zentorch_op, counter_key
            )
            register_graph_pattern(
                patterns[idx],
                pass_dict=matcher_pass,
                extra_check=_binary_binary_output_shape_match(
                    binary_idx_1, binary_idx_2
                ),
            )(replacement_func)


def woq_linear_binary_binary_post_op_fusions(fx_graph: Graph) -> Graph:
    register_woq_linear_binary_binary_patterns()  # type: ignore[arg-type]
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="woq_linear_binary_binary_post_op_fusions",
    )
    if config.pattern_matcher:
        assert fx_graph.owning_module is not None, "Graph has no owning module"
        GraphTransformObserver(
            fx_graph.owning_module, "woq_linear_binary_binary_post_op_fusions"
        ).apply_graph_pass(matcher_pass.apply)
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph
