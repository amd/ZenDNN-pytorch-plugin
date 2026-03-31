# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
from typing import Any, Callable

import torch
from torch._inductor import config
from torch._inductor.fx_passes.mkldnn_fusion import (
    _gelu_fusion_1,
    _gelu_fusion_2,
)
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

from ._utils import counters

matcher_pass = PatternMatcherPass(pass_name="woq_linear_unary_fusion_pass")
aten = torch.ops.aten
prims = torch.ops.prims
zentorch = torch.ops.zentorch


# -----------------------------------------------------------------------------
# WOQ linear compute patterns (for matcher only).
# -----------------------------------------------------------------------------
def create_woq_linear_compute_fn(has_bias: bool, users: int = 1) -> CallFunction:
    """WOQ linear compute: 4 args (no bias) or 5 args (with bias).

    ``users`` matches mkldnn unary GELU patterns: decomposed erf-GELU uses 2,
    tanh-GELU uses 4 (same as ``_unary_fusions.fusions_mapper``).
    """
    if has_bias:
        return CallFunction(
            zentorch.zentorch_woq_linear.default,
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            _users=users,
        )
    return CallFunction(
        zentorch.zentorch_woq_linear.default,
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        _users=users,
    )


def create_woq_decomposed_gelu_pattern_with_prims(
    has_bias: bool,
    unary_fusion: Callable[[CallFunction], CallFunction],
    gelu_users: int,
) -> CallFunction:
    """WOQ (bf16) -> f32 -> decomposed GELU -> bf16.

    Matches graphs that cast WOQ output to float32 before the tanh/erf GELU
    decomposition, then cast back to bfloat16 — same structure as
    ``_unary_fusions.create_pattern(..., with_prims=True)`` for linear.
    """
    woq_fn = create_woq_linear_compute_fn(has_bias, users=1)
    convert_ele_fn = CallFunction(
        prims.convert_element_type,
        woq_fn,
        torch.float32,
        _users=gelu_users,
    )
    unary_out = unary_fusion(convert_ele_fn)
    return CallFunction(
        prims.convert_element_type,
        unary_out,
        torch.bfloat16,
        _users=1,
    )


# -----------------------------------------------------------------------------
# Unary fusion helpers: take compute_fn, return CallFunction(unary, compute_fn).
# -----------------------------------------------------------------------------


def _gelu_fusion(compute_fn: CallFunction) -> CallFunction:
    """Non-decomposed GELU: ``aten.gelu(woq_out, approximate=...)`` (same as ``_unary_fusions``)."""
    return CallFunction(
        aten.gelu,
        compute_fn,
        approximate=KeywordArg("approximate"),
    )


# -----------------------------------------------------------------------------
# Pattern builder and extra checks.
# -----------------------------------------------------------------------------

#            Woq_linear
#                |
#      Unary_op (Woq_linear, arg)


def dummy_extra_check(match: Match) -> bool:
    return True


def gelu_erf_check(match: Match) -> bool:
    return (
        match.kwargs.get("approximate") == "none"
        or match.kwargs.get("approximate") is None
    )


def gelu_tanh_check(match: Match) -> bool:
    return match.kwargs.get("approximate") == "tanh"


# -----------------------------------------------------------------------------
# Generic registration: one pattern + optional extra_check -> fused op.
# -----------------------------------------------------------------------------


def _get_woq_linear_unary_op(post_op_name: str) -> Any:
    """Return zentorch.zentorch_woq_linear_{post_op_name}.default."""
    op_name = f"zentorch_woq_linear_{post_op_name}"
    op = getattr(zentorch, op_name, None)
    if op is None:
        raise ValueError(f"Unknown WOQ unary post-op: {post_op_name}")
    return op.default if hasattr(op, "default") else op


def register_woq_patterns(
    post_op_name: str,
    pattern: CallFunction,
    has_bias: bool,
    extra_check: Callable[[Match], bool] = dummy_extra_check,
) -> None:
    """Register one woq_linear+unary pattern and its replacement."""
    zentorch_op = _get_woq_linear_unary_op(post_op_name)
    counter_key = f"zentorch_woq_linear_{post_op_name}"

    if has_bias:

        @register_graph_pattern(
            pattern, pass_dict=matcher_pass, extra_check=extra_check
        )
        def replacement_fn(
            match: Match,
            input: Any,
            weight: Any,
            weight_scale: Any,
            weight_zero_point: Any,
            bias: Any,
            *,
            approximate: Any = None,
        ) -> None:
            def repl(
                input: Any,
                weight: Any,
                weight_scale: Any,
                weight_zero_point: Any,
                bias: Any,
            ) -> torch.Tensor:
                counters["zentorch"][counter_key] += 1
                return zentorch_op(input, weight, weight_scale, weight_zero_point, bias)

            match.replace_by_example(
                repl, [input, weight, weight_scale, weight_zero_point, bias]
            )

    else:

        @register_graph_pattern(
            pattern, pass_dict=matcher_pass, extra_check=extra_check
        )
        def replacement_fn(
            match: Match,
            input: Any,
            weight: Any,
            weight_scale: Any,
            weight_zero_point: Any,
            *,
            approximate: Any = None,
        ) -> None:
            def repl(
                input: Any,
                weight: Any,
                weight_scale: Any,
                weight_zero_point: Any,
            ) -> torch.Tensor:
                counters["zentorch"][counter_key] += 1
                return zentorch_op(input, weight, weight_scale, weight_zero_point)

            match.replace_by_example(
                repl, [input, weight, weight_scale, weight_zero_point]
            )


# Decomposed GELU on WOQ linear: same ``_gelu_fusion_1`` / ``_gelu_fusion_2`` and
# user counts as ``_unary_fusions.fusions_mapper`` (gelu_erf / gelu_tanh).
WOQ_DECOMPOSED_GELU_PATTERNS: tuple[
    tuple[str, Callable[[CallFunction], CallFunction], int], ...
] = (
    ("gelu_erf", _gelu_fusion_1, 2),
    ("gelu_tanh", _gelu_fusion_2, 4),
)


@functools.cache
def register_woq_linear_unary_fusions() -> None:
    """Register WOQ linear + GELU: decomposed and non-decomposed ``aten.gelu``."""
    for has_bias in [False, True]:
        compute_no_decomp = create_woq_linear_compute_fn(has_bias, users=1)
        pattern_no_decomp = _gelu_fusion(compute_no_decomp)
        register_woq_patterns(
            "gelu_erf",
            pattern_no_decomp,
            has_bias,
            extra_check=gelu_erf_check,
        )
        register_woq_patterns(
            "gelu_tanh",
            pattern_no_decomp,
            has_bias,
            extra_check=gelu_tanh_check,
        )

        for post_op_name, fusion_fn, users in WOQ_DECOMPOSED_GELU_PATTERNS:
            compute_fn_decomp = create_woq_linear_compute_fn(has_bias, users=users)
            pattern_decomp = fusion_fn(compute_fn_decomp)
            register_woq_patterns(post_op_name, pattern_decomp, has_bias)
            pattern_prims = create_woq_decomposed_gelu_pattern_with_prims(
                has_bias, fusion_fn, users
            )
            register_woq_patterns(post_op_name, pattern_prims, has_bias)


def woq_linear_unary_post_op_fusions(fx_graph: Graph) -> Graph:
    register_woq_linear_unary_fusions()  # type: ignore[arg-type]
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="woq_linear_unary_post_op_fusions",
    )
    if config.pattern_matcher:
        assert fx_graph.owning_module is not None, "Graph has no owning module"
        GraphTransformObserver(
            fx_graph.owning_module, "woq_linear_unary_post_op_fusions"
        ).apply_graph_pass(matcher_pass.apply)
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph
