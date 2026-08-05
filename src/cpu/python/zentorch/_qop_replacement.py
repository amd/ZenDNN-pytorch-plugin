# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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
from ._utils import counters, is_valid_fp16


matcher_pass = PatternMatcherPass(pass_name="quantization_replacement_pass")


# Raw graph-mutation rewrites for `dequant -> zentorch_addmm/mm -> ...` chains.
# zentorch_qlinear* needs Tensor per-tensor scale/zp, but PT2E carries them as
# Python scalars. Rather than `replace_by_example` (which synthesised aten.full
# that Inductor fused into per-call `cpp_fused_*_full_*` kernels), we mutate the
# graph directly and lift scalar qparams into `get_attr` 1-element const buffers
# -- no make_fx tracing, so no aten.full/scalar_tensor is emitted.


def _make_real_const_tensor(value, dtype):
    """Real 1-element CPU tensor, bypassing any active FakeTensorMode (a bare
    `torch.tensor` under one yields a data-less FakeTensor that Inductor's
    constant inlining later crashes on)."""
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        return torch.tensor([value], dtype=dtype, device="cpu")


def _resolve_scalar_literal(arg):
    """Best-effort Python literal (int/float/bool) from an FX arg, or None if
    not statically resolvable -- for an `Arg()` that matched a constant node
    (scalar_tensor / full / convert / _to_copy chain) instead of a scalar."""
    if isinstance(arg, (int, float, bool)):
        return arg
    if not isinstance(arg, torch.fx.Node):
        return None

    # A placeholder is a runtime input, not a constant: resolving its meta would
    # bake a per-call value into the graph.
    if arg.op == "placeholder":
        return None

    if arg.op == "get_attr":
        gm = arg.graph.owning_module
        try:
            t = getattr(gm, arg.target)
            if isinstance(t, torch.Tensor) and t.numel() == 1:
                return t.item()
        except Exception:
            return None

    if arg.op == "call_function":
        if arg.target == torch.ops.aten.scalar_tensor.default and len(arg.args) >= 1:
            v = arg.args[0]
            if isinstance(v, (int, float, bool)):
                return v
        if arg.target == torch.ops.aten.full.default and len(arg.args) >= 2:
            size, v = arg.args[0], arg.args[1]
            # Only collapse a statically 1-element full() to a scalar; a
            # multi-element full() is a real tensor qparam, not a scalar.
            if (
                isinstance(v, (int, float, bool))
                and isinstance(size, (list, tuple))
                and all(isinstance(d, int) and d == 1 for d in size)
            ):
                return v
        if (
            arg.target == torch.ops.prims.convert_element_type.default
            and len(arg.args) >= 1
        ):
            return _resolve_scalar_literal(arg.args[0])
        if arg.target == torch.ops.aten._to_copy.default and len(arg.args) >= 1:
            return _resolve_scalar_literal(arg.args[0])

    # Deliberately no arg.meta["val"] fallback: a runtime-computed 1-elem tensor
    # there must not be interned as a compile-time constant.
    return None


def _intern_as_const_buffer(gm, tensor, name_hint):
    """Register `tensor` as a buffer on `gm` under a name unique within the
    module, returning the chosen attribute name."""
    base = f"_zentorch_qop_const_{name_hint}"
    counter = 0
    while hasattr(gm, f"{base}_{counter}"):
        counter += 1
    attr_name = f"{base}_{counter}"
    gm.register_buffer(attr_name, tensor)
    return attr_name


def _try_fakify(real_tensor):
    """Fakify `real_tensor` under the active FakeTensorMode (no-op if none), so a
    freshly-inserted get_attr node's meta["val"] is a FakeTensor -- else later
    re-propagation mixes fake + real args and trips FakeTensor asserts."""
    from torch._inductor.virtualized import V
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake_mode = getattr(V, "fake_mode", None)
    if isinstance(fake_mode, FakeTensorMode):
        return fake_mode.from_tensor(real_tensor, static_shapes=True)
    return real_tensor


def _scalar_to_get_attr_node(graph, value, dtype, name_hint, before_node):
    """Intern `value` as a 1-element const buffer and insert a get_attr node
    (before `before_node`) for it. The buffer stays real (Inductor inlines it);
    the node's meta["val"] is a FakeTensor mirror."""
    gm = graph.owning_module
    const_tensor = _make_real_const_tensor(value, dtype)
    attr_name = _intern_as_const_buffer(gm, const_tensor, name_hint)
    with graph.inserting_before(before_node):
        node = graph.get_attr(attr_name)
    node.meta["val"] = _try_fakify(const_tensor)
    return node


def _ensure_tensor_node(graph, arg, dtype, name_hint, before_node):
    """Return an FX Node for `arg` as a Tensor: scalars (or FX nodes wrapping a
    literal) are interned as 1-element const get_attr buffers; a real-Tensor node
    is returned as-is."""
    if isinstance(arg, (int, float, bool)):
        return _scalar_to_get_attr_node(graph, arg, dtype, name_hint, before_node)
    if isinstance(arg, torch.fx.Node):
        # Reuse an existing matching get_attr rather than interning a duplicate
        # (which would also orphan the original buffer, as buffers aren't erased).
        if arg.op == "get_attr":
            buf = getattr(arg.graph.owning_module, arg.target, None)
            if isinstance(buf, torch.Tensor) and buf.numel() == 1 and buf.dtype == dtype:
                return arg
        lit = _resolve_scalar_literal(arg)
        if lit is not None:
            return _scalar_to_get_attr_node(
                graph, lit, dtype, name_hint, before_node
            )
        return arg
    raise TypeError(
        f"Cannot convert arg of type {type(arg).__name__!r} to an FX Tensor node"
    )


def _convert_to_int32_node(graph, node, before_node):
    """Cast `node` to int32 via prims.convert_element_type before `before_node`
    (per-channel weight zp can arrive int64 but must reach the kernel int32);
    skipped when `node` is already int32."""
    if _get_input_dtype(node) is torch.int32:
        return node
    with graph.inserting_before(before_node):
        out = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(node, torch.int32),
        )
    src_val = node.meta.get("val", None)
    if isinstance(src_val, torch.Tensor):
        out.meta["val"] = src_val.to(torch.int32)
    return out


def _get_input_dtype(input_node):
    """Best-effort dtype of an FX node (prefer meta["tensor_meta"], fall back to
    meta["val"]). Becomes qlinear `output_dtype`; if None the kernel silently
    defaults the output to float32, changing a bf16 path."""
    if isinstance(input_node, torch.fx.Node):
        tensor_meta = input_node.meta.get("tensor_meta", None)
        dtype = getattr(tensor_meta, "dtype", None)
        if dtype is not None:
            return dtype
        val = input_node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return val.dtype
    return None


def _splice_zentorch_qlinear(
    match,
    *,
    input_arg,
    weight_arg,
    bias_arg,
    input_scale_arg,
    input_zp_arg,
    weight_scale_arg,
    weight_zp_arg,
    weight_zp_needs_int32_cast=False,
):
    """Replace the matched root with a `zentorch_qlinear.default` call, interning
    scalar qparams as get_attr buffers and casting per-channel weight zp to int32
    when needed. Per-pattern wrappers just bind matched args to these kwargs."""
    graph = match.graph
    root = match.output_node()

    input_scales = _ensure_tensor_node(
        graph, input_scale_arg, torch.float32, "input_scale", root
    )
    input_zps = _ensure_tensor_node(
        graph, input_zp_arg, torch.int32, "input_zp", root
    )
    weight_scales = _ensure_tensor_node(
        graph, weight_scale_arg, torch.float32, "weight_scale", root
    )
    weight_zps = _ensure_tensor_node(
        graph, weight_zp_arg, torch.int32, "weight_zp", root
    )
    if weight_zp_needs_int32_cast:
        weight_zps = _convert_to_int32_node(graph, weight_zps, root)

    # Fall back to the root's dtype when the input carries no meta, else
    # output_dtype=None and the kernel silently defaults the output to float32.
    output_dtype = _get_input_dtype(input_arg) or _get_input_dtype(root)

    with graph.inserting_before(root):
        new_node = graph.call_function(
            torch.ops.zentorch.zentorch_qlinear.default,
            args=(
                input_arg,
                weight_arg,
                input_scales,
                input_zps,
                weight_scales,
                weight_zps,
                bias_arg,
                None,           # output_scales
                None,           # output_zero_points
                output_dtype,
            ),
        )

    # For view-rooted patterns `root` is the outer aten.view; replacing it is
    # correct as zentorch_qlinear is N-D-capable and subsumes the inner/outer
    # reshapes (matching upstream's weight-prepack pass).
    root.replace_all_uses_with(new_node)
    new_node.meta.update(root.meta)
    match.erase_nodes()
    counters["zentorch"]["zentorch_qlinear"] += 1


# XNNPACKQuantizer patterns


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm
        torch.ops.zentorch.zentorch_addmm.default,
        Arg(),  # bias
        CallFunction(  # Input: dequant(quant(x)), tensor-typed qparams
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                Arg(), Arg(), Arg(), -128, 127, torch.int8,
            ),
            Arg(), Arg(), -128, 127, torch.int8,
        ),
        CallFunction(  # Weight: permute(dequant_per_tensor(w))
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                Arg(), Arg(), Arg(), -127, 127, torch.int8,
            ),
            [1, 0],
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_bias_computed_params_replacement_decorated(
    match: Match,
    bias_arg,
    input_arg,
    scale_arg,
    zp_arg,
    same_scale_arg,
    same_zp_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=bias_arg,
        input_scale_arg=scale_arg,
        input_zp_arg=zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
    )


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm
        torch.ops.zentorch.zentorch_addmm.default,
        Arg(),  # bias
        CallFunction(  # Input: dequant(quant(x)), tensor-typed qparams
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                Arg(), Arg(), Arg(), -128, 127, torch.int8,
            ),
            Arg(), Arg(), -128, 127, torch.int8,
        ),
        CallFunction(  # Weight: permute(dequant_per_channel(w))
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(), Arg(), Arg(), Arg(), -127, 127, torch.int8,
            ),
            [1, 0],
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_bias_per_channel_replacement_decorated(
    match: Match,
    bias_arg,
    input_arg,
    scale_arg,
    zp_arg,
    same_scale_arg,
    same_zp_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
    weight_axis_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=bias_arg,
        input_scale_arg=scale_arg,
        input_zp_arg=zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
        weight_zp_needs_int32_cast=True,
    )


def _convert_float64_replacement_impl(arg1):
    return (arg1,)


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        Arg(),
        torch.float64,
    ),
    extra_check=lambda match: {
        match.args[0].meta["tensor_meta"].dtype == torch.float32
    },
    pass_dict=matcher_pass,
)
def convert_float64_replacement_decorated(
    match: Match,
    arg1,
):
    match.replace_by_example(_convert_float64_replacement_impl, [arg1])


def _convert_int64_clamp_max_replacement_impl(
    arg1,
    arg2,
):
    output = torch.ops.aten.clamp_max.default(arg1, arg2)
    return (output,)


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.aten.clamp_max.default,
            Arg(),
            Arg(),
        ),
        torch.int64,
    ),
    pass_dict=matcher_pass,
)
def convert_int64_clamp_max_replacement_decorated(
    match: Match,
    arg1,
    arg2,
):
    match.replace_by_example(_convert_int64_clamp_max_replacement_impl, [arg1, arg2])


# XNNPACKQuantizer patterns end #


# X86InductorQuantizer patterns


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm
        torch.ops.zentorch.zentorch_addmm.default,
        Arg(),  # primals_3
        CallFunction(  # Input 'x' to addmm
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                Arg(),  # primals_1
                Arg(), Arg(), 0, 255, torch.uint8,
            ),
            Arg(), Arg(), 0, 255, torch.uint8,
        ),
        CallFunction(  # Weight to addmm
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(),  # primals_2
                Arg(), Arg(), 0, -128, 127, torch.int8,
            ),
            [1, 0],
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_1dbias_per_tensor_channel_replacement_decorated(
    match: Match,
    bias_arg,
    input_arg,
    input_scale_arg,
    input_zp_arg,
    same_input_scale_arg,
    same_input_zp_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=bias_arg,
        input_scale_arg=input_scale_arg,
        input_zp_arg=input_zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
        weight_zp_needs_int32_cast=True,
    )


@register_graph_pattern(
    CallFunction(  # Root: aten.view output
        torch.ops.aten.view.default,
        CallFunction(  # zentorch_addmm
            torch.ops.zentorch.zentorch_addmm.default,
            Arg(),  # primals_3
            CallFunction(  # Input 'x' to aten.view
                torch.ops.aten.view.default,
                CallFunction(  # Input 'x' to addmm
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    CallFunction(
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        Arg(),  # primals_1
                        Arg(), Arg(), 0, 255, torch.uint8,
                    ),
                    Arg(), Arg(), 0, 255, torch.uint8,
                ),
                Arg(),
            ),
            CallFunction(  # Weight to addmm
                torch.ops.aten.permute.default,
                CallFunction(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default,
                    Arg(),  # primals_2
                    Arg(), Arg(), 0, -128, 127, torch.int8,
                ),
                [1, 0],
            ),
        ),
        Arg(),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_1dbias_view_per_tensor_channel_replacement_decorated(
    match: Match,
    bias_arg,
    input_arg,
    input_scale_arg,
    input_zp_arg,
    same_input_scale_arg,
    same_input_zp_arg,
    aten_view_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
    output_aten_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=bias_arg,
        input_scale_arg=input_scale_arg,
        input_zp_arg=input_zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
        weight_zp_needs_int32_cast=True,
    )


@register_graph_pattern(
    CallFunction(  # Root: zentorch_mm
        torch.ops.zentorch.zentorch_mm.default,
        CallFunction(  # Input 'x' to addmm
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                Arg(),  # primals_1
                Arg(), Arg(), 0, 255, torch.uint8,
            ),
            Arg(), Arg(), 0, 255, torch.uint8,
        ),
        CallFunction(  # Weight to addmm
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(),  # primals_2
                Arg(), Arg(), 0, -128, 127, torch.int8,
            ),
            [1, 0],
        ),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_per_tensor_channel_replacement_decorated(
    match: Match,
    input_arg,
    input_scale_arg,
    input_zp_arg,
    same_input_scale_arg,
    same_input_zp_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=None,
        input_scale_arg=input_scale_arg,
        input_zp_arg=input_zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
        weight_zp_needs_int32_cast=True,
    )


@register_graph_pattern(
    CallFunction(  # Root: aten.view output
        torch.ops.aten.view.default,
        CallFunction(  # zentorch_mm
            torch.ops.zentorch.zentorch_mm.default,
            CallFunction(  # Input 'x' to aten.view
                torch.ops.aten.view.default,
                CallFunction(  # Input 'x' to addmm
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    CallFunction(
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        Arg(),  # primals_1
                        Arg(), Arg(), 0, 255, torch.uint8,
                    ),
                    Arg(), Arg(), 0, 255, torch.uint8,
                ),
                Arg(),
            ),
            CallFunction(  # Weight to addmm
                torch.ops.aten.permute.default,
                CallFunction(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default,
                    Arg(),  # primals_2
                    Arg(), Arg(), 0, -128, 127, torch.int8,
                ),
                [1, 0],
            ),
        ),
        Arg(),
    ),
    pass_dict=matcher_pass,
    extra_check=functools.partial(is_valid_fp16, "zentorch_qlinear"),
)
def qint8_dq_addmm_view_per_tensor_channel_replacement_decorated(
    match: Match,
    input_arg,
    input_scale_arg,
    input_zp_arg,
    same_input_scale_arg,
    same_input_zp_arg,
    aten_view_arg,
    weight_arg,
    weight_scale_arg,
    weight_zp_arg,
    output_aten_arg,
):
    _splice_zentorch_qlinear(
        match,
        input_arg=input_arg,
        weight_arg=weight_arg,
        bias_arg=None,
        input_scale_arg=input_scale_arg,
        input_zp_arg=input_zp_arg,
        weight_scale_arg=weight_scale_arg,
        weight_zp_arg=weight_zp_arg,
        weight_zp_needs_int32_cast=True,
    )


# X86InductorQuantizer patterns end #


def _has_interned_qop_buffers(gm):
    """True if any `_zentorch_qop_const_*` buffer is registered directly on `gm`
    (recurse=False: `_intern_as_const_buffer` always registers on `gm` itself)."""
    return any(
        name.startswith("_zentorch_qop_const_")
        for name, _ in gm.named_buffers(recurse=False)
    )


def _enable_non_fake_inputs_on_active_fake_mode():
    """Set `allow_non_fake_inputs=True` on the active FakeTensorMode (no-op if
    none). Our get_attr nodes hold real buffers; the later FakeTensorProp (and
    qlinear_reorder_optimizations) feed them into the op under FakeTensorMode and
    would otherwise raise -- flipping the flag makes the mode auto-convert them,
    mirroring upstream's `force_allow_non_fake_inputs` paths."""
    from torch._inductor.virtualized import V
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = getattr(V, "fake_mode", None)
    if isinstance(fake_mode, FakeTensorMode):
        fake_mode.allow_non_fake_inputs = True


def replace_with_zentorch_qops(graph):
    if config.pattern_matcher:
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="replace_with_zentorch_qops",
        )

        assert graph.owning_module is not None, "Graph has no owning module"
        replacements = GraphTransformObserver(
            graph.owning_module, "replace_with_zentorch_qops"
        ).apply_graph_pass(matcher_pass.apply)

        if replacements is not None:
            stable_topological_sort(graph)
            graph.lint()
            # Let later FakeTensorProp passes tolerate our real constant
            # buffers (see `_enable_non_fake_inputs_on_active_fake_mode`).
            if _has_interned_qop_buffers(graph.owning_module):
                _enable_non_fake_inputs_on_active_fake_mode()

    return graph
