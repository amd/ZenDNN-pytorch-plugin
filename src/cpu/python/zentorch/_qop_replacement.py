# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
import sys
import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    Arg,
    Match,
)
from torch.fx.passes.utils.fuser_utils import legalize_graph


matcher_pass = PatternMatcherPass(pass_name="quantization_replacement_pass")

# XNNPACKQuantizer patterns


def _qint8_dq_addmm_bias_per_tensor_replacement_impl(
    primals_1,  # input_arg
    primals_2,  # weight_arg
    primals_3,  # bias_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    scale_arg_dup,  # same scale
    zp_arg_dup,  # same zero_point
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
):
    weight_scale_tensor = torch.ops.aten.full.default(
        [1], weight_scale, dtype=torch.float32
    )
    weight_zero_point_tensor = torch.ops.aten.full.default(
        [1], weight_zero_point, dtype=torch.int32
    )

    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        primals_3,
        input_scale,
        input_zero_point,
        weight_scales=weight_scale_tensor,
        weight_zero_points=weight_zero_point_tensor,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm_1dbias
        torch.ops.zentorch.zentorch_addmm_1dbias.default,
        Arg(),  # bias
        CallFunction(  # Input dequantize
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            CallFunction(  # Input quantize
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                Arg(),  # input tensor
                Arg(),  # scale_arg,
                Arg(),  # zero_point_arg
                -128, 127, torch.int8
            ),
            Arg(),  # same_scale_arg,
            Arg(),  # same_zero_point_arg
            -128, 127, torch.int8
        ),
        CallFunction(  # Weight permute
            torch.ops.aten.permute.default,
            CallFunction(  # Weight dequantize
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                Arg(),  # weight tensor
                Arg(),  # weight scale
                Arg(),  # weight zero_point
                -127, 127, torch.int8
            ),
            [1, 0]
        )
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_bias_computed_params_replacement_decorated(
    match: Match, bias_arg, input_arg, scale_arg, zp_arg, same_scale_arg,
    same_zp_arg, weight_arg, weight_scale_arg, weight_zp_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_bias_per_tensor_replacement_impl,
        [input_arg, weight_arg, bias_arg, scale_arg, zp_arg,
         same_scale_arg, same_zp_arg, weight_scale_arg, weight_zp_arg]
    )


def _qint8_dq_addmm_bias_per_channel_replacement_impl(
    primals_1,  # input_arg
    primals_2,  # weight_arg
    primals_3,  # bias_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    scale_arg_dup,  # same scale
    zp_arg_dup,  # same zero_point
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
    weight_axis,  # weight_axis_arg
):
    weight_zero_point_converted = torch.ops.prims.convert_element_type.default(
        weight_zero_point, torch.int32
    )

    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        primals_3,
        input_scale,
        input_zero_point,
        weight_scales=weight_scale,
        weight_zero_points=weight_zero_point_converted,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm_1dbias
        torch.ops.zentorch.zentorch_addmm_1dbias.default,
        Arg(),  # bias
        CallFunction(  # Input 'x' to addmm
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                Arg(),  # input tensor
                Arg(),  # scale_arg,
                Arg(),  # zero_point_arg
                -128, 127, torch.int8
            ),
            Arg(),  # same scale_arg,
            Arg(),  # same zero_point_arg
            -128, 127, torch.int8
        ),
        CallFunction(  # Weight to addmm
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(),  # weight tensor
                Arg(),  # weight scale
                Arg(),  # weight zero_point
                Arg(),  # axis
                -127, 127, torch.int8
            ),
            [1, 0]
        )
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_bias_per_channel_replacement_decorated(
    match: Match, bias_arg, input_arg, scale_arg, zp_arg, same_scale_arg,
    same_zp_arg, weight_arg, weight_scale_arg, weight_zp_arg, weight_axis_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_bias_per_channel_replacement_impl,
        [input_arg, weight_arg, bias_arg, scale_arg, zp_arg,
         same_scale_arg, same_zp_arg, weight_scale_arg, weight_zp_arg,
         weight_axis_arg]
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
    match: Match, arg1,
):
    match.replace_by_example(
        _convert_float64_replacement_impl,
        [arg1]
    )


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
    match: Match, arg1, arg2,
):
    match.replace_by_example(
        _convert_int64_clamp_max_replacement_impl,
        [arg1, arg2]
    )

# XNNPACKQuantizer patterns end #


# X86InductorQuantizer patterns

def _qint8_dq_addmm_1dbias_per_tensor_channel_replacement_impl(
    primals_1,  # input_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    input_scale_dup,  # same scale
    input_zero_point_dup,  # same zero_point
    primals_2,  # weight_arg
    primals_3,  # bias_arg
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
):
    weight_zero_point_converted = torch.ops.prims.convert_element_type.default(
        weight_zero_point, torch.int32
    )
    input_scales_tensor = torch.ops.aten.full.default(
        [1], input_scale, dtype=torch.float32
    )
    input_zero_points_tensor = torch.ops.aten.full.default(
        [1], input_zero_point, dtype=torch.int32
    )
    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        primals_3,
        input_scales_tensor,
        input_zero_points_tensor,
        weight_scales=weight_scale,
        weight_zero_points=weight_zero_point_converted,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


@register_graph_pattern(
    CallFunction(  # Root: zentorch_addmm_1dbias
        torch.ops.zentorch.zentorch_addmm_1dbias.default,
        Arg(),  # primals_3
        CallFunction(  # Input 'x' to addmm
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                Arg(),  # primals_1
                Arg(),
                Arg(),
                0, 255, torch.uint8
            ),
            Arg(),
            Arg(),
            0, 255, torch.uint8
        ),
        CallFunction(  # Weight to addmm
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(),  # primals_2
                Arg(),
                Arg(),
                0, -128, 127, torch.int8
            ),
            [1, 0]
        )
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_1dbias_per_tensor_channel_replacement_decorated(
    match: Match, bias_arg, input_arg, input_scale_arg, input_zp_arg,
    same_input_scale_arg, same_input_zp_arg, weight_arg, weight_scale_arg, weight_zp_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_1dbias_per_tensor_channel_replacement_impl,
        [input_arg, input_scale_arg, input_zp_arg, same_input_scale_arg,
         same_input_zp_arg, weight_arg, bias_arg, weight_scale_arg,
         weight_zp_arg]
    )


def _qint8_dq_addmm_1dbias_view_per_tensor_channel_replacement_impl(
    primals_1,  # input_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    input_scale_dup,  # same scale
    input_zero_point_dup,  # same zero_point
    aten_view,  # aten_view_arg
    primals_2,  # weight_arg
    primals_3,  # bias_arg
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
    output_aten_arg,  # output_aten_arg
):
    weight_zero_point_converted = torch.ops.prims.convert_element_type.default(
        weight_zero_point, torch.int32
    )

    input_scales_tensor = torch.ops.aten.full.default(
        [1], input_scale, dtype=torch.float32
    )
    input_zero_points_tensor = torch.ops.aten.full.default(
        [1], input_zero_point, dtype=torch.int32
    )
    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        primals_3,
        input_scales_tensor,
        input_zero_points_tensor,
        weight_scales=weight_scale,
        weight_zero_points=weight_zero_point_converted,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


@register_graph_pattern(
    CallFunction(  # Root: aten.view output
        torch.ops.aten.view.default,
        CallFunction(  # zentorch_addmm_1dbias
            torch.ops.zentorch.zentorch_addmm_1dbias.default,
            Arg(),  # primals_3
            CallFunction(  # Input 'x' to aten.view
                torch.ops.aten.view.default,
                CallFunction(  # Input 'x' to addmm
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    CallFunction(
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        Arg(),  # primals_1
                        Arg(),
                        Arg(),
                        0, 255, torch.uint8
                    ),
                    Arg(),
                    Arg(),
                    0, 255, torch.uint8
                ),
                Arg(),
            ),
            CallFunction(  # Weight to addmm
                torch.ops.aten.permute.default,
                CallFunction(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default,
                    Arg(),  # primals_2
                    Arg(),
                    Arg(),
                    0, -128, 127, torch.int8
                ),
                [1, 0]
            )
        ),
        Arg(),
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_1dbias_view_per_tensor_channel_replacement_decorated(
    match: Match, bias_arg, input_arg, input_scale_arg, input_zp_arg,
    same_input_scale_arg, same_input_zp_arg, aten_view_arg, weight_arg,
    weight_scale_arg, weight_zp_arg, output_aten_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_1dbias_view_per_tensor_channel_replacement_impl,
        [input_arg, input_scale_arg, input_zp_arg, same_input_scale_arg,
         same_input_zp_arg, aten_view_arg, weight_arg, bias_arg,
         weight_scale_arg, weight_zp_arg, output_aten_arg]
    )


def _qint8_dq_addmm_per_tensor_channel_replacement_impl(
    primals_1,  # input_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    input_scale_dup,  # same scale
    input_zero_point_dup,  # same zero_point
    primals_2,  # weight_arg
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
):
    weight_zero_point_converted = torch.ops.prims.convert_element_type.default(
        weight_zero_point, torch.int32
    )
    input_scales_tensor = torch.ops.aten.full.default(
        [1], input_scale, dtype=torch.float32
    )
    input_zero_points_tensor = torch.ops.aten.full.default(
        [1], input_zero_point, dtype=torch.int32
    )
    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        None,  # No bias in this case
        input_scales_tensor,
        input_zero_points_tensor,
        weight_scales=weight_scale,
        weight_zero_points=weight_zero_point_converted,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


@register_graph_pattern(
    CallFunction(  # Root: zentorch_mm
        torch.ops.zentorch.zentorch_mm.default,
        CallFunction(  # Input 'x' to addmm
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            CallFunction(
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                Arg(),  # primals_1
                Arg(),
                Arg(),
                0, 255, torch.uint8
            ),
            Arg(),
            Arg(),
            0, 255, torch.uint8
        ),
        CallFunction(  # Weight to addmm
            torch.ops.aten.permute.default,
            CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                Arg(),  # primals_2
                Arg(),
                Arg(),
                0, -128, 127, torch.int8
            ),
            [1, 0]
        )
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_per_tensor_channel_replacement_decorated(
    match: Match, input_arg, input_scale_arg, input_zp_arg,
    same_input_scale_arg, same_input_zp_arg, weight_arg, weight_scale_arg,
    weight_zp_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_per_tensor_channel_replacement_impl,
        [input_arg, input_scale_arg, input_zp_arg, same_input_scale_arg,
         same_input_zp_arg, weight_arg, weight_scale_arg, weight_zp_arg]
    )


def _qint8_dq_addmm_view_per_tensor_channel_replacement_impl(
    primals_1,  # input_arg
    input_scale,  # input_scale_arg
    input_zero_point,  # input_zp_arg
    input_scale_dup,  # same scale
    input_zero_point_dup,  # same zero_point
    aten_view,  # aten_view_arg
    primals_2,  # weight_arg
    weight_scale,  # weight_scale_arg
    weight_zero_point,  # weight_zp_arg
    output_aten_arg,  # output_aten_arg
):
    weight_zero_point_converted = torch.ops.prims.convert_element_type.default(
        weight_zero_point, torch.int32
    )

    input_scales_tensor = torch.ops.aten.full.default(
        [1], input_scale, dtype=torch.float32
    )
    input_zero_points_tensor = torch.ops.aten.full.default(
        [1], input_zero_point, dtype=torch.int32
    )
    output = torch.ops.zentorch.zentorch_qlinear.default(
        primals_1,
        primals_2,
        None,  # No bias in this case
        input_scales_tensor,
        input_zero_points_tensor,
        weight_scales=weight_scale,
        weight_zero_points=weight_zero_point_converted,
        output_dtype=primals_1.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


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
                        Arg(),
                        Arg(),
                        0, 255, torch.uint8
                    ),
                    Arg(),
                    Arg(),
                    0, 255, torch.uint8
                ),
                Arg(),
            ),
            CallFunction(  # Weight to addmm
                torch.ops.aten.permute.default,
                CallFunction(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default,
                    Arg(),  # primals_2
                    Arg(),
                    Arg(),
                    0, -128, 127, torch.int8
                ),
                [1, 0]
            )
        ),
        Arg(),
    ),
    pass_dict=matcher_pass,
)
def qint8_dq_addmm_view_per_tensor_channel_replacement_decorated(
    match: Match, input_arg, input_scale_arg, input_zp_arg,
    same_input_scale_arg, same_input_zp_arg, aten_view_arg, weight_arg,
    weight_scale_arg, weight_zp_arg, output_aten_arg
):
    match.replace_by_example(
        _qint8_dq_addmm_view_per_tensor_channel_replacement_impl,
        [input_arg, input_scale_arg, input_zp_arg, same_input_scale_arg,
         same_input_zp_arg, aten_view_arg, weight_arg,
         weight_scale_arg, weight_zp_arg, output_aten_arg]
    )

# X86InductorQuantizer patterns end #


def replace_with_zentorch_qops(gm):
    if config.pattern_matcher:
        # Quark Quantizer patterns
        # Pattern needs to be registered here to address being missed if Quark
        # is imported after Zentorch.

        if "quark" in sys.modules:
            from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize  # noqa: F401

            def _quark_scaled_fake_quantize_uint8_replacement_impl(
                bias,  # bias_arg
                input_tensor,  # input_tensor_arg
                input_scale,  # input_scale_arg
                input_zero_point,  # input_zp_arg
                weight_tensor,  # weight_tensor_arg
                weight_scale,  # weight_scale_arg
                weight_zero_point,  # weight_zp_arg
            ):
                quantized_weight_tensor = torch.ops.quantized_decomposed.quantize_per_channel.default(
                    weight_tensor,
                    weight_scale,
                    weight_zero_point,
                    0, -128, 127, torch.int8,
                )
                output = torch.ops.zentorch.zentorch_qlinear.default(
                    input_tensor,
                    quantized_weight_tensor,  # Use the newly quantized weight
                    bias,
                    input_scale,
                    input_zero_point,
                    weight_scales=weight_scale,
                    weight_zero_points=weight_zero_point,
                    output_dtype=input_tensor.dtype,
                    output_scales=None,
                    output_zero_points=None,
                )
                return (output,)

            @register_graph_pattern(
                CallFunction(
                    torch.ops.zentorch.zentorch_addmm_1dbias.default,
                    Arg(),
                    CallFunction(
                        torch.ops.quark.scaled_fake_quantize.default,
                        "uint8",
                        Arg(),
                        Arg(),
                        Arg(),
                        1,
                        1,
                        0.0,
                        255.0,
                        8,
                        "per_tensor",
                        "None"
                    ),
                    CallFunction(
                        torch.ops.aten.permute.default,
                        CallFunction(
                            torch.ops.quark.scaled_fake_quantize.default,
                            "int8",
                            Arg(),
                            Arg(),
                            Arg(),
                            0,
                            1,
                            -128.0,
                            127.0,
                            8,
                            "per_channel",
                            "None"
                        ),
                        [1, 0]
                    )
                ),
                pass_dict=matcher_pass,
            )
            def quark_scaled_fake_quantize_uint8_replacement_decorated(
                match: Match, bias_arg, input_tensor_arg, input_scale_arg,
                input_zp_arg, weight_tensor_arg, weight_scale_arg,
                weight_zp_arg
            ):
                match.replace_by_example(
                    _quark_scaled_fake_quantize_uint8_replacement_impl,
                    [bias_arg, input_tensor_arg, input_scale_arg, input_zp_arg,
                        weight_tensor_arg, weight_scale_arg, weight_zp_arg]
                )
        # Quark Quantizer patterns end #

        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="replace_with_zentorch_qops",
        )

        replacements = GraphTransformObserver(
            gm,
            "replace_with_zentorch_qops"
        ).apply_graph_pass(matcher_pass.apply)

        if replacements > 0:
            legalize_graph(gm)
            gm.graph.lint()
            gm.recompile()

    return gm
