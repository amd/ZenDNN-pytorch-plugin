# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))


def qdq_linear(
    inp,
    weight,
    bias,
    inp_scales,
    inp_zero_points,
    weight_scales,
    weight_zero_points,
    eltwise_op,
    output_dtype,
    output_scales=None,
    output_zero_points=None,
):
    inp_min_val = -128 if inp_zero_points.dtype == torch.int8 else 0
    inp_max_val = 127 if inp_zero_points.dtype == torch.int8 else 255
    weight_min_val = -128 if weight_zero_points.dtype == torch.int8 else 0
    weight_max_val = 127 if weight_zero_points.dtype == torch.int8 else 255
    out_features_axis = 0

    if inp.dtype == torch.float32 or inp.dtype == torch.bfloat16:
        # fake_quantize_per_tensor_affine only supports fp32 inputs
        qdq_inp = torch.fake_quantize_per_tensor_affine(
            inp.to(torch.float32), inp_scales, inp_zero_points, inp_min_val, inp_max_val
        )
    else:
        qdq_inp = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            inp,
            inp_scales,
            inp_zero_points,
            inp_min_val,
            inp_max_val,
            inp.dtype,
        )
    if weight_scales.numel() == 1:
        dq_weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            weight,
            weight_scales,
            weight_zero_points,
            weight_min_val,
            weight_max_val,
            weight.dtype,
        )
    else:
        dq_weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            weight,
            weight_scales,
            weight_zero_points,
            out_features_axis,
            weight_min_val,
            weight_max_val,
            weight.dtype,
        )

    qdq_linear_output = torch.nn.functional.linear(qdq_inp, dq_weight, bias)

    if eltwise_op is not None:
        qdq_linear_output = eltwise_op(qdq_linear_output)

    if output_scales is not None and output_zero_points is not None:
        output_min_val = -128 if output_zero_points.dtype == torch.int8 else 0
        output_max_val = 127 if output_zero_points.dtype == torch.int8 else 255
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            qdq_linear_output,
            output_scales,
            output_zero_points,
            output_min_val,
            output_max_val,
            output_zero_points.dtype,
        ).to(output_dtype)
    else:
        return qdq_linear_output.to(output_dtype)
