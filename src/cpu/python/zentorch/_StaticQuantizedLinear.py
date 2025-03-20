# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import torch.nn as nn
import os

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)


# this custom OpContext is created to store weight, weight_scales, weight_zero_points,
# input_scales, input_zero_points, bias, group_size, weight_bits and
# compute_dtype as single parameter for ZenTorchStaticQuantizedLinear module.
# This OpContext helps us in dealing with various issues arising with the
# zentorch.llm.optimize due to ipex's static checks for _op_context parameter
class ZenTorchStaticQuantizedLinearOpContext:
    def __init__(
        self,
        weight,
        weight_scales,
        weight_zero_points,
        weight_bits,
        input_scales,
        input_zero_points,
        input_bits,
        bias=None,
        group_size=None,
        compute_dtype="bfloat16",
        input_symmetric=False,
        weight_symmetric=False,
        enable_weight_prepack=False,
        is_weight_oc_x_ic=True,
    ):
        # TODO : When save and load of quantized model is supported, consider
        # avoiding the dependency between ENV variables during this stage
        zendnn_matmul_algo = os.environ.get(
            "ZENDNN_MATMUL_ALGO", "FP32:4,BF16:4,INT8:4"
        )
        zendnn_weight_caching = os.environ.get("ZENDNN_WEIGHT_CACHING", "1")

        if "INT8:1" in zendnn_matmul_algo or "INT8:2" in zendnn_matmul_algo:
            if enable_weight_prepack and zendnn_weight_caching != "3":
                raise NotImplementedError(
                    "Weight prepacking is not supported when "
                    + "ZENDNN_WEIGHT_CACHING is not set to 3 while ZENDNN_MATMUL_ALGO "
                    + "is set with INT8:1 or INT8:2."
                )

            if not enable_weight_prepack and zendnn_weight_caching == "3":
                raise NotImplementedError(
                    "When ZENDNN_WEIGHT_CACHING is set to 3 while "
                    + "ZENDNN_MATMUL_ALGO is set with INT8:1 or INT8:2, weight "
                    + "prepacking is required."
                )
        # With Quark, the 8-bit quantized LLM models produce zero points
        # with int8/uint8 dtype. Whereas, the 8-bit quantized Linear ops
        # in DLRMv2 generates int32 zero points.
        # Since ZenDNN library is only compatible with int32 format, we
        # convert zero points to compatible data types.
        self.weight_zero_points = (
            None if weight_symmetric else weight_zero_points.to(torch.int32)
        )
        self.input_zero_points = (
            None if input_symmetric else input_zero_points.to(torch.int32)
        )
        if enable_weight_prepack:
            if input_symmetric:
                raise NotImplementedError(
                    "Weight prepacking is not yet supported if "
                    + "input is symmetrically quantized."
                )
            else:
                logger.info("Reordering the weight tensor.")
                self.weight = (
                    torch.ops.zentorch.zentorch_weight_reorder_for_matmul.default(
                        weight,
                        is_weight_oc_x_ic,
                    )
                )
        else:
            self.weight = weight
        self.weight_scales = weight_scales
        self.weight_bits = weight_bits
        self.input_scales = input_scales
        self.bias = bias
        self.group_size = None
        self.input_bits = input_bits
        self.compute_dtype = compute_dtype


# this is a custom ZenTorchStaticQuantizedLinear module to support static Linear
# modules through zentorch optimization and execution flow
class ZenTorchStaticQuantizedLinear(nn.Linear):
    def __init__(
        self,
        mod,
        weight,
        weight_scales,
        weight_zero_points,
        weight_bits,
        input_scales,
        input_zero_points,
        input_bits,
        bias=None,
        group_size=None,
        compute_dtype="bfloat16",
        input_symmetric=False,
        weight_symmetric=False,
        enable_weight_prepack=False,
    ):
        r"""Create a ZenTorchStaticQuantizedLinear module
        from a float module and int8 weight.
        The weight is already in quantized format,
        but the input tensor is quantized at runtime.
        The linear computation occurs with quantized tensors,
        and the dequantized result is achieved through scaling.

        Args:
            mod (Module): a float module provided by the user
            weight (Tensor): weight tensor
            weight_scales (Tensor): weight_scales for weight
            weight_zero_points (Tensor or None): zero points for weight
            weight_bits (int): Number of the weight quantization bits
            input_scales(Tensor): scales for activations
            input_zero_points(Tensor): zeropoints for activattions
            input_bits (int): Number of the input quantization bits
            bias (Tensor or None): bias for linear
            group_size (int): Group size for weight quantization
            compute_dtype (str): Dtype of the module computation
            input_symmetric (bool): True when inputs are symmetrically quantized
            weight_symmetric (bool): True when weight are symmetrically quantized
        """

        float_modules = [torch.nn.Linear]
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert type(mod) in float_modules, (
            "ZenTorchStaticQuantizedLinear only works for one of"
            + str([float_mod.__name__ for float_mod in float_modules])
            + f" or their subclasses, but found {type(mod)}"
        )

        if hasattr(mod, "in_features"):
            self.in_features = mod.in_features
        else:
            self.in_features = mod.weight.size()[1]
        if hasattr(mod, "out_features"):
            self.out_features = mod.out_features
        else:
            self.out_features = mod.weight.size()[0]

        super(ZenTorchStaticQuantizedLinear, self).__init__(
            self.in_features, self.out_features
        )

        if hasattr(self, "bias"):
            del self.bias
        if bias is None:
            bias = mod.bias
        self.bias = False if bias is None else True

        is_weight_oc_x_ic = (
            True
            if self.in_features == weight.size(1)
            and self.out_features == weight.size(0)
            else False
        )

        self._op_context = ZenTorchStaticQuantizedLinearOpContext(
            weight,
            weight_scales,
            weight_zero_points,
            weight_bits,
            input_scales,
            input_zero_points,
            input_bits,
            bias,
            group_size,
            compute_dtype,
            input_symmetric,
            weight_symmetric,
            enable_weight_prepack,
            is_weight_oc_x_ic,
        )
        del (
            weight,
            weight_scales,
            weight_zero_points,
            bias,
            input_scales,
            input_zero_points,
        )

    def _get_name(self):
        return "ZenTorchStaticQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = "in_features={}, out_features={}, dtype={}".format(
            self.in_features, self.out_features, self._op_context.compute_dtype
        )
        extra_repr_str += ", bias={}".format(self.bias)
        extra_repr_str += ", weight_bits={}".format(self._op_context.weight_bits)
        extra_repr_str += ", input_bits={}".format(self._op_context.input_bits)
        extra_repr_str += ", group_size={}".format(self._op_context.group_size)
        return extra_repr_str

    def forward(self, x):
        # zentorch op for qlinear
        linear_out = torch.ops.zentorch.zentorch_qlinear.default(
            x,
            self._op_context.weight,
            self._op_context.bias,
            self._op_context.input_scales,
            self._op_context.input_zero_points,
            self._op_context.weight_scales,
            self._op_context.weight_zero_points,
            output_dtype=x.dtype,
        )
        return linear_out
