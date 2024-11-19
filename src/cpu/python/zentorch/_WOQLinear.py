# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import torch.nn as nn


# this custom OpContext is created to store qweight, scales, zero_points,
# bias, group_size, weight_bits and compute_dtype as single parameter for
# ZenTorchWOQLinear module.
# This OpContext helps us in dealing with various issues arising with the
# zentorch.llm.optimize due to ipex's woq checks for _op_context parameter
class ZenTorchWOQLinearOpContext:
    def __init__(
        self,
        qweight,
        scales,
        zero_points,
        bias,
        group_size,
        weight_bits=4,
        compute_dtype="bfloat16",
    ):
        self.qweight = qweight
        self.scales = scales
        self.zero_points = zero_points
        self.bias = bias
        self.group_size = group_size
        self.weight_bits = weight_bits
        self.compute_dtype = compute_dtype


# this DummyWeight provides us the dummy weight which has dtype
# argument, as ipex checks for linear modules weight dtype and if dtype
# is not available it will throw an error, to bypass this DummyWeight
# provides the dummy dtype which is by deafult set to "None"
class DummyWeight:
    def __init__(self, dtype=None):
        self.dtype = dtype


# this is a custom ZenTorchWOQLinear module to support woq Linear
# modules through zentorch optimization and execution flow
class ZenTorchWOQLinear(nn.Linear):

    def __init__(
        self,
        mod,
        qweight,
        scales,
        zero_points,
        bias,
        group_size,
        weight_bits=4,
        compute_dtype="bfloat16",
        dummy_weight_dtype=None,
    ):
        r"""Create a ZenTorchWOQLinear module from a float module and int4 qweight.
        Weight is dequantized at runtime for computation.

        Args:
            mod (Module): a float module provided by the user
            qweight (Tensor): tensor in int32 dtype but contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor or None): zero points for qweight
            bias (Tensor or None): bias for linear
            group_size (int): Group size for weight quantization
            weight_bits (int): Number of the weight quantization bits
            compute_dtype (str): Dtype of the module computation.
        """
        float_modules = [torch.nn.Linear]
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert type(mod) in float_modules, (
            "ZenTorchWOQLinear only works for one of"
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

        super(ZenTorchWOQLinear, self).__init__(self.in_features, self.out_features)

        if hasattr(self, "bias"):
            del self.bias
        if bias is None:
            bias = mod.bias
        self.bias = False if bias is None else True

        # added to support the zentorch.llm.optimize with ipex >= 2.4.0
        if hasattr(self, "weight"):
            del self.weight
        self.weight = DummyWeight(dummy_weight_dtype)

        self.zentorch_woq = True

        self._op_context = ZenTorchWOQLinearOpContext(
            qweight, scales, zero_points, bias, group_size, weight_bits, compute_dtype
        )
        del qweight, scales, zero_points, bias

    def _get_name(self):
        return "ZenTorchWOQLinear"

    def extra_repr(self):
        extra_repr_str = "in_features={}, out_features={}, dtype={}".format(
            self.in_features, self.out_features, self._op_context.compute_dtype
        )
        extra_repr_str += ", bias={}".format(self.bias)
        extra_repr_str += ", zentorch_woq={}".format(self.zentorch_woq)
        extra_repr_str += ", weight_bits={}".format(self._op_context.weight_bits)
        extra_repr_str += ", group_size={}".format(self._op_context.group_size)
        return extra_repr_str

    def forward(self, x):
        woq_linear = torch.ops.zentorch.zentorch_woq_linear(
            x,
            self._op_context.qweight,
            self._op_context.scales,
            self._op_context.zero_points,
            self._op_context.bias,
            self._op_context.group_size,
            self._op_context.weight_bits,
            self._op_context.compute_dtype,
        )
        return woq_linear
