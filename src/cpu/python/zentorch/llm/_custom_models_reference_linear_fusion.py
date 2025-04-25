# **********************************************************************************************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/fusions/linear_fusion.py
# **********************************************************************************************************************************************************
import torch
import torch.nn as nn
import copy
from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear

from .._logging import get_logger
from .._WOQLinear import ZenTorchWOQLinear

logger = get_logger(__name__)


class _ZenTorchConcatLinearRef(nn.Module):
    def __init__(self, linear_list: list):
        super().__init__()
        self.num_concat = len(linear_list)
        for i in range(self.num_concat):
            attr_name = f"linear_{i}"
            setattr(self, attr_name, copy.deepcopy(linear_list[i]))
        self.concat_linear = None
        if all(
            not isinstance(linear, WeightOnlyQuantizedLinear) for linear in linear_list
        ):
            if all(isinstance(linear, ZenTorchWOQLinear) for linear in linear_list):
                qweight_list = []
                scales_list = []
                zero_points_list = []
                bias_list = []
                group_size_list = []
                weight_bits_list = []
                compute_dtype_list = []
                dummy_weight_dtype_list = []
                out_features = 0
                for i in range(self.num_concat):
                    qweight_list.append(linear_list[i]._op_context.qweight)
                    scales_list.append(linear_list[i]._op_context.scales)
                    if linear_list[i]._op_context.bias is not None:
                        bias_list.append(linear_list[i]._op_context.bias)
                    if linear_list[i]._op_context.zero_points is not None:
                        zero_points_list.append(linear_list[i]._op_context.zero_points)

                    group_size_list.append(linear_list[i]._op_context.group_size)
                    weight_bits_list.append(linear_list[i]._op_context.weight_bits)
                    compute_dtype_list.append(linear_list[i]._op_context.compute_dtype)
                    dummy_weight_dtype_list.append(linear_list[i].weight.dtype)

                    out_features = out_features + linear_list[i].out_features

                if zero_points_list != [] and (len(qweight_list) != len(zero_points_list)):
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because the "
                        "number of qweight and zero_points does "
                        "not match with each other, "
                        "but num(qweight)=%s"
                        "and num(zero_points)=%s", len(qweight_list), len(zero_points_list)
                    )
                    return
                if bias_list != [] and (len(qweight_list) != len(bias_list)):
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because the "
                        "number of qweight and bias does "
                        "not match with each other,"
                        "but num(qweight)=%s"
                        "and num(bias)=%s", len(qweight_list), len(bias_list)
                    )
                    return

                if len(set(group_size_list)) != 1:
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because "
                        "group_size of all woq qkv layers is not equal"
                    )
                    return
                new_group_size = group_size_list[0]

                if len(set(weight_bits_list)) != 1:
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because "
                        "weight_bits of all woq qkv layers is not equal"
                    )
                    return
                new_weight_bits = weight_bits_list[0]

                if len(set(compute_dtype_list)) != 1:
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because "
                        "compute_dtype of all woq qkv layers is not equal"
                    )
                    return
                new_compute_dtype = compute_dtype_list[0]

                if len(set(dummy_weight_dtype_list)) != 1:
                    logger.warning(
                        "ZenTorch woq QKV fusion is not possible because "
                        "dummy_weight_dtype of all woq qkv layers is not equal"
                    )
                    return
                new_dummy_weight_dtype = dummy_weight_dtype_list[0]

                concat_qweight = torch.concat(qweight_list, 1)
                concat_scales = torch.concat(scales_list, 1)
                concat_bias = torch.concat(bias_list, 0) if bias_list != [] else None
                concat_zero_points = (
                    torch.concat(zero_points_list, 1)
                    if zero_points_list != []
                    else None
                )

                float_module = nn.Linear(
                    linear_list[0].in_features,
                    out_features,
                    bias=True if concat_bias is not None else False,
                )
                self.concat_linear = ZenTorchWOQLinear(
                    float_module,
                    concat_qweight,
                    concat_scales,
                    concat_zero_points,
                    concat_bias,
                    new_group_size,
                    new_weight_bits,
                    new_compute_dtype,
                    new_dummy_weight_dtype,
                )
            elif any(isinstance(linear, ZenTorchWOQLinear) for linear in linear_list):
                logger.warning(
                    "QKV fusion is not possible as qkv modules are mix of "
                    "ZenTorchWOQLinear and other Linear modules"
                )
                return
            else:
                weights_list = []
                bias_list = []
                for i in range(self.num_concat):
                    weights_list.append(linear_list[i].weight)
                    if linear_list[i].bias is not None:
                        bias_list.append(linear_list[i].bias)
                concat_weight = torch.concat(weights_list, 0)

                # this additional check is added for zentorch
                if bias_list != [] and (len(weights_list) != len(bias_list)):
                    logger.warning(
                        "QKV fusion is not possible because the "
                        "number of weight and bias does "
                        "not match with each other,"
                        "but num(weight)=%s"
                        "and num(bias)=%s", len(weights_list), len(bias_list)
                    )
                    return

                use_bias = True if bias_list != [] else False
                concat_bias = torch.concat(bias_list, 0) if use_bias else None
                self.concat_linear = nn.Linear(
                    concat_weight.shape[1], concat_weight.shape[0], bias=use_bias
                )
                self.concat_linear.weight = nn.Parameter(concat_weight)
                self.concat_linear.bias = (
                    nn.Parameter(concat_bias) if use_bias else None
                )

    def forward(self, x):
        output_list = []
        for i in range(self.num_concat):
            assert hasattr(self, f"linear_{i}")
            linear = getattr(self, f"linear_{i}")
            y = linear(x)
            output_list.append(y)
        return tuple(output_list)

    def extra_repr(self):
        return f"num_concat = {self.num_concat}"
