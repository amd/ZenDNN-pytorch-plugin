# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    WOQTestCase,
    has_zentorch,
    run_tests,
    bias_opt,
    woq_dtypes,
    input_dim_opt,
    woq_qzeros_opt,
    group_size_opt,
    supported_dtypes,
)


def get_weight_tensor(qweight: torch.Tensor, scales: torch.Tensor, group_size: int):
    if group_size == -1:
        group_size = qweight.shape[0] // scales.shape[0]
    else:
        group_size = group_size

    # unpacking
    from op_tests._pack import create_pack_method

    packmethod = create_pack_method("awq", "int4")
    weight_tensor = packmethod.unpack(qweight, False)

    # dequanitzation
    scales = scales.repeat_interleave(group_size, dim=0)
    weight_tensor = weight_tensor * scales.t()

    weight_tensor = weight_tensor.to(torch.bfloat16)
    return weight_tensor


class Zentorch_Simulated_Woq_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Zentorch_Simulated_Woq_Linear, self).__init__()

    def forward(self, input, qweight, scales, bias, group_size):
        weight_tensor = get_weight_tensor(qweight, scales, group_size)
        output = torch.nn.functional.linear(input, weight_tensor, bias)
        return output


# TODO:raise a separate patch to fill gaps in quantization testing
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear(WOQTestCase):
    @WOQTestCase.hypothesis_params_woq_itr(
        woq_dtypes_list=woq_dtypes,
        input_dim_opt_list=input_dim_opt,
        bias_opt_list=bias_opt,
        woq_qzeros_opt_list=woq_qzeros_opt,
        group_size_opt_list=group_size_opt,
        scales_dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_woq_linear_torch_checks(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
    ):

        # compute_dtype check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
                4,
                "float32",  # unsupported compute_dtype
            )
        self.assertTrue(
            "only bfloat16 compute_dtype is currently supported, but the "
            + "compute_dtype received is float32."
            in str(context.exception)
        )

        # weight_bits check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
                8,  # unsupported weight_bits
            )
        self.assertTrue(
            "only int4 woq is currently supported with qweight packed into int32"
            in str(context.exception)
        )

        # larger group_size check
        group_size = self.data.woq_qweight[scales_dtype].shape[0] + 1
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # larger group_size
            )
        self.assertTrue(
            f"group_size = {group_size} is greater than input "
            + f"channel size = {self.data.woq_qweight[scales_dtype].shape[0]}"
            in str(context.exception)
        )

        # group_size complete divisibilty check
        group_size = self.data.woq_qweight[scales_dtype].shape[0] - 1
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # incompatible group_size
            )
        self.assertTrue(
            f"input channel size = {self.data.woq_qweight[scales_dtype].shape[0]} is "
            + f"not completely divisible by group_size = {group_size}"
            in str(context.exception)
        )

        # group_size and tensor mismatch check
        group_size = 1
        if self.data.woq_group_size == 1:
            group_size = -1
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # incompatible group_size
            )
        self.assertTrue(
            "incompatible dimensions/shape for weight_scales "
            + f"with group_size = {group_size}"
            in str(context.exception)
        )

        # group_size check
        group_size = -2
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # incorrect group_size
            )
        self.assertTrue(
            f"group_size = {group_size} is not supported, only "
            + "group_size = -1 or group_size > 0 is currently supported"
            in str(context.exception)
        )

        # input dtype check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim].to(
                    torch.float32
                ),  # input with unsupported dtype
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "only bfloat16 datatype is currently supported" in str(context.exception)
        )

        # qweight dtype check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype].to(
                    torch.int8
                ),  # qweight with unsupported dtype
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "only int4 woq is currently supported "
            "with qweight packed into int32" in str(context.exception)
        )

        # scales dtype check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype].to(
                    torch.float16
                ),  # scales with unsupported dtype
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "only float32 and bfloat16 weight_scales "
            "are currently supported" in str(context.exception)
        )

        # contiguous qweight check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype].t(),  # non-contiguous qweight
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "qweight is non-contiguous & "
            "it is not supported yet" in str(context.exception)
        )

        # unsupported input and qweight check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.input3d,  # input with incompatible shape
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "unsupported sizes for input and qweight" in str(context.exception)
        )

        # unsupported qweight and scales check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.input3d,  # scales with incompatible shape
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "unsupported dims for "
            "qweight and weight_scales" in str(context.exception)
        )

        # unsupported qzeros check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros_nonzero,  # non-zero qzeros
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "non-zero weight_zero_point "
            "are not supported yet" in str(context.exception)
        )

        # unsupported scales shape check
        woq_scales_shape = self.data.woq_scales[scales_dtype].shape
        woq_packing_ratio = self.data.packing_ratio
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype].view(
                    woq_scales_shape[0]
                    * (
                        woq_scales_shape[1]
                        // (woq_scales_shape[1] // woq_packing_ratio)
                    ),
                    woq_scales_shape[1] // woq_packing_ratio,
                ),  # scales with incompatible shape
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "incompatible dimensions/shape for weight_scales" in str(context.exception)
        )

        # unsupported qzero shape check
        woq_qweight_shape = self.data.woq_qweight[scales_dtype].shape
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qweight[scales_dtype].view(
                    woq_qweight_shape[0] // woq_packing_ratio,
                    woq_qweight_shape[1] * woq_packing_ratio,
                ),  # qzero with incompatible shape
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "incompatible dimensions/shape for "
            "weight_zero_point" in str(context.exception)
        )

        # unsupported bias shape check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.input1d,  # bias with incompatible shape
                self.data.woq_group_size,
            )
        self.assertTrue(
            "incompatible dimensions/shape for bias" in str(context.exception)
        )

        # unsupported qweight dim check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.input3d.to(torch.int32),  # qweight with incompatible dims
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "unsupported dims for "
            "qweight and weight_scales" in str(context.exception)
        )

    @WOQTestCase.hypothesis_params_woq_itr(
        woq_dtypes_list=woq_dtypes,
        input_dim_opt_list=input_dim_opt,
        bias_opt_list=bias_opt,
        woq_qzeros_opt_list=woq_qzeros_opt,
        scales_dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_woq_supported_scale_dtype(
        self, dtype, scales_dtype, woq_input_dim, woq_bias_idx, woq_qzeros_idx
    ):
        self.skip_if_bfloat16_unsupported_hardware()
        op = Zentorch_Simulated_Woq_Linear(1, 8).eval()
        simulated_output = op(
            self.data.woq_input[woq_input_dim],
            self.data.woq_qweight[scales_dtype],
            self.data.woq_scales[scales_dtype],
            self.data.woq_bias[woq_bias_idx],
            -1,  # group_size
        )
        output = torch.ops.zentorch.zentorch_woq_linear(
            self.data.woq_input[woq_input_dim],
            self.data.woq_qweight[scales_dtype],
            self.data.woq_scales[scales_dtype],
            self.data.woq_qzeros[woq_qzeros_idx],
            self.data.woq_bias[woq_bias_idx],
            -1,  # group_size
        )
        self.assertEqual(output, simulated_output)


if __name__ == "__main__":
    run_tests()
