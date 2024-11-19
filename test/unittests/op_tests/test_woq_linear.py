# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    woq_bias_opt,
    woq_dtypes,
    woq_input_dim_opt,
    woq_qzeros_opt,
    group_size_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear(Zentorch_TestCase):
    @parameterized.expand(
        product(
            woq_dtypes, woq_input_dim_opt, woq_bias_opt, woq_qzeros_opt, group_size_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_zentorch_woq_linear_torch_checks(
        self, dtype, woq_input_dim, woq_bias_idx, woq_qzeros_idx, group_size_val
    ):
        self.data.create_data(dtype, group_size_val)

        # compute_dtype check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
        group_size = self.data.woq_qweight.shape[0] + 1
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight,
                self.data.woq_scales,
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # larger group_size
            )
        self.assertTrue(
            f"group_size = {group_size} is greater than input "
            + f"channel size = {self.data.woq_qweight.shape[0]}"
            in str(context.exception)
        )

        # group_size complete divisibilty check
        group_size = self.data.woq_qweight.shape[0] - 1
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight,
                self.data.woq_scales,
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                group_size,  # incompatible group_size
            )
        self.assertTrue(
            f"input channel size = {self.data.woq_qweight.shape[0]} is "
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_qweight.to(torch.int8),  # qweight with unsupported dtype
                self.data.woq_scales,
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
                self.data.woq_qweight,
                self.data.woq_scales.to(
                    torch.bfloat16
                ),  # scales with unsupported dtype
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "only float32 weight_scales "
            "are currently supported" in str(context.exception)
        )

        # contiguous qweight check
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight.t(),  # non-contiguous qweight
                self.data.woq_scales,
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_qweight,
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
                self.data.woq_qweight,
                self.data.woq_scales,
                self.data.woq_qzeros_nonzero,  # non-zero qzeros
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "non-zero weight_zero_point "
            "are not supported yet" in str(context.exception)
        )

        # unsupported scales shape check
        woq_scales_shape = self.data.woq_scales.shape
        woq_packing_ratio = self.data.packing_ratio
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight,
                self.data.woq_scales.view(
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
        woq_qweight_shape = self.data.woq_qweight.shape
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_woq_linear(
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight,
                self.data.woq_scales,
                self.data.woq_qweight.view(
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
                self.data.woq_qweight,
                self.data.woq_scales,
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
                self.data.woq_scales,
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_group_size,
            )
        self.assertTrue(
            "unsupported dims for "
            "qweight and weight_scales" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
