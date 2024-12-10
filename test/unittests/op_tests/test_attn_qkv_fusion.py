# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Attn_QKV_Fusion(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_silu_mul_with_same_params(self, dtype):
        self.data.create_data(dtype)
        self_tensor = self.data.input
        mul_tensors = [self.data.input, self.data.input * 2, self.data.input * 3]
        mat1_tensors = [self.data.x, self.data.x * 2, self.data.x * 3]
        mat2_tensor = self.data.y

        native_addmm_silu_mul_0 = (
            torch.nn.functional.silu(
                torch.addmm(self_tensor, mat1_tensors[0], mat2_tensor)
            )
            * mul_tensors[0]
        )
        native_addmm_silu_mul_1 = (
            torch.nn.functional.silu(
                torch.addmm(self_tensor, mat1_tensors[1], mat2_tensor)
            )
            * mul_tensors[1]
        )
        native_addmm_silu_mul_2 = (
            torch.nn.functional.silu(
                torch.addmm(self_tensor, mat1_tensors[2], mat2_tensor)
            )
            * mul_tensors[2]
        )

        native_output = torch.cat(
            [native_addmm_silu_mul_0, native_addmm_silu_mul_1, native_addmm_silu_mul_2]
        )

        zentorch_addmm_silu_mul_0 = torch.ops.zentorch.zentorch_addmm_silu_mul(
            self_tensor, mat1_tensors[0], mat2_tensor, mul_tensors[0]
        )
        zentorch_addmm_silu_mul_1 = torch.ops.zentorch.zentorch_addmm_silu_mul(
            self_tensor, mat1_tensors[1], mat2_tensor, mul_tensors[1]
        )
        zentorch_addmm_silu_mul_2 = torch.ops.zentorch.zentorch_addmm_silu_mul(
            self_tensor, mat1_tensors[2], mat2_tensor, mul_tensors[2]
        )

        zentorch_output = torch.cat(
            [
                zentorch_addmm_silu_mul_0,
                zentorch_addmm_silu_mul_1,
                zentorch_addmm_silu_mul_2,
            ]
        )

        self.assertEqual(native_output, zentorch_output)

    @parameterized.expand(supported_dtypes)
    def test_attn_qkv_fusion_unsupported_dims_1(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_qkv_fusion(
                [self.data.x1[0]] * 3,
                [self.data.x1[0]] * 3,
                [self.data.y1[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "unsupported dims for self, mat1 and mat2" in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_attn_qkv_fusion_unsupported_dims_2(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_qkv_fusion(
                [self.data.y2[0]] * 3,
                [self.data.x2[0]] * 3,
                [self.data.y2[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "unsupported dims for self, mat1 and mat2" in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_attn_qkv_fusion_input_shape_compatibility(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_qkv_fusion(
                [self.data.M[1]] * 3,
                [self.data.y1[0]] * 3,
                [self.data.y1[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "Tensor shapes incompatible for matrix multiplication"
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_bf16_alpha_not_1(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.data.create_data("bfloat16")
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.7
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.7
            ),
        )


if __name__ == "__main__":
    run_tests()
