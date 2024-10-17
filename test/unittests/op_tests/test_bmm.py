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
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_BMM_Op(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_bmm_variants(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.bmm(self.data.x3d, self.data.y3d),
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d),
        )

    @parameterized.expand(supported_dtypes)
    def test_bmm_unsupported_dims(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x, self.data.y)

        self.assertTrue("unsupported dims for self and mat2" in str(context.exception))
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x, self.data.x)
        self.assertTrue("unsupported dims for self and mat2" in str(context.exception))

    @parameterized.expand([("int",)])
    def test_bmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
