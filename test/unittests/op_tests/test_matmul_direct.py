# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    skip_test_pt_2_0,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Matmul_Direct(Zentorch_TestCase):
    def setUp(self):
        super().setUp()
        self.previous_value = os.environ.get("USE_ZENDNN_MATMUL_DIRECT")
        os.environ["USE_ZENDNN_MATMUL_DIRECT"] = "1"

    def tearDown(self):
        super().tearDown()
        if self.previous_value:
            os.environ["USE_ZENDNN_MATMUL_DIRECT"] = self.previous_value
        else:
            os.environ.pop("USE_ZENDNN_MATMUL_DIRECT")

    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_bmm(self):
        self.assertEqual(
            torch.bmm(self.data.x3d, self.data.y3d),
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d),
            atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    run_tests()
