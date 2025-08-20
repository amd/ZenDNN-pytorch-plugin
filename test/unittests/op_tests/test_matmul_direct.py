# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
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
    # Removing the custom setUp and tearDown functions as these functions would not have any effect until the problem
    # below is solved

    # TODO
    # Check if there is a better way of running this test case. Currently we are using a static map to read the
    # environment variables once and use the values of those variables throughout the session. Because of this, any
    # environment variables exported in between the run of the test suite by any of the tests will not have any
    # effect as the static map cannot be updated. Hence, we have to find a way to run some tests as stand alone
    # tests in a different session to make sure that the testing is happening as we are expecting it to happen.
    # This is one test case which we would like to run as a stand-alone test and one more is
    # /ZenDNN_PyTorch_Plugin/test/unittests/op_tests/test_env_reader.py

    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_bmm(self):
        self.data.create_unittest_data()
        self.assertEqual(
            torch.bmm(self.data.x3d, self.data.y3d),
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d),
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    run_tests()
