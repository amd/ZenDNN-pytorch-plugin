# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from zentorch_test_utils import (  # noqa: 402 # noqa: F401
    BaseZentorchTestCase,
    run_tests,
    zentorch,
    has_zentorch,
    supported_dtypes,
    reset_dynamo,
    set_seed,
    freeze_opt,
    test_with_freeze_opt,
    Test_Data,
)


class Zentorch_TestCase(BaseZentorchTestCase):
    def setUp(self):
        super().setUp()
        self.data = Test_Data()

    def tearDown(self):
        del self.data
