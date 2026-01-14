# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import os
import copy
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
    counters,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Linear_Placeholder_Model(nn.Module):

    def __init__(self, k, dtype) -> None:
        super().__init__()
        self.linear_0 = nn.Linear(k, 512, dtype=dtype)
        self.linear_1 = nn.Linear(512, 256, dtype=dtype)
        self.linear_2 = nn.Linear(256, 64, dtype=dtype)

    def forward(self, inputs):
        outputs = self.linear_0(inputs)
        outputs = self.linear_1(outputs)
        outputs_final = self.linear_2(outputs)
        return outputs_final


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Model_AMP(AddmmTestCase):

    def setUp(self):
        super().setUp()
        self._original_zentorch_linear = os.environ.get("ZENTORCH_LINEAR")
        os.environ["ZENTORCH_LINEAR"] = "1"

    def tearDown(self):
        if self._original_zentorch_linear is None:
            os.environ.pop("ZENTORCH_LINEAR", None)
        else:
            os.environ["ZENTORCH_LINEAR"] = self._original_zentorch_linear
        super().tearDown()

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"], freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_linear_model_nd_pattern_with_autocast(self, dtype, freeze_list=freeze_opt):
        model = Custom_Linear_Placeholder_Model(self.data.k, torch.float32)
        input_3d = torch.randn(4, 8, self.data.k, dtype=torch.float32)

        reset_dynamo()
        # Create deepcopy for non-zentorch (native) run
        native_model = copy.deepcopy(model)
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_linear"], 0)
        # Compile inside autocast context so FX graph includes convert_element_type operations
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            native_output = native_model(input_3d)
            compiled_output = test_with_freeze_opt(compiled_graph, (input_3d,), freeze_opt)

        self.assertEqual(counters["zentorch"]["zentorch_linear"], 3)
        self.assertTrue(torch.allclose(native_output, compiled_output))


if __name__ == "__main__":
    run_tests()
