# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import copy
from parameterized import parameterized
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    zentorch,
    counters,
    reset_dynamo,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Relu_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_relu_model(self, dtype):
        self.data.create_unittest_data(dtype)
        model = nn.Sequential(nn.Linear(self.data.n, self.data.m), nn.ReLU())
        zentorch_model = copy.deepcopy(model)
        if dtype == "bfloat16":
            model = model.bfloat16()
            zentorch_model = zentorch_model.bfloat16()
        model = torch.compile(model, backend="inductor")
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_addmm_1dbias"], 0)
        compiled_model_output = compiled_model(self.data.input)
        self.assertEqual(counters["zentorch"]["zentorch_addmm_1dbias"], 1)
        self.assertEqual(model_output, compiled_model_output)


if __name__ == "__main__":
    run_tests()
