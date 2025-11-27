# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import os
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
class Custom_Model_Linear(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(Custom_Model_Linear, self).__init__()
        self.mlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.mlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.mlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

    def forward(self, inputs):
        outputs = self.mlp_0(inputs)
        outputs = self.mlp_1(outputs)
        outputs_final = self.mlp_2(outputs)

        return outputs_final


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Model(AddmmTestCase):

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
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_linear_model(self, dtype, freeze_opt):

        model = Custom_Model_Linear(self.data.k, self.data.get_torch_type(dtype))

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_linear"], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph, (self.data.x), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_linear"], 3)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
