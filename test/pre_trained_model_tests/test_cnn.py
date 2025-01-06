# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
from itertools import product
from torchvision import models
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pre_trained_model_utils import (  # noqa: 402
    Test_Data,
    TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    reset_dynamo,
    set_seed,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_CNN_Model(TestCase):
    def setUp(self):
        set_seed()

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_resnet18_model(self, dtype, freeze_opt):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")
        data = Test_Data(dtype)
        model = models.__dict__["resnet18"](pretrained=True).eval()
        inductor_model = copy.deepcopy(model)
        zentorch_graph = torch.compile(model, backend="zentorch", dynamic=False)
        zentorch_graph_output = test_with_freeze_opt(
            zentorch_graph,
            (data.input3d),
            freeze_opt
        )
        reset_dynamo()
        inductor_graph = torch.compile(inductor_model, backend="inductor")
        inductor_graph_output = inductor_graph(data.input3d)

        self.assertEqual(inductor_graph_output, zentorch_graph_output)


if __name__ == "__main__":
    run_tests()
