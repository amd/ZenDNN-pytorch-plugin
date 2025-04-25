# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
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
class Test_Addmm_1dbias_Relu_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_relu_model(self, dtype):
        self.data.create_unittest_data(dtype)
        model = nn.Sequential(nn.Linear(self.data.n, self.data.m), nn.ReLU())
        if dtype == "bfloat16":
            model = model.bfloat16()
        fx_g = make_fx(model)(self.data.input)
        fx_g_modified = zentorch.optimize(fx_g)
        fx_g_output = fx_g(self.data.input)
        fx_g_modified_output = fx_g_modified(self.data.input)
        self.assertEqual(fx_g_output, fx_g_modified_output)
        for node in fx_g_modified.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload) and node.target.name() in ["aten::addmm"]:
                self.assertEqual(
                    node.target, torch.ops.zentorch.zentorch_addmm_1dbias
                )


if __name__ == "__main__":
    run_tests()
