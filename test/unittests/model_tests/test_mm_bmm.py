# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
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
    counters
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_MM_BMM(nn.Module):
    def __init__(self, batch_size):
        super(Custom_Model_MM_BMM, self).__init__()
        self.batch_size = batch_size
    # MM followed by BMM: MM output is fed into BMM

    def forward(self, x_2d, y_2d, z_3d):
        mm_res = torch.mm(x_2d, y_2d)
        mm_3d = mm_res.unsqueeze(0).expand(self.batch_size, -1, -1).clone()
        bmm_res = torch.bmm(mm_3d, z_3d)
        return bmm_res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_BMM_Model(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt, time_out=30000
    )
    @torch.inference_mode()
    def test_mm_bmm_model(self, dtype, freeze_opt):

        batch_size = self.data.b
        m = self.data.m
        k = self.data.k
        n = self.data.n
        torch_type = self.data.x.dtype

        x_2d = torch.randn(m, k).type(torch_type)
        y_2d = torch.randn(k, n).type(torch_type)
        z_3d = torch.randn(batch_size, n, m).type(torch_type)

        model = Custom_Model_MM_BMM(batch_size).eval()
        native_output = model(x_2d, y_2d, z_3d)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_mm"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_bmm"], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph, (x_2d, y_2d, z_3d), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_mm"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_bmm"], 1)
        self.assertEqual(native_output, compiled_output, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    run_tests()
