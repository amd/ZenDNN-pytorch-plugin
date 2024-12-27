# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from itertools import product
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_Baddbmm(nn.Module):
    def __init__(self):
        super(Custom_Model_Baddbmm, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        baddbmm_res = torch.baddbmm(add_res, batch1, batch2, beta=1.5, alpha=1.4)
        return baddbmm_res


class Custom_Model_Baddbmm_Unsupport(nn.Module):
    def __init__(self):
        super(Custom_Model_Baddbmm_Unsupport, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        return add_res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Baddbmm_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_baddbmm_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Baddbmm().eval()
        for i in range(len(self.data.x2)):
            for j in range(len(self.data.y2)):
                model_output = model(self.data.M2, self.data.x2[i], self.data.y2[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = test_with_freeze_opt(
                    compiled_graph,
                    (self.data.M2, self.data.x2[i], self.data.y2[j]),
                    freeze_opt
                )
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-5, rtol=1e-3
                )

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_baddbmm_unsupport_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Baddbmm_Unsupport().eval()
        model_output = model(self.data.M3, self.data.x2[0], self.data.y2[0])
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.M3, self.data.x2[0], self.data.y2[0]),
            freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
