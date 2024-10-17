# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
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
)


class Custom_Model_Addmm(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        return add_res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_optimize_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_Addmm().eval()
        self.skip_if_bfloat16_path_issue(dtype)
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    reset_dynamo()
                    zentorch_model = copy.deepcopy(model)
                    inductor_graph = torch.compile(model, backend="inductor")
                    inductor_graph_output = inductor_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    reset_dynamo()
                    zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
                    zentorch_graph_output = zentorch_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )

                    self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_zero_input_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_Addmm().eval()
        for inp in self.data.M:
            model_output = model(inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_inf_input_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_Addmm().eval()
        for inp in self.data.M:
            model_output = model(inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_nan_input_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_Addmm().eval()
        for inp in self.data.M:
            reset_dynamo()
            zentorch_model = copy.deepcopy(model)
            inductor_graph = torch.compile(model, backend="inductor")
            inductor_graph_output = inductor_graph(
                inp * float("nan"),
                self.data.x1[0] * float("nan"),
                self.data.y1[0] * float("nan"),
            )
            reset_dynamo()
            zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
            zentorch_graph_output = zentorch_graph(
                inp * float("nan"),
                self.data.x1[0] * float("nan"),
                self.data.y1[0] * float("nan"),
            )
            self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_identity_input_nan_model(self, dtype):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        model = Custom_Model_Addmm().eval()
        model_output = model(
            torch.eye(self.data.M[0].shape[0], self.data.M[0].shape[1]),
            self.data.x1[0] * float("nan"),
            self.data.y1[0] * float("nan"),
        )
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            torch.eye(self.data.M[0].shape[0], self.data.M[0].shape[1]),
            self.data.x1[0] * float("nan"),
            self.data.y1[0] * float("nan"),
        )
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
