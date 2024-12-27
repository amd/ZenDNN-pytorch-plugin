# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
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
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_Addmm(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        return add_res


class Custom_Model_Addmm_1D(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_1D, self).__init__()

    def forward(self, input, batch1, batch2):
        mm = torch.mm(input, batch1)
        view = torch.ops.aten.view.default(mm, [1, mm.shape[0], mm.shape[1]])
        add_res = torch.add(view, batch2)
        return add_res


class Custom_Model_Addmm_2D(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_2D, self).__init__()

    def forward(self, input, batch1, batch2):
        mm = torch.mm(input, batch1)
        add_res = torch.add(mm, batch2)
        return add_res


class Custom_Model_Addmm_3D(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_3D, self).__init__()

    def forward(self, input, batch1, batch2):
        mm = torch.mm(input, batch1)
        view = torch.ops.aten.view.default(mm, batch2.size())
        add_res = torch.add(view, batch2)
        return add_res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_optimize_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
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
                    zentorch_graph_output = test_with_freeze_opt(
                        zentorch_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_zero_input_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm().eval()
        for inp in self.data.M:
            model_output = model(inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = test_with_freeze_opt(
                compiled_graph,
                (inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0),
                freeze_opt
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_inf_input_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm().eval()
        for inp in self.data.M:
            model_output = model(inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = test_with_freeze_opt(
                compiled_graph,
                (inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0),
                freeze_opt
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_nan_input_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
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
            zentorch_graph_output = test_with_freeze_opt(
                zentorch_graph,
                (
                    inp * float("nan"),
                    self.data.x1[0] * float("nan"),
                    self.data.y1[0] * float("nan"),
                ),
                freeze_opt
            )
            self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_identity_input_nan_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
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
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (
                torch.eye(self.data.M[0].shape[0], self.data.M[0].shape[1]),
                self.data.x1[0] * float("nan"),
                self.data.y1[0] * float("nan"),
            ),
            freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_variable_add_1D_model(self, dtype):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_1D().eval()
        model_output = model(
            self.data.mm_add_1D[0], self.data.mm_add_1D[1], self.data.mm_add_1D[2]
        )
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_view_add"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.mm_add_1D[0], self.data.mm_add_1D[1], self.data.mm_add_1D[2]
        )
        self.assertEqual(model_output, compiled_graph_output)
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_view_add"], 0)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_variable_add_2D_model(self, dtype):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_2D().eval()
        model_output = model(
            self.data.mm_add_2D[0], self.data.mm_add_2D[1], self.data.mm_add_2D[2]
        )
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.mm_add_2D[0], self.data.mm_add_2D[1], self.data.mm_add_2D[2]
        )
        self.assertEqual(model_output, compiled_graph_output)
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 0)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_variable_add_3D_model(self, dtype):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_3D().eval()
        model_output = model(
            self.data.mm_add_3D[0], self.data.mm_add_3D[1], self.data.mm_add_3D[2]
        )
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.mm_add_3D[0], self.data.mm_add_3D[1], self.data.mm_add_3D[2]
        )
        self.assertEqual(model_output, compiled_graph_output)
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 1)


if __name__ == "__main__":
    run_tests()
