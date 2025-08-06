# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
)


class Custom_Model_Addmm_Gelu2(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Gelu2, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.gelu2 = nn.GELU()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = self.gelu(add_res)
        addmm_res = torch.addmm(GELU1_res, batch1, batch2)
        GELU2_res = self.gelu2(addmm_res)
        return GELU2_res


class Custom_Model_Addmm_Gelu_Tanh(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Gelu_Tanh, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU_res = nn.functional.gelu(add_res, approximate="tanh")
        return GELU_res


class Custom_Model_Addmm_Gelu_None(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Gelu_None, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU_res = nn.functional.gelu(add_res, approximate="none")
        return GELU_res


class Custom_Model_Addmm_View(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_View, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        add_res.view(-1, 4)
        GELU1_res = self.gelu(add_res)
        return GELU1_res


# The node being cloned will not always be previous node
# While removing clone op from graph we can encounter this scenario
class Custom_Model_Addmm_Diff_User_In_Btw(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Diff_User_In_Btw, self).__init__()

    def forward(self, input, batch1, batch2):
        mm = torch.mm(batch1, batch2)
        cln = torch.clone(input)
        res = torch.add(mm, cln)
        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Gelu_Model(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_gelu_tanh_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_Gelu_Tanh().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_gelu_none_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_Gelu_None().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_gelu_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_Gelu2().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_view_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_View().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_diff_user_in_btw_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_Diff_User_In_Btw().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
