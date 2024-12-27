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
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Parallel_Baddbmm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.baddbmm_0 = torch.baddbmm
        self.baddbmm_1 = torch.baddbmm
        self.baddbmm_2 = torch.baddbmm

    def forward(self, self_tensor, mat1_tensors, mat2_tensor):
        return torch.cat(
            [
                self.baddbmm_0(self_tensor, mat1_tensors[0], mat2_tensor),
                self.baddbmm_1(self_tensor, mat1_tensors[1], mat2_tensor),
                self.baddbmm_2(self_tensor, mat1_tensors[2], mat2_tensor),
            ]
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Parallel_Addmm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.addmm_0 = torch.addmm
        self.addmm_1 = torch.addmm
        self.addmm_2 = torch.addmm

    def forward(self, self_tensor, mat1_tensors, mat2_tensor):
        return torch.cat(
            [
                self.addmm_0(self_tensor, mat1_tensors[0], mat2_tensor),
                self.addmm_1(self_tensor, mat1_tensors[1], mat2_tensor),
                self.addmm_2(self_tensor, mat1_tensors[2], mat2_tensor),
            ]
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Attn_QKV_Fusion_multi_level(nn.Module):
    def __init__(self, dtype):
        super(Custom_Model_Attn_QKV_Fusion_multi_level, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)

        # Perform three separate view operations
        view1 = bmm_output.view(-1, 60)
        view2 = bmm_output.view(-1, 50)
        view3 = bmm_output.view(-1, 60)

        # Perform three separate view operations
        view4 = view1.view(-1, 60)
        view5 = view2.view(-1, 50)
        view6 = view3.view(-1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)

        view4 = linear1_output.view(-1, 50)
        view5 = linear2_output.view(-1, 50)
        view6 = linear3_output.view(-1, 50)
        output = torch.cat(
            (view4, view5, view6),
        )

        return output


@unittest.skipIf(not has_zentorch, "PT PLUGIN is not installed")
class Custom_Model_Attn_QKV_Fusion_multi_user(nn.Module):
    def __init__(self, arg_1, dtype):
        super(Custom_Model_Attn_QKV_Fusion_multi_user, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)
        self.arg1 = arg_1

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)

        # Perform three separate view operations
        view1 = bmm_output.view(self.arg1, 60)
        view2 = bmm_output.view(self.arg1, 50)
        view3 = bmm_output.view(self.arg1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)

        view4 = linear1_output.view(-1, 50)
        view5 = linear2_output.view(-1, 50)
        view6 = linear3_output.view(-1, 50)
        output = torch.cat(
            (view4, view5, view6),
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Attn_QKV_Fusion_multi_mm(nn.Module):
    def __init__(self, dtype):
        super(Custom_Model_Attn_QKV_Fusion_multi_mm, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)
        self.linear4 = nn.Linear(60, 50, dtype=dtype)

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)

        # Perform three separate view operations
        view1 = bmm_output.view(-1, 60)
        view2 = bmm_output.view(-1, 50)
        view3 = bmm_output.view(-1, 60)
        view4 = bmm_output.view(-1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)
        linear4_output = self.linear4(view4)

        view5 = linear1_output.view(-1, 50)
        view6 = linear2_output.view(-1, 50)
        view7 = linear3_output.view(-1, 50)
        view8 = linear4_output.view(-1, 50)

        output = torch.cat(
            (view5, view6, view7, view8),
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Attn_QKV_Fusion(nn.Module):
    def __init__(self, dtype):
        super(Custom_Model_Attn_QKV_Fusion, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)
        # add_output = torch.add(bmm_output, input)

        # Perform three separate view operations
        view1 = bmm_output.view(-1, 60)
        view2 = bmm_output.view(-1, 50)
        view3 = bmm_output.view(-1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)

        view4 = linear1_output.view(-1, 50)
        view5 = linear2_output.view(-1, 50)
        view6 = linear3_output.view(-1, 50)
        output = torch.cat(
            (view4, view5, view6),
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Attn_QKV_Fusion_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_attn_qkv_fusion_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Attn_QKV_Fusion(self.data.get_torch_type(dtype))
        native_output = model(self.data.x2[0], self.data.y2[0])
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.x2[0], self.data.y2[0]),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_attn_qkv_fusion_multi_mm_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Attn_QKV_Fusion_multi_mm(self.data.get_torch_type(dtype))
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["qkv_fusion"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_graph,
            (self.data.x2[0], self.data.y2[0]),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["qkv_fusion"], 1)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_attn_qkv_fusion_multi_user_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
        model = Custom_Model_Attn_QKV_Fusion_multi_user(
            -1, self.data.get_torch_type(dtype)
        )
        for i in range(len(self.data.x2)):
            for j in range(len(self.data.y2)):
                native_output = model(self.data.x2[i], self.data.y2[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_output = test_with_freeze_opt(
                    compiled_graph,
                    (
                        self.data.x2[i],
                        self.data.y2[j]
                    ),
                    freeze_opt
                )
                self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_attn_qkv_fusion_multi_level_model(self, dtype, freeze_opt):

        self.data.create_unittest_data(dtype)
        model = Custom_Model_Attn_QKV_Fusion_multi_level(
            self.data.get_torch_type(dtype)
        )
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["qkv_fusion"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_graph,
            (self.data.x2[0], self.data.y2[0]),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["qkv_fusion"], 1)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    def test_addmm_with_same_params_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Parallel_Addmm()

        self_tensor = self.data.input
        mat1_tensors = [self.data.x, self.data.x * 2, self.data.x * 3]
        mat2_tensor = self.data.y

        native_output = model(self_tensor, mat1_tensors, mat2_tensor)

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self_tensor, mat1_tensors, mat2_tensor),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_baddbmm_with_same_params_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Parallel_Baddbmm()

        self_tensor = self.data.input3d
        mat1_tensors = [self.data.x3d, self.data.x3d * 2, self.data.x3d * 3]
        mat2_tensor = self.data.y3d

        native_output = model(self_tensor, mat1_tensors, mat2_tensor)

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self_tensor, mat1_tensors, mat2_tensor),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
