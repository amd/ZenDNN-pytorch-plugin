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
    skip_test_pt_2_1,
    freeze_opt,
    test_with_freeze_opt,
)


# mm silu pattern
class Custom_Model_Addmm_Silu_Mul(torch.nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Silu_Mul, self).__init__()

    def forward(self, inp_0, inp_1, inp_2, bias_0):
        view_0 = inp_0.view(inp_0.shape[0] * inp_0.shape[1], inp_0.shape[2])
        mm_0 = torch.ops.zentorch.zentorch_addmm_silu.default(bias_0, view_0, inp_1)
        view_1 = mm_0.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        view_2 = inp_2.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        mul_0 = torch.mul(view_1, view_2)
        return mul_0


# mm silu pattern
class Custom_Model_Addmm_Alpha_Beta_Silu_Mul(torch.nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Alpha_Beta_Silu_Mul, self).__init__()

    def forward(self, inp_0, inp_1, inp_2, bias_0):
        view_0 = inp_0.view(inp_0.shape[0] * inp_0.shape[1], inp_0.shape[2])
        mm_0 = torch.ops.zentorch.zentorch_addmm_silu.default(
            bias_0, view_0, inp_1, alpha=1.3, beta=-3.7
        )
        view_1 = mm_0.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        view_2 = inp_2.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        mul_0 = torch.mul(view_1, view_2)
        return mul_0


class Custom_Model_Gelu_Erf(torch.nn.Module):
    def __init__(self):
        super(Custom_Model_Gelu_Erf, self).__init__()

    def forward(self, input):
        mul_0 = torch.mul(input, 0.5)
        mul_1 = torch.mul(input, 0.7071067811865476)
        erf_0 = torch.erf(mul_1)
        add_0 = torch.add(erf_0, 1)
        mul_2 = torch.mul(mul_0, add_0)
        return mul_2


class Custom_Model_BMM1(nn.Module):
    def __init__(self):
        super(Custom_Model_BMM1, self).__init__()

    def forward(self, arg_0, arg_1):
        exp_0 = arg_0.expand(arg_0.size())
        exp_1 = arg_1.expand(arg_0.size(0), arg_1.size(0), arg_1.size(1))
        bmm_0 = torch.bmm(exp_0, exp_1)
        return bmm_0


class Custom_Model_BMM2(nn.Module):
    def __init__(self):
        super(Custom_Model_BMM2, self).__init__()

    def forward(self, arg_0, arg_1, run_bmm=True):
        # BMM and MM can have numerical differences hence we want
        # to compare mm with mm. For zenorch, run_bmm pattern will be replaced
        # with mm pattern in torch.compile.
        # For torch, we will run else block (with mm)
        if run_bmm:
            exp_0 = arg_0.expand(arg_0.size())
            view_0 = exp_0.view(arg_0.size())
            exp_1 = arg_1.expand(arg_0.size(0), arg_1.size(0), arg_1.size(1))
            bmm_0 = torch.bmm(view_0, exp_1)
            return bmm_0
        else:
            exp_0 = arg_0.squeeze(1)
            arg_1 = arg_1.transpose(0, 1)
            linear_0 = torch.nn.functional.linear(exp_0, arg_1)
            unsqueeze_0 = linear_0.unsqueeze(1)
            return unsqueeze_0


class Custom_Model_BMM3(nn.Module):
    def __init__(self):
        super(Custom_Model_BMM3, self).__init__()

    def forward(self, arg_0, arg_1):
        exp_0 = arg_0.expand(arg_1.size(0), arg_0.size(0), arg_0.size(1))
        exp_1 = arg_1.expand(arg_1.size())
        bmm_0 = torch.bmm(exp_0, exp_1)
        return bmm_0


# add a pattern for mm split
class Custom_Model_MM_Silu(torch.nn.Module):
    def __init__(self, batch_size, seq_len, dim: int = 30):
        super(Custom_Model_MM_Silu, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.linear = torch.nn.Linear(dim, 40, bias=False)
        self.silu = torch.nn.SiLU()

    def forward(self, inp):
        # inp shape should be [bs, sl, dim]
        lin_op = self.linear(inp)
        split_0 = torch.split(lin_op, lin_op.shape[-1] // 2, -1)
        half_0 = split_0[0]
        half_1 = split_0[1]
        silu_0 = self.silu(half_0)
        mul_0 = torch.mul(half_1, silu_0)
        return mul_0


# pattern matcher tests
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2")
class Test_Pattern_Matcher_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_silu_mul_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        decomp_mm_silu_model = Custom_Model_Addmm_Silu_Mul()
        model = decomp_mm_silu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        amp_enabled = True if dtype == "bfloat16" else False
        new_dtype = self.data.get_torch_type(dtype)
        inp_0 = torch.rand((2, 2, 11), dtype=new_dtype)
        inp_1 = torch.rand((11, 53), dtype=new_dtype)
        inp_2 = torch.rand((4, 53), dtype=new_dtype)
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 0)
        with torch.autocast(device_type="cpu", enabled=amp_enabled):
            _ = test_with_freeze_opt(
                compiled_model, (inp_0, inp_1, inp_2, inp_2), freeze_opt
            )
            # test for both dtypes, two separate tests will be run
            self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 1)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_alpha_beta_silu_mul_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        decomp_mm_silu_model = Custom_Model_Addmm_Alpha_Beta_Silu_Mul()
        model = decomp_mm_silu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        amp_enabled = True if dtype == "bfloat16" else False
        new_dtype = self.data.get_torch_type(dtype)
        inp_0 = torch.rand((2, 2, 11), dtype=new_dtype)
        inp_1 = torch.rand((11, 53), dtype=new_dtype)
        inp_2 = torch.rand((4, 53), dtype=new_dtype)
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 0)
        with torch.autocast(device_type="cpu", enabled=amp_enabled):
            _ = test_with_freeze_opt(
                compiled_model, (inp_0, inp_1, inp_2, inp_2), freeze_opt
            )
            self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 1)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_gelu_erf_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        decomp_gelu_model = Custom_Model_Gelu_Erf()
        model = decomp_gelu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        new_dtype = self.data.get_torch_type(dtype)
        inp = torch.empty((4, 11), dtype=new_dtype)
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 0)
        _ = test_with_freeze_opt(compiled_model, (inp), freeze_opt)
        # test for both dtypes, two separate tests will be run
        self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 1)

    @parameterized.expand(freeze_opt)
    @torch.inference_mode()
    def test_gelu_erf_autocast_pattern_model(self, freeze_opt):
        reset_dynamo()
        inp = torch.empty((5, 13))
        decomp_gelu_model = Custom_Model_Gelu_Erf()
        model = decomp_gelu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 0)
        with torch.autocast("cpu"):
            _ = test_with_freeze_opt(compiled_model, (inp), freeze_opt)
            self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 1)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_bmm1_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        custom_expand_model = Custom_Model_BMM1()
        new_dtype = self.data.get_torch_type(dtype)
        # case 1: arg_o.size(1) == 1
        arg_0 = torch.randn((512, 1, 64), dtype=new_dtype)
        arg_1 = torch.randn((64, 32), dtype=new_dtype)
        model_case1 = custom_expand_model.to("cpu").eval()
        model_case2 = copy.deepcopy(model_case1)
        native_output = model_case1(arg_0, arg_1)
        reset_dynamo()
        compiled_model = torch.compile(model_case1, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        zentorch_graph_output = test_with_freeze_opt(
            compiled_model, (arg_0, arg_1), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 1)
        self.assertEqual(native_output, zentorch_graph_output)

        # case 2: arg_0.size(1) != 1
        arg_0 = torch.randn((512, 64, 32), dtype=new_dtype)
        arg_1 = torch.randn((32, 64), dtype=new_dtype)
        native_output = model_case2(arg_0, arg_1)
        reset_dynamo()
        compiled_model = torch.compile(model_case2, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        zentorch_graph_output = test_with_freeze_opt(
            compiled_model, (arg_0, arg_1), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        self.assertEqual(native_output, zentorch_graph_output)

        # Case 3: arg_0 is 2D and arg_1 is 3D
        neg_expand_model = Custom_Model_BMM3()
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.randn((64, 1), dtype=new_dtype)
        arg_1 = torch.randn((64, 1, 64), dtype=new_dtype)
        model_case1 = neg_expand_model.to("cpu").eval()
        native_output = model_case1(arg_0, arg_1)
        reset_dynamo()
        compiled_model = torch.compile(model_case1, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        zentorch_graph_output = test_with_freeze_opt(
            compiled_model, (arg_0, arg_1), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        self.assertEqual(native_output, zentorch_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_bmm2_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        custom_expand_model = Custom_Model_BMM2()
        model = custom_expand_model.eval()
        model_copy_mm = copy.deepcopy(model).eval()

        new_dtype = self.data.get_torch_type(dtype)
        inner_dim = 4096
        arg_0 = torch.randn((512, 1, inner_dim), dtype=new_dtype, requires_grad=False)
        arg_1 = torch.randn((inner_dim, 4096), dtype=new_dtype, requires_grad=False)

        model_copy_mm = torch.compile(model_copy_mm)
        native_output_mm = model_copy_mm(arg_0, arg_1, run_bmm=False)

        reset_dynamo()
        counters.clear()

        compiled_model = torch.compile(model, backend="zentorch")
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        zentorch_graph_output = test_with_freeze_opt(
            compiled_model, (arg_0, arg_1), freeze_opt
        )

        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 1)

        # Range for ALGO 1 (from blis) is 2 * inner_dim * (1e-6)
        self.assertEqual(
            native_output_mm,
            zentorch_graph_output,
            atol=2 * inner_dim * (1e-6),
            rtol=2 * inner_dim * (1e-6),
        )

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_mm_silu_pattern_model(self, dtype, freeze_opt):
        reset_dynamo()
        mm_split_model = Custom_Model_MM_Silu(4, 64, 30)
        model = mm_split_model.to("cpu").eval()
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        new_dtype = self.data.get_torch_type(dtype)
        inp = torch.rand((4, 64, 30), dtype=new_dtype)
        model_op = model(inp)
        reset_dynamo()
        compiled_model = torch.compile(model, backend="zentorch")
        counters.clear()
        # autocast subtest
        with self.subTest(dtype="float32"):
            self.skip_if_bfloat16_path_issue(dtype)
            self.skip_if_bfloat16_unsupported_hardware()
            self.assertEqual(counters["zentorch"]["pattern_matcher_split_mm"], 0)
            with torch.autocast("cpu"):
                _ = test_with_freeze_opt(compiled_model, (inp), freeze_opt)
                self.assertEqual(counters["zentorch"]["pattern_matcher_split_mm"], 1)
                counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_split_mm"], 0)
        compiled_model_op = test_with_freeze_opt(compiled_model, (inp), freeze_opt)
        self.assertEqual(counters["zentorch"]["pattern_matcher_split_mm"], 1)
        self.assertEqual(model_op, compiled_model_op)


if __name__ == "__main__":
    run_tests()
