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
    skip_test_pt_2_1,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_MM_Silu_Mul(nn.Module):
    def __init__(self, data, bias):
        super(Custom_Model_MM_Silu_Mul, self).__init__()
        self.m = data.m
        self.n = data.n
        self.k = data.k
        self.linear_1 = torch.nn.Linear(self.n, self.k, bias=bias)
        self.linear_2 = torch.nn.Linear(self.n, self.k, bias=bias)
        self.silu = torch.nn.SiLU()

    def forward(self, inp):
        inp_shape = inp.shape
        inp_view = inp.view(inp_shape[0] * inp_shape[1], inp_shape[2])
        inp1_view = inp.view(inp_shape[0] * inp_shape[1], inp_shape[2])
        linear_silu = self.silu(self.linear_1(inp_view))
        linear_silu_view = linear_silu.view(inp_shape[0], self.m, self.k)
        linear = self.linear_2(inp1_view)
        linear_view = linear.view(inp_shape[0], self.m, self.k)

        return linear_silu_view * linear_view


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2"
)
class Test_MM_SiLU_Mul_Model(Zentorch_TestCase):

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_mm_silu_mul_with_bias_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        model = Custom_Model_MM_Silu_Mul(self.data, bias=True)
        model_input = self.data.input.view(1, self.data.m, self.data.n)
        if dtype == "bfloat16" and zentorch._C.is_bf16_supported():
            model = model.bfloat16()
        else:
            self.skipTest(
                "Warning: Skipping Bfloat16 Testcases since they are not "
                + "supported on this hardware"
            )
        model_output = model(model_input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        # autocast subtest
        with self.subTest(dtype="float32"):
            self.assertEqual(
                counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
            )
            with torch.autocast("cpu"):
                _ = compiled_graph(model_input)
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
                )
                counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (model_input),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_mm_silu_mul_without_bias_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        model = Custom_Model_MM_Silu_Mul(self.data, bias=False)
        model_input = self.data.input.view(1, self.data.m, self.data.n)
        if dtype == "bfloat16" and zentorch._C.is_bf16_supported():
            model = model.bfloat16()
        else:
            self.skipTest(
                "Warning: Skipping Bfloat16 Testcases since they are not "
                + "supported on this hardware"
            )
        model_output = model(model_input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        # autocast subtest
        with self.subTest(dtype="float32"):
            self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
            with torch.autocast("cpu"):
                _ = compiled_graph(model_input)
                self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 1)
                counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (model_input),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 1)
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
