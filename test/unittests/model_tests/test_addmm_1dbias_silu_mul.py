# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
import unittest
import copy
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    reset_dynamo,
    counters,
    run_tests,
    zentorch,
    skip_test_pt_2_1,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_Addmm_1dbias_SiLU_Mul(nn.Module):
    def __init__(self, data, bias):
        super(Custom_Model_Addmm_1dbias_SiLU_Mul, self).__init__()
        self.m = data.m
        self.n = data.n
        self.k = data.k
        self.linear = torch.nn.Linear(self.n, self.k, bias=bias)
        self.silu = torch.nn.SiLU()

    def forward(self, inp, mul_tensor):
        linear_silu = self.silu(self.linear(inp))
        return linear_silu * mul_tensor


class Custom_Model_Addmm_1dbias_Alpha_Beta_SiLU_Mul(nn.Module):
    def __init__(self, bias, weight, mul_tensor):
        super(Custom_Model_Addmm_1dbias_Alpha_Beta_SiLU_Mul, self).__init__()
        self.silu = torch.nn.SiLU()
        self.bias = nn.Parameter(bias)
        self.weight = nn.Parameter(weight)
        self.mul_tensor = nn.Parameter(mul_tensor)

    def forward(self, mat1):
        addmm_silu = self.silu(torch.addmm(self.bias, mat1, self.weight, alpha=1.1, beta=1.8))
        return addmm_silu * self.mul_tensor


@unittest.skipIf(skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2")
class Test_Pattern_Matcher_Test_With_Different_Dtypes_Model(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_float32_addmm_1dbias_silu_float32_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=True)

        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.float32
        )
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        zentorch_model = copy.deepcopy(model)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_float32_addmm_1dbias_alpha_beta_silu_float32_mul_pattern_model(
        self,
        freeze_opt
    ):
        reset_dynamo()

        mat1_tensor = self.data.x
        mat2_tensor = self.data.y
        bias_tensor = self.data.input1d
        mul_tensor = self.data.input

        model = Custom_Model_Addmm_1dbias_Alpha_Beta_SiLU_Mul(bias_tensor, mat2_tensor, mul_tensor)

        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )

        zentorch_model = copy.deepcopy(model)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (mat1_tensor,),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_float32_addmm_1dbias_silu_bfloat16_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=True)
        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.bfloat16
        )
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        zentorch_model = copy.deepcopy(model)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["bfloat16"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_bfloat16_addmm_1dbias_silu_float32_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        self.skip_if_bfloat16_unsupported_hardware()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=True)

        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.float32
        )
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        zentorch_model = copy.deepcopy(model).to(dtype=torch.bfloat16)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["bfloat16"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_bfloat16_addmm_1dbias_silu_bfloat16_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        self.skip_if_bfloat16_unsupported_hardware()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=True)
        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.bfloat16
        )
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        zentorch_model = model.to(dtype=torch.bfloat16)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_float32_mm_silu_float32_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=False)

        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.float32
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        zentorch_model = model
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 1)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["float32"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_float32_mm_silu_bfloat16_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=False)
        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.bfloat16
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        zentorch_model = model
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["bfloat16"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_bfloat16_mm_silu_float32_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        self.skip_if_bfloat16_unsupported_hardware()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=False)

        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.float32
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        zentorch_model = model.to(dtype=torch.bfloat16)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=["bfloat16"],
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_bfloat16_mm_silu_bfloat16_mul_pattern_model(self, freeze_opt):
        reset_dynamo()
        self.skip_if_bfloat16_unsupported_hardware()
        model = Custom_Model_Addmm_1dbias_SiLU_Mul(self.data, bias=False)
        mul_tensor = torch.reshape(self.data.x, (1, self.data.m, self.data.k)).to(
            torch.bfloat16
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        zentorch_model = model.to(dtype=torch.bfloat16)
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        _ = test_with_freeze_opt(
            compiled_model,
            (
                self.data.input.view(1, self.data.m, self.data.n), mul_tensor
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 1)


if __name__ == "__main__":
    run_tests()
