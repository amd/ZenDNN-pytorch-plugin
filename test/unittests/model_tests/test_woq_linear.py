# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
from itertools import product
import torch
from parameterized import parameterized
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
    bias_opt,
    woq_dtypes,
    supported_dtypes,
    input_dim_opt,
    woq_qzeros_opt,
    group_size_opt,
    skip_test_pt_2_1,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


# Check number of user sequential
class Custom_Model_WOQ_Linear_Add_Sequential(nn.Module):
    def __init__(self):
        super(Custom_Model_WOQ_Linear_Add_Sequential, self).__init__()

    def forward(
        self, inp, qweight, woq_scales, woq_qzeros, woq_bias, add1, add2, group_size
    ):
        woq_out = torch.ops.zentorch.zentorch_woq_linear(
            inp,
            qweight,
            woq_scales,
            woq_qzeros,
            woq_bias,
            group_size,
        )
        add_1_res = torch.add(woq_out, add1)
        add_res = torch.add(add_1_res, add2)
        y = torch.ops.zentorch.zentorch_woq_linear(
            add_res,
            qweight,
            woq_scales,
            woq_qzeros,
            woq_bias,
            group_size,
        )
        add_2_res = torch.add(y, add1)
        add3 = add_res * add_2_res
        return add3


# Check number of user parallel
class Custom_Model_WOQ_Linear_Add_Parallel(nn.Module):
    def __init__(self):
        super(Custom_Model_WOQ_Linear_Add_Parallel, self).__init__()

    def forward(
        self, inp, qweight, woq_scales, woq_qzeros, woq_bias, add1, add2, group_size
    ):
        woq_out = torch.ops.zentorch.zentorch_woq_linear(
            inp,
            qweight,
            woq_scales,
            woq_qzeros,
            woq_bias,
            group_size,
        )
        add_1_res = torch.add(woq_out, add1)
        add_res = torch.add(add_1_res, add2)
        y = torch.ops.zentorch.zentorch_woq_linear(
            inp,
            qweight,
            woq_scales,
            woq_qzeros,
            woq_bias,
            group_size,
        )
        add_2_res = torch.add(y, add1)
        add3 = add_res * add_2_res
        return add3


class Custom_Model_WOQ_Linear_Silu_Mul(nn.Module):
    def __init__(self):
        super(Custom_Model_WOQ_Linear_Silu_Mul, self).__init__()

    def forward(self, inp, qweight, woq_scales, woq_qzeros, woq_bias, mul, group_size):
        woq_out = torch.ops.zentorch.zentorch_woq_linear(
            inp,
            qweight,
            woq_scales,
            woq_qzeros,
            woq_bias,
            group_size,
        )
        silu_res = torch.nn.functional.silu(woq_out)
        res = torch.mul(silu_res, mul)
        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2")
class Test_WOQ_Linear_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            woq_dtypes,
            supported_dtypes,
            input_dim_opt,
            bias_opt,
            woq_qzeros_opt,
            group_size_opt,
            freeze_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_woq_linear_add_sequential_model(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
        freeze_opt
    ):
        self.data.create_data(dtype, group_size_val)
        model = Custom_Model_WOQ_Linear_Add_Sequential().eval()
        zentorch_model = model
        _ = model(
            self.data.woq_input[woq_input_dim],
            self.data.woq_qweight[scales_dtype],
            self.data.woq_scales[scales_dtype],
            self.data.woq_qzeros[woq_qzeros_idx],
            self.data.woq_bias[woq_bias_idx],
            self.data.woq_add[woq_input_dim],
            self.data.woq_add[woq_input_dim],
            self.data.woq_group_size,
        )
        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)
        _ = test_with_freeze_opt(
            compiled_graph,
            (
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_add[woq_input_dim],
                self.data.woq_add[woq_input_dim],
                self.data.woq_group_size,
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 1)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 1)

    # TODO:
    # Add op level test cases for woq_linear_add and woq_linear_mul
    @parameterized.expand(
        product(
            woq_dtypes,
            supported_dtypes,
            input_dim_opt,
            bias_opt,
            woq_qzeros_opt,
            group_size_opt,
            freeze_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_woq_linear_add_sequential_postop_float_model(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
        freeze_opt
    ):
        self.data.create_data(dtype, group_size_val)
        model = Custom_Model_WOQ_Linear_Add_Sequential().eval()
        zentorch_model = copy.deepcopy(model)

        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")

        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)
        with self.assertRaises(RuntimeError) as context:
            _ = test_with_freeze_opt(
                compiled_graph,
                (
                    self.data.woq_input[woq_input_dim],
                    self.data.woq_qweight[scales_dtype],
                    self.data.woq_scales[scales_dtype],
                    self.data.woq_qzeros[woq_qzeros_idx],
                    self.data.woq_bias[woq_bias_idx],
                    self.data.woq_add[woq_input_dim].to(torch.float32),
                    self.data.woq_add[woq_input_dim].to(torch.float32),
                    self.data.woq_group_size,
                ),
                freeze_opt
            )
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)

        self.assertTrue(
            "only bfloat16 datatype is currently supported" in str(context.exception)
        )

    @parameterized.expand(
        product(
            woq_dtypes,
            supported_dtypes,
            input_dim_opt,
            bias_opt,
            woq_qzeros_opt,
            group_size_opt,
            freeze_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_woq_linear_silu_mul_postop_float_model(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
        freeze_opt
    ):
        self.data.create_data(dtype, group_size_val)
        model = Custom_Model_WOQ_Linear_Silu_Mul().eval()
        zentorch_model = copy.deepcopy(model)

        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")

        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_silu_mul"], 0)
        _ = test_with_freeze_opt(
            compiled_graph,
            (
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_mul[woq_input_dim].to(torch.float32),
                self.data.woq_group_size,
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_silu_mul"], 0)

    @parameterized.expand(
        product(
            woq_dtypes,
            supported_dtypes,
            input_dim_opt,
            bias_opt,
            woq_qzeros_opt,
            group_size_opt,
            freeze_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_woq_linear_add_parallel_model(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
        freeze_opt
    ):
        self.data.create_data(dtype, group_size_val)
        model = Custom_Model_WOQ_Linear_Add_Parallel().eval()
        zentorch_model = copy.deepcopy(model)
        _ = model(
            self.data.woq_input[woq_input_dim],
            self.data.woq_qweight[scales_dtype],
            self.data.woq_scales[scales_dtype],
            self.data.woq_qzeros[woq_qzeros_idx],
            self.data.woq_bias[woq_bias_idx],
            self.data.woq_add[woq_input_dim],
            self.data.woq_add[woq_input_dim],
            self.data.woq_group_size,
        )
        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 0)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 0)
        _ = test_with_freeze_opt(
            compiled_graph,
            (
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_add[woq_input_dim],
                self.data.woq_add[woq_input_dim],
                self.data.woq_group_size,
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add_add"], 1)
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_add"], 1)

    @parameterized.expand(
        product(
            woq_dtypes,
            supported_dtypes,
            input_dim_opt,
            bias_opt,
            woq_qzeros_opt,
            group_size_opt,
            freeze_opt
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_woq_linear_silu_mul_model(
        self,
        dtype,
        scales_dtype,
        woq_input_dim,
        woq_bias_idx,
        woq_qzeros_idx,
        group_size_val,
        freeze_opt
    ):
        self.data.create_data(dtype, group_size_val)
        model = Custom_Model_WOQ_Linear_Silu_Mul().eval()
        zentorch_model = copy.deepcopy(model)
        _ = model(
            self.data.woq_input[woq_input_dim],
            self.data.woq_qweight[scales_dtype],
            self.data.woq_scales[scales_dtype],
            self.data.woq_qzeros[woq_qzeros_idx],
            self.data.woq_bias[woq_bias_idx],
            self.data.woq_mul[woq_input_dim],
            self.data.woq_group_size,
        )
        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_silu_mul"], 0)
        _ = test_with_freeze_opt(
            compiled_graph,
            (
                self.data.woq_input[woq_input_dim],
                self.data.woq_qweight[scales_dtype],
                self.data.woq_scales[scales_dtype],
                self.data.woq_qzeros[woq_qzeros_idx],
                self.data.woq_bias[woq_bias_idx],
                self.data.woq_mul[woq_input_dim],
                self.data.woq_group_size,
            ),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_woq_silu_mul"], 1)


if __name__ == "__main__":
    run_tests()
