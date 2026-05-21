# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import operator
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
    update_supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
    counters,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_linear")

LINEAR_UNARY_OPS = {
    "silu": {
        "op": torch.nn.functional.silu,
        "counter": "zentorch_linear_silu",
    },
}

LINEAR_BINARY_OPS = {
    "mul": {
        "op": operator.mul,
        "counter": "zentorch_linear_mul",
    },
}

# Define combinations of unary + binary operations
LINEAR_UNARY_BINARY_OPS = {
    "silu_mul": {
        "unary_op": torch.nn.functional.silu,
        "binary_op": operator.mul,
        "counter": "zentorch_linear_silu_mul",
    },
}

LINEAR_BIAS_CASES = {
    "with_bias": True,
    "no_bias": False,
}

LINEAR_TOLERANCES = {
    "float32": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 5e-2, "rtol": 5e-2},
}


class LinearUnaryBinaryModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        unary_op,
        binary_op,
        bias: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.unary_op = unary_op
        self.binary_op = binary_op

    def forward(
        self, input_tensor: torch.Tensor, binary_tensor: torch.Tensor
    ) -> torch.Tensor:
        linear_out = self.linear(input_tensor)
        unary_out = self.unary_op(linear_out)
        return self.binary_op(unary_out, binary_tensor)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Unary_Binary_Model(AddmmTestCase):

    def _run_unary_binary_post_op(
        self, key: str, bias_flag: bool, dtype: str, freeze_flag: bool
    ) -> None:
        if dtype == "bfloat16":
            self.skip_if_bfloat16_unsupported_hardware()
        torch_dtype = self.data.get_torch_type(dtype)
        model = LinearUnaryBinaryModel(
            self.data.n,
            self.data.m,
            torch_dtype,
            LINEAR_UNARY_BINARY_OPS[key]["unary_op"],
            LINEAR_UNARY_BINARY_OPS[key]["binary_op"],
            bias=bias_flag,
        )
        reference_linear = model.linear(self.data.input)
        binary_tensor = torch.randn_like(reference_linear)

        # Compute native output
        unary_out = LINEAR_UNARY_BINARY_OPS[key]["unary_op"](reference_linear)
        native_output = LINEAR_UNARY_BINARY_OPS[key]["binary_op"](
            unary_out, binary_tensor
        )

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        counter_key = LINEAR_UNARY_BINARY_OPS[key]["counter"]
        self.assertEqual(counters["zentorch"][counter_key], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input, binary_tensor),
            freeze_flag,
        )
        self.assertEqual(counters["zentorch"][counter_key], 1)
        if freeze_flag:
            self.assertEqual(
                counters["zentorch"]["zentorch_weight_prepack_for_linear"], 1
            )
        tolerance = LINEAR_TOLERANCES.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
        self.assertEqual(native_output, compiled_output, **tolerance)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_linear_silu_mul_model(self, dtype, freeze_opt):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_unary_binary_post_op("silu_mul", bias_flag, dtype, freeze_opt)


if __name__ == "__main__":
    run_tests()
