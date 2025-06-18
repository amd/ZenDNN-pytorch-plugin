# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path
from parameterized import parameterized

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
    supported_dtypes,
    reset_dynamo,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Quant_Embedding_Group(nn.Module):
    def __init__(self):
        super(Custom_Model_Quant_Embedding_Group, self).__init__()

    def forward(self, weights, indices, offsets, cat_input, output_dtype):
        eb1 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )

        eb2 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )
        res = torch.cat([eb1, eb2, cat_input], dim=1)
        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Embedding_Bag_Group(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_quant_embedding_bag_group(self, dtype):
        torch_type = self.data.get_torch_type(dtype)
        weight = torch.randint(low=0, high=15, size=(4, 16), dtype=torch_type)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2], dtype=torch.long)
        # TODO Check with zendnn for decimal place rounding
        scales = torch.rand(weight.size(0), 1).round(decimals=2)
        zero_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        dequant_weight = weight * scales
        dequant_weight = dequant_weight.to(torch_type)

        from op_tests._pack import create_pack_method

        packmethod = create_pack_method("awq", "int4")
        packed_weight = packmethod.pack(
            (weight.to(torch.int32)), False, transpose=False
        )

        zentorch_packed_weights = zentorch._C.zentorch_get_packed_embedding_weight(
            packed_weight, scales, zero_points
        )

        cat_input = torch.randn(2, 8).type(torch_type)

        eb1 = torch.nn.functional.embedding_bag(
            indices,
            dequant_weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        eb2 = torch.nn.functional.embedding_bag(
            indices,
            dequant_weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        ref_result = torch.cat([eb1, eb2, cat_input], dim=1)

        reset_dynamo()
        model = Custom_Model_Quant_Embedding_Group()
        model = torch.compile(model, backend="zentorch")
        model_result = model(
            zentorch_packed_weights, indices, offsets, cat_input, torch_type
        )
        self.assertEqual(ref_result, model_result, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
