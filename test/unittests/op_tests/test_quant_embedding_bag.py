# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path
from parameterized import parameterized

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Embedding_Bag(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_quant_embedding_bag(self, dtype):
        torch_type = self.data.get_torch_type(dtype)
        weight = torch.randint(low=0, high=15, size=(4, 16), dtype=torch_type)
        indices = torch.tensor([1, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)
        scales = torch.tensor([[1.0], [4.0], [5.0], [7.0]])
        zero_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        # 1   2    3    4   5   6      7  8
        # will be packed as
        # 8     7   6    5    4    3    2    1
        # 1000 0111 0110 0101 0100 0011 0010 0001

        from op_tests._pack import create_pack_method

        packmethod = create_pack_method("awq", "int4")
        packed_weight = packmethod.pack(
            (weight.to(torch.int32)), False, transpose=False
        )
        dequant_weight = weight * scales

        ref_result = torch.nn.functional.embedding_bag(
            indices, dequant_weight, offsets, mode="sum"
        ).to(torch_type)

        zentorch_packed_weights = zentorch._C.zentorch_get_packed_embedding_weight(
            packed_weight, scales, zero_points
        )
        op_result = torch.ops.zentorch.zentorch_quant_embedding_bag(
            zentorch_packed_weights,
            indices,
            offsets,
            4,  # assumes that weights has been quantized to uint4 hence 4 bits
            torch_type,
            False,
            0,
            False,
            None,
            0,
            -1,
        )
        self.assertEqual(ref_result, op_result)


if __name__ == "__main__":
    run_tests()
