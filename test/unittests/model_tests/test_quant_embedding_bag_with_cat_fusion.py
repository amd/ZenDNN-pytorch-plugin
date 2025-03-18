# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    zentorch,
    has_zentorch,
    include_last_offset_opt,
    run_tests,
    supported_dtypes,
)

concat_dims = [0, 1]
other_tensor_positions = [0, -1]
exact_sizes = [True, False]
use_zendnn_eb = ["0", "1"]


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Quant_Embedding_Group(torch.nn.Module):
    def __init__(self, dtype, cat_dim, other_tensor_pos, include_last_offset):
        super(Custom_Model_Quant_Embedding_Group, self).__init__()
        self.cat_dim = cat_dim
        self.other_tensor_pos = other_tensor_pos
        self.include_last_offset = include_last_offset
        self.dtype = dtype

    def forward(self, weights, indices, offsets, other_tensor):
        eb1 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,  # Weight
            indices,  # indices
            offsets,  # offsets
            num_bits_per_weight=4,
            # assumes that weights has been quantized to uint4
            # hence 4 bits
            output_dtype=self.dtype,  # output_dtype
            scale_grad_by_freq=False,
            mode=0,
            sparse=False,
            per_sample_weights=None,
            include_last_offset=self.include_last_offset,
            padding_idx=-1,
        )
        eb2 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,  # Weight
            indices,  # indices
            offsets,  # offsets
            num_bits_per_weight=4,
            # assumes that weights has been quantized to uint4
            # hence 4 bits
            output_dtype=self.dtype,  # output_dtype
            scale_grad_by_freq=False,
            mode=0,
            sparse=False,
            per_sample_weights=None,
            include_last_offset=self.include_last_offset,
            padding_idx=-1,
        )
        tensor_list = []
        if self.other_tensor_pos == -1:
            tensor_list = [
                eb1,
                eb2,
                other_tensor,
            ]
        elif self.other_tensor_pos == 0:
            tensor_list = [
                other_tensor,
                eb1,
                eb2,
            ]
        else:
            raise ValueError(
                "Only zeroth or last position is permitted for concatenation"
            )

        res = torch.cat(tensor_list, dim=self.cat_dim)

        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_EmbeddingBag(Zentorch_TestCase):
    @parameterized.expand(
        product(
            supported_dtypes,
            concat_dims,
            other_tensor_positions,
            include_last_offset_opt,
            exact_sizes,
            use_zendnn_eb,
        )
    )
    @torch.inference_mode()
    def test_quant_embedding_bag_group(
        self,
        dtype,
        cat_dim,
        other_tensor_pos,
        include_last_offset,
        exact_size,
        zendnn_eb,
    ):
        os.environ["USE_ZENDNN_EB"] = zendnn_eb
        torch_type = self.data.get_torch_type(dtype)
        weight = torch.randint(low=0, high=15, size=(4, 16), dtype=torch_type)
        indices = torch.tensor([1, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)
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

        # TODO:
        # Add support for other aggregation modes for group quant embedding bag
        # aggregation = {
        #     0: "sum",
        #     1: "mean",
        #     2: "max"
        # }

        eb1 = torch.nn.functional.embedding_bag(
            indices,
            dequant_weight,
            offsets,
            mode="sum",
            include_last_offset=include_last_offset,
        )
        eb2 = torch.nn.functional.embedding_bag(
            indices,
            dequant_weight,
            offsets,
            mode="sum",
            include_last_offset=include_last_offset,
        )

        other_tensor_shape = list(eb1.shape)
        if exact_size:
            if cat_dim == 0:
                other_tensor_shape[0] += torch.randint(11, 20, (1,)).item()
            elif cat_dim == 1:
                other_tensor_shape[1] += torch.randint(11, 20, (1,)).item()

        other_tensor = torch.randn(*other_tensor_shape, dtype=torch_type)

        ref_result = None
        if other_tensor_pos == 0:
            ref_result = torch.cat(
                [
                    other_tensor,
                    eb1,
                    eb2,
                ],
                dim=cat_dim,
            )
        elif other_tensor_pos == -1:
            ref_result = torch.cat(
                [
                    eb1,
                    eb2,
                    other_tensor,
                ],
                dim=cat_dim,
            )
        else:
            raise ValueError(
                "Only zeroth or last position is permitted for concatenation"
            )

        model = Custom_Model_Quant_Embedding_Group(
            torch_type, cat_dim, other_tensor_pos, include_last_offset
        )
        model = torch.compile(model, backend="zentorch")
        model_result = model(
            zentorch_packed_weights,
            indices,
            offsets,
            other_tensor,
        )

        self.assertEqual(ref_result, model_result, atol=1e-3, rtol=1e-3)
        reset_dynamo()


if __name__ == "__main__":
    run_tests()
    os.environ.pop("USE_ZENDNN_EB")
