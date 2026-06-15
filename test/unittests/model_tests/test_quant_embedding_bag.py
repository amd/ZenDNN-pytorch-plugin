# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    QuantEmbTestCase,
    has_zentorch,
    zentorch,
    run_tests,
    counters,
    supported_dtypes,
    update_supported_dtypes,
    reset_dynamo,
    Range,
    test_with_freeze_opt_and_cpp_wrapper,
    freeze_opt,
    cpp_wrapper_opt,
)

supported_dtypes = update_supported_dtypes(supported_dtypes)
NUM_BITS_UINT4 = 4


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Quant_Embedding_Group(nn.Module):
    def __init__(self):
        super(Custom_Model_Quant_Embedding_Group, self).__init__()

    def forward(
        self,
        weights: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        cat_input: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        eb1 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            NUM_BITS_UINT4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )

        eb2 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            NUM_BITS_UINT4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )
        res = torch.cat([eb1, eb2, cat_input], dim=1)
        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Quant_Embedding_Group_Out(nn.Module):
    def __init__(self):
        super(Custom_Model_Quant_Embedding_Group_Out, self).__init__()

    def forward(
        self,
        weights: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        eb1 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            NUM_BITS_UINT4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )

        eb2 = torch.ops.zentorch.zentorch_quant_embedding_bag(
            weights,
            indices,
            offsets,
            NUM_BITS_UINT4,  # assumes that weights has been quantized to uint4 hence 4 bits
            output_dtype,
            include_last_offset=True,
        )
        res = torch.cat([eb1, eb2], dim=1)
        return res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Embedding_Bag_Group(QuantEmbTestCase):

    # TODO: Test fails while generalizing the test - tracked in ZENAI-3863.
    # Revert to hypothesis-generated values once the issue is resolved.
    @QuantEmbTestCase.hypothesis_params_quant_emb_itr(
        dtype_list=supported_dtypes,
        include_last_offset_opt_list=[True],
        num_embeddings_range=Range(4, 4),
        embedding_dim_range=[16],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
    )
    @torch.inference_mode()
    def test_quant_embedding_bag_group(self, dtype, include_last_offset, freeze_opt, cpp_wrapper):
        torch_type = self.data.get_torch_type(dtype)
        weight = self.data.weight
        # TODO: Use hypothesis-generated indices and offsets once ZENAI-3863 is resolved.
        # indices = self.data.indices
        # offsets = self.data.offsets
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2], dtype=torch.long)
        scales = self.data.scales
        zero_points = self.data.zero_points

        zero_points_expanded = zero_points.unsqueeze(1).expand(weight.shape)
        dequant_weight = (weight - zero_points_expanded) * scales
        dequant_weight = dequant_weight.to(torch_type)

        from op_tests._pack import create_pack_method

        packmethod = create_pack_method("awq", "int4")
        packed_weight = packmethod.pack(
            (weight.to(torch.int32)), False, transpose=False
        )

        zentorch_packed_weights = zentorch._C.zentorch_get_packed_embedding_weight(
            packed_weight, scales, zero_points
        )

        # TODO: Use hypothesis-generated cat_input once ZENAI-3863 is resolved.
        # cat_input = self.data.cat_input
        cat_input = torch.randn(2, 8).type(torch_type)

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
        ref_result = torch.cat([eb1, eb2, cat_input], dim=1)

        reset_dynamo()
        model = Custom_Model_Quant_Embedding_Group()
        compiled_model = torch.compile(model, backend="zentorch")
        model_result = test_with_freeze_opt_and_cpp_wrapper(
            compiled_model,
            (zentorch_packed_weights, indices, offsets, cat_input, torch_type),
            freeze_opt,
            cpp_wrapper,
        )

        # TODO:
        # Remove dtype based tolerance once ZENAI-2844 is resolved
        if torch_type == torch.bfloat16:
            self.assertEqual(ref_result, model_result, atol=0.04, rtol=0.04)
        else:  # float32
            self.assertEqual(ref_result, model_result, atol=0.01, rtol=0.01)

    # TODO: Test fails while generalizing the test - tracked in ZENAI-3863.
    # Revert to hypothesis-generated values once the issue is resolved.
    @QuantEmbTestCase.hypothesis_params_quant_emb_itr(
        dtype_list=supported_dtypes,
        include_last_offset_opt_list=[True],
        num_embeddings_range=Range(4, 4),
        embedding_dim_range=[16],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
    )
    @torch.inference_mode()
    def test_quant_embedding_bag_group_out(self, dtype, include_last_offset, freeze_opt, cpp_wrapper):
        torch_type = self.data.get_torch_type(dtype)
        weight = self.data.weight
        # TODO: Use hypothesis-generated indices and offsets once ZENAI-3863 is resolved.
        # indices = self.data.indices
        # offsets = self.data.offsets
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2], dtype=torch.long)
        scales = self.data.scales
        zero_points = self.data.zero_points

        zero_points_expanded = zero_points.unsqueeze(1).expand(weight.shape)
        dequant_weight = (weight - zero_points_expanded) * scales
        dequant_weight = dequant_weight.to(torch_type)

        from op_tests._pack import create_pack_method

        packmethod = create_pack_method("awq", "int4")
        packed_weight = packmethod.pack(
            (weight.to(torch.int32)), False, transpose=False
        )

        zentorch_packed_weights = zentorch._C.zentorch_get_packed_embedding_weight(
            packed_weight, scales, zero_points
        )

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
        ref_result = torch.cat([eb1, eb2], dim=1)

        reset_dynamo()
        model = Custom_Model_Quant_Embedding_Group_Out()
        counters.clear()
        compiled_model = torch.compile(model, backend="zentorch")
        model_result = test_with_freeze_opt_and_cpp_wrapper(
            compiled_model,
            (zentorch_packed_weights, indices, offsets, torch_type),
            freeze_opt,
            cpp_wrapper,
        )

        self.assertEqual(counters["zentorch"]["out_variant"], 2)
        # TODO:
        # Remove dtype based tolerance once ZENAI-2844 is resolved
        if torch_type == torch.bfloat16:
            self.assertEqual(ref_result, model_result, atol=0.04, rtol=0.04)
        else:  # float32
            self.assertEqual(ref_result, model_result, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    run_tests()
