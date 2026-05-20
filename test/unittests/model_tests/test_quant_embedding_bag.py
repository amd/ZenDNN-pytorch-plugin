# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
    counters,
    supported_dtypes,
    reset_dynamo,
)

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
        zero_points = torch.randint(low=0, high=15, size=(weight.size(0),), dtype=torch.int32)
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

        # TODO:
        # Remove dtype based tolerance once ZENAI-2844 is resolved
        if torch_type == torch.bfloat16:
            self.assertEqual(ref_result, model_result, atol=0.04, rtol=0.04)
        else:  # float32
            self.assertEqual(ref_result, model_result, atol=0.01, rtol=0.01)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_quant_embedding_bag_group_out(self, dtype):
        torch_type = self.data.get_torch_type(dtype)
        weight = torch.randint(low=0, high=15, size=(4, 16), dtype=torch_type)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2], dtype=torch.long)
        # TODO Check with zendnn for decimal place rounding
        scales = torch.rand(weight.size(0), 1).round(decimals=2)
        zero_points = torch.randint(low=0, high=15, size=(weight.size(0),), dtype=torch.int32)
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
            include_last_offset=True,
        )
        eb2 = torch.nn.functional.embedding_bag(
            indices,
            dequant_weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        ref_result = torch.cat([eb1, eb2], dim=1)

        reset_dynamo()
        model = Custom_Model_Quant_Embedding_Group_Out()
        counters.clear()
        model = torch.compile(model, backend="zentorch")
        model_result = model(zentorch_packed_weights, indices, offsets, torch_type)

        self.assertEqual(counters["zentorch"]["out_variant"], 2)
        # TODO:
        # Remove dtype based tolerance once ZENAI-2844 is resolved
        if torch_type == torch.bfloat16:
            self.assertEqual(ref_result, model_result, atol=0.04, rtol=0.04)
        else:  # float32
            self.assertEqual(ref_result, model_result, atol=0.01, rtol=0.01)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Embedding_Bag_Group_CppWrapper(Test_WOQ_Embedding_Bag_Group):
    """Re-runs the Test_WOQ_Embedding_Bag_Group cases with freezing +
    cpp_wrapper enabled, exercising the AOTI shim + lowering codegen path
    for both `.default` and `.out` variants of the horizontally-fused quant
    embedding-bag group op (plus the `aot_inductor.custom_ops_to_c_shims`
    cache-key handling registered by zentorch's `_lowerings.py`).

    * `.out` parameterizations exercise
      `aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group_out`
      via `_ZentorchHorizontalQuantEmbBagGroupOut`, whose codegen override
      drops the spurious `&out_handle` Inductor would otherwise append for
      tensor-returning shims (since this overload returns `()`).
    * `.default` parameterizations exercise
      `aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group` via
      `_ZentorchHorizontalQuantEmbBagGroupDefault`, whose codegen override
      replaces Inductor's standard `&handle_0, ..., &handle_{N-1}`
      multi-output pattern with `(handle_array, N)` to match the shim's
      fixed `(AtenTensorHandle* ret0_handles, int64_t ret0_len_)` signature
      (the only way to express a variable-length `Tensor[]` return in a
      single shim signature).
    """

    # TODO: convert to hypothesis-style parameterized tests so we exercise
    # the cpp_wrapper path across more (dtype, num_bags, indices_size,
    # padding_idx, include_last_offset) combos like the rest of the
    # quant embedding-bag unit tests.

    def setUp(self):
        super().setUp()
        self._prev_freezing = torch._inductor.config.freezing
        self._prev_cpp_wrapper = torch._inductor.config.cpp_wrapper
        torch._inductor.config.freezing = True
        torch._inductor.config.cpp_wrapper = True

    def tearDown(self):
        torch._inductor.config.cpp_wrapper = self._prev_cpp_wrapper
        torch._inductor.config.freezing = self._prev_freezing
        super().tearDown()


if __name__ == "__main__":
    run_tests()
