# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    include_last_offset_opt,
    mode_opt,
    run_tests,
    scale_grad_opt,
    sparse_opt,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Horizontal_Embedding_Bag_Group(Zentorch_TestCase):
    @parameterized.expand(
        product(
            supported_dtypes,
            mode_opt,
            include_last_offset_opt,
            sparse_opt,
            scale_grad_opt,
        )
    )
    def test_horizontal_embedding_bag_group(
        self, dtype, mode, include_last_offset, sprs_opt, scale_opt
    ):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_horizontal_embedding_bag_group(
                    [self.data.embedding_matrix] * 3,
                    [self.data.emb_input] * 3,
                    [self.data.offsets] * 3,
                    [scale_opt] * 3,
                    [mode] * 3,
                    [sprs_opt] * 3,
                    [None] * 3,
                    [include_last_offset] * 3,
                    [-1] * 3,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN EmbeddingBag!"
                in str(context.exception)
            )

        else:
            y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
                self.data.embedding_matrix,
                self.data.emb_input,
                self.data.offsets,
                scale_opt,
                mode,
                sprs_opt,
                None,
                include_last_offset,
            )

            y_ebz_list = torch.ops.zentorch.zentorch_horizontal_embedding_bag_group(
                [self.data.embedding_matrix] * 3,
                [self.data.emb_input] * 3,
                [self.data.offsets] * 3,
                [scale_opt] * 3,
                [mode] * 3,
                [sprs_opt] * 3,
                [None] * 3,
                [include_last_offset] * 3,
                [-1] * 3,
            )

            for i in range(0, int(len(y_ebz_list) / 4)):
                self.assertEqual(y_eb, y_ebz_list[i * 4])


if __name__ == "__main__":
    run_tests()
