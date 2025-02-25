# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
class Test_Embedding_Bag(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    def test_embedding_bag(self, dtype):

        self.data.create_unittest_data(dtype)
        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            self.data.embedding_matrix,
            self.data.emb_input,
            self.data.offsets,
            False,
            0,
            False,
            None,
            False,
        )
        y_ebz = torch.ops.zentorch.zentorch_embedding_bag(
            self.data.embedding_matrix,
            self.data.emb_input,
            self.data.offsets,
            False,
            0,
            False,
            None,
            False,
            -1,
        )
        self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(
        product(
            supported_dtypes,
            mode_opt,
            include_last_offset_opt,
            sparse_opt,
            scale_grad_opt,
        )
    )
    def test_embedding_bag_sparse_scale_mode(
        self, dtype, mode, include_last_offset, sprs_opt, scale_opt
    ):

        self.data.create_unittest_data(dtype)

        # max mode is not supported whenever any of the sparse_opt
        # or scale_grad_opt is True
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

        y_ebz = torch.ops.zentorch.zentorch_embedding_bag(
            self.data.embedding_matrix,
            self.data.emb_input,
            self.data.offsets,
            scale_opt,
            mode,
            sprs_opt,
            None,
            include_last_offset,
            -1,
        )
        self.assertEqual(y_eb, y_ebz)


if __name__ == "__main__":
    run_tests()
