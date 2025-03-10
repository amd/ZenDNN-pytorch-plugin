# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    def test_embedding(self, dtype):
        self.data.create_unittest_data(dtype)
        y_eb = torch._C._VariableFunctions.embedding(
            self.data.embedding_matrix, self.data.emb_input
        )
        y_ebz = torch.ops.zentorch.zentorch_embedding(
            self.data.embedding_matrix, self.data.emb_input
        )
        self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    def test_embedding_sparse_scale(self, dtype):
        self.data.create_unittest_data(dtype)
        sparse_opt = [True, False]
        scale_grad_opt = [True, False]

        for sprs_opt in sparse_opt:
            for scale_opt in scale_grad_opt:
                y_eb = torch._C._VariableFunctions.embedding(
                    self.data.embedding_matrix,
                    self.data.emb_input,
                    -1,
                    scale_opt,
                    sprs_opt,
                )
                y_ebz = torch.ops.zentorch.zentorch_embedding(
                    self.data.embedding_matrix,
                    self.data.emb_input,
                    -1,
                    scale_opt,
                    sprs_opt,
                )
                self.assertEqual(y_eb, y_ebz)


if __name__ == "__main__":
    run_tests()
