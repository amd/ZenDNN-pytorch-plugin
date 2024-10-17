# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
class Test_Horizontal_Embedding_Group(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    def test_horizontal_embedding_group(self, dtype):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_horizontal_embedding_group(
                    [self.data.embedding_matrix] * 3,
                    [self.data.emb_input] * 3,
                    [-1] * 3,
                    [False] * 3,
                    [False] * 3,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN Embedding!"
                in str(context.exception)
            )

        else:
            y_eb = torch._C._VariableFunctions.embedding(
                self.data.embedding_matrix, self.data.emb_input
            )

            y_ebz_list = torch.ops.zentorch.zentorch_horizontal_embedding_group(
                [self.data.embedding_matrix] * 3,
                [self.data.emb_input] * 3,
                [-1] * 3,
                [False] * 3,
                [False] * 3,
            )

            for i in range(0, int(len(y_ebz_list))):
                self.assertEqual(y_eb, y_ebz_list[i])


if __name__ == "__main__":
    run_tests()
