# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    EmbTestCase,
    has_zentorch,
    include_last_offset_opt,
    mode_opt,
    run_tests,
    scale_grad_opt,
    sparse_opt,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Horizontal_Embedding_Bag_Group(EmbTestCase):
    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes,
        mode_opt_list=mode_opt,
        include_last_offset_opt_list=include_last_offset_opt,
        sparse_opt_list=sparse_opt,
        scale_grad_opt_list=scale_grad_opt,

    )
    def test_horizontal_embedding_bag_group(
        self, dtype, mode, include_last_offset, sprs_opt, scale_opt
    ):
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
