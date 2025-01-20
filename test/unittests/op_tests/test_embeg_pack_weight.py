# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import zentorch

from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Packed_Weight(Zentorch_TestCase):
    def test_embedding_packed_weight(self):
        weight = torch.randint(low=0, high=255, size=(20, 40), dtype=torch.int32)
        zero_points = torch.zeros(20).type(torch.int32)
        weight_scales = torch.randn(20)
        packed_weight = zentorch._C.zentorch_get_packed_embedding_weight(
            weight, weight_scales, zero_points
        )
        self.assertEqual(weight, packed_weight[:, :-1])
