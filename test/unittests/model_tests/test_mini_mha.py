# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from unittests.unittest_utils import (  # noqa: 402
    TestCase,
    run_tests,
    skip_test_pt_2_3,
)
from llm_tests.test_masked_mha import Test_Masked_MHA  # noqa: 402


@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class Test_MHA_Model(TestCase):
    def setUp(self):
        self.mha = Test_Masked_MHA()
        self.beam_size = 1
        self.batch_size = 1
        self.head_size = 256
        self.head_num = 16
        self.head_num_kv = 1
        self.max_seq_len = 64
        self.first_seq_len = 32

    def tearDown(self):
        del self.mha

    def test_mha_model(self):
        self.mha._test_mha(
            self.beam_size,
            self.batch_size,
            self.head_size,
            self.head_num,
            self.head_num_kv,
            self.max_seq_len,
            self.first_seq_len,
        )


if __name__ == "__main__":
    run_tests()
