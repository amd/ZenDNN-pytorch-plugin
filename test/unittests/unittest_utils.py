# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path
import os
import shutil
sys.path.append(str(Path(__file__).parent.parent))

from utils import (  # noqa: 402 # noqa: F401
    BaseZentorchTestCase,
    run_tests,
    zentorch,
    has_zentorch,
    counters,
    supported_dtypes,
    qlinear_dtypes,
    skip_test_pt_2_0,
    skip_test_pt_2_1,
    skip_test_pt_2_3,
    reset_dynamo,
    set_seed,
    freeze_opt,
    test_with_freeze_opt,
    Test_Data,
    woq_dtypes,
    include_last_offset_opt,
    scale_grad_opt,
    mode_opt,
    sparse_opt,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    woq_qzeros_opt,
    group_size_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
    conv_stride,
    conv_padding,
    at_ops,
    zt_ops,
    qlinear_eltwise_map,
    seq_length_opt,
    batch_size_opt,
)


path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Zentorch_TestCase(BaseZentorchTestCase):
    def setUp(self):
        super().setUp()
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        self.data = Test_Data()

    def tearDown(self):
        del self.data
        shutil.rmtree(os.path.join(path, "data"))
