# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from utils import (  # noqa: 402 # noqa: F401
    TestCase,
    run_tests,
    zentorch,
    has_zentorch,
    counters,
    supported_dtypes,
    skip_test_pt_2_0,
    skip_test_pt_2_1,
    skip_test_pt_2_3,
    reset_dynamo,
    set_seed,
)

woq_dtypes = []

if has_zentorch and zentorch._C.is_bf16_supported():
    woq_dtypes.append("bfloat16")

include_last_offset_opt = [True, False]
scale_grad_opt = [True, False]
mode_opt = [0, 1, 2]
sparse_opt = [True, False]
woq_input_dim_opt = [2, 3, 4]
woq_bias_opt = [0, 1]
woq_qzeros_opt = [0, 1]
group_size_opt = [-1, 1, 2, 3, 4, 5, 7, 8, 10]


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Test_Data(metaclass=Singleton):
    def create_data(self, dtype, group_size=-1):
        torch_type = self.get_torch_type(dtype)
        self.b, self.m, self.k, self.n = (
            torch.randint(1, 11, (1,)).item() for _ in range(4)
        )

        # m*k, k*n, m*n
        self.x = torch.randn(self.m, self.k).type(torch_type)
        self.y = torch.randn(self.k, self.n).type(torch_type)
        self.result = torch.zeros(self.m, self.n).type(torch_type)

        self.input = torch.randn(self.m, self.n).type(torch_type)
        self.input1d = torch.randn(self.n).type(torch_type)

        self.empty_bias = torch.zeros(0).type(torch_type)
        self.result_m = torch.zeros(int(self.m)).type(torch_type)
        self.result_1 = torch.zeros(1).type(torch_type)

        self.A = torch.randn(self.m, 1).type(torch_type)
        self.B = torch.randn(1, self.m).type(torch_type)

        # b*m*k, b*k*n, b*m*n
        self.x3d = torch.randn(self.b, self.m, self.k).type(torch_type)
        self.y3d = torch.randn(self.b, self.k, self.n).type(torch_type)
        self.input3d = torch.randn(self.b, self.m, self.n).type(torch_type)

        self.R = torch.randint(11, 20, (1,)).item()
        self.W = torch.randint(1, 15, (1,)).item()
        self.embedding_matrix = torch.randn(self.R, 3).type(torch_type)
        self.emb_input = torch.randint(0, self.R, (self.W,))
        self.offsets = torch.tensor([0, self.W])
        self.mlp_inputs = torch.randn(2, self.k)

        self.M = [
            torch.randn(60, 30).type(torch_type),
            torch.randn(30).type(torch_type),
        ]
        self.T1 = [torch.randn(2, 30, 30).type(torch_type)]
        self.x1 = [
            torch.randn(60, 40).type(torch_type),
            torch.randn(40, 60).transpose(0, 1).type(torch_type),
        ]

        self.y1 = [
            torch.randn(40, 30).type(torch_type),
            torch.randn(30, 40).transpose(1, 0).type(torch_type),
        ]

        self.M2 = torch.randn(60, 30, 50).type(torch_type)
        self.M3 = torch.randn(50).type(torch_type)

        self.x2 = [
            torch.randn(60, 30, 40).type(torch_type),
            torch.randn(60, 40, 30).transpose(1, 2).type(torch_type),
            torch.randn(30, 60, 40).transpose(0, 1).type(torch_type),
        ]

        self.y2 = [
            torch.randn(60, 40, 50).type(torch_type),
            torch.randn(60, 50, 40).transpose(1, 2).type(torch_type),
            torch.randn(50, 40, 60).transpose(0, 2).type(torch_type),
        ]

        self.woq_m, self.woq_x, self.woq_y = (
            torch.randint(1, 11, (1,)).item() for _ in range(3)
        )
        self.woq_group_size = group_size
        self.packing_ratio = 8
        if group_size == -1:
            self.woq_k = torch.randint(3, 11, (1,)).item() * self.packing_ratio
            group_size = self.woq_k
        else:
            self.woq_k = (
                torch.randint(3, 11, (1,)).item() * self.packing_ratio * group_size
            )
        # This is done for supporting the add/mul operation in unit test.
        self.woq_n = self.woq_k

        self.woq_input = {
            2: torch.randn(self.woq_m, self.woq_k).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_k).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_k).type(
                torch_type
            ),
        }
        self.woq_add = {
            2: torch.randn(self.woq_m, self.woq_n).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_n).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_n).type(
                torch_type
            ),
        }
        self.woq_mul = {
            2: torch.randn(self.woq_m, self.woq_n).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_n).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_n).type(
                torch_type
            ),
        }
        self.woq_qweight = torch.randn(
            self.woq_k, self.woq_n // self.packing_ratio
        ).type(torch.int32)
        self.woq_scales = torch.randn(self.woq_k // group_size, self.woq_n).type(
            torch.float32
        )
        self.woq_qzeros = [
            None,
            torch.zeros(
                self.woq_k // group_size, self.woq_n // self.packing_ratio
            ).type(torch.int32),
        ]
        self.woq_qzeros_nonzero = torch.randint(
            1, 15, (self.woq_k // group_size, self.woq_n // self.packing_ratio)
        ).type(torch.int32)
        self.woq_bias = [
            None,
            torch.randn(self.woq_n).type(torch_type),
        ]

    def get_torch_type(self, str_type):
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int": torch.int,
        }
        return dtypes[str_type]


class Zentorch_TestCase(TestCase):
    def setUp(self):
        set_seed()
        self.data = Test_Data()

    def tearDown(self):
        del self.data

    def skip_if_bfloat16_path_issue(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")

    def skip_if_bfloat16_unsupported_hardware(self):
        if not zentorch._C.is_bf16_supported():
            self.skipTest(
                "Warning: Skipping Bfloat16 Testcases since they are not "
                + "supported on this hardware"
            )
