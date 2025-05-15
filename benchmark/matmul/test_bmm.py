# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import zentorch  # noqa: F401
import os
import time
import csv
import json

zentorch_test_data_dir = os.environ.get("ZENTORCH_TEST_DATA_DIR")
if not zentorch_test_data_dir or not os.path.exists(zentorch_test_data_dir):
    # Exiting with 1 to represent the absence of ZENTORCH_TEST_DATA_DIR directory
    exit(1)

# Testing over batch size 1 to 75 both inclusive
batch_sizes = range(1, 76)
iterations = 100
previous_env_value = os.environ.get("USE_ZENDNN_MATMUL_DIRECT")

with open(os.path.join(zentorch_test_data_dir, "shapes.json"), "r") as file:
    test_data = json.load(file)
data = []
for _, tensor_info in test_data["bmm"].items():
    m_, k_, n_, stridedA, stridedB, transA, transB = tensor_info.values()
    for batch_size in sorted(batch_sizes):
        mat1 = torch.randn(batch_size, m_, k_)
        mat2 = torch.randn(batch_size, k_, n_)

        if stridedA:
            mat1 = torch.randn(batch_size, k_, m_).permute(0, 2, 1)
        if stridedB:
            mat2 = torch.randn(batch_size, k_, n_).permute(0, 2, 1)

        # Eager
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.bmm(mat1, mat2)
        end_time = time.time()
        eager_time = end_time - start_time

        # ZenDNN Direct Kernel
        os.environ["USE_ZENDNN_MATMUL_DIRECT"] = "1"
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.ops.zentorch.zentorch_bmm(mat1, mat2)
        end_time = time.time()
        zt_time_direct = end_time - start_time

        os.environ["USE_ZENDNN_MATMUL_DIRECT"] = "0"
        # ZenDNN
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.ops.zentorch.zentorch_bmm(mat1, mat2)
        end_time = time.time()
        zt_time = end_time - start_time

        data.append(
            [
                batch_size,
                m_,
                k_,
                n_,
                round(eager_time * 1000, 4),
                round(zt_time_direct * 1000, 4),
                round(zt_time * 1000, 4),
            ]
        )

if previous_env_value:
    os.environ["USE_ZENDNN_MATMUL_DIRECT"] = previous_env_value
else:
    os.environ.pop("USE_ZENDNN_MATMUL_DIRECT")

header = [
    "bs",
    "m",
    "k",
    "n",
    "eager (time in ms)",
    "zentorch-direct (time in ms)",
    "zentorch-algo (time in ms)",
]
with open(
    os.path.join(zentorch_test_data_dir, "bmm.csv"), "w", newline=""
) as file:
    csvw = csv.writer(file)
    csvw.writerow(header)
    csvw.writerows(data)
