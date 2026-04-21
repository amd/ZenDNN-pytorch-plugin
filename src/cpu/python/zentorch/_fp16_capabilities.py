# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os

# TODO: Remove this once the fp16 support is fully implemented
FP16_CAPABLE_OPS = {
    "zentorch_embedding": False,
    "zentorch_mm": False,
    "zentorch_bmm": False,
    "zentorch_addmm": False,
    "zentorch_baddbmm": False,
    "zentorch_convolution": False,
    "zentorch_sdpa": False,
    "zentorch_rope": False,
    "zentorch_mmha": False,
    "zentorch_linear": False,
    "zentorch_woq_linear": False,
    "zentorch_weight_prepack": False,
}


# This function is responsible for enable the ops based on the env varible.
# Only the ops in the env should be enabled and those should be supported.
def update_fp16_registry():
    ops = os.environ.get("ZENTORCH_FP16_OPS", None)
    if ops:
        supported_ops = [key for (key, value) in FP16_CAPABLE_OPS.items() if value]
        for ops in FP16_CAPABLE_OPS:
            FP16_CAPABLE_OPS[ops] = False
        for op in ops.split(","):
            op = op.strip().lower()
            if op not in supported_ops:
                raise ValueError(
                    f"Unsupported fp16 op: {op} has been enabled by env variblle ZENTORCH_FP16_OPS"
                )
            if op in FP16_CAPABLE_OPS:
                FP16_CAPABLE_OPS[op] = True


def is_fp16_capable(op):
    if op in FP16_CAPABLE_OPS:
        return FP16_CAPABLE_OPS[op]
    return False


def get_fp16_registry():
    return FP16_CAPABLE_OPS
