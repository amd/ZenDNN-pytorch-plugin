# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import importlib.util

import zentorch

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None

# Means "the zentorch vLLM plugin can be exercised", which every skipUnless in
# this directory uses it for. zentorch.vllm imports zentorch._utils and
# zentorch._C, so it needs libzentorch.so: with only the portable library
# loaded these tests would fail at import rather than skip.
VLLM_AVAILABLE = (
    importlib.util.find_spec("vllm") is not None and not zentorch.__stable_abi_only__
)

if VLLM_AVAILABLE:
    import vllm
else:
    vllm = None
