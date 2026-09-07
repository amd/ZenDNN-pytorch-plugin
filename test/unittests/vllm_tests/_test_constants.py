# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import importlib.util

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None
VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

if VLLM_AVAILABLE:
    import vllm
else:
    vllm = None
