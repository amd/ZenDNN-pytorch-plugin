# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Gates shared by the zentorch SDPA patches."""

from __future__ import annotations

import os

import torch

__all__ = ["zentorch_sdpa_enabled", "zentorch_sdpa_supports_dtype"]


def zentorch_sdpa_enabled(op_name: str = "zentorch_sdpa") -> bool:
    """False when ``ZENTORCH_SDPA=0`` opts out or ``op_name`` is unregistered."""
    if os.environ.get("ZENTORCH_SDPA", "1") == "0":
        return False
    return hasattr(torch.ops.zentorch, op_name)


def zentorch_sdpa_supports_dtype(dtype: torch.dtype) -> bool:
    """Mirror of the dtype/ISA gate in zentorch_sdpa_common (Sdpa_ref.cpp).

    Below that gate zentorch_sdpa falls back to the native ATen flash kernel,
    which is what vLLM would have run anyway, so the replacement only adds the
    wrapper cost.
    """
    from zentorch._C import (
        is_avx512_supported,
        is_bf16_supported,
        is_fp16_supported,
    )

    if dtype == torch.bfloat16:
        return is_bf16_supported()
    if dtype == torch.float16:
        return is_fp16_supported()
    if dtype == torch.float32:
        return is_avx512_supported()
    return False
