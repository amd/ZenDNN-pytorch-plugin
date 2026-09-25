# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Skip CPUWNA16 N/K % 32 in ``ZentorchWNA16LinearKernel.can_implement``.

Zentorch WOQ does not need the parent oneDNN packing alignment. Quant type,
``has_g_idx``, and ``group_size`` dividing K stay.
"""

from __future__ import annotations

import sys
from functools import wraps

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.kernels.linear.mixed_precision.zentorch"
_PATCH_MARKER = "_zentorch_wna16_can_implement_patched"


def _can_implement_without_cpu_align(_cls, config):
    """WNA16 eligibility without the CPUWNA16 N/K % 32 gate."""
    from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
        _CPUWNA16_SUPPORTED_QUANT_TYPES,
    )

    if config.weight_type not in _CPUWNA16_SUPPORTED_QUANT_TYPES:
        return (
            False,
            f"Quant type ({config.weight_type}) not supported by "
            "CPUWNA16, supported types are: "
            f"{_CPUWNA16_SUPPORTED_QUANT_TYPES}",
        )

    # group_size == -1 is per-channel (one group of size K). Otherwise
    # ZenDNN WOQ requires group_size to divide K (partition_weight_shape[0]).
    if config.group_size != -1:
        in_features = config.partition_weight_shape[0]
        if config.group_size <= 0 or in_features % config.group_size != 0:
            return (
                False,
                f"Group size ({config.group_size}) must divide input size "
                f"({in_features})",
            )

    if getattr(config, "has_g_idx", False):
        return False, "ZentorchWNA16 does not support activation re-ordering."
    return True, None


def _do_patch_wna16_can_implement() -> bool:
    """Replace ``ZentorchWNA16LinearKernel.can_implement`` to skip N/K % 32."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False

    cls = getattr(module, "ZentorchWNA16LinearKernel", None)
    if cls is None or not callable(getattr(cls, "can_implement", None)):
        return False
    if getattr(cls, _PATCH_MARKER, False):
        return True

    @classmethod
    @wraps(cls.can_implement)
    def _patched_can_implement(kcls, config):
        return _can_implement_without_cpu_align(kcls, config)

    cls.can_implement = _patched_can_implement
    setattr(cls, _PATCH_MARKER, True)
    logger.info(
        "[zentorch] Patched ZentorchWNA16LinearKernel.can_implement to skip "
        "CPUWNA16 N/K % 32 alignment"
    )
    return True


def _apply_wna16_can_implement_patch() -> bool:
    """Apply after vLLM's mixed-precision zentorch kernel module imports."""
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_wna16_can_implement)
