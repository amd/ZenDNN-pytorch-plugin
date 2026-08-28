# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Register zentorch providers for the vLLM IR norm ops.

RMSNorm/GemmaRMSNorm dispatch to ``vllm.ir.ops.rms_norm`` /
``fused_add_rms_norm`` (impl picked by IrOpPriorityConfig), not through
``nn.Module.forward``. We register a ``zentorch`` provider only for the residual
``fused_add_rms_norm`` (in-place); the non-residual ``rms_norm`` stays on native.
"""

import torch
from torch import Tensor

from zentorch._logging import get_logger

logger = get_logger(__name__)

_ZENTORCH_IR_NORM_REGISTERED = False

_SUPPORTED_ACT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


# supports_args signature must match the native op's exactly (param names +
# defaults); vLLM validates this at registration time.
def _add_rms_supports_args(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        variance_size is None
        and weight is not None
        and x.dim() >= 2
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.is_contiguous()
        and x.dtype in _SUPPORTED_ACT_DTYPES
        and x.shape == x_residual.shape
        and x.dtype == x_residual.dtype
    )


def register_zentorch_ir_norm_impls() -> bool:
    """Register the ``zentorch`` provider on ``vllm.ir.ops.fused_add_rms_norm``.

    Idempotent; returns True once the provider is registered.
    """
    global _ZENTORCH_IR_NORM_REGISTERED
    if _ZENTORCH_IR_NORM_REGISTERED:
        return True

    from vllm import ir

    fused_add_rms_norm_op = ir.ops.fused_add_rms_norm

    # Already registered (e.g. re-import); nothing to do.
    if "zentorch" in fused_add_rms_norm_op.impls:
        _ZENTORCH_IR_NORM_REGISTERED = True
        return True

    @fused_add_rms_norm_op.register_impl(
        "zentorch",
        supports_args=_add_rms_supports_args,
        supported=True,
        inplace=True,
    )
    def fused_add_rms_norm(
        x: Tensor,
        x_residual: Tensor,
        weight: Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        x_2d = x.view(-1, x.shape[-1])
        residual_2d = x_residual.view(-1, x_residual.shape[-1])
        # In-place: writes normalized x back into x_2d and the updated
        # residual (x + x_residual) back into residual_2d.
        torch.ops.zentorch.zentorch_add_rms_norm_(
            x_2d, weight, residual_2d, epsilon
        )
        return x, x_residual

    _ZENTORCH_IR_NORM_REGISTERED = True
    logger.info(
        "[zentorch] Registered zentorch IR impl for fused_add_rms_norm; "
        "rms_norm left on native provider"
    )
    return True
