# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Run multimodal-encoder attention on zentorch. Set ZENTORCH_SDPA=0 to opt out.

Encoder attention for the Whisper audio tower and for the ViT towers all ends
up in one function, ``vit_attn_wrappers.apply_sdpa``, so swapping that single
function covers every caller. The swap is a plain module attribute assignment
and still reaches callers that were imported earlier, because the wrapper above
it looks ``apply_sdpa`` up by name on each call instead of holding a reference.

Quantization does not change anything here: the CompressedTensors Whisper
checkpoints quantize only Linear layers and leave the KV cache alone, so their
W8A8 and W4A16 variants reach this code in the same activation dtype as BF16.
The dtype is what matters, and each call is gated on it.
"""

from __future__ import annotations

from functools import wraps
import sys

import torch

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import
from zentorch.vllm._sdpa_utils import (
    zentorch_sdpa_enabled,
    zentorch_sdpa_supports_dtype,
)

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.v1.attention.ops.vit_attn_wrappers"
_PATCH_MARKER = "_zentorch_mm_encoder_sdpa_patched"
_OP_NAME = "zentorch_sdpa_attn"
# Shared tail for the bail-out warnings, so a run that silently lost the
# optimization is greppable in the logs.
_FALLBACK = "multimodal encoder attention stays on the stock SDPA"


def _do_patch_mm_encoder_sdpa() -> bool:
    """Rebind ``vit_attn_wrappers.apply_sdpa`` onto zentorch_sdpa_attn."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        # patch_now_or_on_import runs this either with the module already in
        # sys.modules or straight after it executes, so a miss is anomalous.
        logger.warning("[zentorch] %s did not load; %s", _TARGET_MODULE, _FALLBACK)
        return False

    original = getattr(module, "apply_sdpa", None)
    if original is None:
        logger.warning(
            "[zentorch] %s has no apply_sdpa on this vLLM; %s",
            _TARGET_MODULE,
            _FALLBACK,
        )
        return False
    if getattr(module, _PATCH_MARKER, False):
        return True

    @wraps(original)
    def _patched_apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        if not zentorch_sdpa_supports_dtype(q.dtype):
            return original(q, k, v, scale=scale, enable_gqa=enable_gqa)

        # We pass a {B, H, S, D} view; the op transposes `out` back to
        # BSHD inside. The default AVX-512 kernel wants that view
        # contiguous (otherwise it copies through a temporary), while the
        # opt-in ZenDNN kernel (ZENTORCH_USE_ZENDNN_SDPA=1) reads strides
        # directly and does not care. empty() gives a fresh contiguous
        # buffer even on the cu_seqlens path, where q is a non-contiguous
        # slice; empty_like() would inherit q's layout.
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        # No alibi_slopes and no window, so zentorch_sdpa_attn builds no mask.
        # It also takes no enable_gqa: the KV head count comes from the key
        # tensor and each query head maps onto its KV head, so GQA/MQA needs
        # no expansion here.
        torch.ops.zentorch.zentorch_sdpa_attn(
            q.movedim(1, 2),
            k.movedim(1, 2),
            v.movedim(1, 2),
            output.movedim(1, 2),
            scale=scale,
            is_causal=False,  # encoder attention is bidirectional
        )
        return output

    module.apply_sdpa = _patched_apply_sdpa
    setattr(module, _PATCH_MARKER, True)
    logger.info(
        "[zentorch] Patched vit_attn_wrappers.apply_sdpa -> %s "
        "(multimodal encoder attention)",
        _OP_NAME,
    )
    return True


def _apply_mm_encoder_sdpa_patch() -> bool:
    """Apply after vLLM's ViT attention wrapper module imports."""
    if not zentorch_sdpa_enabled(_OP_NAME):
        logger.debug(
            "[zentorch] MM encoder SDPA patch skipped (ZENTORCH_SDPA=0 or "
            "%s unavailable)",
            _OP_NAME,
        )
        return False
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_mm_encoder_sdpa)
