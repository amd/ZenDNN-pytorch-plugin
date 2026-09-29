# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Force Whisper audio truncation to the 30s / 3000-frame window."""

from __future__ import annotations

import sys
from functools import wraps

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.models.whisper"
_PATCH_MARKER = "_zentorch_whisper_truncation_patched"


def _with_truncation(kwargs):
    out = dict(kwargs)
    out["truncation"] = True
    return out


def _do_patch_whisper_truncation() -> bool:
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False

    cls = getattr(module, "WhisperMultiModalProcessor", None)
    if cls is None:
        return False
    if getattr(cls, _PATCH_MARKER, False):
        return True

    orig_pre = cls.__dict__.get("_preprocess_hf_mm_data")
    orig_get = cls.__dict__.get("_get_hf_mm_inputs")
    applied = False

    if callable(orig_pre):

        @wraps(orig_pre)
        def _patched_pre(self, mm_data, hf_processor_mm_kwargs):
            mm_data, kwargs = orig_pre(self, mm_data, hf_processor_mm_kwargs)
            return mm_data, _with_truncation(kwargs)

        cls._preprocess_hf_mm_data = _patched_pre
        applied = True

    if callable(orig_get):

        @wraps(orig_get)
        def _patched_get(self, mm_items, hf_kwargs):
            hf_inputs = orig_get(self, mm_items, hf_kwargs)
            return hf_inputs._replace(
                hf_kwargs=_with_truncation(hf_inputs.hf_kwargs)
            )

        cls._get_hf_mm_inputs = _patched_get
        applied = True

    if not applied:
        return False

    setattr(cls, _PATCH_MARKER, True)
    logger.info(
        "[zentorch] Patched WhisperMultiModalProcessor to force truncation=True"
    )
    return True


def _apply_whisper_truncation_patch() -> bool:
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_whisper_truncation)
