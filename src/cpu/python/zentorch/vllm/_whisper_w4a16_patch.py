# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Fix packed W4A16 Whisper k_proj bias loading.

vLLM 0.27 and above misses ``.k_proj.weight_packed`` and sizes the fake bias
from the packed dim. Inject a ``d_model``-sized zero bias, then run the
original loader.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
import sys

import torch

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.models.whisper"
_PATCH_MARKER = "_zentorch_whisper_w4a16_patched"
_K_PROJ_WEIGHT_PACKED = ".k_proj.weight_packed"


def _create_fake_bias_for_k_proj(
    weights: Iterable[tuple[str, torch.Tensor]],
    fake_bias_key_name: str,
    out_features: int | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield weights plus a zero k_proj bias when the checkpoint has none."""
    # ".weight_packed" / ".weight" -> ".bias" (packed suffix first).
    bias_key_name = (
        fake_bias_key_name.removesuffix(".weight_packed").removesuffix(".weight")
        + ".bias"
    )

    real_bias_names: set[str] = set()
    pending: dict[str, torch.Tensor] = {}
    for name, weight in weights:
        yield name, weight

        if name.endswith(bias_key_name):
            real_bias_names.add(name)
            pending.pop(name, None)
            continue

        if name.endswith(fake_bias_key_name):
            bias_name = name[: -len(fake_bias_key_name)] + bias_key_name
            if bias_name not in real_bias_names:
                pending[bias_name] = torch.zeros(
                    out_features if out_features is not None else weight.size(0)
                )

    for bias_name, bias in pending.items():
        yield bias_name, bias


def _do_patch_whisper_loader() -> bool:
    """Wrap Whisper ``load_weights`` for packed k_proj weights."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False

    cls = getattr(module, "WhisperForConditionalGeneration", None)
    if cls is None or not hasattr(cls, "load_weights"):
        return False
    if getattr(cls, _PATCH_MARKER, False):
        return True

    original_load_weights = cls.load_weights

    @wraps(original_load_weights)
    def _patched_load_weights(self, weights):
        d_model = self.config.d_model

        def _helper(weights, key, out_features=None):
            return _create_fake_bias_for_k_proj(
                weights,
                key,
                out_features=d_model if out_features is None else out_features,
            )

        # load_weights looks this up in the module globals.
        module._create_fake_bias_for_k_proj = _helper
        # Packed keys first; stock still walks ``.k_proj.weight``.
        weights = _helper(weights, _K_PROJ_WEIGHT_PACKED)
        return original_load_weights(self, weights)

    cls.load_weights = _patched_load_weights
    setattr(cls, _PATCH_MARKER, True)
    logger.info(
        "[zentorch] Patched WhisperForConditionalGeneration.load_weights "
        "for packed k_proj bias"
    )
    return True


def _apply_whisper_w4a16_patch() -> bool:
    """Apply after the vLLM Whisper model module imports."""
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_whisper_loader)
