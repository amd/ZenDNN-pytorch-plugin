# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Load Mixtral checkpoints with alternate per-expert projection names.

vLLM 0.27-0.28 expects ``w1``/``w2``/``w3``. Some compressed checkpoints use
``gate_proj``/``up_proj``/``down_proj``. vLLM 0.29 maps those names natively.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import wraps
import sys
from typing import TypeVar

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.models.mixtral"
_EXPERT_PATH = ".block_sparse_moe.experts."
_PROJECTION_NAMES = (
    (".gate_proj.", ".w1."),
    (".up_proj.", ".w3."),
    (".down_proj.", ".w2."),
)

_Weight = TypeVar("_Weight")


def _remap_mixtral_expert_names(
    weights: Iterable[tuple[str, _Weight]],
) -> Iterator[tuple[str, _Weight]]:
    """Translate alternate expert projection names to Mixtral's native names."""
    for name, weight in weights:
        if _EXPERT_PATH in name:
            for source, target in _PROJECTION_NAMES:
                name = name.replace(source, target)
        yield name, weight


def _has_native_expert_name_mapping(cls) -> bool:
    mapper = getattr(cls, "hf_to_vllm_mapper", None)
    mapping = getattr(mapper, "orig_to_new_substr", {})
    return all(mapping.get(source) == target for source, target in _PROJECTION_NAMES)


def _do_patch_mixtral_loader() -> bool:
    """Wrap Mixtral weight loading only when vLLM lacks the native mapping."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False

    cls = getattr(module, "MixtralModel", None)
    if cls is None or not hasattr(cls, "load_weights"):
        return False
    if getattr(cls, "_zentorch_mixtral_loader_patched", False):
        return True

    if _has_native_expert_name_mapping(cls):
        cls._zentorch_mixtral_loader_patched = True
        logger.info("[zentorch] Mixtral expert-name mapping is native")
        return True

    original_load_weights = cls.load_weights

    @wraps(original_load_weights)
    def _patched_load_weights(self, weights):
        return original_load_weights(
            self, _remap_mixtral_expert_names(weights)
        )

    cls.load_weights = _patched_load_weights
    cls._zentorch_mixtral_loader_patched = True
    logger.info(
        "[zentorch] Patched MixtralModel.load_weights for alternate expert names"
    )
    return True


def _apply_mixtral_loader_patch_impl() -> bool:
    """Apply after the vLLM Mixtral model module imports."""
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_mixtral_loader)
