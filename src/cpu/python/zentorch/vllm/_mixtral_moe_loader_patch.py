# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Load Mixtral checkpoints with per-expert projection names on vLLM 0.27.

vLLM's Mixtral loader expects expert projections named ``w1``/``w2``/``w3``.
Some compressed checkpoints instead use ``gate_proj``/``up_proj``/``down_proj``.
Only those expert-path components are remapped; native names remain unchanged.
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


def _do_patch_mixtral_loader() -> bool:
    """Wrap the vLLM 0.27 Mixtral weight loader."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False

    cls = getattr(module, "MixtralModel", None)
    if cls is None or not hasattr(cls, "load_weights"):
        return False
    if getattr(cls, "_zentorch_mixtral_loader_patched", False):
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
