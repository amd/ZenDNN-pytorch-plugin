# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Out-of-tree Mixtral per-expert W8A8 checkpoint loading (no vLLM changes).

vanilla ``MixtralModel.get_expert_mapping`` builds its expert-params mapping
with the Mixtral-native checkpoint names (``w1``/``w2``/``w3``). LLM-Compressor
W8A8 checkpoints (e.g. ``amd/Mixtral-8x7B-Instruct-v0.1-w8a8-llmcompressor``)
instead store experts per-projection as
``...block_sparse_moe.experts.N.{gate,up,down}_proj.*`` -- those names don't
match the ``w1/w2/w3`` mapping, so ``AutoWeightsLoader`` falls through to a
direct ``params_dict`` lookup and raises
``KeyError: '...experts.0.down_proj.weight'`` at load time.

This wraps ``get_expert_mapping`` to ADD a second mapping keyed on the
``gate_proj``/``up_proj``/``down_proj`` names (gate->w1 shard, up->w3 shard,
down->w2), so both the native and compressed-tensors layouts load. The
native ``w1/w2/w3`` entries are preserved, so nothing else changes.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from zentorch._logging import get_logger

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.models.mixtral"
_TARGET_CLASS = "MixtralModel"


def _register_mixtral_loader_patch(mod) -> None:
    cls = getattr(mod, _TARGET_CLASS)
    orig_get_expert_mapping = cls.get_expert_mapping

    def _zen_get_expert_mapping(self):
        # Native w1/w2/w3 mapping (unchanged behaviour for stock checkpoints).
        mapping = list(orig_get_expert_mapping(self))
        # Add compressed-tensors gate_proj/up_proj/down_proj naming so
        # LLM-Compressor per-expert W8A8 checkpoints resolve. Same shard
        # semantics: gate->w1, up->w3, down->w2.
        mapping += mod.fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
            num_redundant_experts=self.num_redundant_experts,
        )
        return mapping

    cls.get_expert_mapping = _zen_get_expert_mapping


def _apply_mixtral_loader_patch_to_module(mod) -> bool:
    try:
        cls = getattr(mod, _TARGET_CLASS, None)
        if cls is None or not hasattr(cls, "get_expert_mapping"):
            return False
        if getattr(cls, "_zentorch_mixtral_loader_patched", False):
            return True
        _register_mixtral_loader_patch(mod)
        cls._zentorch_mixtral_loader_patched = True
        logger.info(
            "[zentorch] Patched %s.get_expert_mapping: gate/up/down_proj "
            "per-expert W8A8 loading",
            _TARGET_CLASS,
        )
        return True
    except Exception:
        logger.warning("[zentorch] mixtral loader patch FAILED", exc_info=True)
        return False


_HOOK_INSTALLED = False


class _MixtralLoaderImportHook:
    """Defer patch until the mixtral model module loads."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _apply_mixtral_loader_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_mixtral_loader_patch_impl() -> bool:
    """Schedule the Mixtral per-expert loader patch on first import."""
    global _HOOK_INSTALLED

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        return _apply_mixtral_loader_patch_to_module(mod)

    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _MixtralLoaderImportHook())
        _HOOK_INSTALLED = True
    return True
