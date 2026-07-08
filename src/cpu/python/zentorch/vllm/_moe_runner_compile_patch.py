# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Make CPU FusedMoE torch.compile-safe.

vLLM's ``MoERunner._select_forward`` returns the *raw* ``_moe_forward`` function
on CPU (it assumes CPU MoE is not compiled), whereas GPU uses the opaque
``torch.ops.vllm.moe_forward`` custom op. Under ``aot_compile`` (e.g. gpt-oss's
DYNAMO_TRACE_ONCE mode) Dynamo then traces into ``_forward_impl`` and trips on
its non-traceable Python (dp_metadata / context managers). Routing CPU through
the same opaque custom op keeps the MoE a single graph node, matching GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
_TARGET_CLASS = "MoERunner"


def _register_moe_runner_patch(mod) -> None:
    cls = getattr(mod, _TARGET_CLASS)

    def _zen_select_forward(self):
        # Always use the opaque custom op (as GPU does) so aot_compile treats
        # the MoE as one node instead of tracing into _forward_impl.
        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    cls._select_forward = _zen_select_forward


def _apply_moe_runner_patch_to_module(mod) -> bool:
    try:
        cls = getattr(mod, _TARGET_CLASS, None)
        if cls is None or not hasattr(cls, "_select_forward"):
            return False
        if getattr(cls, "_zentorch_compile_patched", False):
            return True
        _register_moe_runner_patch(mod)
        cls._zentorch_compile_patched = True
        logger.info(
            "[zentorch] Patched %s._select_forward: CPU MoE via opaque custom "
            "op (torch.compile-safe)",
            _TARGET_CLASS,
        )
        return True
    except Exception:
        logger.warning("[zentorch] MoE runner compile patch FAILED", exc_info=True)
        return False


_HOOK_INSTALLED = False


class _MoeRunnerImportHook:
    """Defer patch until the moe_runner module loads."""

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
            _apply_moe_runner_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_moe_runner_compile_patch_impl() -> bool:
    """Schedule the CPU MoE compile patch on first import of moe_runner."""
    global _HOOK_INSTALLED

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        return _apply_moe_runner_patch_to_module(mod)

    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _MoeRunnerImportHook())
        _HOOK_INSTALLED = True
    return True
