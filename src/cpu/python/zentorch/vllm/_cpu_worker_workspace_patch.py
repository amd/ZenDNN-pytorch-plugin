# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Initialize the modular-kernel workspace manager on the CPU worker.

The modular MoE kernel allocates scratch via ``current_workspace_manager()``,
which asserts the manager was initialized. The GPU worker calls
``init_workspace_manager`` but vanilla ``CPUWorker`` does not, so modular MoE
kernels on CPU hit ``WorkspaceManager not initialized``. Wrap
``CPUWorker.init_device`` to initialize it (mirrors the in-tree cpu_worker fix).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from zentorch._logging import get_logger

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.v1.worker.cpu_worker"
_TARGET_CLASS = "CPUWorker"


def _register_cpu_worker_workspace_patch(mod) -> None:
    cls = getattr(mod, _TARGET_CLASS)
    orig_init_device = cls.init_device

    def _zen_init_device(self):
        orig_init_device(self)
        from vllm.v1.worker.workspace import init_workspace_manager

        # Idempotent enough: init_workspace_manager warns + reinitializes if
        # already set. num_ubatches=1 matches the in-tree CPU default.
        init_workspace_manager(self.device, 1)
        logger.info("[zentorch] Initialized CPU workspace manager for modular MoE")

    cls.init_device = _zen_init_device


def _apply_cpu_worker_workspace_patch_to_module(mod) -> bool:
    try:
        cls = getattr(mod, _TARGET_CLASS, None)
        if cls is None or not hasattr(cls, "init_device"):
            return False
        if getattr(cls, "_zentorch_workspace_patched", False):
            return True
        _register_cpu_worker_workspace_patch(mod)
        cls._zentorch_workspace_patched = True
        logger.info(
            "[zentorch] Patched %s.init_device: workspace manager init",
            _TARGET_CLASS,
        )
        return True
    except Exception:
        logger.warning("[zentorch] CPU worker workspace patch FAILED", exc_info=True)
        return False


_HOOK_INSTALLED = False


class _CpuWorkerImportHook:
    """Defer patch until the cpu_worker module loads."""

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
            _apply_cpu_worker_workspace_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_cpu_worker_workspace_patch_impl() -> bool:
    """Schedule the CPU worker workspace patch on first import."""
    global _HOOK_INSTALLED

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        return _apply_cpu_worker_workspace_patch_to_module(mod)

    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _CpuWorkerImportHook())
        _HOOK_INSTALLED = True
    return True
