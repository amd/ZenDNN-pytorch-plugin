# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Monkey-patcher: replace ``GatedDeltaNetAttention.forward_cpu`` with
``forward_cpu_zen``."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from typing import Any

from zentorch._logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "apply",
    "apply_deferred",
    "restore",
    "is_applied",
    "PATCHES",
]


_TRIGGER_MODULE = "vllm.model_executor.layers.mamba.gdn_linear_attn"

_DEFERRED_HOOK_INSTALLED = False

PATCHES: dict[tuple[str, str], Any] = {}
_TARGETS: dict[tuple[str, str], Any] = {}
_APPLY_IN_PROGRESS = False


def _target_dotted_name(target: Any) -> str:
    if isinstance(target, types.ModuleType):
        return target.__name__
    return f"{target.__module__}.{target.__qualname__}"


class _GatedDeltaNetImportHook:
    """Defer applying the GDN patch until ``gdn_linear_attn`` loads."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TRIGGER_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            try:
                apply()
            except Exception:
                logger.warning(
                    "[zentorch] GatedDeltaNet patch failed after %s loaded",
                    _TRIGGER_MODULE,
                    exc_info=True,
                )

        spec.loader.exec_module = _exec_then_patch
        return spec


def apply_deferred() -> bool:
    """Install the deferred patcher: defer :func:`apply` until vLLM is ready."""
    global _DEFERRED_HOOK_INSTALLED

    if _DEFERRED_HOOK_INSTALLED:
        return True

    if _TRIGGER_MODULE in sys.modules:
        try:
            apply()
        except Exception:
            logger.warning(
                "[zentorch] GatedDeltaNet immediate apply failed",
                exc_info=True,
            )
            return False
    else:
        sys.meta_path.insert(0, _GatedDeltaNetImportHook())
        logger.debug(
            "[zentorch] Installed GatedDeltaNet deferred-patch import hook"
        )

    _DEFERRED_HOOK_INSTALLED = True
    return True


def is_applied() -> bool:
    return bool(PATCHES)


def apply() -> None:
    """Install the ``forward_cpu`` override. Idempotent + re-entrancy safe."""
    global _APPLY_IN_PROGRESS

    if is_applied() or _APPLY_IN_PROGRESS:
        return
    _APPLY_IN_PROGRESS = True

    try:
        _do_apply()
    finally:
        _APPLY_IN_PROGRESS = False


def _do_apply() -> None:
    from . import forward as _gdn_forward
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
    )

    target = GatedDeltaNetAttention
    name = "forward_cpu"
    key = (_target_dotted_name(target), name)
    PATCHES[key] = getattr(target, name, None)
    _TARGETS[key] = target
    setattr(target, name, _gdn_forward.forward_cpu_zen)
    logger.info(
        "[zentorch] Installed GatedDeltaNetAttention.forward_cpu -> forward_cpu_zen"
    )


def restore() -> None:
    """Undo the patch installed by :func:`apply`. Idempotent."""
    global _DEFERRED_HOOK_INSTALLED

    sys.meta_path[:] = [
        h for h in sys.meta_path if not isinstance(h, _GatedDeltaNetImportHook)
    ]
    _DEFERRED_HOOK_INSTALLED = False

    if not is_applied():
        return

    for key, original in PATCHES.items():
        target = _TARGETS.get(key)
        if target is None:
            continue
        attr = key[1]
        if original is None:
            with contextlib.suppress(AttributeError):
                delattr(target, attr)
        else:
            setattr(target, attr, original)

    PATCHES.clear()
    _TARGETS.clear()
