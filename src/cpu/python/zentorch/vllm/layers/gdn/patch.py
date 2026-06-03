# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Monkey-patcher: replace ``GatedDeltaNetAttention.forward_cpu`` with
``forward_cpu_zen``."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
from typing import Any

from packaging import version as pkg_version

from zentorch._logging import get_logger
from zentorch.vllm._core import _base_version, get_vllm_version

logger = get_logger(__name__)

__all__ = [
    "apply",
    "apply_deferred",
    "restore",
    "is_applied",
    "PATCHES",
]


# vLLM 0.22 restructured GatedDeltaNet: the module moved under ``mamba.gdn``
# and the concrete class was renamed ``QwenGatedDeltaNetAttention``. Support
# both layouts additively (newest first) so 0.21 behavior is unchanged.
_GDN_CANDIDATES = (
    (
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "QwenGatedDeltaNetAttention",
    ),
    (
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "GatedDeltaNetAttention",
    ),
)

_resolved_target: tuple[str, str] | None = None


def _resolve_gdn_target() -> tuple[str, str]:
    """Return the ``(module_name, class_name)`` for the running vLLM version.

    Gated on the vLLM version string (no eager submodule imports, which keeps
    plugin registration side-effect free): vLLM >= 0.22.0 uses the
    restructured ``mamba.gdn.qwen_gdn_linear_attn`` / ``QwenGatedDeltaNetAttention``
    layout; older versions use the legacy ``mamba.gdn_linear_attn`` /
    ``GatedDeltaNetAttention`` layout. Cached after first resolution.
    """
    global _resolved_target
    if _resolved_target is not None:
        return _resolved_target

    ver = get_vllm_version()
    use_new_layout = False
    if ver is not None:
        try:
            use_new_layout = pkg_version.parse(_base_version(ver)) >= pkg_version.parse(
                "0.22.0"
            )
        except Exception:
            use_new_layout = False

    _resolved_target = _GDN_CANDIDATES[0] if use_new_layout else _GDN_CANDIDATES[1]
    return _resolved_target


def _trigger_module() -> str:
    return _resolve_gdn_target()[0]


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
        if fullname != _trigger_module():
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
                    _trigger_module(),
                    exc_info=True,
                )

        spec.loader.exec_module = _exec_then_patch
        return spec


def apply_deferred() -> bool:
    """Install the deferred patcher: defer :func:`apply` until vLLM is ready."""
    global _DEFERRED_HOOK_INSTALLED

    if _DEFERRED_HOOK_INSTALLED:
        return True

    if _trigger_module() in sys.modules:
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

    mod_name, cls_name = _resolve_gdn_target()
    module = importlib.import_module(mod_name)
    target = getattr(module, cls_name)
    name = "forward_cpu"
    key = (_target_dotted_name(target), name)
    PATCHES[key] = getattr(target, name, None)
    _TARGETS[key] = target
    setattr(target, name, _gdn_forward.forward_cpu_zen)
    logger.info(
        "[zentorch] Installed %s.forward_cpu -> forward_cpu_zen", cls_name
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
