# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM – ZenTorch integration

This module is discovered via *both* platform & general plugin entry-points.
Platform:  ``vllm.platform_plugins``  –>  returns dotted path to
            :class:`ZenCPUPlatform` so vLLM treats ZenTorch as a new device.
General :  ``vllm.general_plugins``   –>  executes light-weight runtime.
                For now we patch PagedAttention
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from importlib import import_module
from typing import Optional
import sys

from packaging.version import parse as V

from zentorch._logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional PagedAttention replacement
# ---------------------------------------------------------------------------

_PAGED_ATTN_MODULES = {
    "vllm.attention.ops.ipex_attn"
}


def _maybe_get_zt_paged_attn():
    try:
        return import_module("zentorch.vllm.zentorch_attention").PagedAttention
    except Exception as exc:  # pragma: no-cover – best effort
        logger.debug("[zentorch] No ZenTorch PagedAttention found", exc_info=exc)
        return None


def _replace_paged_attention(module: ModuleType, zt_cls) -> None:  # noqa: D401
    if not hasattr(module, "PagedAttention"):
        return
    if module.PagedAttention is zt_cls:
        return
    orig_cls = module.PagedAttention  # type: ignore[attr-defined]
    module.PagedAttention = zt_cls  # type: ignore[assignment]

    # Update already-imported refs
    for m in list(sys.modules.values()):
        if m is None:
            continue
        for attr in dir(m):
            with contextlib.suppress(Exception):
                if getattr(m, attr) is orig_cls:
                    setattr(m, attr, zt_cls)


def _apply_paged_attention_patch() -> None:  # noqa: D401
    zt_cls = _maybe_get_zt_paged_attn()
    if zt_cls is None:
        return

    # Patch already-loaded modules first
    for name in _PAGED_ATTN_MODULES:
        mod = sys.modules.get(name)
        if mod is not None:
            _replace_paged_attention(mod, zt_cls)

    # Install import-hook for future imports
    orig_import = __import__

    def _hook(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: D401
        mod = orig_import(name, globals, locals, fromlist, level)
        if name in _PAGED_ATTN_MODULES and isinstance(mod, ModuleType):
            _replace_paged_attention(mod, zt_cls)
        return mod

    # Check explicitly if torch.compiler and allow_in_graph exist
    # instead of suppressing all exceptions.
    import importlib
    try:
        _tc = importlib.import_module("torch.compiler")
        if hasattr(_tc, "allow_in_graph"):
            # Ensure both the custom hook and the original import are graph-safe
            _tc.allow_in_graph(_hook)
            _tc.allow_in_graph(orig_import)
    except ImportError:
        # torch.compiler doesn't exist (e.g., PyTorch < 2.0), nothing to do.
        pass

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = _hook  # type: ignore[index]
    else:
        __builtins__.__import__ = _hook  # type: ignore[attr-defined]

    logger.info("[zentorch] PagedAttention patch installed")


# ---------------------------------------------------------------------------
# Plugin entry-points
# ---------------------------------------------------------------------------


def register() -> Optional[str]:  # noqa: D401
    """Entry-point for *both* platform & general plugin groups.

    If vLLM calls this via the *platform* group, we must **return** the dotted
    class path so that it can import the platform.  When executed via the
    *general* group the return value is ignored – that is fine.
    """
    if "vllm" not in sys.modules:
        logger.warning(
            "[zentorch] vllm not found in sys.modules. "
            "The plugin should be loaded by vLLM. "
            "ZenTorch platform will not be available."
        )
        return None

    vllm_module = sys.modules["vllm"]
    vllm_version = getattr(vllm_module, "__version__", None)

    if vllm_version is None:
        logger.warning(
            "[zentorch] Could not determine vLLM version. "
            "ZenTorch platform will not be available."
        )
        return None

    # NOTE: zentorch-vllm plugin requires vLLM>=0.9.0
    if V(vllm_version) < V("0.9.0"):
        logger.warning(
            "[zentorch] Mismatched vLLM version. "
            "Found v%s, expected v0.9.0 or greater.",
            vllm_version,
        )
        return None

    # Light-weight runtime patch (safe to execute multiple times)
    _apply_paged_attention_patch()

    # Ensure the custom torch.compile backend is registered so that
    # `@torch.compile(backend=current_platform.simple_compile_backend)` can
    # resolve to "zentorch".  Importing the module has the side-effect of
    # calling `torch._dynamo.register_backend("zentorch", ...)`.
    try:
        import importlib

        importlib.import_module("zentorch._compile_backend")
        # Backend successfully imported, provide the platform class path to vLLM
        logger.debug("[zentorch] Compile backend initialised successfully.")
        return "zentorch.vllm.platform.ZenCPUPlatform"
    except Exception as exc:  # pragma: no-cover – best effort
        logger.warning(
            "[zentorch] Failed to initialise compile backend. "
            "ZenTorch platform will not be available.",
            exc_info=exc,
        )
        # Return None to signal that the platform registration failed.
        return None
