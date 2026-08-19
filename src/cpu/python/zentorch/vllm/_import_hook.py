# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Run a patch function immediately after its target vLLM module imports."""

from __future__ import annotations

import importlib.util
import sys
from typing import Callable

from zentorch._logging import get_logger

logger = get_logger(__name__)

_handled: set[str] = set()


def _run_patch(module_name: str, patch_fn: Callable[[], bool]) -> bool:
    """Run an optional patch without turning failure into an import failure."""
    try:
        return patch_fn()
    except Exception:
        logger.warning(
            "[zentorch] Patch for %s failed; continuing without it",
            module_name,
            exc_info=True,
        )
        return False


class _PostImportPatcher:
    """meta_path finder that runs ``patch_fn`` once ``module_name`` has loaded."""

    def __init__(self, module_name: str, patch_fn: Callable[[], bool]) -> None:
        self._module_name = module_name
        self._patch_fn = patch_fn

    def find_spec(self, fullname, path, target=None):
        if fullname != self._module_name:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _run_patch(self._module_name, self._patch_fn)

        spec.loader.exec_module = _exec_then_patch
        return spec


def patch_now_or_on_import(module_name: str, patch_fn: Callable[[], bool]) -> bool:
    """Apply ``patch_fn`` now if ``module_name`` is loaded, else defer to import.

    Returns ``patch_fn``'s result when applied immediately, otherwise ``True``.
    Repeated calls for the same module are no-ops.
    """
    if module_name in _handled:
        return True
    _handled.add(module_name)

    if module_name in sys.modules:
        return _run_patch(module_name, patch_fn)
    sys.meta_path.insert(0, _PostImportPatcher(module_name, patch_fn))
    return True
