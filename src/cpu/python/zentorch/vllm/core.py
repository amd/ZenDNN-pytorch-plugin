# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Core patching infrastructure for zentorch vLLM plugin.

- @vllm_version(): Decorator for version-specific patches
- PatchManager: Register and apply patches

Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Type

from zentorch._logging import get_logger
logger = get_logger(__name__)

# Version -> family mapping
_VERSION_MAP = {
    "0.11.0": "v11",
    "0.11.1": "v11",
    "0.11.2": "v11",
    "0.12.0": "v12",
    "0.13.0": "v13",
}


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def get_vllm_version() -> Optional[str]:
    """Get current vLLM version."""
    if "vllm" not in sys.modules:
        return None
    return getattr(sys.modules["vllm"], "__version__", None)


def _base_version(ver: str) -> str:
    """Strip dev/rc/local suffixes: '0.12.0.dev1+cpu' -> '0.12.0'"""
    return ver.split("+")[0].split(".dev")[0].split("rc")[0]


def get_version_family() -> Optional[str]:
    """Return 'v11', 'v12', 'v13' or None."""
    ver = get_vllm_version()
    if ver is None:
        return None
    return _VERSION_MAP.get(_base_version(ver))


def is_v11() -> bool:
    return get_version_family() == "v11"


def is_v12() -> bool:
    return get_version_family() == "v12"


def is_v13() -> bool:
    return get_version_family() == "v13"


# ---------------------------------------------------------------------------
# Version decorators
# ---------------------------------------------------------------------------


def vllm_version(*versions: str) -> Callable[[Type], Type]:
    """Decorator: apply patch only for specific vLLM versions.

    Usage:
        @vllm_version("0.11.0", "0.11.1")
        class MyPatch:
            pass
    """

    def decorator(patch_cls: Type) -> Type:
        original_apply = patch_cls.apply

        @classmethod
        def versioned_apply(cls) -> bool:
            current = get_vllm_version()
            if current is None:
                return False

            base = _base_version(current)
            if base not in versions and current not in versions:
                logger.debug("Skipping %s: not for %s", cls.__name__, current)
                return False

            return original_apply.__func__(cls)

        patch_cls.apply = versioned_apply
        patch_cls._target_versions = set(versions)
        return patch_cls

    return decorator


def vllm_version_range(
    min_ver: Optional[str] = None, max_ver: Optional[str] = None
) -> Callable[[Type], Type]:
    """Decorator: apply patch for a version range.

    Usage:
        @vllm_version_range(min_ver="0.11.0", max_ver="0.13.99")
        class MyPatch:
            pass
    """
    from packaging import version as pkg_version

    def decorator(patch_cls: Type) -> Type:
        original_apply = patch_cls.apply

        @classmethod
        def versioned_apply(cls) -> bool:
            current_str = get_vllm_version()
            if current_str is None:
                return False

            current = pkg_version.parse(current_str.split("+")[0])

            if min_ver and current < pkg_version.parse(min_ver):
                logger.debug("Skipping %s: requires >= %s", cls.__name__, min_ver)
                return False

            if max_ver and current > pkg_version.parse(max_ver):
                logger.debug("Skipping %s: requires <= %s", cls.__name__, max_ver)
                return False

            return original_apply.__func__(cls)

        patch_cls.apply = versioned_apply
        return patch_cls

    return decorator


# ---------------------------------------------------------------------------
# Patch Manager
# ---------------------------------------------------------------------------


class PatchManager:
    """Manages registration and application of vLLM patches."""

    def __init__(self):
        self.patches = {}
        self.applied = []

    def register(self, name: str, patch_cls: Type) -> None:
        """Register a patch by name."""
        self.patches[name] = patch_cls

    def apply(self, name: str) -> bool:
        """Apply a single patch by name."""
        if name not in self.patches:
            logger.error("Unknown patch: %s", name)
            return False

        result = self.patches[name].apply()
        if result:
            self.applied.append(name)
        return result

    def apply_all(self) -> None:
        """Apply all registered patches (version decorators filter)."""
        for name in self.patches:
            self.apply(name)


# Global manager
manager = PatchManager()
