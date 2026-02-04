# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM - zentorch integration using the plugin pattern.

Entry points:
- vllm.platform_plugins -> returns ZenCPUPlatform class path
- vllm.general_plugins  -> applies early patches

Patches (each with version decorator):
- CompilationConfig repr (all versions)
- oneDNN disable (all versions)

Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""

from __future__ import annotations

import sys
from typing import Optional
import torch

from zentorch._logging import get_logger
from zentorch.vllm.core import (
    vllm_version,
    vllm_version_range,
    manager,
    get_version_family,
    is_v12,
    VLLM_MIN_VERSION,
    VLLM_MAX_VERSION,
    VLLM_V12,
    VLLM_V13,
    VLLM_V14,
    VLLM_V14_1,
)


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patches (all versions)
# ---------------------------------------------------------------------------


@vllm_version_range(min_ver=VLLM_MIN_VERSION, max_ver=VLLM_MAX_VERSION)
class CompilationConfigReprPatch:
    """Fix CompilationConfig repr for pydantic serialization."""

    @classmethod
    def apply(cls) -> bool:
        try:
            from vllm.config import CompilationConfig
            from pydantic import TypeAdapter
            from vllm.config.compilation import PassConfig

            if hasattr(CompilationConfig.__repr__, "_zentorch_patched"):
                return True

            def patched_repr(self):
                exclude = {
                    "static_forward_context": True,
                    "enabled_custom_ops": True,
                    "disabled_custom_ops": True,
                    "compilation_time": True,
                    "bs_to_padded_graph_size": True,
                    "traced_files": True,
                    "inductor_compile_config": {
                        "post_grad_custom_post_pass": True,
                        "joint_custom_post_pass": True,
                        "joint_custom_pre_pass": True,
                    },
                }

                pass_config_exclude = {}
                try:
                    for attr, default_val in vars(PassConfig()).items():
                        if getattr(self.pass_config, attr) == default_val:
                            pass_config_exclude[attr] = True
                    if pass_config_exclude:
                        exclude["pass_config"] = pass_config_exclude
                except Exception:
                    pass

                try:
                    return (
                        TypeAdapter(CompilationConfig)
                        .dump_json(self, exclude=exclude, exclude_unset=True)
                        .decode()
                    )
                except Exception:
                    mode = getattr(self, "mode", getattr(self, "level", "?"))
                    return f"CompilationConfig(mode={mode}, backend={self.backend!r})"

            patched_repr._zentorch_patched = True
            CompilationConfig.__repr__ = patched_repr
            CompilationConfig.__str__ = patched_repr
            logger.info("[zentorch] Patched CompilationConfig repr")
            return True
        except ImportError:
            return False


@vllm_version_range(min_ver=VLLM_MIN_VERSION, max_ver=VLLM_MAX_VERSION)
class OneDNNDisablePatch:
    """Disable oneDNN GEMM to use zentorch.zentorch_linear_unary()."""

    @classmethod
    def apply(cls) -> bool:
        try:
            import vllm._custom_ops as ops

            ops._supports_onednn = False
            logger.info("[zentorch] Disabled oneDNN GEMM")
            return True
        except ImportError:
            return False


@vllm_version_range(min_ver=VLLM_MIN_VERSION, max_ver=VLLM_MAX_VERSION)
class DispatchCPUUnquantizedGemmPatch:
    """Fix Freezing issue by overwriting dispatch_cpu_unquantized_gemm."""

    @classmethod
    def apply(cls) -> bool:
        def patched_dispatch_cpu_unquantized_gemm(
            layer: torch.nn.Module,
            remove_weight: bool,
        ) -> None:
            weights_copy = layer.weight.detach()
            layer.cpu_linear = (
                lambda x, weight, bias: torch.ops.zentorch.zentorch_linear_unary(
                    x, weights_copy, bias, is_weight_prepacked=False
                )
            )
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return

        try:
            from vllm.model_executor.layers import utils

            utils.dispatch_cpu_unquantized_gemm = patched_dispatch_cpu_unquantized_gemm
            logger.info(
                "[zentorch] Successfully overwritten dispatch_cpu_unquantized_gemm"
            )
            return True
        except ImportError:
            logger.debug("[zentorch] Failed to overwrite dispatch_cpu_unquantized_gemm")
            return False


@vllm_version(VLLM_V12)
class CPUProfilerPatchV12:
    """Stub: Actual patching happens in platform.py check_and_update_config."""

    @classmethod
    def apply(cls) -> bool:
        # Patching is done in platform.py's check_and_update_config()
        # which is called in every process (main and subprocess)
        return True


@vllm_version(VLLM_V13, VLLM_V14, VLLM_V14_1)
class CPUProfilerPatchV13:
    """Stub: Actual patching happens in platform.py check_and_update_config."""

    @classmethod
    def apply(cls) -> bool:
        # Patching is done in platform.py's check_and_update_config()
        return True


# ---------------------------------------------------------------------------
# vLLM 0.12 Profiler Fix (Post-import hook)
# ---------------------------------------------------------------------------

_V12_PROFILER_HOOK_INSTALLED = False


def _do_patch_cpuworker():
    """Actually patch CPUWorker class. Called when module is available."""
    import torch

    try:
        import vllm.v1.worker.cpu_worker as cpu_worker_module
    except ImportError:
        return False

    CPUWorker = cpu_worker_module.CPUWorker
    if hasattr(CPUWorker, "_zentorch_profiler_patched"):
        return True  # Already patched

    orig_init = CPUWorker.__init__

    class _CPUProfilerWrapper:
        """Wrapper to give raw profiler the TorchProfilerWrapper interface."""

        def __init__(self, raw_profiler):
            self._profiler = raw_profiler
            self._running = False

        def start(self):
            self._profiler.start()
            self._running = True

        def stop(self):
            self._profiler.stop()
            self._running = False

        def step(self):
            pass  # Not needed for CPU profiling

        def shutdown(self):
            if self._running:
                self.stop()

        def annotate_context_manager(self, name: str):
            return torch.profiler.record_function(name)

        def key_averages(self):
            return self._profiler.key_averages()

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Wrap raw profiler with TorchProfilerWrapper-compatible interface
        if self.profiler is not None:
            self.profiler = _CPUProfilerWrapper(self.profiler)

    CPUWorker.__init__ = patched_init
    CPUWorker._zentorch_profiler_patched = True
    logger.info("[zentorch] Patched CPUWorker profiler for 0.12")
    return True


class _V12ProfilerImportHook:
    """Intercept imports to patch CPUWorker when cpu_worker is imported."""

    def find_module(self, fullname, path=None):
        if fullname == "vllm.v1.worker.cpu_worker":
            return self
        return None

    def load_module(self, fullname):
        # Remove ourselves to avoid recursion
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        # Let Python do the actual import
        import importlib

        module = importlib.import_module(fullname)

        # Now patch the class
        _do_patch_cpuworker()

        return module


def _apply_profiler_patch_v12():
    """Fix vLLM 0.12 bug: CPUWorker uses raw torch.profiler.profile but
    inherits methods expecting TorchProfilerWrapper interface.

    Installs a meta_path hook to patch CPUWorker when it's imported.
    """
    global _V12_PROFILER_HOOK_INSTALLED

    if not is_v12():
        return

    if _V12_PROFILER_HOOK_INSTALLED:
        return

    # Check if module is already imported
    if "vllm.v1.worker.cpu_worker" in sys.modules:
        _do_patch_cpuworker()
    else:
        # Install import hook to patch when module is imported
        sys.meta_path.insert(0, _V12ProfilerImportHook())
        logger.debug("[zentorch] Installed v12 profiler import hook")

    _V12_PROFILER_HOOK_INSTALLED = True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_REGISTERED = False


def _register_patches():
    """Register all patches with the manager (only once)."""
    global _REGISTERED
    if _REGISTERED:
        return

    manager.register("CompilationConfigRepr", CompilationConfigReprPatch)
    manager.register("OneDNNDisable", OneDNNDisablePatch)
    manager.register("DispatchCPUUnquantizedGemm", DispatchCPUUnquantizedGemmPatch)
    manager.register("CPUProfilerV12", CPUProfilerPatchV12)
    manager.register("CPUProfilerV13", CPUProfilerPatchV13)

    _REGISTERED = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_INITIALIZED = False


def register() -> Optional[str]:
    """Entry-point for vllm.platform_plugins and vllm.general_plugins.

    This is called multiple times by vLLM (both entry points). We only
    apply patches once but always return the platform class path.
    """
    global _INITIALIZED

    if "vllm" not in sys.modules:
        logger.warning("[zentorch] vllm not loaded")
        return None

    vllm_ver = getattr(sys.modules["vllm"], "__version__", None)
    family = get_version_family()

    if family is None:
        logger.warning(
            "[zentorch] Unsupported vLLM %s. Supports: %s, %s, %s, %s",
            vllm_ver,
            VLLM_V12,
            VLLM_V13,
            VLLM_V14,
            VLLM_V14_1,
        )
        return None

    # Only initialize once per process
    if not _INITIALIZED:
        logger.info("[zentorch] vLLM %s detected (family: %s)", vllm_ver, family)

        # Register and apply all patches (decorators handle version filtering)
        _register_patches()
        manager.apply_all()

        # Apply profiler patches early (before worker creation)
        # Must be done here in register(), not in check_and_update_config(),
        # because VllmConfig may be passed from main process (not recreated)
        _apply_profiler_patch_v12()

        logger.info("[zentorch] Applied patches: %s", manager.applied)
        _INITIALIZED = True

    return "zentorch.vllm.platform.ZenCPUPlatform"
