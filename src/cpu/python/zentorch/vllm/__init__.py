# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM - zentorch integration using the plugin pattern.

Entry points:
- vllm.platform_plugins -> returns ZenCPUPlatform class path
- vllm.general_plugins  -> applies early patches

Patches (each with version decorator):
- PagedAttention (0.11 only)
- IPEX flash attention (0.11 only)
- CompilationConfig repr (all versions)
- oneDNN disable (all versions)
- InternVL dtype fix (all versions)

Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from zentorch._logging import get_logger
from zentorch.vllm.core import (
    vllm_version,
    vllm_version_range,
    manager,
    get_version_family,
    is_v12,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patches (0.11 only)
# ---------------------------------------------------------------------------


@vllm_version("0.11.0", "0.11.1", "0.11.2")
class PagedAttentionPatch:
    """Replace vLLM's PagedAttention with zentorch implementation."""

    @classmethod
    def apply(cls) -> bool:
        try:
            import vllm.v1.attention.backends.cpu_attn as cpu_attn_module
            from zentorch.vllm.attention import PagedAttention

            cpu_attn_module._get_paged_attn_impl = lambda: PagedAttention
            logger.info("[zentorch] Patched PagedAttention")
            return True
        except ImportError:
            logger.debug("[zentorch] PagedAttention patch deferred")
            return False


@vllm_version("0.11.0", "0.11.1", "0.11.2")
class IPEXFlashAttentionPatch:
    """Replace IPEX flash_attn_varlen_func with zentorch."""

    @classmethod
    def apply(cls) -> bool:
        if os.environ.get("DISABLE_ZENTORCH_FLASH_ATTENTION_VARLEN", "0") == "1":
            logger.info("[zentorch] IPEX patch disabled via env")
            return False

        try:
            import intel_extension_for_pytorch.llm.modules as ipex_modules
            from zentorch.vllm.attention import PagedAttention

            ipex_modules.PagedAttention.flash_attn_varlen_func = (
                PagedAttention.flash_attn_varlen_func
            )
            logger.info("[zentorch] Patched IPEX flash_attn_varlen_func")
            return True
        except ImportError:
            logger.debug("[zentorch] IPEX patch deferred")
            return False


# ---------------------------------------------------------------------------
# Patches (all versions)
# ---------------------------------------------------------------------------


@vllm_version_range(min_ver="0.11.0", max_ver="0.13.99")
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


@vllm_version_range(min_ver="0.11.0", max_ver="0.13.99")
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


@vllm_version_range(min_ver="0.11.0", max_ver="0.13.99")
class InternVLDtypePatch:
    """Fix InternVL video dtype issue."""

    @classmethod
    def apply(cls) -> bool:
        try:
            import numpy as np
            from vllm.multimodal import profiling as profiling_module

            if hasattr(profiling_module.BaseDummyInputsBuilder, "_zentorch_patched"):
                return True

            def patched_get_dummy_videos(self, width, height, num_frames, num_videos):
                if num_videos == 0:
                    return []
                return [
                    np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
                ] * num_videos

            profiling_module.BaseDummyInputsBuilder._get_dummy_videos = (
                patched_get_dummy_videos
            )
            profiling_module.BaseDummyInputsBuilder._zentorch_patched = True
            logger.info("[zentorch] Patched InternVL dtype")
            return True
        except ImportError:
            return False


@vllm_version("0.11.0", "0.11.1", "0.11.2")
class CPUProfilerPatch:
    """Patch Worker profiler for CPU-only profiling (0.11 only)."""

    @classmethod
    def apply(cls) -> bool:
        import torch

        try:
            import vllm.envs as envs
            import vllm.v1.worker.gpu_worker as worker_module
        except ImportError:
            return False

        if hasattr(worker_module.Worker, "_zentorch_profiler_patched"):
            return True

        Worker = worker_module.Worker
        orig_init = Worker.__init__

        def patched_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            if self.profiler:
                record_shapes = getattr(
                    envs, "VLLM_TORCH_PROFILER_RECORD_SHAPES", False
                )
                profile_memory = getattr(
                    envs, "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY", False
                )
                with_stack = getattr(envs, "VLLM_TORCH_PROFILER_WITH_STACK", False)
                with_flops = getattr(envs, "VLLM_TORCH_PROFILER_WITH_FLOPS", False)
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    with_flops=with_flops,
                )

        def patched_profile(self, is_start=True):
            if not self.profiler:
                raise RuntimeError("Profiler not enabled")
            if is_start:
                self.profiler.start()
            else:
                self.profiler.stop()
                logger.info(
                    "\n%s",
                    self.profiler.key_averages().table(sort_by="self_cpu_time_total"),
                )

        Worker.__init__ = patched_init
        Worker.profile = patched_profile
        Worker._zentorch_profiler_patched = True
        logger.info("[zentorch] Patched Worker profiler (0.11)")

        return True


@vllm_version("0.12.0")
class CPUProfilerPatchV12:
    """Stub: Actual patching happens in platform.py check_and_update_config."""

    @classmethod
    def apply(cls) -> bool:
        # Patching is done in platform.py's check_and_update_config()
        # which is called in every process (main and subprocess)
        return True


@vllm_version("0.13.0")
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

    manager.register("PagedAttention", PagedAttentionPatch)
    manager.register("IPEXFlashAttention", IPEXFlashAttentionPatch)
    manager.register("CompilationConfigRepr", CompilationConfigReprPatch)
    manager.register("OneDNNDisable", OneDNNDisablePatch)
    manager.register("InternVLDtype", InternVLDtypePatch)
    manager.register("CPUProfiler", CPUProfilerPatch)
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
            "[zentorch] Unsupported vLLM %s. Supports: 0.11.x, 0.12.x, 0.13.x",
            vllm_ver,
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
