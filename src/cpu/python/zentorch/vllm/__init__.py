# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM - zentorch integration

This module is discovered via *both* platform & general plugin entry-points.
Platform:  ``vllm.platform_plugins``  ->  returns dotted path to
            :class:`ZenCPUPlatform`
General :  ``vllm.general_plugins``   ->  executes light-weight runtime.
            Applies early monkey-patches for:
            - PagedAttention (zentorch implementation)
            - CompilationConfig repr (pydantic serialization fix)
            - oneDNN GEMM disable (use zentorch GEMM instead)
            - CPU profiler (remove CUDA dependencies)
            - InternVL video input dtype bug fix

Supported vLLM versions:
- 0.11.0
- 0.11.1.dev0+gb8b302cde.d20251203.cpu
"""

from __future__ import annotations

from typing import Optional
import os
import sys

from zentorch._logging import get_logger

logger = get_logger(__name__)

SUPPORTED_VLLM_VERSIONS = {
    "0.11.0",
    "0.11.1.dev0+gb8b302cde.d20251203.cpu",
}

# Track if we've already logged circular import warnings
_logged_circular_import_warning = False

# ---------------------------------------------------------------------------
# Plugin entry-points
# ---------------------------------------------------------------------------


def register() -> Optional[str]:  # noqa: D401
    """Entry-point for *both* platform & general plugin groups.

    If vLLM calls this via the *platform* group, we must **return** the dotted
    class path so that it can import the platform.  When executed via the
    *general* group the return value is ignored - that is fine.
    """
    if "vllm" not in sys.modules:
        logger.warning(
            "[zentorch] vllm not found in sys.modules. "
            "The plugin should be loaded by vLLM. "
            "zentorch platform will not be available."
        )
        return None

    vllm_module = sys.modules["vllm"]
    vllm_version = getattr(vllm_module, "__version__", None)

    if vllm_version is None:
        logger.warning(
            "[zentorch] Could not determine vLLM version. "
            "zentorch platform will not be available."
        )
        return None

    if vllm_version not in SUPPORTED_VLLM_VERSIONS:
        logger.warning(
            "[zentorch] Unsupported vLLM version: %s. "
            "Plugin supports versions: %s.",
            vllm_version,
            ", ".join(sorted(SUPPORTED_VLLM_VERSIONS)),
        )
        return None

    # Apply monkey-patches early (runs in all processes)
    _apply_paged_attention_monkey_patch()

    # Monkey-patch IPEX's flash_attn_varlen_func with zentorch's implementation
    # Set DISABLE_ZENTORCH_FLASH_ATTENTION_VARLEN=1 to use IPEX's native implementation
    if os.environ.get("DISABLE_ZENTORCH_FLASH_ATTENTION_VARLEN", "0") != "1":
        _apply_ipex_flash_attention_monkey_patch()
    else:
        logger.info("[zentorch] Skipping flash_attn_varlen_func patch (DISABLE_ZENTORCH_FLASH_ATTENTION_VARLEN=1)")

    _apply_compilation_config_repr_patch()
    _apply_internvl_video_input_dtype_bug_fix()

    # Critical: oneDNN must be disabled for zentorch to work correctly
    if not _disable_onednn_gemm():
        logger.error(
            "[zentorch] Failed to disable oneDNN GEMM. "
            "Plugin cannot function correctly with oneDNN enabled."
        )
        return None

    return "zentorch.vllm.platform.ZenCPUPlatform"


def _apply_ipex_flash_attention_monkey_patch():
    """
    Monkey-patch IPEX's flash_attn_varlen_func with ZenTorch's implementation.
    """
    import intel_extension_for_pytorch.llm.modules as ipex_modules
    from zentorch.vllm.attention import PagedAttention

    # Replace IPEX's flash_attn_varlen_func with ZenTorch's
    ipex_modules.PagedAttention.flash_attn_varlen_func = (
        PagedAttention.flash_attn_varlen_func
    )

    logger.info(
        "[zentorch] Monkey-patched IPEX flash_attn_varlen_func with ZenTorch implementation"
    )


def _apply_paged_attention_monkey_patch():
    """
    Monkey-patch vLLM's PagedAttention implementation with zentorch's.

    This is called early during plugin registration to ensure that all
    subsequent imports of vLLM modules get the zentorch implementation.
    """
    try:
        import vllm.v1.attention.backends.cpu_attn as cpu_attn_module
        from zentorch.vllm.attention import PagedAttention

        def _get_zentorch_paged_attn_impl():
            return PagedAttention

        cpu_attn_module._get_paged_attn_impl = _get_zentorch_paged_attn_impl

        logger.info(
            "[zentorch] Monkey-patched vLLM PagedAttention with zentorch implementation"
        )

    except ImportError:
        # Deferred - will be applied in platform.py
        global _logged_circular_import_warning
        if not _logged_circular_import_warning:
            logger.debug("[zentorch] PagedAttention patch deferred")
            _logged_circular_import_warning = True
    except Exception:
        logger.exception("[zentorch] Failed to patch PagedAttention")


def _apply_compilation_config_repr_patch():
    """
    Monkey-patch CompilationConfig.__repr__ early to prevent serialization errors.
    """
    try:
        from vllm.config import CompilationConfig
        from pydantic import TypeAdapter
        from vllm.config.compilation import PassConfig

        if hasattr(CompilationConfig.__repr__, "_zentorch_patched"):
            logger.info("[zentorch] CompilationConfig.__repr__ already patched")
            return

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
                try:
                    return (
                        f"CompilationConfig("
                        f"level={getattr(self, 'level', '?')}, "
                        f"backend={getattr(self, 'backend', '?')!r}, "
                        f"use_inductor={getattr(self, 'use_inductor', '?')}, "
                        f"custom_ops={getattr(self, 'custom_ops', '?')!r}"
                        f")"
                    )
                except Exception as e:
                    return f"CompilationConfig(<error during repr: {e}>)"

        patched_repr._zentorch_patched = True

        CompilationConfig.__repr__ = patched_repr
        CompilationConfig.__str__ = patched_repr
        logger.info(
            "[zentorch] Patched CompilationConfig.__repr__ to handle custom passes"
        )

    except ImportError:
        # Deferred - will be applied later
        global _logged_circular_import_warning
        if not _logged_circular_import_warning:
            logger.debug("[zentorch] CompilationConfig repr patch deferred")
            _logged_circular_import_warning = True
    except Exception:
        logger.exception("[zentorch] Failed to patch CompilationConfig.__repr__")


def _disable_onednn_gemm() -> bool:
    """
    Disable oneDNN GEMM in vLLM to use zentorch.mm() instead.

    This ensures zentorch optimizations are used instead of native oneDNN.

    Returns:
        True if successfully disabled, False otherwise.
    """
    try:
        import vllm._custom_ops as ops

        ops._supports_onednn = False

        logger.info("[zentorch] Disabled oneDNN GEMM to use zentorch.mm() instead")
        return True

    except ImportError:
        # Module not available yet - will be retried in platform.py
        logger.debug("[zentorch] oneDNN disable deferred (module not loaded)")
        return True  # Not a failure, just deferred
    except Exception:
        logger.exception("[zentorch] Failed to disable oneDNN GEMM")
        return False


def _apply_internvl_video_input_dtype_bug_fix():
    """
    Patch vLLM's multimodal profiling to fix InternVL video dtype issue.

    The original code creates dummy videos without specifying dtype, which
    causes issues with InternVL models. This patch adds dtype=np.uint8.

    Bug fix: np.full((num_frames, width, height, 3), 255) ->
             np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
    """
    try:
        import numpy as np
        from vllm.multimodal import profiling as profiling_module

        if hasattr(
            profiling_module.BaseDummyInputsBuilder, "_zentorch_internvl_patched"
        ):
            return

        def patched_get_dummy_videos(
            self,
            width: int,
            height: int,
            num_frames: int,
            num_videos: int,
        ):
            if num_videos == 0:
                return []
            video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
            return [video] * num_videos

        profiling_module.BaseDummyInputsBuilder._get_dummy_videos = (
            patched_get_dummy_videos
        )
        profiling_module.BaseDummyInputsBuilder._zentorch_internvl_patched = True

        logger.info("[zentorch] Patched multimodal profiling for InternVL video dtype")

    except ImportError:
        logger.debug("[zentorch] InternVL patch deferred (module not loaded)")
    except Exception:
        logger.exception("[zentorch] Failed to apply InternVL video dtype patch")
