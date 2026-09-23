# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Backport Qwen3-VL inner-text architectures for vLLM 0.29.

vLLM 0.29 rebuilds the model config for Qwen3-VL's nested text model without
supplying an architecture. On CPU, ``CpuPlatform.check_and_update_config``
then indexes the empty architecture list while checking MLA support. Upstream
fixed this after 0.29 by passing the dense/MoE causal-LM architecture
explicitly at each Qwen3-VL construction site.

This compatibility shim applies the same mapping centrally when
``VllmConfig.with_hf_config`` receives either affected nested text config.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

from packaging import version as pkg_version

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.config.vllm"
_MARKER = "_zentorch_qwen3_vl_text_config_patched"
_TEXT_ARCHITECTURES = {
    "qwen3_vl_text": ["Qwen3ForCausalLM"],
    "qwen3_vl_moe_text": ["Qwen3MoeForCausalLM"],
}


def _make_patched_with_hf_config(
    original: Callable,
) -> Callable:
    """Inject the missing architecture for Qwen3-VL nested text configs."""

    def patched(self, hf_config, architectures=None):
        if architectures is None and not getattr(
            hf_config,
            "architectures",
            None,
        ):
            architectures = _TEXT_ARCHITECTURES.get(
                getattr(hf_config, "model_type", None)
            )
        return original(
            self,
            hf_config,
            architectures=architectures,
        )

    return patched


def _do_patch_qwen3_vl_text_config() -> bool:
    """Wrap ``VllmConfig.with_hf_config`` once."""
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        module = importlib.import_module(_TARGET_MODULE)

    vllm_config_cls = getattr(module, "VllmConfig", None)
    if vllm_config_cls is None:
        return False
    if getattr(vllm_config_cls, _MARKER, False):
        return True

    original = vllm_config_cls.with_hf_config
    vllm_config_cls._zentorch_orig_with_hf_config = original
    vllm_config_cls.with_hf_config = _make_patched_with_hf_config(original)
    setattr(vllm_config_cls, _MARKER, True)
    logger.info(
        "[zentorch] Backported Qwen3-VL nested text architectures "
        "for vLLM 0.29"
    )
    return True


def _apply_qwen3_vl_text_config_patch() -> bool:
    """Apply only to vLLM 0.29, before the upstream fix."""
    vllm_module = sys.modules.get("vllm")
    vllm_version = getattr(vllm_module, "__version__", None)
    if vllm_version is None:
        return False
    if pkg_version.parse(vllm_version.split("+")[0]) != pkg_version.parse(
        "0.29.0"
    ):
        return False
    return patch_now_or_on_import(
        _TARGET_MODULE,
        _do_patch_qwen3_vl_text_config,
    )
