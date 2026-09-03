# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Out-of-tree generic dense-MLP -> fused FFN replacement for Zen CPU.

This is the plugin-side detects any dense MLP by structure and swaps its
``forward`` to a single ``torch.ops.zentorch.fused_ffn_concat`` call
(W13 -> gated activation -> W2):

  * Pattern: merged ``gate_up_proj`` + ``down_proj`` (Llama, Mistral, Gemma,
    Qwen2, ...).

Only fp32/bf16 single-expert (non-MoE) layers on Zen CPU are fused; everything
else falls back to the original ``forward`` unchanged.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from zentorch._logging import get_logger
from zentorch._utils import _SUPPORTED_MOE_ACTIVATIONS
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

# Map vLLM activation module class name -> zentorch gated-activation string.
# Mirrors the in-tree fused_ffn ACTIVATION_MAPPING.
ACTIVATION_MAPPING = {
    "SiluAndMul": "silu",
    "GeluAndMul": "gelu",
    "NewGELU": "gelu_tanh",
}

# Original (unpatched) ``forward`` per MLP class, so the shim can fall back.
_ORIGINAL_MLP_FORWARDS: Dict[type, Callable] = {}

_TARGET_MODULE = "vllm.model_executor.model_loader.base_loader"

# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------


def _get_activation_string(act_fn) -> Optional[str]:
    """Convert a vLLM activation instance to a zentorch activation string."""
    if act_fn is None:
        return None
    return ACTIVATION_MAPPING.get(type(act_fn).__name__)


def _should_use_fused_ffn(
    activation: str, dtype: torch.dtype
) -> bool:
    """ Fused only on fp32/bf16, supported gated act, single expert."""

    if dtype not in (torch.float32, torch.bfloat16):
        return False

    if activation not in _SUPPORTED_MOE_ACTIVATIONS:
        return False

    return True


# ---------------------------------------------------------------------------
# Per-module fusion setup
# ---------------------------------------------------------------------------


def _install_fused_forward(
    mlp_module: nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    activation: str,
) -> None:
    """Stash fused weights on the module and set ``cpu_mlp_forward``."""
    mlp_module._fused_w13_weight = w13
    mlp_module._fused_w2_weight = w2
    mlp_module._fused_w13_bias = w13_bias
    mlp_module._fused_w2_bias = w2_bias
    mlp_module._fused_activation = activation

    def fused_forward_fn(x):
        # Allocate a fresh output (MLP output shares the input's [*, hidden]
        # shape). Using empty_like for both patterns avoids mutating the
        # caller's input tensor in place.
        output = torch.empty_like(x)
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            x,
            w13_weight=mlp_module._fused_w13_weight,
            w2_weight=mlp_module._fused_w2_weight,
            w13_bias=mlp_module._fused_w13_bias,
            w2_bias=mlp_module._fused_w2_bias,
            activation=mlp_module._fused_activation,
        )
        return output

    mlp_module.cpu_mlp_forward = fused_forward_fn


def _free_weight(linear: nn.Module, attr: str) -> None:
    """Replace a projection weight/bias with an empty parameter."""
    if getattr(linear, attr, None) is not None:
        setattr(
            linear, attr, nn.Parameter(torch.empty(0), requires_grad=False)
        )


def dispatch_cpu_fused_mlp(
    mlp_module: nn.Module,
    activation_fn,
    remove_weights: bool = True,
) -> None:
    """Set up ``mlp_module.cpu_mlp_forward`` when the module is fusable.

    Leaves ``cpu_mlp_forward = None`` (native path) when any eligibility check
    fails, so this is always safe to call on every module.
    """
    mlp_module.cpu_mlp_forward = None

    activation = _get_activation_string(activation_fn)
    if activation is None:
        logger.info(
            "[zentorch] CPU MLP fusion: unsupported activation %s",
            type(activation_fn).__name__,
        )
        return

    # Pattern A: merged gate_up_proj.
    if hasattr(mlp_module, "gate_up_proj") and hasattr(mlp_module, "down_proj"):
        gate_up_weight = mlp_module.gate_up_proj.weight
        down_weight = mlp_module.down_proj.weight

        if gate_up_weight.numel() == 0 or down_weight.numel() == 0:
            logger.debug("[zentorch] CPU MLP fusion: weights already empty")
            return
        if gate_up_weight.dim() != 2 or down_weight.dim() != 2:
            logger.debug("[zentorch] CPU MLP fusion: weights not 2D")
            return

        dtype = gate_up_weight.dtype
        if not _should_use_fused_ffn(activation, dtype):
            return

        gate_up_bias = getattr(mlp_module.gate_up_proj, "bias", None)
        down_bias = getattr(mlp_module.down_proj, "bias", None)

        w13_data = gate_up_weight.clone().detach().contiguous()
        w2_data = down_weight.clone().detach().contiguous()
        w13_bias_data = (
            gate_up_bias.clone().detach().contiguous()
            if gate_up_bias is not None
            else None
        )
        w2_bias_data = (
            down_bias.clone().detach().contiguous()
            if down_bias is not None
            else None
        )

        _install_fused_forward(mlp_module, w13_data, w2_data, w13_bias_data, w2_bias_data, activation)

        if remove_weights:
            _free_weight(mlp_module.gate_up_proj, "weight")
            _free_weight(mlp_module.down_proj, "weight")
            if gate_up_bias is not None:
                _free_weight(mlp_module.gate_up_proj, "bias")
            if down_bias is not None:
                _free_weight(mlp_module.down_proj, "bias")

        logger.debug(
            "[zentorch] CPU MLP fusion: fused_ffn_concat "
            "(Pattern A, activation=%s, dtype=%s)",
            activation,
            dtype,
        )
        return
    logger.debug("[zentorch] CPU MLP fusion: unsupported MLP structure")


# ---------------------------------------------------------------------------
# Class-level forward shim + model iteration
# ---------------------------------------------------------------------------


def _create_fused_forward(original_forward: Callable) -> Callable:
    """Wrap a class ``forward`` to prefer ``cpu_mlp_forward`` when present."""

    def fused_forward(self, x):
        cpu_forward = getattr(self, "cpu_mlp_forward", None)
        if cpu_forward is not None:
            return cpu_forward(x)
        return original_forward(self, x)

    return fused_forward


def install_fused_mlp_forward(mlp_class: type) -> None:
    """Monkey-patch an MLP class ``forward`` to check for fusion (idempotent)."""
    if mlp_class in _ORIGINAL_MLP_FORWARDS:
        return
    _ORIGINAL_MLP_FORWARDS[mlp_class] = mlp_class.forward
    mlp_class.forward = _create_fused_forward(mlp_class.forward)
    logger.debug(
        "[zentorch] Installed fused MLP forward for %s", mlp_class.__name__
    )


def _is_mlp_module(module: nn.Module) -> bool:
    """Structural MLP detection (Pattern)."""
    has_pattern = hasattr(module, "gate_up_proj") and hasattr(
        module, "down_proj"
    )
    return has_pattern


def install_fused_mlp_forwards_for_model(model: nn.Module) -> None:
    """Patch ``forward`` on every detected MLP class in ``model``."""
    patched_classes = set()
    for module in model.modules():
        if _is_mlp_module(module):
            mlp_class = type(module)
            if mlp_class not in patched_classes:
                install_fused_mlp_forward(mlp_class)
                patched_classes.add(mlp_class)


def process_mlp_weights_after_loading(model: nn.Module) -> None:
    """Build fused weights for every eligible MLP module in ``model``."""
    for module in model.modules():
        if _is_mlp_module(module):
            act_fn = getattr(module, "act_fn", None)
            dispatch_cpu_fused_mlp(module, act_fn, remove_weights=True)


# ---------------------------------------------------------------------------
# base_loader.process_weights_after_loading wrapper + deferred import hook
# ---------------------------------------------------------------------------


def _wrap_process_weights_after_loading(base_loader_mod) -> bool:
    """Wrap ``process_weights_after_loading`` in the base_loader namespace.

    ``base_loader`` binds ``process_weights_after_loading`` by value at import
    but calls it by bare name inside ``load_model``, so rebinding the module
    attribute is picked up on the next call. Running fusion here (before the
    original, which contains the quant-processing loop) preserves the required
    "fuse before weights are freed" ordering.
    """
    if getattr(base_loader_mod, "_zentorch_fused_mlp_patched", False):
        return True

    orig_pwal = getattr(base_loader_mod, "process_weights_after_loading", None)
    if orig_pwal is None or not callable(orig_pwal):
        return False

    def _zen_process_weights_after_loading(model, *args, **kwargs):
        try:
            install_fused_mlp_forwards_for_model(model)
            process_mlp_weights_after_loading(model)
        except Exception:
            logger.warning(
                "[zentorch] fused MLP setup failed; falling back to native MLP",
                exc_info=True,
            )
        return orig_pwal(model, *args, **kwargs)

    base_loader_mod.process_weights_after_loading = (
        _zen_process_weights_after_loading
    )
    base_loader_mod._zentorch_fused_mlp_patched = True
    logger.info(
        "[zentorch] Patched base_loader.process_weights_after_loading "
        "for fused MLP"
    )
    return True


def _do_patch_fused_mlp() -> bool:
    """Wrap ``process_weights_after_loading`` once ``base_loader`` is loaded."""
    try:
        import vllm.model_executor.model_loader.base_loader as base_loader_mod
    except ImportError:
        return False
    return _wrap_process_weights_after_loading(base_loader_mod)


def _apply_fused_mlp_patch_impl() -> bool:
    """Opt in with ``ZENTORCH_FUSED_FFN=1`` to fuse dense MLPs on Zen CPU.

    Disabled by default. When enabled, arms the fused-MLP wrapper on
    ``base_loader.process_weights_after_loading`` (deferred until that module
    imports) via the shared post-import hook.
    """
    if os.environ.get("ZENTORCH_FUSED_FFN", "0") != "1":
        logger.debug(
            "[zentorch] Fused MLP replacement disabled; using native MLP "
            "(set ZENTORCH_FUSED_FFN=1 to enable the zentorch fused FFN)"
        )
        return False
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_fused_mlp)
