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

fp32/bf16 and int8 DA8W8 (dynamic per-token activations, per-channel int8
weights) single-expert (non-MoE) layers on Zen CPU are fused; everything else
falls back to the original ``forward`` unchanged.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

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

_FP_WEIGHT_DTYPES = (torch.float32, torch.bfloat16)
_QUANT_SCALE_DTYPES = (torch.float32, torch.bfloat16)
_DA8W8_SCALE_DTYPES = _QUANT_SCALE_DTYPES


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------


def _get_activation_string(act_fn) -> Optional[str]:
    """Convert a vLLM activation instance to a zentorch activation string."""
    if act_fn is None:
        return None
    return ACTIVATION_MAPPING.get(type(act_fn).__name__)


def _should_use_fused_ffn(activation: str, dtype: torch.dtype) -> bool:
    """Fuse fp32/bf16 and int8 DA8W8 with a gated activation."""
    if dtype not in (*_FP_WEIGHT_DTYPES, torch.int8):
        return False
    if activation not in _SUPPORTED_MOE_ACTIVATIONS:
        return False
    return True


def _scheme_weight_bits(linear: nn.Module) -> Optional[int]:
    """Weight bit width from the layer's quant scheme, or None if unknown.

    Schemes disagree on where they keep it. Most set ``num_bits``, but the
    W4A8 ones only keep a ``quant_type`` scalar type, so checking ``num_bits``
    alone reads 4-bit layers as 8-bit.
    """
    scheme = getattr(linear, "scheme", None)
    if scheme is None:
        return None
    bits = getattr(scheme, "num_bits", None)
    if bits is None:
        bits = getattr(getattr(scheme, "quant_type", None), "size_bits", None)
    return bits


def _as_tensor(value) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return value


def _normalize_da8w8_scale(scale: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a per-channel DA8W8 scale as 1D ``[N]`` (f32/bf16).

    Compressed-tensors often stores per-channel scales as ``[N, 1]``. The
    group-matmul kernel treats 1D ``[N]`` as ``{1, N}``; a 2D ``[N, 1]``
    would be read as ``G=N`` groups, so squeeze the trailing singleton.
    """
    if scale is None:
        return None
    if scale.dtype not in _DA8W8_SCALE_DTYPES:
        return None
    if scale.dim() == 2 and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    if scale.dim() not in (1, 2):
        return None
    return scale.contiguous()


def _unwrap_proj_weight(
    linear: nn.Module,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(weight_2d, scale_or_None)`` for fused FFN.

    Compressed-tensors W8A8 -> int8 ``weight`` plus ``weight_scale``.
    Floating-point weights return ``(weight, None)``. Scales are not squeezed
    here; the caller normalizes.
    """
    # Sub-8-bit schemes (W4A16, W4A8) only pack their weights during vLLM's
    # own sweep, which runs after this. Fuse only what is already final.
    bits = _scheme_weight_bits(linear)
    if bits is not None and bits < 8:
        return None, None

    weight = getattr(linear, "weight", None)
    if weight is None:
        return None, None

    if weight.dtype != torch.int8:
        return weight, None

    scale = getattr(linear, "weight_scale", None)
    if scale is None:
        scale = getattr(linear, "weight_scales", None)
    return weight, _as_tensor(scale)


def _take_bias(
    bias: Optional[torch.Tensor], *, force_bf16: bool
) -> Optional[torch.Tensor]:
    if bias is None or not isinstance(bias, torch.Tensor) or bias.numel() == 0:
        return None
    if force_bf16 and bias.dtype != torch.bfloat16:
        return bias.to(torch.bfloat16)
    return bias


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
    w13_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> None:
    """Stash fused weights on the module and set ``cpu_mlp_forward``."""
    mlp_module._fused_w13_weight = w13
    mlp_module._fused_w2_weight = w2
    mlp_module._fused_w13_bias = w13_bias
    mlp_module._fused_w2_bias = w2_bias
    mlp_module._fused_activation = activation
    mlp_module._fused_w13_scale = w13_scale
    mlp_module._fused_w2_scale = w2_scale
    # Shared experts (e.g. Qwen2-MoE / Qwen3-Next / Qwen3.5-MoE) scale their
    # output by sigmoid(expert_gate(x)) inside the replaced ``forward``.
    expert_gate = getattr(mlp_module, "expert_gate", None)

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
            w13_scale=mlp_module._fused_w13_scale,
            w2_scale=mlp_module._fused_w2_scale,
        )
        if expert_gate is not None:
            gate_logits = expert_gate(x)
            if isinstance(gate_logits, tuple):
                gate_logits = gate_logits[0]
            output.mul_(torch.sigmoid(gate_logits))
        return output

    mlp_module.cpu_mlp_forward = fused_forward_fn


def _free_weight(linear: nn.Module, attr: str) -> None:
    """Replace a projection weight/bias/scale with an empty parameter."""
    if getattr(linear, attr, None) is not None:
        setattr(
            linear, attr, nn.Parameter(torch.empty(0), requires_grad=False)
        )


def _free_pattern_weights(mlp_module: nn.Module) -> None:
    """Drop projection weights, biases, and scales, and retire the layers.

    Retiring matters: vLLM's own post-load sweep would otherwise process a
    projection whose weight we just emptied. ``quant_method`` is the gate it
    checks on 0.27-0.29; ``_cpu_skip_gemm_dispatch`` is 0.29's name for it.
    """
    for proj in (mlp_module.gate_up_proj, mlp_module.down_proj):
        _free_weight(proj, "weight")
        if getattr(proj, "bias", None) is not None:
            _free_weight(proj, "bias")
        for scale_attr in ("weight_scale", "weight_scales"):
            if getattr(proj, scale_attr, None) is not None:
                _free_weight(proj, scale_attr)
        proj._cpu_skip_gemm_dispatch = True
        proj.quant_method = None


def dispatch_cpu_fused_mlp(
    mlp_module: nn.Module,
    activation_fn,
    remove_weights: bool = True,
) -> None:
    """Set up ``mlp_module.cpu_mlp_forward`` when the module is fusable.

    Leaves ``cpu_mlp_forward = None`` (native path) when any eligibility check
    fails, so this is always safe to call on every module. Idempotent: an
    already-fused module is left unchanged.
    """
    if getattr(mlp_module, "cpu_mlp_forward", None) is not None:
        return

    mlp_module.cpu_mlp_forward = None

    activation = _get_activation_string(activation_fn)
    if activation is None:
        logger.info(
            "[zentorch] CPU MLP fusion: unsupported activation %s",
            type(activation_fn).__name__,
        )
        return

    if hasattr(mlp_module, "gate_up_proj") and hasattr(mlp_module, "down_proj"):
        w13_data, w13_scale = _unwrap_proj_weight(mlp_module.gate_up_proj)
        w2_data, w2_scale = _unwrap_proj_weight(mlp_module.down_proj)

        if w13_data is None or w2_data is None:
            logger.debug("[zentorch] CPU MLP fusion: missing projection weights")
            return
        if w13_data.numel() == 0 or w2_data.numel() == 0:
            logger.debug("[zentorch] CPU MLP fusion: weights already empty")
            return
        if w13_data.dim() != 2 or w2_data.dim() != 2:
            logger.debug("[zentorch] CPU MLP fusion: weights not 2D")
            return
        if w13_data.dtype != w2_data.dtype:
            logger.debug("[zentorch] CPU MLP fusion: mixed weight dtypes")
            return

        if w13_data.size(0) % 2 != 0:
            logger.debug(
                "[zentorch] CPU MLP fusion: W13 is not gated (2I rows)"
            )
            return

        hidden = w2_data.size(0)
        intermediate = w13_data.size(0) // 2

        force_bf16 = False
        weight_dtype = w13_data.dtype
        if weight_dtype == torch.int8:
            w13_scale = _normalize_da8w8_scale(w13_scale)
            w2_scale = _normalize_da8w8_scale(w2_scale)
            if w13_scale is None or w2_scale is None:
                logger.debug(
                    "[zentorch] CPU MLP fusion: DA8W8 weights missing scales"
                )
                return
            if (
                w13_scale.numel() != w13_data.size(0)
                or w2_scale.numel() != w2_data.size(0)
            ):
                logger.debug(
                    "[zentorch] CPU MLP fusion: DA8W8 scales are not per-channel"
                )
                return
            force_bf16 = True
        elif weight_dtype in _FP_WEIGHT_DTYPES:
            if w13_data.size(1) != hidden or w2_data.size(1) != intermediate:
                logger.debug(
                    "[zentorch] CPU MLP fusion: incompatible W13/W2 packed shapes"
                )
                return
        else:
            logger.debug(
                "[zentorch] CPU MLP fusion: weight dtype is not fp or DA8W8"
            )
            return

        if not _should_use_fused_ffn(activation, weight_dtype):
            return

        w13_bias_data = _take_bias(
            getattr(mlp_module.gate_up_proj, "bias", None),
            force_bf16=force_bf16,
        )
        w2_bias_data = _take_bias(
            getattr(mlp_module.down_proj, "bias", None),
            force_bf16=force_bf16,
        )

        _install_fused_forward(
            mlp_module,
            w13_data.contiguous(),
            w2_data.contiguous(),
            w13_bias_data,
            w2_bias_data,
            activation,
            w13_scale,
            w2_scale,
        )

        if remove_weights:
            _free_pattern_weights(mlp_module)

        logger.debug(
            "[zentorch] CPU MLP fusion: fused_ffn_concat "
            "(activation=%s, dtype=%s)",
            activation,
            weight_dtype,
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
    "fuse before weights are freed" ordering for fp32/bf16 and DA8W8.
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
