# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""CPU dispatch for the ``_moe_C`` top-k router ops (no vLLM changes).

vLLM's MoE router calls ``torch.ops._moe_C.topk_softmax`` / ``topk_sigmoid``,
which only exist in the CUDA ``_moe_C`` extension -> on CPU the router raises
``AttributeError: '_moe_C' object has no attribute 'topk_softmax'``. Register a
pure-PyTorch CPU implementation for those ops so the existing router works
unchanged (mirrors the in-tree fix / the "register a CPU dispatch key" review).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)


def _native_topk_score_then_select(
    score_fn,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
) -> None:
    # Pure-PyTorch fallback for CPU, since _moe_C is CUDA-only.
    scores = (
        score_fn(gating_output.float(), dim=-1)
        if score_fn is torch.softmax
        else score_fn(gating_output.float())
    )
    if e_score_correction_bias is not None:
        scores_for_selection = scores + e_score_correction_bias.float()
    else:
        scores_for_selection = scores
    top_k = topk_weights.shape[-1]
    _, idx = torch.topk(scores_for_selection, k=top_k, dim=-1, sorted=False)
    vals = torch.gather(scores, dim=-1, index=idx)
    if renormalize:
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    topk_weights.copy_(vals.to(topk_weights.dtype))
    topk_ids.copy_(idx.to(topk_ids.dtype))
    M = topk_weights.shape[0]
    token_expert_indices.copy_(
        torch.arange(
            top_k, device=gating_output.device, dtype=token_expert_indices.dtype
        )
        .unsqueeze(0)
        .expand(M, top_k)
    )


def _topk_softmax_cpu(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> None:
    _native_topk_score_then_select(
        torch.softmax,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )


def _topk_sigmoid_cpu(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> None:
    _native_topk_score_then_select(
        torch.sigmoid,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )


def _topk_cpu_fake(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> None:
    return None


# Keep the Library object alive for the process lifetime.
_moe_C_cpu_lib = None
_HOOK_INSTALLED = False

# Deferral target: a fused-MoE module that loads after vLLM init but before any
# router forward, avoiding the init-time circular import of torch_utils.
_DEFER_TARGET = "vllm.model_executor.layers.fused_moe.layer"


def _register_moe_C_topk_cpu() -> bool:
    """Register the CPU ``_moe_C.topk_softmax``/``topk_sigmoid`` fallback.

    Requires ``vllm.utils.torch_utils`` to be fully importable; raises
    ``ImportError`` (circular import) if called mid-``import vllm``.
    """
    global _moe_C_cpu_lib
    if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "topk_softmax"):
        return True  # already available (CUDA build or already patched)

    from vllm.utils.torch_utils import direct_register_custom_op

    _moe_C_cpu_lib = torch.library.Library("_moe_C", "FRAGMENT")
    direct_register_custom_op(
        "topk_softmax",
        _topk_softmax_cpu,
        mutates_args=["topk_weights", "topk_indices", "token_expert_indices"],
        fake_impl=_topk_cpu_fake,
        target_lib=_moe_C_cpu_lib,
        dispatch_key="CPU",
    )
    direct_register_custom_op(
        "topk_sigmoid",
        _topk_sigmoid_cpu,
        mutates_args=["topk_weights", "topk_indices", "token_expert_indices"],
        fake_impl=_topk_cpu_fake,
        target_lib=_moe_C_cpu_lib,
        dispatch_key="CPU",
    )
    logger.info("[zentorch] Registered CPU _moe_C.topk_softmax/topk_sigmoid fallback")
    return True


class _MoeTopkImportHook:
    """Defer the topk-CPU registration until ``_DEFER_TARGET`` loads, so the
    ``direct_register_custom_op`` import happens after vLLM is fully
    initialized (avoids the init-time circular import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _DEFER_TARGET:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_register(module):
            original_exec(module)
            try:
                _register_moe_C_topk_cpu()
            except Exception:
                logger.warning(
                    "[zentorch] deferred _moe_C topk CPU patch FAILED", exc_info=True
                )

        spec.loader.exec_module = _exec_then_register
        return spec


def _apply_moe_topk_cpu_patch_impl() -> bool:
    """Register the CPU ``_moe_C`` top-k fallback (idempotent).

    Tries eagerly; if ``vllm.utils.torch_utils`` isn't fully importable yet
    (init-time circular import while running during ``import vllm``), defers
    the registration to when the fused-MoE layer module loads.
    """
    global _HOOK_INSTALLED
    try:
        return _register_moe_C_topk_cpu()
    except ImportError:
        # torch_utils partially initialized (called mid-`import vllm`): defer.
        if _DEFER_TARGET in sys.modules:
            # Target already loaded (torch_utils must be ready) -> retry now.
            try:
                return _register_moe_C_topk_cpu()
            except Exception:
                logger.warning(
                    "[zentorch] _moe_C topk CPU patch FAILED", exc_info=True
                )
                return False
        if not _HOOK_INSTALLED:
            sys.meta_path.insert(0, _MoeTopkImportHook())
            _HOOK_INSTALLED = True
        logger.info(
            "[zentorch] _moe_C topk CPU registration deferred until %s loads",
            _DEFER_TARGET,
        )
        return True
    except Exception:
        logger.warning("[zentorch] _moe_C topk CPU patch FAILED", exc_info=True)
        return False
