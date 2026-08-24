# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Shared helpers for the out-of-tree INT8 and WNA16 fused-MoE vLLM patches."""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)

_SELECT_EXPERTS = None


def import_select_experts():
    """``select_experts`` lives in ``fused_moe.cpu_fused_moe`` through vLLM
    0.27.1 and moves to ``fused_moe.experts.cpu_moe`` in 0.28.0.
    """
    global _SELECT_EXPERTS
    if _SELECT_EXPERTS is None:
        try:
            from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
                select_experts,
            )
        except ImportError:
            # For vLLM v0.28.0 and above:
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                select_experts,
            )
        _SELECT_EXPERTS = select_experts
    return _SELECT_EXPERTS


def run_select_experts(
    hidden_states,
    router_logits,
    moe_config,
    *,
    num_expert_group=None,
    e_score_correction_bias=None,
    routed_scaling_factor=None,
    topk_group=None,
):
    """Monolithic-experts top-k routing used by both INT8 and WNA16 apply()."""
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    return import_select_experts()(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=num_expert_group is not None,
        top_k=moe_config.experts_per_token,
        renormalize=moe_config.routing_method
        in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ),
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func="softmax",
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        e_score_correction_bias=e_score_correction_bias,
    )


def allocate_expert_biases(
    layer,
    moe,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype,
    extra_weight_attrs: dict,
) -> None:
    """Allocate ``w13_bias`` / ``w2_bias`` when the vanilla method never did.

    Both the INT8 and WNA16 compressed-tensors methods skip per-expert biases,
    so gpt-oss-style checkpoints have nowhere to land them.
    """
    if not getattr(moe, "has_bias", False):
        return
    if getattr(layer, "w13_bias", None) is not None:
        return

    from vllm.model_executor.utils import set_weight_attrs

    bias_attrs = {
        k: v for k, v in extra_weight_attrs.items() if k != "intermediate_size_full"
    }
    w13_num_shards = 2 if moe.is_act_and_mul else 1
    w13_bias = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            w13_num_shards * intermediate_size_per_partition,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_bias", w13_bias)
    set_weight_attrs(w13_bias, bias_attrs)

    w2_bias = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_size, dtype=params_dtype),
        requires_grad=False,
    )
    layer.register_parameter("w2_bias", w2_bias)
    set_weight_attrs(w2_bias, bias_attrs)


def run_moe_patch_apply(
    mod,
    *,
    target,
    flag: str,
    register_fn: Callable,
    success_log: str,
    fail_log: str,
    missing_log: str | None = None,
    missing_ok: bool = False,
    missing_debug: bool = False,
    extra_guard: Callable[[], str | None] | None = None,
) -> bool:
    """Shared apply() skeleton for the INT8 and WNA16 MoE module patches.

    Resolves ``target`` (the class or module that carries ``flag``), returns
    early if it is already patched, optionally runs ``extra_guard``, then
    calls ``register_fn(mod)``. ``target is None`` means this module is not
    the one to patch: ``missing_ok=True`` is a skip (the other candidate),
    otherwise a failure.
    ``extra_guard`` returns a warning string to abort, or ``None`` to proceed.
    """
    try:
        if target is None:
            if missing_log:
                if missing_debug:
                    logger.debug(missing_log)
                else:
                    logger.warning(missing_log)
            return missing_ok
        if getattr(target, flag, False):
            return True
        if extra_guard is not None:
            reason = extra_guard()
            if reason is not None:
                logger.warning(reason)
                return False
        register_fn(mod)
        setattr(target, flag, True)
        logger.info(success_log)
        return True
    except Exception:
        logger.warning(fail_log, exc_info=True)
        return False


def schedule_module_patches(targets: dict[str, Callable]) -> bool:
    """Patch already-imported modules; otherwise defer via ``patch_now_or_on_import``.

    ``targets`` maps a fully-qualified module name to ``(module) -> bool``.
    """
    from zentorch.vllm._import_hook import patch_now_or_on_import

    ok = True
    for name, apply_fn in targets.items():

        def _do(name=name, apply_fn=apply_fn) -> bool:
            return apply_fn(sys.modules[name])

        ok = patch_now_or_on_import(name, _do) and ok
    return ok
