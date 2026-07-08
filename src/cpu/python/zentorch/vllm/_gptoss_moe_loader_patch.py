# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Out-of-tree GPT-OSS per-expert W8A8 checkpoint loading (no vLLM changes).

Wraps ``GptOssModel._load_weights_other`` to route the compressed-tensors
per-expert keys (``...mlp.experts.experts.N.{gate,up,down}_proj.*``) into the
stacked ``w13_*/w2_*`` params, delegating all other weights to the original
loader.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)

# _load_weights_other lives on the inner GptOssModel (GptOssForCausalLM
# delegates loading to it via AutoWeightsLoader), so patch GptOssModel.
_TARGET_MODULE = "vllm.model_executor.models.gpt_oss"
_TARGET_CLASS = "GptOssModel"


def _load_per_expert_moe_weight(
    in_name: str,
    in_weight: torch.Tensor,
    *,
    params_dict: dict,
    loaded_params: set,
    use_ep: bool,
    ep_rank_start: int,
    ep_rank_end: int,
    tp_rank: int,
    tp_rank_start: int,
    tp_rank_end: int,
    per_rank_intermediate_size: int,
    remap_fn=None,
) -> bool:
    """Route ``experts.experts.N.{w1,w3,w2}_*`` keys into the stacked
    ``w13_*/w2_*`` params, honoring EP/TP. Returns True iff the name matched a
    per-expert pattern (caller should skip it)."""
    if ".mlp.experts.experts." not in in_name:
        return False
    parts = in_name.split(".")
    digits = [int(p) for p in parts if p.isdigit()]
    if len(digits) != 2:
        return False
    layer_id, expert_id = digits

    # EP gating: skip experts owned by other ranks; remap to local id.
    if use_ep:
        if not (ep_rank_start <= expert_id < ep_rank_end):
            return True
        local_expert_id = expert_id - ep_rank_start
    else:
        local_expert_id = expert_id

    # Actual TP slice size (shorter than per_rank_intermediate_size only on the
    # last rank when intermediate_size % tp_size != 0).
    local_intermediate = tp_rank_end - tp_rank_start

    # Match projection role (gate/up/down) + kind (weight/weight_scale/bias) by
    # name: keys arrive as raw gate_proj/up_proj/down_proj or renamed w1/w3/w2.
    last = parts[-1]
    proj = parts[-2] if len(parts) >= 2 else ""
    if last in ("weight", "weight_scale", "bias"):
        kind = last
        role = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}.get(proj)
    else:
        role = None
        kind = None
        for _pfx, _role in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            if last.startswith(_pfx + "_"):
                role, kind = _role, last[len(_pfx) + 1 :]
                break
    if role is None or kind not in ("weight", "weight_scale", "bias"):
        return False

    if role == "gate":
        fused_suffix = f"w13_{kind}"
        dim1_start, dim1_end = 0, local_intermediate
        if not use_ep:
            in_weight = in_weight[tp_rank_start:tp_rank_end, ...]
    elif role == "up":
        fused_suffix = f"w13_{kind}"
        dim1_start, dim1_end = (
            per_rank_intermediate_size,
            per_rank_intermediate_size + local_intermediate,
        )
        if not use_ep:
            in_weight = in_weight[tp_rank_start:tp_rank_end, ...]
    else:  # down
        fused_suffix = f"w2_{kind}"
        dim1_start = dim1_end = None
        # w2_weight: TP-slice the INPUT (intermediate) dim.
        if not use_ep and kind == "weight":
            in_weight = in_weight[:, tp_rank_start:tp_rank_end]
        # w2_bias: replicated in TP; only rank 0 contributes to the
        # post-all-reduce sum (mirrors the stacked loader's zero-on-nonzero-rank).
        if not use_ep and kind == "bias" and tp_rank != 0:
            in_weight = torch.zeros_like(in_weight)

    fused_name = f"layers.{layer_id}.mlp.experts.{fused_suffix}"
    # Remap the target name for the RoutedExperts refactor (>=0.23.1rc0) that
    # nests params under mlp.experts.routed_experts.*; no-op if remap_fn is None.
    if remap_fn is not None:
        fused_name = remap_fn(fused_name, params_dict)
    if fused_name not in params_dict:
        # Model didn't allocate this param (e.g. has_bias=False).
        return True

    target_slot = params_dict[fused_name].data[local_expert_id]
    if dim1_start is None:
        dst = target_slot
    else:
        dst = target_slot[dim1_start:dim1_end]
    dst.copy_(in_weight.view(dst.shape) if dst.shape != in_weight.shape else in_weight)
    loaded_params.add(fused_name)
    return True


def _register_gptoss_loader_patch(mod) -> None:
    cls = getattr(mod, _TARGET_CLASS)
    orig_load_weights_other = cls._load_weights_other

    def _zen_load_weights_other(
        self,
        ep_rank_end,
        ep_rank_start,
        heads_per_rank,
        head_start,
        weights,
        stacked_params_mapping,
    ):
        params_dict = dict(self.named_parameters())
        loaded_params: set = set()

        use_ep = self.parallel_config.enable_expert_parallel
        tp_size, tp_rank = mod.FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=mod.get_tensor_model_parallel_world_size(),
            dp_size=mod.get_dp_group().world_size,
            dp_rank=mod.get_dp_group().rank_in_group,
            pcp_size=mod.get_pcp_group().world_size,
            pcp_rank=mod.get_pcp_group().rank_in_group,
        )
        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = mod.cdiv(intermediate_size, tp_size)
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        # RoutedExperts target-name remap helper (imported from weight_utils;
        # gpt_oss doesn't re-export it). None on genuinely pre-refactor vLLM.
        try:
            from vllm.model_executor.model_loader.weight_utils import (
                maybe_remap_moe_expert_param_name as remap_fn,
            )
        except Exception:
            remap_fn = None

        # Route per-expert keys; pass everything else to the original loader.
        remaining = []
        _n_loaded_before = len(loaded_params)
        for name, weight in weights:
            if _load_per_expert_moe_weight(
                name,
                weight,
                params_dict=params_dict,
                loaded_params=loaded_params,
                use_ep=use_ep,
                ep_rank_start=ep_rank_start,
                ep_rank_end=ep_rank_end,
                tp_rank=tp_rank,
                tp_rank_start=tp_rank_start,
                tp_rank_end=tp_rank_end,
                per_rank_intermediate_size=per_rank_intermediate_size,
                remap_fn=remap_fn,
            ):
                continue
            remaining.append((name, weight))

        # Safety net: if nothing routed, the stacked w13_*/w2_* params stay
        # zero-initialized and the model emits garbage. Warn loudly (visible at
        # the default log level) rather than fail silently.
        if len(loaded_params) - _n_loaded_before == 0:
            logger.warning(
                "[zentorch] gpt-oss per-expert loader routed 0 MoE weights "
                "(remap_fn=%s). Expert tensors will be UNINITIALIZED -> garbage "
                "output. Check the per-expert key matching / routed_experts remap.",
                remap_fn is not None,
            )

        loaded_params |= orig_load_weights_other(
            self,
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            remaining,
            stacked_params_mapping,
        )
        return loaded_params

    cls._load_weights_other = _zen_load_weights_other


def _apply_gptoss_loader_patch_to_module(mod) -> bool:
    try:
        cls = getattr(mod, _TARGET_CLASS, None)
        if cls is None or not hasattr(cls, "_load_weights_other"):
            return False
        if getattr(cls, "_zentorch_gptoss_loader_patched", False):
            return True
        _register_gptoss_loader_patch(mod)
        cls._zentorch_gptoss_loader_patched = True
        logger.info(
            "[zentorch] Patched %s._load_weights_other: per-expert W8A8 loading",
            _TARGET_CLASS,
        )
        return True
    except Exception:
        logger.warning("[zentorch] gpt-oss loader patch FAILED", exc_info=True)
        return False


_HOOK_INSTALLED = False


class _GptOssLoaderImportHook:
    """Defer patch until the gpt_oss model module loads."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _apply_gptoss_loader_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_gptoss_loader_patch_impl() -> bool:
    """Schedule the gpt-oss per-expert loader patch on first import."""
    global _HOOK_INSTALLED

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        return _apply_gptoss_loader_patch_to_module(mod)

    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _GptOssLoaderImportHook())
        _HOOK_INSTALLED = True
    return True
