# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Out-of-tree GPT-OSS per-expert ("streamed") checkpoint loading.

Backport of the loader half of vLLM PR #52209. gpt-oss checkpoints that store
MoE experts one tensor per expert -- e.g.

    ...mlp.experts.<id>.gate_up_proj[.weight|.bias|.weight_scale]
    ...mlp.experts.<id>.down_proj[.weight|.bias|.weight_scale]

-- get normalized by ``hf_to_vllm_mapper`` to per-expert
``...mlp.experts.<id>.w13_*/w2_*`` names. Stock ``GptOssModel._load_weights_other``
then looks those up directly in ``params_dict``, where only the *fused*
``...mlp.experts.routed_experts.w13_*/w2_*`` params exist, raising e.g.
``KeyError: '...experts.0.w2_bias'``.

This patch, applied only on a supported vLLM via the plugin's import hook:

  * installs a ``RoutedExperts`` subclass whose ``weight_loader`` fills one
    expert slice of the fused param at a time, and
  * replaces ``GptOssModel._load_weights_other`` with a copy that intercepts the
    per-expert keys and routes them to that fused param's ``weight_loader``.

Scope is deliberately minimal: only the compressed-tensors / BF16 path
(``_load_weights_other``) is covered -- native mxfp4 and quark keep their stock
loaders -- and PR #52209's RL weight-sync reload (``reload/meta.py``) is omitted.
"""

from __future__ import annotations

import sys
import typing
from typing import Callable

import torch

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

# _load_weights_other lives on the inner GptOssModel (GptOssForCausalLM delegates
# to it via AutoWeightsLoader), so the patch targets that module/class.
_TARGET_MODULE = "vllm.model_executor.models.gpt_oss"
_MARKER = "_zentorch_gptoss_streamed_patched"

# Fused-param suffix -> the per-expert shard id understood by the RoutedExperts
# subclass below. Covers every per-expert key emitted by hf_to_vllm_mapper for
# the BF16 and compressed-tensors int8 gpt-oss checkpoints.
_STREAMED_EXPERT_SUFFIX_TO_SHARD = {
    "w13_weight": "gpt_oss_w13",
    "w2_weight": "gpt_oss_w2",
    "w13_bias": "gpt_oss_w13",
    "w2_bias": "gpt_oss_w2",
    "w13_weight_scale": "gpt_oss_w13",
    "w2_weight_scale": "gpt_oss_w2",
}

# Built lazily (needs vLLM imported) and cached for the process.
_ROUTED_EXPERTS_CLS: type | None = None


def _get_routed_experts_cls() -> type:
    """Return (building once) the GPT-OSS per-expert ``RoutedExperts`` subclass."""
    global _ROUTED_EXPERTS_CLS
    if _ROUTED_EXPERTS_CLS is not None:
        return _ROUTED_EXPERTS_CLS

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    class GptOssRoutedExperts(RoutedExperts):
        """Load one GPT-OSS expert at a time without assembling the global stack."""

        @staticmethod
        def _narrow_for_rank(
            loaded_weight: torch.Tensor, dim: int, rank: int, size: int
        ) -> torch.Tensor:
            start = rank * size
            available = loaded_weight.shape[dim] - start
            return loaded_weight.narrow(dim, start, min(size, max(available, 0)))

        @staticmethod
        def _copy_to_expert(
            expert_data: torch.Tensor, loaded_weight: torch.Tensor
        ) -> None:
            if expert_data.numel() == loaded_weight.numel() == 1:
                expert_data.copy_(loaded_weight.reshape(()))
                return
            while loaded_weight.ndim > expert_data.ndim and loaded_weight.shape[-1] == 1:
                loaded_weight = loaded_weight.squeeze(-1)
            slices = tuple(slice(0, size) for size in loaded_weight.shape)
            expert_data[slices].copy_(loaded_weight)

        def _load_expert_bias(
            self, expert_data: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str
        ) -> None:
            tp_rank = self.moe_config.moe_parallel_config.tp_rank
            if shard_id == "gpt_oss_w13":
                loaded_weight = self._narrow_for_rank(
                    loaded_weight, 0, tp_rank, expert_data.shape[0]
                )
            elif tp_rank != 0:
                # w2 bias is replicated; only rank 0 owns it to avoid double-add.
                loaded_weight = torch.zeros_like(loaded_weight)
            self._copy_to_expert(expert_data, loaded_weight)

        def _load_unquantized_expert(
            self, expert_data: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str
        ) -> None:
            tp_rank = self.moe_config.moe_parallel_config.tp_rank
            if shard_id == "gpt_oss_w13":
                loaded_weight = self._narrow_for_rank(
                    loaded_weight, 1, tp_rank, expert_data.shape[0]
                )
            else:
                loaded_weight = self._narrow_for_rank(
                    loaded_weight, 0, tp_rank, expert_data.shape[1]
                )
            loaded_weight = loaded_weight.t().contiguous()
            self._copy_to_expert(expert_data, loaded_weight)

        def weight_loader(
            self,
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            if shard_id not in ("gpt_oss_w13", "gpt_oss_w2"):
                return super().weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success,
                )

            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id == -1:
                return False if return_success else None

            expert_data = param.data[expert_id]
            if weight_name.endswith("_bias"):
                self._load_expert_bias(expert_data, loaded_weight, shard_id)
            else:
                self._load_unquantized_expert(expert_data, loaded_weight, shard_id)
            return True if return_success else None

    _ROUTED_EXPERTS_CLS = GptOssRoutedExperts
    return GptOssRoutedExperts


def _get_streamed_expert_info(
    name: str, params_dict: dict
) -> tuple[int, str, str] | None:
    """Parse ``...mlp.experts.<expert_id>.<fused_param>`` checkpoint keys.

    Returns ``(expert_id, fused_param_name, shard_id)`` when ``name`` is a
    per-expert key whose fused ``routed_experts`` param exists, else ``None``.
    """
    if ".mlp.experts." not in name:
        return None
    suffix = name.rsplit(".", 1)[-1]
    shard_id = _STREAMED_EXPERT_SUFFIX_TO_SHARD.get(suffix)
    if shard_id is None:
        return None

    prefix, expert_suffix = name.split(".mlp.experts.", maxsplit=1)
    expert_id_str, separator, param_suffix = expert_suffix.partition(".")
    if not separator or not expert_id_str.isdigit():
        return None
    expert_id = int(expert_id_str)
    for base_layer_prefix in ("", "base_layer."):
        fused_name = (
            f"{prefix}.mlp.experts.{base_layer_prefix}routed_experts.{param_suffix}"
        )
        if fused_name in params_dict:
            return expert_id, fused_name, shard_id
    return None


def _try_load_streamed_expert(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    loaded_params: set,
) -> bool:
    """Load a per-expert key via its fused param's weight_loader.

    Returns True iff ``name`` was a per-expert key we handled (caller skips it).
    """
    info = _get_streamed_expert_info(name, params_dict)
    if info is None:
        return False

    expert_id, fused_name, shard_id = info
    param = params_dict[fused_name]
    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
    success = weight_loader(
        param,
        loaded_weight,
        weight_name=fused_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )
    if success:
        loaded_params.add(fused_name)
    return True


def _patched_load_weights_other(
    self,
    ep_rank_end: int,
    ep_rank_start: int,
    heads_per_rank: int,
    head_start: int,
    weights,
    stacked_params_mapping,
) -> set:
    """Copy of stock ``GptOssModel._load_weights_other`` with a single added
    per-expert interception (marked ``# zentorch``)."""
    from vllm.distributed import (
        get_dp_group,
        get_pcp_group,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        remap_moe_expert_weights,
    )
    from vllm.model_executor.models.utils import is_pp_missing_parameter
    from vllm.utils.math_utils import cdiv

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    use_ep = self.parallel_config.enable_expert_parallel

    tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
        tp_size=get_tensor_model_parallel_world_size(),
        dp_size=get_dp_group().world_size,
        dp_rank=get_dp_group().rank_in_group,
        pcp_size=get_pcp_group().world_size,
        pcp_rank=get_pcp_group().rank_in_group,
    )

    intermediate_size = self.config.intermediate_size
    per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
    tp_rank_start = tp_rank * per_rank_intermediate_size
    tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

    for name, weight in remap_moe_expert_weights(weights, params_dict):
        if is_pp_missing_parameter(name, self):
            continue

        # zentorch: route per-expert checkpoint keys into the fused param.
        if _try_load_streamed_expert(name, weight, params_dict, loaded_params):
            continue

        if ".w13_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]
            narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
            param = params_dict[name]
            param.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        elif ".w2_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
            narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
            param = params_dict[name]
            param.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        elif ".w13_bias" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
            param = params_dict[name]
            param.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        elif ".w2_bias" in name:
            if use_ep:
                weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                if tp_rank != 0:
                    weight.zero_()
            param = params_dict[name]
            param.copy_(weight)
            loaded_params.add(name)
            continue
        elif "sinks" in name:
            param = params_dict[name]
            narrow_weight = weight.narrow(0, head_start, heads_per_rank)
            param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, weight)
            else:
                weight_loader(param, weight, shard_id)
            break
        else:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
        loaded_params.add(name)
    return loaded_params


def _make_patched_mlpblock_init(orig_init, routed_cls):
    """Wrap ``MLPBlock.__init__`` to retype its RoutedExperts to ``routed_cls``.

    The subclass only *adds* methods (no new fields), so swapping ``__class__``
    on the already-constructed instance injects the per-expert ``weight_loader``
    without re-implementing ``MLPBlock.__init__`` or threading a
    ``routed_experts_cls`` kwarg through ``FusedMoEFactory``. Params created
    before the swap still reference the base ``weight_loader``, so they are
    rebound to the subclass method below.
    """

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        experts = getattr(self, "experts", None)
        routed = getattr(experts, "routed_experts", None)
        if routed is not None and not isinstance(routed, routed_cls):
            routed.__class__ = routed_cls
            # create_weights (routed_experts.py) captured the *base* class's
            # weight_loader as a bound method on every fused param, before this
            # __class__ swap. Rebind those params to the subclass's bound method
            # so the gpt_oss_* shard ids route to the per-expert loader instead
            # of the base loader (which rejects them).
            new_loader = routed.weight_loader
            for param in routed.parameters(recurse=True):
                loader = getattr(param, "weight_loader", None)
                if loader is not None and getattr(loader, "__self__", None) is routed:
                    param.weight_loader = new_loader

    return _patched_init


def _do_patch() -> bool:
    """Graft the streamed-expert loader onto GptOssModel/MLPBlock."""
    mod = sys.modules.get(_TARGET_MODULE)
    if mod is None:
        import vllm.model_executor.models.gpt_oss as mod  # noqa: F811

    gpt_oss_model = getattr(mod, "GptOssModel", None)
    mlp_block = getattr(mod, "MLPBlock", None)
    if gpt_oss_model is None or mlp_block is None:
        return False
    if getattr(gpt_oss_model, _MARKER, False):
        return True

    routed_cls = _get_routed_experts_cls()

    gpt_oss_model._load_weights_other = _patched_load_weights_other
    mlp_block.__init__ = _make_patched_mlpblock_init(mlp_block.__init__, routed_cls)

    setattr(gpt_oss_model, _MARKER, True)
    logger.info(
        "[zentorch] Patched GptOssModel for per-expert (streamed) checkpoint loading"
    )
    return True


def _apply_gptoss_streamed_expert_patch_impl() -> bool:
    """Arm the streamed-expert loader for GptOssModel (deferred via import hook)."""
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch)
