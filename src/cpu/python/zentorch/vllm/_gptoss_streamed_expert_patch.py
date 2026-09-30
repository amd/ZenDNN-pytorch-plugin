# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Load gpt-oss checkpoints that keep one expert per key.

vLLM's gpt-oss loader only accepts expert tensors already stacked across
experts, while compressed-tensors checkpoints store them per expert. Ports the
streamed-expert path from vllm-project/vllm#52209.
vLLM 0.29 ships this loader natively, so the patch stands aside there.
"""

from __future__ import annotations

import sys
import typing
from collections.abc import Iterable, Iterator
from functools import wraps
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
    "w13_weight_packed": "gpt_oss_w13",
    "w2_weight_packed": "gpt_oss_w2",
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

            if weight_name.endswith("_bias"):
                pass  # [N] per expert, nothing to reorder
            elif weight_name.endswith("_weight_scale"):
                # Stored [N, G]; the fused param is [G, N].
                loaded_weight = loaded_weight.t().contiguous()
            else:
                # Experts are stored [out, in]; the fused param is [in, out].
                loaded_weight = loaded_weight.t().contiguous()
            self._copy_to_expert(param.data[expert_id], loaded_weight)
            return True if return_success else None

    _ROUTED_EXPERTS_CLS = GptOssRoutedExperts
    return GptOssRoutedExperts


def _get_streamed_expert_info(
    name: str, params_dict: dict
) -> tuple[int, str, str] | None:
    """Resolve ``...experts.<expert_id>.<fused_param>`` to its stacked param."""
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
    fused_name = f"{prefix}.mlp.experts.routed_experts.{param_suffix}"
    if fused_name not in params_dict:
        return None
    return expert_id, fused_name, shard_id


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


def _route_streamed_experts(
    weights: Iterable[tuple[str, torch.Tensor]],
    params_dict: dict,
    loaded_params: set,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume per-expert keys, yielding the rest for vLLM's own loader.

    Stays a generator so the checkpoint is not buffered a second time.
    """
    for name, weight in weights:
        if not _try_load_streamed_expert(name, weight, params_dict, loaded_params):
            yield name, weight


def _make_patched_load_weights_other(orig_load_weights_other):
    """Wrap ``_load_weights_other``, intercepting only the per-expert keys.

    Every other key reaches the stock method, so rank sharding stays vLLM's.
    """

    @wraps(orig_load_weights_other)
    def _patched(
        self,
        ep_rank_end,
        ep_rank_start,
        heads_per_rank,
        head_start,
        weights,
        stacked_params_mapping,
    ) -> set:
        params_dict = dict(self.named_parameters())
        streamed_params: set[str] = set()
        remaining = _route_streamed_experts(weights, params_dict, streamed_params)
        loaded_params = orig_load_weights_other(
            self,
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            remaining,
            stacked_params_mapping,
        )
        return loaded_params | streamed_params

    return _patched


def _make_patched_mlpblock_init(orig_init, routed_cls):
    """Retype ``MLPBlock``'s RoutedExperts to ``routed_cls`` after construction.

    The subclass only adds methods, so swapping ``__class__`` is enough.
    """

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        experts = getattr(self, "experts", None)
        routed = getattr(experts, "routed_experts", None)
        if routed is not None and not isinstance(routed, routed_cls):
            routed.__class__ = routed_cls
            # create_weights bound the base weight_loader onto every fused
            # param before this swap, so the gpt_oss_* shard ids would still
            # reach the base loader, which rejects them.
            new_loader = routed.weight_loader
            for param in routed.parameters(recurse=True):
                loader = getattr(param, "weight_loader", None)
                if loader is not None and getattr(loader, "__self__", None) is routed:
                    param.weight_loader = new_loader

    return _patched_init


def _extend_native_streamed_loader(mod) -> None:
    """Teach vLLM's own streamed loader about pack-quantized experts.

    A compressed-tensors int4 checkpoint names its per-expert keys
    ``w13/w2_weight_packed``, which the native suffix table omits, so they miss
    the per-expert path and reach a slice that assumes stacked experts.
    """
    table = getattr(mod, "_GPT_OSS_STREAMED_EXPERT_SUFFIX_TO_SHARD", None)
    if table is None:
        logger.debug("[zentorch] native streamed loader not as expected; left alone")
        return
    for suffix, shard in _STREAMED_EXPERT_SUFFIX_TO_SHARD.items():
        table.setdefault(suffix, shard)
    logger.info(
        "[zentorch] Extended native GPT-OSS streamed loader for packed experts"
    )


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

    if (
        getattr(mod, "GptOssRoutedExperts", None) is not None
        and hasattr(gpt_oss_model, "_try_load_streamed_expert")
    ):
        _extend_native_streamed_loader(mod)
        setattr(gpt_oss_model, _MARKER, True)
        return True

    routed_cls = _get_routed_experts_cls()

    gpt_oss_model._load_weights_other = _make_patched_load_weights_other(
        gpt_oss_model._load_weights_other
    )
    mlp_block.__init__ = _make_patched_mlpblock_init(mlp_block.__init__, routed_cls)

    setattr(gpt_oss_model, _MARKER, True)
    logger.info(
        "[zentorch] Patched GptOssModel for per-expert (streamed) checkpoint loading"
    )
    return True


def _apply_gptoss_streamed_expert_patch_impl() -> bool:
    """Arm the streamed-expert loader for GptOssModel (deferred via import hook)."""
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch)
