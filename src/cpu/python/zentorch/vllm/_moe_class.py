# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch
from torch.nn.parameter import Parameter

from zentorch._logging import get_logger

logger = get_logger(__name__)

_TORCHAO_MOE_TARGET_MODULE = "vllm.model_executor.layers.quantization.torchao"


def _register_torchao_moe_patches(torchao_mod) -> None:
    """Install FusedMoE-related changes on the loaded ``torchao_mod``.

    All patched methods and the ``TorchAOFusedMoEMethod`` class are defined
    inside this function (nested-handlers style, mirroring
    ``torchao_int8_patch.py``). vLLM imports are local so this module stays
    importable before vLLM's layer modules exist.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.utils import set_weight_attrs

    TorchAOConfig = torchao_mod.TorchAOConfig
    _get_weight_attrs = torchao_mod._get_weight_attrs
    _restore_weight_attrs = torchao_mod._restore_weight_attrs
    convert_to_packed = torchao_mod.convert_to_packed_tensor_based_on_current_hardware
    should_skip = torchao_mod.should_skip
    torchao_quantize_param_data = torchao_mod.torchao_quantize_param_data

    def _resolve_torchao_config_for_prefix(self, prefix):
        """Return the effective ``TorchAOConfig`` for ``prefix``, or ``None`` to
        use unquantized weights for that module."""
        import re
        from torchao.quantization import ModuleFqnToConfig

        if should_skip(prefix, self.skip_modules):
            return None

        module_fqn = prefix
        if isinstance(self.torchao_config, ModuleFqnToConfig):
            _FQN_CONFIG_MISSING = (
                object()
            )  # Sentinel for dict.get(): None is a valid config value (skip quant).
            module_fqn_to_config = self.torchao_config.module_fqn_to_config
            c = module_fqn_to_config.get(module_fqn, _FQN_CONFIG_MISSING)
            if c is not _FQN_CONFIG_MISSING:
                assert not module_fqn.startswith("re:"), (
                    "module fqn should not start with"
                    "`re:`, which is used for specifying regex"
                )
            else:
                c = None
                regex_matched = False
                for maybe_module_fqn_pattern in module_fqn_to_config:
                    if not maybe_module_fqn_pattern.startswith("re:"):
                        continue
                    if re.fullmatch(maybe_module_fqn_pattern[3:], module_fqn):
                        c = module_fqn_to_config.get(maybe_module_fqn_pattern)
                        regex_matched = True
                        break
                if not regex_matched:
                    c = module_fqn_to_config.get("_default")
            if c is not None:
                return TorchAOConfig(
                    c, self.skip_modules, self.is_checkpoint_torchao_serialized
                )
            return None

        return self

    def _patched_get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            resolved = self._resolve_torchao_config_for_prefix(prefix)
            if resolved is None:
                # Layer is explicitly skipped (e.g. via `modules_to_not_convert`
                # / `skip_modules`); keep it unquantized end-to-end.
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return torchao_mod.TorchAOFusedMoEMethod(resolved, layer.moe_config)

        if not isinstance(layer, LinearBase):
            return None

        resolved = self._resolve_torchao_config_for_prefix(prefix)
        if resolved is None:
            return UnquantizedLinearMethod()
        return torchao_mod.TorchAOLinearMethod(resolved)

    class TorchAOFusedMoEMethod(UnquantizedFusedMoEMethod):
        """MoE method for torchao-quantized expert weights (CPU). Forward
        inherited from :class:`UnquantizedFusedMoEMethod`."""

        def __init__(self, quant_config, moe):
            assert isinstance(moe, FusedMoEConfig)
            super().__init__(moe)
            self.quant_config = quant_config

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            if self.moe.is_act_and_mul:
                w13_up_dim = 2 * intermediate_size_per_partition
            else:
                w13_up_dim = intermediate_size_per_partition

            w13_raw = torch.empty(
                num_experts * w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            )
            w2_raw = torch.empty(
                num_experts * hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            )
            if self.quant_config.is_checkpoint_torchao_serialized:
                # torchao_quantize_param_data uses nn.Linear and returns a 2D
                # tensor. Keep 2D for checkpoint load; view to 3D in
                # process_weights_after_loading (Int8Tensor view is patched).
                w13_raw = torchao_quantize_param_data(
                    Parameter(
                        w13_raw.view(
                            num_experts * w13_up_dim, hidden_size
                        ).contiguous(),
                        requires_grad=False,
                    ),
                    self.quant_config.torchao_config,
                ).view(num_experts, w13_up_dim, hidden_size)
                w2_raw = torchao_quantize_param_data(
                    Parameter(
                        w2_raw.view(
                            num_experts * hidden_size,
                            intermediate_size_per_partition,
                        ).contiguous(),
                        requires_grad=False,
                    ),
                    self.quant_config.torchao_config,
                ).view(num_experts, hidden_size, intermediate_size_per_partition)

            w13_weight = Parameter(w13_raw, requires_grad=False)
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)
            if self.moe.has_bias:
                w13_bias = Parameter(
                    torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                    requires_grad=False,
                )
                layer.register_parameter("w13_bias", w13_bias)
                set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_weight = Parameter(w2_raw, requires_grad=False)
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)
            if self.moe.has_bias:
                w2_bias = Parameter(
                    torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                    requires_grad=False,
                )
                layer.register_parameter("w2_bias", w2_bias)
                set_weight_attrs(w2_bias, extra_weight_attrs)

        def process_weights_after_loading(self, layer):

            num_experts = layer.w13_weight.size(0)
            w13_up_dim = layer.w13_weight.size(1)
            hidden_size = layer.w13_weight.size(2)
            w2_weight = layer.w2_weight
            if w2_weight.dim() == 3:
                intermediate_size_per_partition = w2_weight.size(2)
            else:
                intermediate_size_per_partition = w2_weight.size(1)

            def _pack(name, out_dim, in_dim):
                p = getattr(layer, name)
                rec = _get_weight_attrs(p)
                tensor = p.data if isinstance(p, Parameter) else p
                if not self.quant_config.is_checkpoint_torchao_serialized:
                    stacked = torchao_quantize_param_data(
                        Parameter(
                            tensor.reshape(num_experts * out_dim, in_dim).contiguous(),
                            requires_grad=False,
                        ),
                        self.quant_config.torchao_config,
                    ).view(num_experts, out_dim, in_dim)
                    packed = convert_to_packed(stacked)
                else:
                    if tensor.dim() == 2:
                        tensor = tensor.view(num_experts, out_dim, in_dim)
                    packed = convert_to_packed(tensor)
                new_p = Parameter(packed, requires_grad=False)
                layer.register_parameter(name, new_p)
                _restore_weight_attrs(getattr(layer, name), rec)

            _pack("w13_weight", w13_up_dim, hidden_size)
            _pack("w2_weight", hidden_size, intermediate_size_per_partition)
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)

    TorchAOFusedMoEMethod.__module__ = torchao_mod.__name__

    torchao_mod.TorchAOFusedMoEMethod = TorchAOFusedMoEMethod
    TorchAOConfig._resolve_torchao_config_for_prefix = (
        _resolve_torchao_config_for_prefix
    )
    TorchAOConfig.get_quant_method = _patched_get_quant_method


def _apply_torchao_moe_patch_to_module(torchao_mod) -> bool:
    """Install the FusedMoE monkey-patch on an already-loaded
    ``vllm.model_executor.layers.quantization.torchao`` module.

    Idempotent via the ``_zentorch_moe_patched`` flag on ``TorchAOConfig``.
    """
    try:
        if getattr(torchao_mod.TorchAOConfig, "_zentorch_moe_patched", False):
            return True
        _register_torchao_moe_patches(torchao_mod)
        torchao_mod.TorchAOConfig._zentorch_moe_patched = True
        logger.info(
            "[zentorch] Patched vLLM TorchAOConfig: FusedMoE -> "
            "TorchAOFusedMoEMethod"
        )
        return True
    except Exception:
        logger.warning("[zentorch] torchao FusedMoE patch FAILED", exc_info=True)
        return False


_MOE_HOOK_INSTALLED = False


class _TorchAOMoeImportHook:
    """Defer patch until vLLM's torchao quant module loads (avoids circular import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TORCHAO_MOE_TARGET_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _apply_torchao_moe_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_torchao_moe_patch_impl() -> bool:
    """Schedule FusedMoE patch on first import of ``vllm...quantization.torchao``."""
    global _MOE_HOOK_INSTALLED

    torchao_mod = sys.modules.get(_TORCHAO_MOE_TARGET_MODULE)
    if torchao_mod is not None:
        return _apply_torchao_moe_patch_to_module(torchao_mod)

    if not _MOE_HOOK_INSTALLED:
        sys.meta_path.insert(0, _TorchAOMoeImportHook())
        _MOE_HOOK_INSTALLED = True
    return True
