# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Out-of-tree W8A8 INT8 fused-MoE via zentorch (no vLLM changes).

Patches ``CompressedTensorsW8A8Int8MoEMethod`` to run the FFN through
``zentorch_fused_moe``, as monolithic experts on the standard FusedMoE path.
"""

from __future__ import annotations

import inspect

import torch

from zentorch._logging import get_logger
from zentorch._utils import _SUPPORTED_MOE_ACTIVATIONS
from zentorch.vllm._moe_patch_utils import (
    allocate_expert_biases,
    import_select_experts,
    run_moe_patch_apply,
    run_select_experts,
    schedule_module_patches,
)

logger = get_logger(__name__)

_TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
_TARGET_CLASS = "CompressedTensorsW8A8Int8MoEMethod"


# --------------------------------------------------------------------------- #
# Opaque, torch.compile-safe dispatch op. Scales/biases are passed as tensors
# (no Python handles) so the compiled graph carries no process-local state.
# --------------------------------------------------------------------------- #
@torch.library.custom_op("zentorch_vllm::cpu_int8_moe", mutates_args={"output"})
def _cpu_int8_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
) -> None:
    x = hidden_states
    if apply_router_weight_on_input:
        top_k = topk_ids.shape[1]
        if top_k != 1:
            raise NotImplementedError(
                "zen int8 MoE: apply_router_weight_on_input=True is only "
                f"supported for top_k=1 (got top_k={top_k})."
            )
        x = hidden_states.mul(topk_weights.to(hidden_states.dtype))

    torch.ops.zentorch.zentorch_fused_moe(
        output,
        x,
        w1,
        w2,
        w13_bias,
        w2_bias,
        topk_weights.to(torch.float32).contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        apply_router_weight_on_input,  # skip_weighted
        activation,
        w13_scale,
        w2_scale,
    )


@_cpu_int8_moe.register_fake
def _cpu_int8_moe_fake(
    output,
    hidden_states,
    w1,
    w2,
    w13_scale,
    w2_scale,
    w13_bias,
    w2_bias,
    topk_weights,
    topk_ids,
    activation,
    apply_router_weight_on_input,
) -> None:
    return None


# --------------------------------------------------------------------------- #
# Method patch (nested-handlers style, mirrors _moe_class.py). vLLM imports are
# local so this module stays importable before vLLM's layers exist.
# --------------------------------------------------------------------------- #
def _register_int8_moe_patches(mod) -> None:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.oracle.int8 import (
        Int8MoeBackend,
        make_int8_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        QuantKey,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import_select_experts()

    class CPUInt8Experts(mk.FusedMoEExpertsMonolithic):
        """CPU FusedMoE experts for W8A8 int8 dispatching through zentorch."""

        def __init__(self, moe_config, quant_config):
            super().__init__(moe_config, quant_config)
            assert (
                self.w1_scale is not None and self.w2_scale is not None
            ), "CPUInt8Experts requires per-channel weight scales on the layer."
            E = self.w1_scale.shape[0]
            self._w13_scale = (
                self.w1_scale.detach().to(torch.bfloat16).reshape(E, -1).contiguous()
            )
            self._w2_scale = (
                self.w2_scale.detach().to(torch.bfloat16).reshape(E, -1).contiguous()
            )
            self._w13_bias = (
                None
                if self.w1_bias is None
                else self.w1_bias.detach().to(torch.bfloat16).contiguous()
            )
            self._w2_bias = (
                None
                if self.w2_bias is None
                else self.w2_bias.detach().to(torch.bfloat16).contiguous()
            )

        @staticmethod
        def activation_format() -> mk.FusedMoEActivationFormat:
            return mk.FusedMoEActivationFormat.Standard

        @property
        def expects_unquantized_inputs(self) -> bool:
            # zentorch_fused_moe quantizes activations itself.
            return True

        @staticmethod
        def _supports_current_device() -> bool:
            return has_zentorch_op(["zentorch_fused_moe"])

        @staticmethod
        def _supports_no_act_and_mul() -> bool:
            return False

        @staticmethod
        def _supports_quant_scheme(
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
        ) -> bool:
            return (
                weight_key == kInt8StaticChannelSym
                and activation_key == kInt8DynamicTokenSym
            )

        @staticmethod
        def _supports_activation(activation: MoEActivation) -> bool:
            return activation.value in _SUPPORTED_MOE_ACTIVATIONS

        @staticmethod
        def _supports_parallel_config(
            moe_parallel_config: FusedMoEParallelConfig,
        ) -> bool:
            return True

        @staticmethod
        def _supports_routing_method(
            routing_method: RoutingMethodType,
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
        ) -> bool:
            # Routing runs in select_experts(), so opt into CPUExpertsInt8's set.
            # vllm/model_executor/layers/fused_moe/experts/cpu_moe.py
            return routing_method in (
                RoutingMethodType.Default,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            )

        @staticmethod
        def _supports_router_logits_dtype(
            router_logits_dtype: torch.dtype | None,
            routing_method: RoutingMethodType,
        ) -> bool:
            return True

        def supports_expert_map(self) -> bool:
            return False

        def apply(
            self,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            router_logits: torch.Tensor,
            activation: MoEActivation,
            global_num_experts: int,
            expert_map: torch.Tensor | None,
            a1q_scale: torch.Tensor | None,
            apply_router_weight_on_input: bool,
            num_expert_group: int | None = None,
            e_score_correction_bias: torch.Tensor | None = None,
            routed_scaling_factor: float | None = None,
            topk_group: int | None = None,
        ) -> torch.Tensor:
            topk_weights, topk_ids = run_select_experts(
                hidden_states,
                router_logits,
                self.moe_config,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                topk_group=topk_group,
            )
            output = torch.empty_like(hidden_states)
            torch.ops.zentorch_vllm.cpu_int8_moe(
                output,
                hidden_states,
                w1,
                w2,
                self._w13_scale,
                self._w2_scale,
                self._w13_bias,
                self._w2_bias,
                topk_weights,
                topk_ids,
                activation.value,
                apply_router_weight_on_input,
            )
            return output

    method_cls = getattr(mod, _TARGET_CLASS)
    orig_init = method_cls.__init__
    orig_create_weights = method_cls.create_weights
    orig_process_weights = method_cls.process_weights_after_loading
    int8_quant_config_params = inspect.signature(make_int8_moe_quant_config).parameters

    def _zen_init(self, *args, **kwargs):
        # select_int8_moe_backend() has no out-of-tree hook and raises on Zen CPU.
        # vllm/model_executor/layers/fused_moe/oracle/int8.py
        saved = getattr(mod, "select_int8_moe_backend", None)
        if saved is not None:
            mod.select_int8_moe_backend = lambda *a, **k: (None, None)
        try:
            orig_init(self, *args, **kwargs)
        finally:
            if saved is not None:
                mod.select_int8_moe_backend = saved
        self.experts_cls = CPUInt8Experts
        # From 0.25 on the helpers take a backend; before that there is no CPU
        # member to name.
        self.int8_backend = getattr(Int8MoeBackend, "CPU", None)

    def _zen_create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Capture the model dtype before the original overrides it to int8; the
        # biases stay in the model (bf16) dtype.
        bias_dtype = params_dtype
        orig_create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # The int8 method never allocates per-expert biases (e.g. gpt-oss).
        # vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py
        allocate_expert_biases(
            layer,
            self.moe,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=bias_dtype,
            extra_weight_attrs=extra_weight_attrs,
        )

    def _zen_get_fused_moe_quant_config(self, layer) -> "FusedMoEQuantConfig":
        # make_int8_moe_quant_config takes w1_bias/w2_bias; the method never does.
        # vllm/model_executor/layers/fused_moe/oracle/int8.py
        quant_config_kwargs = {
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "a1_scale": layer.w13_input_scale,
            "a2_scale": layer.w2_input_scale,
            "w1_bias": getattr(layer, "w13_bias", None),
            "w2_bias": getattr(layer, "w2_bias", None),
            "per_act_token_quant": True,
        }
        # vLLM 0.25 adds int8_backend to the quant-config constructor; keep
        # older supported releases on the original call shape.
        if "int8_backend" in int8_quant_config_params:
            quant_config_kwargs["int8_backend"] = self.int8_backend
        return make_int8_moe_quant_config(
            **quant_config_kwargs,
        )

    def _maybe_permute_swigluoai(layer) -> None:
        _act = getattr(layer, "activation", None)
        _act_str = getattr(_act, "value", _act)
        if not (isinstance(_act_str, str) and _act_str.lower() == "swigluoai"):
            return
        two_i = layer.w13_weight.size(1)
        i = two_i // 2
        device = layer.w13_weight.device
        perm = torch.stack(
            [
                torch.arange(0, i, device=device),
                torch.arange(i, two_i, device=device),
            ],
            dim=1,
        ).flatten()
        has_w13_bias = getattr(layer, "w13_bias", None) is not None
        logger.info(
            "[zentorch][swigluoai-permute] Reordering w13 half-split -> "
            "interleaved for ZenDNN swiglu_oai_mul: E=%d, 2I=%d, I=%d, "
            "has_w13_bias=%s",
            layer.w13_weight.size(0),
            two_i,
            i,
            has_w13_bias,
        )
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data[:, perm, :].contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data[:, perm].contiguous(), requires_grad=False
        )
        if has_w13_bias:
            layer.w13_bias = torch.nn.Parameter(
                layer.w13_bias.data[:, perm].contiguous(), requires_grad=False
            )

    def _keep_weights_unpacked(int8_backend, w13, w2, layer=None, w13_scale=None):
        return w13, w2

    def _zen_process_weights_after_loading(self, layer) -> None:
        # Relayout first: the original then reads w13 scales/biases to build
        # the quant config and the MoE kernel.
        _maybe_permute_swigluoai(layer)
        # vLLM <= 0.26.0 VNNI-prepacks w13/w2; zentorch needs plain [N, K] int8.
        # TODO: drop once 0.26.0 and earlier are out of the supported set.
        # vllm/model_executor/layers/fused_moe/oracle/int8.py
        saved = getattr(mod, "convert_to_int8_moe_kernel_format", None)
        if saved is not None:
            mod.convert_to_int8_moe_kernel_format = _keep_weights_unpacked
        try:
            orig_process_weights(self, layer)
        finally:
            if saved is not None:
                mod.convert_to_int8_moe_kernel_format = saved
        logger.info(
            "[zentorch] W8A8 int8 MoE kernel built via OOT patch "
            "(experts=%d, has_bias=%s)",
            layer.w13_weight.shape[0],
            getattr(layer, "w13_bias", None) is not None,
        )

    def _zen_apply_monolithic(self, layer, x, router_logits, input_ids=None):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    method_cls.__init__ = _zen_init
    method_cls.create_weights = _zen_create_weights
    method_cls.get_fused_moe_quant_config = _zen_get_fused_moe_quant_config
    method_cls.process_weights_after_loading = _zen_process_weights_after_loading
    # On 0.22.1-0.24 the base apply_monolithic only raises, so supply upstream's
    # forwarding body. TODO: drop once VLLM_MIN_VERSION moves past 0.24.
    if method_cls.apply_monolithic is FusedMoEMethodBase.apply_monolithic:
        method_cls.apply_monolithic = _zen_apply_monolithic


def _apply_int8_moe_patch_to_module(mod) -> bool:
    def _ops_guard() -> str | None:
        zt = getattr(torch.ops, "zentorch", None)
        if zt is None or not hasattr(zt, "zentorch_fused_moe"):
            return (
                "[zentorch] zentorch_fused_moe not available; leaving vLLM's "
                "int8 MoE backend unpatched."
            )
        return None

    return run_moe_patch_apply(
        mod,
        target=getattr(mod, _TARGET_CLASS, None),
        flag="_zentorch_int8_moe_patched",
        register_fn=_register_int8_moe_patches,
        success_log=(
            f"[zentorch] Patched {_TARGET_CLASS}: "
            "monolithic zentorch W8A8 int8 MoE"
        ),
        fail_log="[zentorch] int8 MoE patch FAILED",
        missing_log=(
            f"[zentorch] {_TARGET_CLASS} not found in {_TARGET_MODULE}; "
            "int8 MoE patch skipped"
        ),
        extra_guard=_ops_guard,
    )


def _apply_int8_moe_patch_impl() -> bool:
    """Schedule the W8A8 int8 MoE patch on first import of the target module."""
    return schedule_module_patches(
        {_TARGET_MODULE: _apply_int8_moe_patch_to_module},
    )
