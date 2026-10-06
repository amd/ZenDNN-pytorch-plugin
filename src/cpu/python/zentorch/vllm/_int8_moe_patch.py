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
    permute_swigluoai_w13,
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
    from vllm.model_executor.layers.fused_moe.experts import cpu_moe
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
                self.w1_scale.detach().to(torch.float32).reshape(E, -1).contiguous()
            )
            self._w2_scale = (
                self.w2_scale.detach().to(torch.float32).reshape(E, -1).contiguous()
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
            if native_zen_experts_cls is None:
                return True
            return getattr(moe_parallel_config, "ep_size", 1) == 1

        @staticmethod
        def _supports_routing_method(
            routing_method: RoutingMethodType,
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
        ) -> bool:
            # Routing runs in select_experts() for the in-tree methods;
            # Custom (Gemma4) is handled via custom_routing_function in apply().
            return routing_method in (
                RoutingMethodType.Default,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
                RoutingMethodType.Custom,
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
                custom_routing_fn=getattr(self, "_custom_routing_fn", None),
                renormalize=getattr(self, "_renormalize", None),
            )
            x = hidden_states
            if apply_router_weight_on_input:
                top_k = topk_ids.shape[1]
                if top_k != 1:
                    raise NotImplementedError(
                        "zen int8 MoE: apply_router_weight_on_input=True is only "
                        f"supported for top_k=1 (got top_k={top_k})."
                    )
                x = hidden_states.mul(topk_weights.to(hidden_states.dtype))

            # The schema declares ``Tensor(a!) output`` and zentorch registers a
            # meta kernel, so this traces without a wrapper op.
            output = torch.empty_like(hidden_states)
            torch.ops.zentorch.zentorch_fused_moe(
                output,
                x,
                w1,
                w2,
                self._w13_bias,
                self._w2_bias,
                topk_weights.to(torch.float32).contiguous(),
                topk_ids.to(torch.int32).contiguous(),
                apply_router_weight_on_input,  # skip_weighted
                activation.value,
                self._w13_scale,
                self._w2_scale,
            )
            return output

    method_cls = getattr(mod, _TARGET_CLASS)
    orig_init = method_cls.__init__
    orig_create_weights = method_cls.create_weights
    orig_get_fused_moe_quant_config = method_cls.get_fused_moe_quant_config
    orig_process_weights = method_cls.process_weights_after_loading
    int8_quant_config_params = inspect.signature(make_int8_moe_quant_config).parameters
    native_zen_experts_cls = getattr(cpu_moe, "ZenCPUExpertsInt8", None)

    def _zen_init(self, *args, **kwargs):
        moe = kwargs.get("moe", args[2] if len(args) > 2 else None)
        use_native = (
            native_zen_experts_cls is not None
            and moe is not None
            and moe.routing_method != RoutingMethodType.Custom
        )
        if use_native:
            orig_init(self, *args, **kwargs)
            self._zentorch_uses_native_int8 = True
            return

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
        if getattr(self, "_zentorch_uses_native_int8", False):
            return
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
        if getattr(self, "_zentorch_uses_native_int8", False):
            return orig_get_fused_moe_quant_config(self, layer)

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

    def _keep_weights_unpacked(int8_backend, w13, w2, layer=None, w13_scale=None):
        return w13, w2

    def _zen_process_weights_after_loading(self, layer) -> None:
        if getattr(self, "_zentorch_uses_native_int8", False):
            orig_process_weights(self, layer)
            # In-tree ZenCPUExpertsInt8 reaches the same ZenDNN reorder, so it
            # needs the slack too. Skipped when the routed kernel will read it.
            _pad_moe_weights_for_inplace(layer)
            return

        # Relayout first: the original then reads w13 scales/biases to build
        # the quant config and the MoE kernel.
        permute_swigluoai_w13(layer)
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
        _pad_moe_weights_for_inplace(layer)
        kernel = getattr(self, "moe_kernel", None)
        experts = getattr(kernel, "fused_experts", None) if kernel is not None else None
        if experts is not None:
            experts._custom_routing_fn = getattr(
                layer, "custom_routing_function", None
            )
            experts._renormalize = getattr(layer, "renormalize", True)
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


def _pad_experts_for_inplace(w: torch.Tensor) -> torch.Tensor:
    """Re-allocate an [E, N, K] int8 weight with per-expert trailing slack.

    ZenDNN's blocked layout needs ``N * 4`` bytes beyond the weight for its
    compensation row, so an exactly-sized tensor never passes the in-place gate
    and every expert weight is duplicated. The gap has to be per-expert: in a
    packed tensor the bytes after one expert belong to the next.
    """
    if w.dim() != 3 or w.dtype != torch.int8 or not w.is_contiguous():
        return w
    E, N, K = w.shape
    per_expert = N * K  # elements
    # Bytes, not elements: the row is one int32 per output column whatever the
    # weight dtype. Rounded to 64 to match the reorder's overscan, plus a spare
    # line so an exactly-equal fit is not the only passing case.
    esz = w.element_size()
    pad_bytes = ((N * 4 + 63) // 64) * 64 + 64
    pad = (pad_bytes + esz - 1) // esz
    storage = torch.empty(E * (per_expert + pad), dtype=w.dtype, device=w.device)
    out = storage.as_strided((E, N, K), (per_expert + pad, K, 1))
    out.copy_(w)
    return out


def _pad_moe_weights_for_inplace(layer) -> None:
    """Trade ``E * N * 4`` bytes per projection for the in-place reorder: on a
    256-expert Qwen3.6 that is 120 MiB against a 30 GiB duplicate."""
    padded = []
    for name in ("w13_weight", "w2_weight"):
        w = getattr(layer, name, None)
        if w is None:
            continue
        new_w = _pad_experts_for_inplace(w.data)
        if new_w.data_ptr() == w.data.data_ptr():
            continue  # not eligible (dtype/layout); left untouched
        setattr(layer, name, torch.nn.Parameter(new_w, requires_grad=False))
        padded.append(name)
    if padded:
        logger.info(
            "[zentorch] MoE weights re-allocated with per-expert slack for the "
            "in-place int8 reorder: %s",
            ", ".join(padded),
        )


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
