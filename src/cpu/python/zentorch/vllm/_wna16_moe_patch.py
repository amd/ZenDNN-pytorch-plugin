# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Out-of-tree DA8W4 WNA16 MoE for vLLM 0.27+ (``CompressedTensorsWNA16MoEMethod``).

Prepends ``ZentorchExpertsInt4DA8W4`` to the WNA16 oracle, patches the method
for per-expert bias allocation and WOQ repack, and routes through
``zentorch_fused_moe``. Enabled by default; ``VLLM_CPU_INT4_W4A8=0`` keeps the
native ``CPUExpertsInt4`` backend.
"""

from __future__ import annotations

import os
import sys

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

_ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"

_METHOD_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16"
)
_METHOD_CLASS = "CompressedTensorsWNA16MoEMethod"

_DA8W4_MOE_OPS = ["zentorch_fused_moe", "zentorch_woq_repack_weight"]


def use_zentorch_da8w4_moe() -> bool:
    """True unless disabled by env var, and the Zen CPU + ops are available.

    On by default, matching the DA8W4 linear path so one variable governs
    both; ``VLLM_CPU_INT4_W4A8=0`` forces W4A16.
    """
    if os.environ.get("VLLM_CPU_INT4_W4A8", "1") == "0":
        return False
    try:
        from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op

        return has_zentorch_op(_DA8W4_MOE_OPS)
    except Exception:
        logger.debug(
            "[zentorch] DA8W4 MoE gate: has_zentorch_op probe failed",
            exc_info=True,
        )
        return False


# --------------------------------------------------------------------------- #
# Experts class. Built lazily: its base lives in a vLLM module that must not be
# imported at plugin-registration time.
# --------------------------------------------------------------------------- #

_EXPERTS_CLS = None


def _get_zentorch_experts_cls():
    """Build (once) and return ``ZentorchExpertsInt4DA8W4``."""
    global _EXPERTS_CLS
    if _EXPERTS_CLS is not None:
        return _EXPERTS_CLS

    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        QuantKey,
        kInt4Static,
    )

    import_select_experts()

    class ZentorchExpertsInt4DA8W4(mk.FusedMoEExpertsMonolithic):
        """Zen CPU DA8W4 group-quantized monolithic MoE experts.

        Symmetric int4 weights in the zentorch WOQ layout; the bf16 activation
        is quantized to s8 per token inside the kernel and the matmul runs as
        s8 x s4 -> bf16 on the AOCL-DLP backend.
        """

        def __init__(
            self,
            moe_config: FusedMoEConfig,
            quant_config: FusedMoEQuantConfig,
            max_num_tokens: int | None = None,
            num_dispatchers: int | None = None,
        ):
            super().__init__(
                moe_config,
                quant_config,
            )
            # w1_bias / w2_bias are read-only properties on FusedMoEExperts;
            # they already return None when the quant config has no bias.
            # apply() passes them through to the op.

        @property
        def expects_unquantized_inputs(self) -> bool:
            return True

        @staticmethod
        def activation_format() -> mk.FusedMoEActivationFormat:
            return mk.FusedMoEActivationFormat.Standard

        @staticmethod
        def _supports_current_device() -> bool:
            # has_zentorch_op already requires is_zen_cpu() plus the ops.
            return use_zentorch_da8w4_moe()

        @staticmethod
        def _supports_no_act_and_mul() -> bool:
            return False

        @staticmethod
        def _supports_activation(activation: MoEActivation) -> bool:
            # The enum value goes straight through to the op's gated-act post-op
            # (GroupMatmul.cpp::map_activation_to_gated_act).
            return activation.value in _SUPPORTED_MOE_ACTIVATIONS

        @staticmethod
        def _supports_parallel_config(
            moe_parallel_config: FusedMoEParallelConfig,
        ) -> bool:
            # The monolithic op takes no expert_map and indexes the stacked
            # weights by global expert id. This is the only parallel gate the
            # oracle consults, so True under EP would fail later in apply().
            return moe_parallel_config.ep_size == 1

        @staticmethod
        def _supports_quant_scheme(
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
        ) -> bool:
            # The bf16 activation is quantized at runtime, hence no static
            # activation key.
            SUPPORTED_W_A = [
                (kInt4Static, None),
            ]
            return (weight_key, activation_key) in SUPPORTED_W_A

        @staticmethod
        def _supports_routing_method(
            routing_method: RoutingMethodType,
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
        ) -> bool:
            return routing_method in [
                RoutingMethodType.Default,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ]

        @staticmethod
        def _supports_router_logits_dtype(
            router_logits_dtype: torch.dtype | None,
            routing_method: RoutingMethodType,
        ) -> bool:
            return True

        def supports_expert_map(self) -> bool:
            # Base-class parity only; _supports_parallel_config is the gate that
            # keeps this class off EP layers.
            return False

        # A staticmethod taking cls explicitly, as the base declares it and the
        # oracle calls it: k_cls.is_supported_config(k_cls, config, ...). A
        # classmethod would bind cls and shift the arguments along by one.
        @staticmethod
        def is_supported_config(
            cls: type[mk.FusedMoEExperts],
            moe_config: FusedMoEConfig,
            weight_key: QuantKey | None,
            activation_key: QuantKey | None,
            activation_format: mk.FusedMoEActivationFormat,
        ) -> tuple[bool, str | None]:
            supported, reason = mk.FusedMoEExpertsMonolithic.is_supported_config(
                cls, moe_config, weight_key, activation_key, activation_format
            )
            if not supported:
                return supported, reason
            # FusedMoE.cpp guards E_a > 1: with top_k == 1 a single-token decode
            # routes everything to one expert and trips it. Declining lets the
            # oracle fall through to CPUExpertsInt4.
            if moe_config.experts_per_token < 2:
                return False, (
                    "kernel does not support experts_per_token "
                    f"{moe_config.experts_per_token} (< 2 active experts)"
                )
            return True, None

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
            # grouped topk + fused topk bias parameters
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

            assert (
                activation.value in _SUPPORTED_MOE_ACTIVATIONS
            ), f"zentorch DA8W4 fused MoE does not support activation {activation}"

            # DA8W4 kernel is bf16-in / bf16-out.
            x = hidden_states.to(torch.bfloat16)
            if apply_router_weight_on_input:
                assert (
                    topk_ids.shape[1] == 1
                ), "apply_router_weight_on_input requires top_k == 1"
                x = x * topk_weights.to(x.dtype)

            x_c = x.contiguous()
            output = torch.empty_like(x_c)

            # The op carries no expert_map arg, so assert the contract rather
            # than silently dropping a non-trivial map.
            assert expert_map is None, (
                "zentorch DA8W4 fused MoE does not support expert_map "
                "(_supports_parallel_config() rejects ep_size > 1)"
            )

            torch.ops.zentorch.zentorch_fused_moe(
                output,
                x_c,
                w1,  # w13  [E, 2*I, H/8] int32 (packed s4)
                w2,  # w2   [E, H, I/8] int32 (packed s4)
                self.w1_bias,  # [E, 2*I] bf16 or None
                self.w2_bias,  # [E, H] bf16 or None
                topk_weights.to(torch.float32).contiguous(),
                topk_ids.to(torch.int32).contiguous(),
                bool(apply_router_weight_on_input),  # skip_weighted
                activation.value,
                self.w1_scale,  # w13_scales  [E, num_g_w13, 2*I]
                self.w2_scale,  # w2_scales   [E, num_g_w2, H]
            )
            return output.to(hidden_states.dtype)

    _EXPERTS_CLS = ZentorchExpertsInt4DA8W4
    return _EXPERTS_CLS


def _is_zentorch_experts(experts_cls) -> bool:
    """True iff ``experts_cls`` is the (already built) zentorch experts class."""
    return _EXPERTS_CLS is not None and experts_cls is _EXPERTS_CLS


# --------------------------------------------------------------------------- #
# Weight post-processing
# --------------------------------------------------------------------------- #


def _da8w4_repack_unsupported_reason(quant_config) -> str | None:
    """Why this checkpoint cannot use the zentorch WOQ repack, or None if it can.

    ``symmetric is False`` is the compressed-tensors flag (asymmetric /
    unsigned int4). Leftover zero-point tensors on a ``symmetric=True``
    config are ignored, matching the linear DA8W4 path. GROUP act-order
    needs a runtime ``g_idx`` the op does not implement; WEIGHT act-order
    is baked into the saved weights and is fine.
    """
    if getattr(quant_config, "symmetric", True) is False:
        return "asymmetric / unsigned int4 checkpoints are not supported"
    if getattr(quant_config, "actorder", None) == "group":
        return "group act-ordering is not supported"
    return None


def _native_wna16_cpu_experts():
    """First non-zentorch CPU experts class from the (possibly patched) oracle.

    ``backend_to_kernel_cls`` lives on the oracle module in vLLM 0.27, not on
    the compressed-tensors method module.
    """
    oracle = sys.modules.get(_ORACLE_MODULE)
    backend = getattr(oracle, "WNA16MoEBackend", None)
    to_cls = getattr(oracle, "backend_to_kernel_cls", None)
    if backend is None or to_cls is None:
        return None
    try:
        classes = to_cls(backend.CPU)
    except Exception:
        return None
    for cls in classes or ():
        if not _is_zentorch_experts(cls):
            return cls
    return None


def _process_weights_zentorch(
    quant_config,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Repack symmetric int4 MoE weights into the zentorch WOQ layout:

        w13: ``[E, K//8, 2*N]`` int32  ->  ``[E, 2*N, K//8]`` int32
        w2:  ``[E, N//8, K]``   int32  ->  ``[E, K,   N//8]`` int32

    Per expert: unpack to signed int8 ``[K, N]``, transpose, repack with
    ``zentorch_woq_repack_weight``. Per-group scales ``[E, num_groups, N]``
    already match what the op expects and pass through unchanged. DA8W4 is
    symmetric only; leftover zero-point tensors on a symmetric config are
    ignored.
    """
    from compressed_tensors.quantization import QuantizationArgs

    if not isinstance(quant_config, QuantizationArgs):
        raise TypeError(
            "Zentorch DA8W4 MoE backend requires compressed-tensors "
            f"QuantizationArgs, got {type(quant_config).__name__}."
        )
    reason = _da8w4_repack_unsupported_reason(quant_config)
    if reason is not None:
        raise NotImplementedError(f"Zentorch DA8W4 MoE backend: {reason}")

    # Reuse vLLM's own cross-version compressed-tensors importer.
    from vllm.model_executor.kernels.linear.mixed_precision.zentorch import (
        _import_unpack_from_int32,
    )

    unpack_from_int32 = _import_unpack_from_int32()
    repack_op = torch.ops.zentorch.zentorch_woq_repack_weight.default
    bits = 4

    def _repack_stacked(w_packed: torch.Tensor) -> torch.Tensor:
        # [E, K//8, N] int32, packed along K -> [E, N, K//8] in the WOQ layout.
        num_experts = w_packed.shape[0]
        k_packed, n = w_packed.shape[1], w_packed.shape[2]
        k = k_packed * (32 // bits)  # 8 nibbles per int32
        repacked_experts = []
        for e in range(num_experts):
            unpacked = unpack_from_int32(
                w_packed[e].contiguous(),
                bits,
                torch.Size([k, n]),
                packed_dim=0,
            )
            repacked_experts.append(repack_op(unpacked.t().to(torch.int8).contiguous()))
        return torch.stack(repacked_experts, dim=0).contiguous()

    return (
        _repack_stacked(w13),
        _repack_stacked(w2),
        w13_scale.contiguous(),
        w2_scale.contiguous(),
        # Bias must match the bf16 activation; convert once, not per forward.
        w13_bias.to(torch.bfloat16) if w13_bias is not None else None,
        w2_bias.to(torch.bfloat16) if w2_bias is not None else None,
    )


# --------------------------------------------------------------------------- #
# Oracle patch: offer the zentorch experts ahead of CPUExpertsInt4.
# --------------------------------------------------------------------------- #


def _register_oracle_patch(mod) -> None:
    orig_backend_to_kernel_cls = mod.backend_to_kernel_cls

    def _zen_backend_to_kernel_cls(backend):
        classes = orig_backend_to_kernel_cls(backend)
        if backend == mod.WNA16MoEBackend.CPU and use_zentorch_da8w4_moe():
            # The oracle walks this list in order and keeps the first class
            # whose is_supported_config passes, so this is only a preference.
            return [_get_zentorch_experts_cls(), *classes]
        return classes

    mod.backend_to_kernel_cls = _zen_backend_to_kernel_cls


def _apply_oracle_patch_to_module(mod) -> bool:
    has_cls = hasattr(mod, "backend_to_kernel_cls")
    return run_moe_patch_apply(
        mod,
        target=mod if has_cls else None,
        flag="_zentorch_wna16_oracle_patched",
        register_fn=_register_oracle_patch,
        success_log=(
            "[zentorch] Patched WNA16 MoE oracle: DA8W4 experts offered on CPU"
        ),
        fail_log="[zentorch] WNA16 MoE oracle patch FAILED",
        missing_log=(
            f"[zentorch] backend_to_kernel_cls not found in {_ORACLE_MODULE}; "
            "DA8W4 MoE oracle patch skipped"
        ),
    )


# --------------------------------------------------------------------------- #
# Quant-method patch: biases, weight repack, kernel construction.
# --------------------------------------------------------------------------- #


def _modular_method_cls(mod):
    """Return ``CompressedTensorsWNA16MoEMethod`` from ``mod``, or ``None``.

    ``make_wna16_moe_kernel`` is imported onto this module from the oracle in
    vLLM 0.27; its presence is the check that we have the modular method.
    """
    if getattr(mod, "__name__", "") != _METHOD_MODULE:
        return None
    if not hasattr(mod, "make_wna16_moe_kernel"):
        return None
    return getattr(mod, _METHOD_CLASS, None)


def _register_wna16_method_patch(mod) -> None:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.utils import replace_parameter

    method_cls = _modular_method_cls(mod)
    assert method_cls is not None
    orig_create_weights = method_cls.create_weights
    orig_process = method_cls.process_weights_after_loading
    orig_make_kernel = mod.make_wna16_moe_kernel

    def _zen_make_wna16_moe_kernel(
        moe_quant_config, moe_config, experts_cls, *args, **kwargs
    ):
        # Vanilla asserts experts_cls against a hardcoded tuple of in-tree
        # classes, so build the kernel here and delegate everything else.
        if not _is_zentorch_experts(experts_cls):
            return orig_make_kernel(
                moe_quant_config, moe_config, experts_cls, *args, **kwargs
            )

        prepare_finalize = maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=moe_quant_config,
            routing_tables=kwargs.get("routing_tables"),
            allow_new_interface=True,
            use_monolithic=True,
        )
        assert prepare_finalize is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )
        return mk.FusedMoEKernel(prepare_finalize, experts)

    def _zen_create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        orig_create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # Vanilla WNA16 never allocates per-expert biases, so the loader has no
        # target and bias-capable backends cannot apply them. Biases stay in the
        # model dtype; only the packed weights are int32.
        allocate_expert_biases(
            layer,
            self.moe,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            extra_weight_attrs=extra_weight_attrs,
        )

    def _zen_get_fused_moe_quant_config(self, layer):
        # Vanilla plus w1_bias/w2_bias; backends without bias support ignore
        # them.
        return mod.make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.group_size,
            num_bits=self.num_bits,
            w1_zp=getattr(layer, "w13_weight_zero_point", None),
            w2_zp=getattr(layer, "w2_weight_zero_point", None),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
        )

    def _zen_process_weights_after_loading(self, layer) -> None:
        if not _is_zentorch_experts(getattr(self, "experts_cls", None)):
            orig_process(self, layer)
            return

        reason = _da8w4_repack_unsupported_reason(self.weight_quant)
        if reason is not None:
            native_cls = _native_wna16_cpu_experts()
            if native_cls is None:
                raise NotImplementedError(
                    f"Zentorch DA8W4 MoE backend: {reason}"
                )
            logger.warning(
                "[zentorch] DA8W4 MoE skipping zentorch path (%s); "
                "falling back to the native WNA16 backend",
                reason,
            )
            self.experts_cls = native_cls
            orig_process(self, layer)
            return

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_bias,
            w2_bias,
        ) = _process_weights_zentorch(
            self.weight_quant,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            w13_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

        replace_parameter(layer, "w13_weight_packed", w13_qweight)
        replace_parameter(layer, "w2_weight_packed", w2_qweight)
        replace_parameter(layer, "w13_weight_scale", w13_scales)
        replace_parameter(layer, "w2_weight_scale", w2_scales)
        if w13_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
        if w2_bias is not None:
            replace_parameter(layer, "w2_bias", w2_bias)

        # Alias packed weights to w13_weight/w2_weight for the modular kernel.
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        self._setup_kernel(layer)
        logger.info(
            "[zentorch] W4A8 (DA8W4) MoE kernel built via OOT patch "
            "(experts=%d, has_bias=%s)",
            layer.w13_weight.shape[0],
            w13_bias is not None,
        )

    mod.make_wna16_moe_kernel = _zen_make_wna16_moe_kernel
    method_cls.create_weights = _zen_create_weights
    method_cls.get_fused_moe_quant_config = _zen_get_fused_moe_quant_config
    method_cls.process_weights_after_loading = _zen_process_weights_after_loading


def _apply_wna16_method_patch_to_module(mod) -> bool:
    cls = _modular_method_cls(mod)
    return run_moe_patch_apply(
        mod,
        target=cls,
        flag="_zentorch_wna16_moe_patched",
        register_fn=_register_wna16_method_patch,
        success_log=(
            f"[zentorch] Patched {cls.__name__}: per-expert biases + "
            "zentorch DA8W4 MoE"
            if cls is not None
            else ""
        ),
        fail_log="[zentorch] WNA16 MoE method patch FAILED",
        missing_log=(
            f"[zentorch] {_METHOD_CLASS} not found in {_METHOD_MODULE}; "
            "WNA16 MoE method patch skipped"
        ),
    )


# --------------------------------------------------------------------------- #
# Deferred application
# --------------------------------------------------------------------------- #

_TARGETS = {
    _ORACLE_MODULE: _apply_oracle_patch_to_module,
    _METHOD_MODULE: _apply_wna16_method_patch_to_module,
}


def _apply_wna16_moe_patch_impl() -> bool:
    """Schedule the WNA16 MoE patches on first import of their target modules."""
    return schedule_module_patches(_TARGETS)
