# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Out-of-tree W8A8 INT8 fused-MoE via zentorch (no vLLM changes).

Mirrors the in-tree modular structure so the FusedMoE runner (and torch.compile)
drive it through the standard modular-kernel path. Monkey-patches
``CompressedTensorsW8A8Int8MoEMethod`` at import time to:
* allocate per-expert biases in ``create_weights`` (gpt-oss),
* reorder w13 to the interleaved SwiGLU-OAI layout (Zen-gated) and build a
  ``FusedMoEKernel`` backed by a plugin-defined ``CPUInt8Experts`` in
  ``process_weights_after_loading``,
* pass biases through ``get_fused_moe_quant_config``.

``CPUInt8Experts`` runs the FFN via ``zentorch_fused_moe`` (>=2 active experts)
with a per-expert ``zentorch_dynamic_qlinear`` fallback (<2), wrapped in an
opaque custom op for torch.compile.

Only installed on Zen CPU (enforced by the plugin's register()) and when the
zentorch MoE ops are present; otherwise vLLM's own int8 MoE backend is used.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import torch
import torch.nn.functional as F

from zentorch._logging import get_logger

logger = get_logger(__name__)

_TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
_TARGET_CLASS = "CompressedTensorsW8A8Int8MoEMethod"


# --------------------------------------------------------------------------- #
# Native activation fallbacks (per-expert loop path).
# --------------------------------------------------------------------------- #
def _silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


def _swigluoai_and_mul_native(
    x: torch.Tensor,
    *,
    alpha: float = 1.702,  # gpt-oss defaults; match ZenDNN swiglu_oai_mul.
    limit: float = 7.0,
) -> torch.Tensor:
    # Interleaved SwiGLU-OAI (gate=x[..., 0::2], up=x[..., 1::2]); w13 is
    # reordered to interleaved in process_weights_after_loading so this
    # fallback and the fused op share the layout.
    gate = x[..., 0::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * gate * torch.sigmoid(alpha * gate)


_ACT_FN = {
    "silu": _silu_and_mul_native,
    "gelu": _gelu_and_mul_native,
    "swigluoai": _swigluoai_and_mul_native,
}


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
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    E, _, K = w1.shape
    M, top_k = topk_ids.shape

    # Map global -> local expert ids; -1 marks remote-rank tokens.
    local_topk_ids_raw = (
        expert_map[topk_ids.to(torch.long)] if expert_map is not None else topk_ids
    )

    flat_local = local_topk_ids_raw.reshape(-1)
    valid_local = flat_local[flat_local >= 0]
    num_active_local = (
        int(valid_local.unique().numel()) if valid_local.numel() > 0 else 0
    )

    # Fast path: >=2 active local experts -> single fused-MoE call.
    if num_active_local >= 2:
        if expert_map is not None and (local_topk_ids_raw < 0).any():
            invalid = local_topk_ids_raw < 0
            local_topk_ids_fast = local_topk_ids_raw.masked_fill(invalid, 0)
            topk_weights_fast = topk_weights.masked_fill(invalid, 0.0)
        else:
            local_topk_ids_fast = local_topk_ids_raw
            topk_weights_fast = topk_weights

        output.zero_()
        input_for_op = hidden_states
        if apply_router_weight_on_input:
            if top_k != 1:
                raise NotImplementedError(
                    "zen int8 MoE: apply_router_weight_on_input=True is only "
                    f"supported for top_k=1 (got top_k={top_k})."
                )
            input_for_op = hidden_states.mul(topk_weights_fast.to(hidden_states.dtype))

        torch.ops.zentorch.zentorch_fused_moe(
            output,
            input_for_op,
            w1,
            w2,
            w13_bias,
            w2_bias,
            topk_weights_fast.to(torch.float32).contiguous(),
            local_topk_ids_fast.to(torch.int32).contiguous(),
            apply_router_weight_on_input,  # skip_weighted
            activation,
            w13_scale,
            w2_scale,
        )
        return

    # Fallback: per-expert zentorch_dynamic_qlinear loop (<2 active experts,
    # e.g. M==1/top_k==1 decode). Handles arbitrary (M, top_k) shapes.
    if num_active_local == 0:
        output.zero_()
        return

    act_fn = _ACT_FN[activation]
    local_topk_ids = local_topk_ids_raw

    input_for_loop = hidden_states
    if apply_router_weight_on_input:
        if top_k != 1:
            raise NotImplementedError(
                "zen int8 MoE: apply_router_weight_on_input=True is only "
                f"supported for top_k=1 (got top_k={top_k})."
            )
        input_for_loop = hidden_states.mul(topk_weights.to(hidden_states.dtype))

    flat_ids = local_topk_ids.reshape(-1)
    sort_idx = torch.argsort(flat_ids.to(torch.int64), stable=True)
    sorted_local_ids = flat_ids[sort_idx]
    token_src = sort_idx // top_k
    sorted_tokens = input_for_loop.index_select(0, token_src)

    # Exclude remote tokens (-1, produced by expert_map under EP) from the
    # per-expert counts. Using clamp(min=0) would miscount every -1 as expert 0,
    # inflating counts_cpu[0] by num_remote and corrupting the cursor slicing
    # below (the -1 block is skipped separately via `cursor = num_remote`).
    num_remote = int((sorted_local_ids == -1).sum().item())
    per_expert_counts = torch.bincount(
        sorted_local_ids[sorted_local_ids >= 0], minlength=E
    )

    sorted_out = sorted_tokens.new_zeros(sorted_tokens.size(0), K)
    counts_cpu = per_expert_counts.cpu().tolist()
    cursor = num_remote
    for e in range(E):
        n_e = int(counts_cpu[e])
        if n_e == 0:
            continue
        tokens_e = sorted_tokens[cursor : cursor + n_e]
        bias13_e = None if w13_bias is None else w13_bias[e]
        gate_up = torch.ops.zentorch.zentorch_dynamic_qlinear(
            tokens_e, w1[e], w13_scale[e], bias13_e
        )
        act_out = act_fn(gate_up)
        bias2_e = None if w2_bias is None else w2_bias[e]
        sorted_out[cursor : cursor + n_e] = torch.ops.zentorch.zentorch_dynamic_qlinear(
            act_out, w2[e], w2_scale[e], bias2_e
        )
        cursor += n_e

    unsorted_out = torch.empty_like(sorted_out)
    unsorted_out.index_copy_(0, sort_idx, sorted_out)
    per_topk = unsorted_out.view(M, top_k, K)
    if not apply_router_weight_on_input:
        per_topk = per_topk * topk_weights.view(M, top_k, 1).to(per_topk.dtype)
    torch.sum(per_topk, dim=1, out=output)


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
    expert_map,
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
    )
    from vllm.model_executor.layers.fused_moe.oracle.int8 import (
        make_int8_moe_kernel,
        make_int8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        QuantKey,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )
    from vllm.model_executor.utils import set_weight_attrs

    class CPUInt8Experts(mk.FusedMoEExpertsModular):
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
            # zentorch_dynamic_qlinear quantizes activations itself.
            return True

        @staticmethod
        def _supports_current_device() -> bool:
            return has_zentorch_op(["zentorch_fused_moe", "zentorch_dynamic_qlinear"])

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
            return activation.value in _ACT_FN

        @staticmethod
        def _supports_parallel_config(
            moe_parallel_config: FusedMoEParallelConfig,
        ) -> bool:
            return True

        def supports_expert_map(self) -> bool:
            return True

        def workspace_shapes(
            self,
            M: int,
            N: int,
            K: int,
            topk: int,
            global_num_experts: int,
            local_num_experts: int,
            expert_tokens_meta,
            activation: MoEActivation,
        ):
            # Framework uses these only for activation-chunking heuristics; we
            # allocate our own scratch inside the op.
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            workspace13 = (M, topk, max(activation_out_dim, K))
            workspace2 = (M, topk, max(N, K))
            output = (M, K)
            return (workspace13, workspace2, output)

        def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
            return TopKWeightAndReduceNoOP()

        def apply(
            self,
            output: torch.Tensor,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: MoEActivation,
            global_num_experts: int,
            expert_map: torch.Tensor | None,
            a1q_scale: torch.Tensor | None,
            a2_scale: torch.Tensor | None,
            workspace13: torch.Tensor,
            workspace2: torch.Tensor,
            expert_tokens_meta,
            apply_router_weight_on_input: bool,
        ) -> None:
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
                expert_map,
                apply_router_weight_on_input,
            )

    method_cls = getattr(mod, _TARGET_CLASS)
    orig_init = method_cls.__init__
    orig_create_weights = method_cls.create_weights

    def _zen_init(self, *args, **kwargs):
        # Vanilla __init__'s select_int8_moe_backend() raises on CPU; neutralize
        # it, then install our own experts class for the modular kernel.
        saved = getattr(mod, "select_int8_moe_backend", None)
        if saved is not None:
            mod.select_int8_moe_backend = lambda *a, **k: (None, None)
        try:
            orig_init(self, *args, **kwargs)
        finally:
            if saved is not None:
                mod.select_int8_moe_backend = saved
        self.experts_cls = CPUInt8Experts

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
        # Per-expert biases (e.g. gpt-oss) are not allocated by vanilla 0.23
        # create_weights; add them so the per-expert loader has a target.
        if getattr(self.moe, "has_bias", False) and (
            getattr(layer, "w13_bias", None) is None
        ):
            w13_num_shards = 2 if self.moe.is_act_and_mul else 1
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    dtype=bias_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=bias_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _zen_get_fused_moe_quant_config(self, layer) -> "FusedMoEQuantConfig":
        # Pass per-expert biases through so CPUInt8Experts can add them
        # (vanilla's config omits w1_bias/w2_bias).
        return make_int8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=True,
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

    def _zen_process_weights_after_loading(self, layer) -> None:
        _maybe_permute_swigluoai(layer)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_int8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=CPUInt8Experts,
            routing_tables=layer._expert_routing_tables(),
        )
        logger.info(
            "[zentorch] W8A8 int8 MoE kernel built via OOT patch "
            "(experts=%d, has_bias=%s)",
            layer.w13_weight.shape[0],
            getattr(layer, "w13_bias", None) is not None,
        )

    method_cls.__init__ = _zen_init
    method_cls.create_weights = _zen_create_weights
    method_cls.get_fused_moe_quant_config = _zen_get_fused_moe_quant_config
    method_cls.process_weights_after_loading = _zen_process_weights_after_loading


def _apply_int8_moe_patch_to_module(mod) -> bool:
    try:
        cls = getattr(mod, _TARGET_CLASS, None)
        if cls is None:
            logger.warning(
                "[zentorch] %s not found in %s; int8 MoE patch skipped",
                _TARGET_CLASS,
                _TARGET_MODULE,
            )
            return False
        if getattr(cls, "_zentorch_int8_moe_patched", False):
            return True
        # Skip patching if this zentorch build lacks the MoE ops.
        zt = getattr(torch.ops, "zentorch", None)
        if zt is None or not all(
            hasattr(zt, op) for op in ("zentorch_fused_moe", "zentorch_dynamic_qlinear")
        ):
            logger.warning(
                "[zentorch] zentorch_fused_moe/zentorch_dynamic_qlinear not "
                "available; leaving vLLM's int8 MoE backend unpatched."
            )
            return False
        _register_int8_moe_patches(mod)
        cls._zentorch_int8_moe_patched = True
        logger.info(
            "[zentorch] Patched %s: modular zentorch W8A8 int8 MoE",
            _TARGET_CLASS,
        )
        return True
    except Exception:
        logger.warning("[zentorch] int8 MoE patch FAILED", exc_info=True)
        return False


_HOOK_INSTALLED = False


class _Int8MoeImportHook:
    """Defer patch until the target vLLM module loads (avoids import cycles)."""

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
            _apply_int8_moe_patch_to_module(module)

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_int8_moe_patch_impl() -> bool:
    """Schedule the W8A8 int8 MoE patch on first import of the target module."""
    global _HOOK_INSTALLED

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        return _apply_int8_moe_patch_to_module(mod)

    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _Int8MoeImportHook())
        _HOOK_INSTALLED = True
    return True
